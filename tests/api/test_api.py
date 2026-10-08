from pathlib import Path
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from medellin_rent import __version__
from medellin_rent.api.main import ComunaName, app, get_model
from medellin_rent.features.build import COMUNA_NAMES
from medellin_rent.inference.predict import load_model

client = TestClient(app)
LISTING = {"comuna": "EL POBLADO", "estrato": 5, "area_m2": 80, "bedrooms": 2, "parking": "yes"}


@pytest.fixture
def with_model(saved_model: Path):  # type: ignore[no-untyped-def]
    loaded = load_model(saved_model)
    app.dependency_overrides[get_model] = lambda: loaded
    yield
    app.dependency_overrides.clear()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_comuna_names_match_the_official_list() -> None:
    assert list(get_args(ComunaName)) == list(COMUNA_NAMES.values())


@pytest.mark.usefixtures("with_model")
def test_predict_returns_rent_and_interval() -> None:
    response = client.post("/predict", json=LISTING)
    assert response.status_code == 200
    body = response.json()
    assert body["interval_low_cop"] < body["predicted_rent_cop"] < body["interval_high_cop"]
    assert body["interval_level"] == 0.8
    assert body["model"] == "lightgbm"
    assert "2020" in body["prices"]


@pytest.mark.usefixtures("with_model")
def test_predict_with_only_the_comuna() -> None:
    response = client.post("/predict", json={"comuna": "LAURELES"})
    assert response.status_code == 200
    assert response.json()["predicted_rent_cop"] > 0


@pytest.mark.usefixtures("with_model")
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"comuna": "NARNIA"},
        {**LISTING, "estrato": 7},
        {**LISTING, "area_m2": 5},
        {**LISTING, "parking": "maybe"},
        {**LISTING, "lat": 4.6},
        {**LISTING, "bedrooms": -1},
    ],
)
def test_predict_rejects_invalid_requests(payload: dict[str, object]) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_without_a_model_is_unavailable() -> None:
    app.state.model = None
    response = client.post("/predict", json=LISTING)
    assert response.status_code == 503
    assert "make train" in response.json()["detail"]


def test_openapi_documents_the_fields() -> None:
    schema = client.get("/openapi.json").json()["components"]["schemas"]["PredictRequest"]
    assert schema["required"] == ["comuna"]
    assert "description" in schema["properties"]["area_m2"]
    assert len(schema["properties"]["comuna"]["enum"]) == 21


@pytest.mark.usefixtures("with_model")
def test_api_and_batch_inference_agree(saved_model: Path) -> None:
    import pandas as pd

    from medellin_rent.inference.predict import predict

    model, metadata = load_model(saved_model)
    batch = predict(model, metadata, pd.DataFrame([LISTING]))["predicted_rent_cop"].iloc[0]
    assert client.post("/predict", json=LISTING).json()["predicted_rent_cop"] == batch


def _config_with_models(models_dir: Path):  # type: ignore[no-untyped-def]
    from medellin_rent.utils.config import load_config

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "conf" / "base.yaml")
    return config.model_copy(
        update={"paths": config.paths.model_copy(update={"models": models_dir})}
    )


def test_startup_loads_the_model(saved_model: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "medellin_rent.api.main.get_config", lambda: _config_with_models(saved_model)
    )
    with TestClient(app) as started:
        assert started.post("/predict", json=LISTING).status_code == 200


def test_startup_without_a_model_keeps_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("medellin_rent.api.main.get_config", lambda: _config_with_models(tmp_path))
    with TestClient(app) as started:
        assert started.get("/health").status_code == 200
        assert started.post("/predict", json=LISTING).status_code == 503
