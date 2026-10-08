from pathlib import Path

import pandas as pd
import pandera.errors as pe
import pytest

from medellin_rent.features.build import FEATURES
from medellin_rent.inference.predict import load_model, predict, to_features

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "01_raw" / "new_listings_sample.csv"


def _listing(**overrides: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "comuna": "EL POBLADO",
        "estrato": 5,
        "area_m2": 80.0,
        "bedrooms": 2,
        "bathrooms": 2,
        "parking": "yes",
        "lat": None,
        "lon": None,
    }
    return pd.DataFrame([row | overrides])


def test_to_features_matches_training_format() -> None:
    features = to_features(_listing())
    assert features.columns.tolist() == ["listing_id", *FEATURES]
    row = features.iloc[0]
    assert row["comuna_code"] == "14"
    assert row["estrato"] == "5"
    assert row["parking"] == "yes"
    assert row["listing_quarter"] == "2021Q3"  # predictions at the latest data prices
    assert not row["has_pool"]  # missing amenity columns default to False
    assert not row["area_missing"]


def test_to_features_handles_unknown_values() -> None:
    features = to_features(_listing(estrato=None, area_m2=None, parking="unknown"))
    row = features.iloc[0]
    assert row["estrato"] == "unknown"
    assert row["area_missing"]
    assert row["parking"] == "unknown"


@pytest.mark.parametrize(
    ("column", "value"),
    [("comuna", "NARNIA"), ("estrato", 7), ("area_m2", 5.0), ("parking", "maybe"), ("lat", 4.6)],
)
def test_to_features_rejects_invalid_listings(column: str, value: object) -> None:
    with pytest.raises(pe.SchemaErrors, match=column):
        to_features(_listing(**{column: value}))


def test_the_sample_file_is_valid() -> None:
    assert len(to_features(pd.read_csv(SAMPLE))) == 5


def test_predict_with_interval(saved_model: Path) -> None:
    model, metadata = load_model(saved_model)
    listings = pd.concat([_listing(), _listing(comuna="LAURELES", estrato=4)], ignore_index=True)
    listings.index = [10, 20]  # keeps the caller's index
    out = predict(model, metadata, listings)
    assert out.index.tolist() == [10, 20]
    assert (out["interval_low_cop"] < out["predicted_rent_cop"]).all()
    assert (out["predicted_rent_cop"] < out["interval_high_cop"]).all()
    assert (out["predicted_rent_cop"] % 1000 == 0).all()  # rounded to thousands of COP


def test_predict_without_interval(saved_model: Path) -> None:
    model, _ = load_model(saved_model)
    out = predict(model, {"model": "x"}, _listing())
    assert out.columns.tolist() == ["predicted_rent_cop"]


def test_only_the_comuna_is_required() -> None:
    row = to_features(pd.DataFrame([{"comuna": "BELEN"}])).iloc[0]
    assert row["comuna_code"] == "16"
    assert row["estrato"] == "unknown"
    assert row["parking"] == "unknown"
    assert row["area_missing"]


def test_predict_at_current_prices(saved_model: Path) -> None:
    model, metadata = load_model(saved_model)
    out = predict(model, metadata, _listing(), {"factor": 2.0}).iloc[0]
    assert out["predicted_rent_cop_current"] == pytest.approx(
        out["predicted_rent_cop"] * 2, abs=1000
    )
    assert out["interval_low_cop_current"] < out["predicted_rent_cop_current"]
    assert out["predicted_rent_cop_current"] < out["interval_high_cop_current"]


def test_download_model_fetches_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    from medellin_rent.inference import predict as module

    requested: list[str] = []

    class _Response(io.BytesIO):
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            self.close()

    def fake_urlopen(request: object, timeout: float) -> _Response:
        url = request.full_url  # type: ignore[attr-defined]
        requested.append(url)
        return _Response(url.encode())

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    models = tmp_path / "models"
    models.mkdir()
    (models / "model_metadata.json").write_text("{}")  # already there: not downloaded
    base = "https://github.com/RafaGM1108/medellin-rent-predictor/releases/download/v1.0.0"

    module.download_model(models, base)

    assert requested == [f"{base}/model.joblib"]
    assert (models / "model.joblib").read_bytes() == f"{base}/model.joblib".encode()
    assert (models / "model_metadata.json").read_text() == "{}"
    assert not list(models.glob("*.part"))


def test_download_model_only_from_github(tmp_path: Path) -> None:
    from medellin_rent.inference.predict import download_model

    with pytest.raises(ValueError, match="GitHub"):
        download_model(tmp_path, "http://example.com/model")
