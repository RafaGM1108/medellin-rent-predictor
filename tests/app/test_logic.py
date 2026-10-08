from pathlib import Path

import pandas as pd
import pytest

from medellin_rent.app.logic import LABELS, explain, verdict
from medellin_rent.features.build import FEATURES
from medellin_rent.inference.predict import load_model


@pytest.mark.parametrize(
    ("listed", "label"),
    [(700_000, "good deal"), (800_000, "fair"), (1_000_000, "fair"), (1_300_000, "expensive")],
)
def test_verdict(listed: float, label: str) -> None:
    result = verdict(listed, predicted=1_000_000, low=800_000, high=1_250_000)
    assert result.label == label
    assert result.difference_pct == pytest.approx(100 * (listed / 1_000_000 - 1))


def test_explain_one_listing(saved_model: Path) -> None:
    model, _ = load_model(saved_model)
    listing = pd.DataFrame([{"comuna": "EL POBLADO", "estrato": 6, "area_m2": 90.0}])
    table = explain(model, listing, top=4)
    assert len(table) == 4
    assert table["effect_pct"].abs().is_monotonic_decreasing
    assert set(table["feature"]) <= set(LABELS.values())
    shown = dict(zip(table["feature"], table["value"], strict=True))
    if "Comuna" in shown:
        assert shown["Comuna"] == "El Poblado"
    if "Area" in shown:
        assert shown["Area"] == "90 m²"


def test_every_model_input_has_a_label() -> None:
    assert set(FEATURES) - {"barrio_name"} <= set(LABELS)


def test_values_are_readable() -> None:
    import numpy as np

    from medellin_rent.app.logic import _value

    assert _value("has_pool", np.bool_(False)) == "no"
    assert _value("has_pool", True) == "yes"
    assert _value("bathrooms", np.float64(2.0)) == "2"
    assert _value("lat", float("nan")) == "not given"
    assert _value("estrato", "unknown") == "unknown"
