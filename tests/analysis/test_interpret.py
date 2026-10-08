from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medellin_rent.analysis import interpret
from medellin_rent.features.build import FEATURES
from medellin_rent.model.baseline import ComunaMedianBaseline
from medellin_rent.model.gradient_boosting import make_lightgbm

PARAMS = {"num_leaves": 7, "learning_rate": 0.1, "n_estimators": 50, "min_child_samples": 5}


def _model(model_input: pd.DataFrame):  # type: ignore[no-untyped-def]
    return make_lightgbm(PARAMS, seed=0).fit(model_input[FEATURES], model_input["log_rent"])


def test_shap_values_add_up_to_the_prediction(model_input: pd.DataFrame) -> None:
    model = _model(model_input)
    values, inputs = interpret.shap_values(model, model_input)
    assert np.allclose(values.sum(axis=1), model.predict(model_input[FEATURES]))
    assert "barrio_name" not in values.columns
    assert list(values.columns[:-1]) == list(inputs.columns)


def test_importance_and_levels(model_input: pd.DataFrame) -> None:
    values, inputs = interpret.shap_values(_model(model_input), model_input)
    table = interpret.importance(values)
    assert table["mean_abs_shap"].is_monotonic_decreasing
    assert "base_value" not in set(table["feature"])
    levels = interpret.by_level(values, inputs, "comuna_code")
    assert set(levels["level"]) == {"11", "14"}
    assert levels["n"].sum() == len(model_input)
    assert levels["mean_effect_pct"].is_monotonic_decreasing


def test_as_percent() -> None:
    assert interpret.as_percent(0.0) == 0.0
    assert interpret.as_percent(np.log(1.5)) == pytest.approx(50.0)


def test_rejects_models_without_tree_shap(model_input: pd.DataFrame) -> None:
    baseline = ComunaMedianBaseline().fit(model_input[FEATURES], model_input["log_rent"])
    with pytest.raises(TypeError, match="LightGBM"):
        interpret.shap_values(baseline, model_input)


def test_run_writes_tables_and_figures(tmp_path: Path, model_input: pd.DataFrame) -> None:
    paths = interpret.run(_model(model_input), model_input, tmp_path)
    assert len(paths) == 6
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
