import numpy as np
import pandas as pd
import pytest

from medellin_rent.model.baseline import ComunaMedianBaseline
from medellin_rent.model.evaluate import cross_validate, folds, metrics, predict_rent, summarize


def test_metrics() -> None:
    result = metrics(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert result["mae"] == 15.0
    assert result["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert result["mape"] == pytest.approx((10 + 10) / 2)


def test_folds_cover_every_row_once_and_are_reproducible() -> None:
    splits = folds(23, 5, seed=42)
    validation = np.concatenate([val for _, val in splits])
    assert sorted(validation.tolist()) == list(range(23))
    assert all(len(set(fit) & set(val)) == 0 for fit, val in splits)
    assert [v.tolist() for _, v in folds(23, 5, seed=42)] == [v.tolist() for _, v in splits]


def test_cross_validate_and_summarize(model_input: pd.DataFrame) -> None:
    cv = cross_validate(ComunaMedianBaseline, model_input, folds(len(model_input), 4, seed=1))
    assert cv["fold"].tolist() == [0, 1, 2, 3]
    assert cv["n"].sum() == len(model_input)
    summary = summarize(cv)
    assert summary["mae_mean"] == pytest.approx(cv["mae"].mean())
    assert set(summary) == {f"{m}_{s}" for m in ("mae", "rmse", "mape") for s in ("mean", "std")}


def test_predict_rent_returns_cop(model_input: pd.DataFrame) -> None:
    model = ComunaMedianBaseline().fit(model_input, model_input["log_rent"])
    rent = predict_rent(model, model_input)
    assert rent.min() > 1e5  # COP, not log
