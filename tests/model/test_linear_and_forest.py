import pickle

import numpy as np
import pandas as pd
import pytest

from medellin_rent.features.build import FEATURES
from medellin_rent.model.baseline import ComunaMedianBaseline
from medellin_rent.model.evaluate import cross_validate, folds
from medellin_rent.model.linear_and_forest import (
    AreaByBedroomsImputer,
    make_random_forest,
    make_ridge,
)
from medellin_rent.model.registry import get_models


def test_area_imputer_uses_median_area_by_bedrooms() -> None:
    train = pd.DataFrame({"bedrooms": [1.0, 1.0, 3.0, 3.0], "area_m2": [40.0, 50.0, 90.0, np.nan]})
    imputer = AreaByBedroomsImputer().fit(train)
    new = pd.DataFrame({"bedrooms": [1.0, 3.0, 5.0, np.nan], "area_m2": [np.nan] * 4})
    assert imputer.transform(new)["area_m2"].tolist() == [45.0, 90.0, 50.0, 50.0]
    assert new["area_m2"].isna().all()  # input not modified


@pytest.mark.parametrize("factory", [make_ridge, lambda: make_random_forest(seed=0)])
def test_models_fit_predict_and_pickle(factory, model_input: pd.DataFrame) -> None:  # type: ignore[no-untyped-def]
    model = factory().fit(model_input[FEATURES], model_input["log_rent"])
    prediction = model.predict(model_input[FEATURES])
    assert np.isfinite(prediction).all()
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(restored.predict(model_input[FEATURES]), prediction)


def test_models_beat_the_baseline_on_synthetic_data(model_input: pd.DataFrame) -> None:
    splits = folds(len(model_input), 4, seed=0)
    baseline = cross_validate(ComunaMedianBaseline, model_input, splits)["mae"].mean()
    ridge = cross_validate(make_ridge, model_input, splits)["mae"].mean()
    forest = cross_validate(lambda: make_random_forest(0), model_input, splits)["mae"].mean()
    assert ridge < baseline
    assert forest < baseline


def test_registry_names() -> None:
    assert list(get_models(seed=1)) == ["baseline", "ridge", "random_forest"]
