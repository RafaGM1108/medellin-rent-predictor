import pickle

import numpy as np
import pandas as pd

from medellin_rent.features.build import FEATURES
from medellin_rent.model.gradient_boosting import SEARCH_SPACE, make_lightgbm, tune
from medellin_rent.model.registry import NAMES, get_models

PARAMS = {"num_leaves": 7, "learning_rate": 0.1, "n_estimators": 50, "min_child_samples": 5}


def test_lightgbm_variants_fit_predict_and_pickle(model_input: pd.DataFrame) -> None:
    for barrio in (False, True):
        model = make_lightgbm(PARAMS, seed=0, barrio=barrio)
        model.fit(model_input[FEATURES], model_input["log_rent"])
        prediction = model.predict(model_input[FEATURES])
        assert np.isfinite(prediction).all()
        restored = pickle.loads(pickle.dumps(model))
        assert np.allclose(restored.predict(model_input[FEATURES]), prediction)


def test_barrio_only_used_by_the_barrio_variant(model_input: pd.DataFrame) -> None:
    plain = make_lightgbm(PARAMS, seed=0).fit(model_input[FEATURES], model_input["log_rent"])
    with_barrio = make_lightgbm(PARAMS, seed=0, barrio=True)
    with_barrio.fit(model_input[FEATURES], model_input["log_rent"])
    assert "barrio_name" not in plain.named_steps["model"].feature_name_
    assert "barrio_name" in with_barrio.named_steps["model"].feature_name_


def test_tune_returns_best_configuration_from_the_space(model_input: pd.DataFrame) -> None:
    best, results = tune(model_input, seed=0, n_iter=3, inner_folds=2)
    assert len(results) == 3
    assert results["mae"].is_monotonic_increasing
    assert set(best) == set(SEARCH_SPACE)
    assert all(best[k] in SEARCH_SPACE[k] for k in SEARCH_SPACE)
    assert best["num_leaves"] == results.loc[0, "num_leaves"]
    assert isinstance(best["num_leaves"], int)  # plain Python types, JSON-serializable


def test_registry_includes_lightgbm_variants() -> None:
    assert list(get_models(seed=0)) == NAMES
