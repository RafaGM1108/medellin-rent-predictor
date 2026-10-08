"""The models compared by the training pipeline, by name."""

from functools import partial
from typing import Any

from medellin_rent.model.baseline import ComunaMedianBaseline
from medellin_rent.model.evaluate import ModelFactory
from medellin_rent.model.gradient_boosting import DEFAULT_PARAMS, make_lightgbm
from medellin_rent.model.linear_and_forest import make_random_forest, make_ridge

NAMES = ["baseline", "ridge", "random_forest", "lightgbm", "lightgbm_barrio"]


def get_models(seed: int, lightgbm_params: dict[str, Any] | None = None) -> dict[str, ModelFactory]:
    """Factories of every model, seeded with the project seed.

    Args:
        seed: Project random seed.
        lightgbm_params: Tuned LightGBM hyperparameters (defaults if omitted).
    """
    params = lightgbm_params or DEFAULT_PARAMS
    return {
        "baseline": ComunaMedianBaseline,
        "ridge": make_ridge,
        "random_forest": partial(make_random_forest, seed),
        "lightgbm": partial(make_lightgbm, params, seed),
        "lightgbm_barrio": partial(make_lightgbm, params, seed, barrio=True),
    }
