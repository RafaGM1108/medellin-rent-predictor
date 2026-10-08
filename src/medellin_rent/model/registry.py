"""The models compared by the training pipeline, by name."""

from functools import partial

from medellin_rent.model.baseline import ComunaMedianBaseline
from medellin_rent.model.evaluate import ModelFactory
from medellin_rent.model.linear_and_forest import make_random_forest, make_ridge


def get_models(seed: int) -> dict[str, ModelFactory]:
    """Factories of every model, seeded with the project seed."""
    return {
        "baseline": ComunaMedianBaseline,
        "ridge": make_ridge,
        "random_forest": partial(make_random_forest, seed),
    }
