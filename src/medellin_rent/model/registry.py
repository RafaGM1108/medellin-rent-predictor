"""The models compared by the training pipeline, by name."""

from medellin_rent.model.baseline import ComunaMedianBaseline
from medellin_rent.model.evaluate import ModelFactory

MODELS: dict[str, ModelFactory] = {
    "baseline": ComunaMedianBaseline,
}
