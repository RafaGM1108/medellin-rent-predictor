"""Inference pipeline: saved model + new listings -> predictions (``07_model_output``)."""

from pathlib import Path

import pandas as pd

from medellin_rent.inference.predict import load_model, predict
from medellin_rent.utils.config import Config, get_config
from medellin_rent.utils.log import get_logger

PREDICTIONS_FILE = "predictions.csv"


def run(config: Config | None = None, listings_path: Path | None = None) -> Path:
    """Predict the rent of every listing in a CSV and write the predictions.

    Args:
        config: Project configuration. Loaded from ``conf/base.yaml`` if omitted.
        listings_path: CSV in the ``NewListingSchema`` format; defaults to
            ``paths.new_listings``.

    Returns:
        Path to the predictions CSV: the input columns plus the predicted rent and interval.
    """
    config = config or get_config()
    logger = get_logger(__name__, config.logging.level)
    listings = pd.read_csv(listings_path or config.paths.new_listings)
    model, metadata = load_model(config.paths.models)
    predictions = pd.concat([listings, predict(model, metadata, listings)], axis=1)

    config.paths.model_output.mkdir(parents=True, exist_ok=True)
    out = config.paths.model_output / PREDICTIONS_FILE
    predictions.to_csv(out, index=False)
    logger.info("Predicted %d listings with %s: %s", len(listings), metadata["model"], out)
    return out
