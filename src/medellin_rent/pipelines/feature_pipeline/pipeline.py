"""Feature pipeline: raw data -> model input."""

from pathlib import Path

import pandas as pd

from medellin_rent.data.clean import clean_listings
from medellin_rent.data.listings import RAW_FILE, load_listings
from medellin_rent.data.parse import parse_listings
from medellin_rent.data.schemas import IntermediateSchema, PrimarySchema
from medellin_rent.features.build import build_features
from medellin_rent.features.model_input import make_model_input
from medellin_rent.geo.boundaries import load_layer
from medellin_rent.geo.mapping import assign_estrato, assign_location
from medellin_rent.utils.config import Config, get_config
from medellin_rent.utils.log import get_logger

LISTINGS_FILE = "listings.parquet"
FEATURES_FILE = "features.parquet"
TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"
GEO_LAYERS = ("barrios", "estrato")


def run(config: Config | None = None) -> Path:
    """Run the feature pipeline.

    Steps: raw listings -> typed intermediate table (``02_intermediate``) -> located, cleaned
    primary table (``03_primary``) -> features (``04_feature``) -> train/test model input
    (``05_model_input``). Intermediate and primary tables are validated before they are
    written; the split uses ``params.test_size`` and ``project.random_seed`` from the config.

    Args:
        config: Project configuration. Loaded from ``conf/base.yaml`` if omitted.

    Returns:
        Path to the model input directory.
    """
    config = config or get_config()
    logger = get_logger(__name__, config.logging.level)
    paths = config.paths

    geo = {name: paths.raw_geo / f"{name}.geojson" for name in GEO_LAYERS}
    missing = [str(p) for p in geo.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{missing} not found: run `make geo` first")

    intermediate = parse_listings(load_listings(paths.raw_listings / RAW_FILE))
    IntermediateSchema.validate(intermediate, lazy=True)
    _write(intermediate, paths.intermediate / LISTINGS_FILE)

    located = assign_location(intermediate, load_layer("barrios", geo["barrios"]))
    located = assign_estrato(located, load_layer("estrato", geo["estrato"]))
    primary = clean_listings(located)
    PrimarySchema.validate(primary, lazy=True)
    _write(primary, paths.primary / LISTINGS_FILE)
    logger.info(
        "Primary: %d listings, %d with a comuna", len(primary), primary["comuna_code"].notna().sum()
    )

    features = build_features(primary)
    _write(features, paths.feature / FEATURES_FILE)
    train, test = make_model_input(
        features,
        primary,
        test_size=float(config.params["test_size"]),
        seed=config.project.random_seed,
    )
    _write(train, paths.model_input / TRAIN_FILE)
    _write(test, paths.model_input / TEST_FILE)
    logger.info("Model input: %d train, %d test listings", len(train), len(test))
    return paths.model_input


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
