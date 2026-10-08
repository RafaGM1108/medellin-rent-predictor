"""Feature pipeline: raw data -> model input."""

from pathlib import Path

from medellin_rent.data.listings import RAW_FILE, load_listings
from medellin_rent.data.parse import parse_listings
from medellin_rent.data.schemas import IntermediateSchema
from medellin_rent.utils.config import Config, get_config
from medellin_rent.utils.log import get_logger

INTERMEDIATE_FILE = "listings.parquet"


def run(config: Config | None = None) -> Path:
    """Run the feature pipeline.

    Steps so far: raw listings -> typed intermediate table (``02_intermediate``).

    Args:
        config: Project configuration. Loaded from ``conf/base.yaml`` if omitted.

    Returns:
        Path to the intermediate listings table.
    """
    config = config or get_config()
    logger = get_logger(__name__, config.logging.level)

    listings = parse_listings(load_listings(config.paths.raw_listings / RAW_FILE))
    IntermediateSchema.validate(listings, lazy=True)
    config.paths.intermediate.mkdir(parents=True, exist_ok=True)
    out = config.paths.intermediate / INTERMEDIATE_FILE
    listings.to_parquet(out, index=False)
    logger.info("Wrote %d listings to %s", len(listings), out)
    # TODO: primary (#14, #15), features (#22, #23) and model input steps.
    return out
