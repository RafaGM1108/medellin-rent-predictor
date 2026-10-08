import pandas as pd

from medellin_rent.analysis import eda, price_m2
from medellin_rent.pipelines.feature_pipeline.pipeline import LISTINGS_FILE
from medellin_rent.utils.config import get_config
from medellin_rent.utils.log import get_logger

if __name__ == "__main__":
    paths = get_config().paths
    primary = pd.read_parquet(paths.primary / LISTINGS_FILE)
    for path in [*eda.run(primary, paths.reporting), *price_m2.run(primary, paths.reporting)]:
        get_logger("medellin_rent.analysis").info("Wrote %s", path)
