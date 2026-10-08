import pandas as pd

from medellin_rent.analysis import eda, hypothesis, price_m2, rent_map
from medellin_rent.geo.boundaries import load_layer
from medellin_rent.pipelines.feature_pipeline.pipeline import LISTINGS_FILE
from medellin_rent.utils.config import get_config
from medellin_rent.utils.log import get_logger

if __name__ == "__main__":
    config = get_config()
    paths = config.paths
    primary = pd.read_parquet(paths.primary / LISTINGS_FILE)
    for path in [
        *eda.run(primary, paths.reporting),
        *price_m2.run(primary, paths.reporting),
        *hypothesis.run(primary, paths.reporting, seed=config.project.random_seed),
        *rent_map.run(
            primary, load_layer("comunas", paths.raw_geo / "comunas.geojson"), paths.reporting
        ),
    ]:
        get_logger("medellin_rent.analysis").info("Wrote %s", path)
