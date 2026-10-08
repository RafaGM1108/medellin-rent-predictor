import shutil
from pathlib import Path

import pandas as pd

from medellin_rent.data.listings import RAW_FILE
from medellin_rent.pipelines.feature_pipeline.pipeline import run
from medellin_rent.utils.config import PathsConfig, load_config

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "data" / "fixtures" / "co_properties.csv"


def test_run_writes_intermediate_parquet(tmp_path: Path) -> None:
    config = load_config(ROOT / "conf" / "base.yaml")
    paths = {name: tmp_path / name for name in PathsConfig.model_fields}
    config = config.model_copy(update={"paths": PathsConfig(**paths)})
    paths["raw_listings"].mkdir()
    shutil.copy(FIXTURE, paths["raw_listings"] / RAW_FILE)

    out = run(config)

    listings = pd.read_parquet(out)
    assert out == paths["intermediate"] / "listings.parquet"
    assert listings["listing_id"].tolist() == ["fx1", "fx2"]
    assert "description" not in listings.columns
