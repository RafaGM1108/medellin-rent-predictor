from pathlib import Path

import pandas as pd

from medellin_rent.pipelines.inference_pipeline.pipeline import run
from medellin_rent.utils.config import PathsConfig, load_config

ROOT = Path(__file__).resolve().parents[3]


def test_run_writes_predictions_for_the_sample(tmp_path: Path, saved_model: Path) -> None:
    config = load_config(ROOT / "conf" / "base.yaml")
    paths = config.paths.model_copy(
        update={"models": saved_model, "model_output": tmp_path / "out"}
    )
    out = run(config.model_copy(update={"paths": PathsConfig(**paths.model_dump())}))

    predictions = pd.read_csv(out)
    assert out == tmp_path / "out" / "predictions.csv"
    assert len(predictions) == 5
    assert {"comuna", "predicted_rent_cop", "interval_low_cop", "interval_high_cop"} <= set(
        predictions.columns
    )
    assert predictions["predicted_rent_cop"].gt(0).all()
