import json
from pathlib import Path

import pandas as pd

from medellin_rent.pipelines.training_pipeline.pipeline import run
from medellin_rent.utils.config import PathsConfig, load_config

ROOT = Path(__file__).resolve().parents[3]


def test_run_writes_comparison(tmp_path: Path, model_input: pd.DataFrame) -> None:
    config = load_config(ROOT / "conf" / "base.yaml")
    paths = PathsConfig(**{name: tmp_path / name for name in PathsConfig.model_fields})
    config = config.model_copy(update={"paths": paths})
    paths.model_input.mkdir()
    model_input.to_parquet(paths.model_input / "train.parquet", index=False)

    out = run(config, models=["baseline"])

    comparison = pd.read_csv(out)
    assert comparison["model"].tolist() == ["baseline"]
    assert {"mae_mean", "rmse_mean", "mape_mean"} <= set(comparison.columns)
    folds = pd.read_csv(paths.reporting / "model_cv_folds.csv")
    assert len(folds) == config.params["cv_folds"]
    summary = json.loads((paths.reporting / "model_comparison.json").read_text())
    assert summary["n_train"] == 200
