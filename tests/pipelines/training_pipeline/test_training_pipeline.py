import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from medellin_rent.features.build import FEATURES
from medellin_rent.pipelines.training_pipeline.pipeline import run
from medellin_rent.utils.config import Config, PathsConfig, load_config

ROOT = Path(__file__).resolve().parents[3]


def _config(tmp_path: Path, model_input: pd.DataFrame) -> Config:
    config = load_config(ROOT / "conf" / "base.yaml")
    paths = PathsConfig(**{name: tmp_path / name for name in PathsConfig.model_fields})
    paths.model_input.mkdir()
    model_input.iloc[:160].to_parquet(paths.model_input / "train.parquet", index=False)
    model_input.iloc[160:].to_parquet(paths.model_input / "test.parquet", index=False)
    return config.model_copy(update={"paths": paths})


def test_run_compares_selects_saves_and_tracks(tmp_path: Path, model_input: pd.DataFrame) -> None:
    config = _config(tmp_path, model_input)
    reporting = config.paths.reporting

    model_path = run(config, models=["baseline", "ridge"])

    comparison = pd.read_csv(reporting / "model_comparison.csv")
    assert set(comparison["model"]) == {"baseline", "ridge"}
    assert comparison["mae_mean"].is_monotonic_increasing
    best = str(comparison.loc[0, "model"])
    assert len(pd.read_csv(reporting / "model_cv_folds.csv")) == 2 * config.params["cv_folds"]

    test_metrics = json.loads((reporting / "test_metrics.json").read_text())
    assert test_metrics["model"] == best
    assert test_metrics["n_test"] == 40
    assert {"mae", "rmse", "mape"} <= set(test_metrics)

    model = joblib.load(model_path)
    assert np.isfinite(model.predict(model_input[FEATURES])).all()
    metadata = json.loads((config.paths.models / "model_metadata.json").read_text())
    assert metadata["model"] == best
    assert metadata["features"] == FEATURES
    assert metadata["test"]["mae"] == test_metrics["mae"]

    runs = mlflow.search_runs(experiment_names=["medellin-rent-predictor"])
    assert sorted(runs["tags.mlflow.runName"]) == sorted(["baseline", "ridge", f"{best} (final)"])
    assert (config.paths.mlruns / "mlflow.db").is_file()
