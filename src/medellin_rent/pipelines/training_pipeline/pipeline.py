"""Training pipeline: model input -> model comparison, best model and its test metrics.

Steps:

1. Tune LightGBM on the training set (when a LightGBM variant is selected).
2. Cross-validate every model on the same folds; log each one as an MLflow run.
3. Pick the model with the lowest CV MAE, refit it on the whole training set and score it
   once on the held-out test set.
4. Save the model to ``06_models`` and the comparison and test metrics to ``08_reporting``.

MLflow tracks to a local SQLite store in ``paths.mlruns`` (git-ignored); see the runs with
``uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

from medellin_rent.features.build import FEATURES
from medellin_rent.features.model_input import LOG_TARGET, TARGET
from medellin_rent.model.evaluate import cross_validate, folds, metrics, predict_rent, summarize
from medellin_rent.model.gradient_boosting import tune
from medellin_rent.model.registry import NAMES, get_models
from medellin_rent.pipelines.feature_pipeline.pipeline import TEST_FILE, TRAIN_FILE
from medellin_rent.utils.config import Config, get_config
from medellin_rent.utils.log import get_logger

CV_FILE = "model_cv_folds.csv"
COMPARISON_FILE = "model_comparison.csv"
TUNING_FILE = "lightgbm_tuning.csv"
TEST_METRICS_FILE = "test_metrics.json"
MODEL_FILE = "model.joblib"
METADATA_FILE = "model_metadata.json"
EXPERIMENT = "medellin-rent-predictor"


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _start_tracking(mlruns: Path) -> None:
    """Point MLflow at a local SQLite store and the project experiment."""
    mlruns.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{mlruns / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(mlruns / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def run(config: Config | None = None, models: list[str] | None = None) -> Path:
    """Train, compare and select the models.

    Args:
        config: Project configuration. Loaded from ``conf/base.yaml`` if omitted.
        models: Names from ``NAMES`` to run (all by default). LightGBM is tuned first
            (``params.lightgbm_tuning_iter`` configurations) when one of its variants is run.

    Returns:
        Path to the saved best model in ``06_models``.
    """
    config = config or get_config()
    logger = get_logger(__name__, config.logging.level)
    paths, seed = config.paths, config.project.random_seed
    train = pd.read_parquet(paths.model_input / TRAIN_FILE)
    test = pd.read_parquet(paths.model_input / TEST_FILE)
    splits = folds(len(train), int(config.params["cv_folds"]), seed)
    names = models or NAMES
    out_dir = paths.reporting
    out_dir.mkdir(parents=True, exist_ok=True)
    _start_tracking(paths.mlruns)

    lightgbm_params = None
    if any(name.startswith("lightgbm") for name in names):
        n_iter = int(config.params["lightgbm_tuning_iter"])
        lightgbm_params, tuning = tune(train, seed, n_iter=n_iter)
        tuning.to_csv(out_dir / TUNING_FILE, index=False, float_format="%.4f")
        logger.info("LightGBM tuned over %d configurations: %s", n_iter, lightgbm_params)

    factories = get_models(seed, lightgbm_params)
    per_fold, summary = [], []
    for name in names:
        with mlflow.start_run(run_name=name):
            cv = cross_validate(factories[name], train, splits).assign(model=name)
            scores = summarize(cv)
            mlflow.log_params({"model": name, "cv_folds": len(splits), "seed": seed})
            if name.startswith("lightgbm") and lightgbm_params:
                mlflow.log_params(lightgbm_params)
            mlflow.log_metrics({f"cv_{k}": v for k, v in scores.items()})
        per_fold.append(cv)
        summary.append({"model": name, **scores})
        logger.info(
            "%s: CV MAE %.0f COP, MAPE %.1f%%", name, scores["mae_mean"], scores["mape_mean"]
        )

    pd.concat(per_fold).to_csv(out_dir / CV_FILE, index=False, float_format="%.4f")
    comparison = pd.DataFrame(summary).sort_values("mae_mean", ignore_index=True)
    comparison.to_csv(out_dir / COMPARISON_FILE, index=False, float_format="%.4f")
    _write_json(
        {
            "n_train": len(train),
            "cv_folds": len(splits),
            "lightgbm_params": lightgbm_params,
            "models": summary,
        },
        out_dir / "model_comparison.json",
    )

    best = str(comparison.loc[0, "model"])
    model = factories[best]().fit(train[FEATURES], train[LOG_TARGET])
    test_scores = metrics(test[TARGET], predict_rent(model, test)) if len(test) else {}
    test_report = {"model": best, "n_test": len(test), **test_scores}
    _write_json(test_report, out_dir / TEST_METRICS_FILE)

    paths.models.mkdir(parents=True, exist_ok=True)
    model_path = paths.models / MODEL_FILE
    joblib.dump(model, model_path)
    metadata = {
        "model": best,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_train": len(train),
        "features": FEATURES,
        "target": f"{LOG_TARGET} (predictions are exp() of the output, in COP)",
        "lightgbm_params": lightgbm_params if best.startswith("lightgbm") else None,
        "cv": {k: v for k, v in comparison.loc[0].items() if k != "model"},
        "test": test_scores,
        "data_period": "Properati listings published 2020-07-26 to 2021-08-19",
    }
    _write_json(metadata, paths.models / METADATA_FILE)
    with mlflow.start_run(run_name=f"{best} (final)"):
        mlflow.log_params({"model": best, "final": True, "seed": seed})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_scores.items()})
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(paths.models / METADATA_FILE))
    logger.info("Best model %s: test MAE %s COP", best, test_scores.get("mae"))
    return model_path
