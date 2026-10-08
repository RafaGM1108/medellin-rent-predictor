"""Training pipeline: model input -> cross-validated model comparison."""

import json
from pathlib import Path

import pandas as pd

from medellin_rent.model.evaluate import cross_validate, folds, summarize
from medellin_rent.model.gradient_boosting import tune
from medellin_rent.model.registry import NAMES, get_models
from medellin_rent.pipelines.feature_pipeline.pipeline import TRAIN_FILE
from medellin_rent.utils.config import Config, get_config
from medellin_rent.utils.log import get_logger

CV_FILE = "model_cv_folds.csv"
COMPARISON_FILE = "model_comparison.csv"
TUNING_FILE = "lightgbm_tuning.csv"


def run(config: Config | None = None, models: list[str] | None = None) -> Path:
    """Cross-validate every model on the same folds and write the comparison.

    Args:
        config: Project configuration. Loaded from ``conf/base.yaml`` if omitted.
        models: Names from ``NAMES`` to run (all by default). LightGBM is tuned first
            (``params.lightgbm_tuning_iter`` configurations) when one of its variants is run.

    Returns:
        Path to the comparison table in ``08_reporting``.
    """
    config = config or get_config()
    logger = get_logger(__name__, config.logging.level)
    train = pd.read_parquet(config.paths.model_input / TRAIN_FILE)
    splits = folds(len(train), int(config.params["cv_folds"]), config.project.random_seed)

    seed = config.project.random_seed
    names = models or NAMES
    out_dir = config.paths.reporting
    out_dir.mkdir(parents=True, exist_ok=True)

    lightgbm_params = None
    if any(name.startswith("lightgbm") for name in names):
        n_iter = int(config.params["lightgbm_tuning_iter"])
        lightgbm_params, tuning = tune(train, seed, n_iter=n_iter)
        tuning.to_csv(out_dir / TUNING_FILE, index=False, float_format="%.4f")
        logger.info("LightGBM tuned over %d configurations: %s", n_iter, lightgbm_params)

    factories = get_models(seed, lightgbm_params)
    per_fold, summary = [], []
    for name in names:
        cv = cross_validate(factories[name], train, splits).assign(model=name)
        per_fold.append(cv)
        summary.append({"model": name, **summarize(cv)})
        logger.info("%s: CV MAE %.0f COP, MAPE %.1f%%", name, cv["mae"].mean(), cv["mape"].mean())

    pd.concat(per_fold).to_csv(out_dir / CV_FILE, index=False, float_format="%.4f")
    comparison = pd.DataFrame(summary).sort_values("mae_mean", ignore_index=True)
    comparison.to_csv(out_dir / COMPARISON_FILE, index=False, float_format="%.4f")
    (out_dir / "model_comparison.json").write_text(
        json.dumps(
            {
                "n_train": len(train),
                "cv_folds": len(splits),
                "lightgbm_params": lightgbm_params,
                "models": summary,
            },
            indent=2,
        )
        + "\n"
    )
    return out_dir / COMPARISON_FILE
