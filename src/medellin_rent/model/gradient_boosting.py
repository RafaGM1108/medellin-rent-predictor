"""LightGBM: native missing values and categories, tuned by random search.

Tuning uses only the training set, on folds drawn with a different seed from the evaluation
folds; the test set is never touched. Because the evaluation folds reuse the same rows, the
tuned model's CV score is slightly optimistic; the held-out test score is the honest one.

Two variants (decision 5 in ``docs/decisions.md``): without barrio, and with barrio as a
target-encoded feature (scikit-learn's ``TargetEncoder`` cross-fits internally, so the
encoding of a row never uses its own rent).
"""

from typing import Any

import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, TargetEncoder

from medellin_rent.model.evaluate import cross_validate, folds

SEARCH_SPACE: dict[str, list[Any]] = {
    "num_leaves": [15, 31, 63, 127],
    "learning_rate": [0.02, 0.05, 0.1],
    "n_estimators": [300, 600, 1000],
    "min_child_samples": [10, 20, 50],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}
DEFAULT_PARAMS: dict[str, Any] = {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 600}


def _drop_barrio(x: pd.DataFrame) -> pd.DataFrame:
    """All features except ``barrio_name`` (a named function, so the model can be pickled)."""
    return x.drop(columns="barrio_name")


def make_lightgbm(params: dict[str, Any], seed: int, barrio: bool = False) -> Pipeline:
    """LightGBM regressor, optionally with a target-encoded barrio.

    Args:
        params: LightGBM hyperparameters (e.g. from :func:`tune`).
        seed: Random seed.
        barrio: Add ``barrio_name`` as a target-encoded feature.
    """
    if barrio:
        features: Any = ColumnTransformer(
            [
                (
                    "barrio",
                    TargetEncoder(
                        target_type="continuous",
                        cv=KFold(5, shuffle=True, random_state=seed),
                    ),
                    ["barrio_name"],
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    else:
        features = FunctionTransformer(_drop_barrio)
    model = lgb.LGBMRegressor(**params, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)
    return Pipeline([("features", features), ("model", model)])


def tune(
    train: pd.DataFrame, seed: int, n_iter: int, inner_folds: int = 3
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Random search of LightGBM hyperparameters by inner CV MAE on the training set.

    Args:
        train: Training set from ``05_model_input``.
        seed: Seed for the sampled configurations and the inner folds.
        n_iter: Number of configurations to try.
        inner_folds: Number of inner CV folds.

    Returns:
        ``(best_params, results)``: the best configuration and one row per configuration
        with its parameters and mean inner MAE / MAPE, best first.
    """
    splits = folds(len(train), inner_folds, seed + 1)  # not the evaluation folds
    rows = []
    for params in ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=seed):
        cv = cross_validate(lambda p=params: make_lightgbm(p, seed), train, splits)  # type: ignore[misc]
        rows.append({**params, "mae": cv["mae"].mean(), "mape": cv["mape"].mean()})
    results = pd.DataFrame(rows).sort_values("mae", ignore_index=True)
    best = {k: results.loc[0, k] for k in SEARCH_SPACE}
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    return best, results
