"""Model input: features + target, restricted to listings with a comuna, split train/test.

Decisions (``docs/decisions.md``): train only on listings with a comuna (2), target is the
monthly rent in COP and its log (1), a held-out test set of ``test_size`` (8).
"""

import numpy as np
import pandas as pd

TARGET = "rent_cop"
LOG_TARGET = "log_rent"


def make_model_input(
    features: pd.DataFrame, primary: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join features with the target, keep listings with a comuna and split them.

    Args:
        features: Output of :func:`medellin_rent.features.build.build_features`.
        primary: Primary listings (for ``listing_id`` and ``rent_cop``).
        test_size: Share of listings held out for the test set (0-1).
        seed: Random seed for the split.

    Returns:
        ``(train, test)``: features + ``rent_cop`` + ``log_rent``, fresh indexes.

    Raises:
        ValueError: If ``test_size`` is not strictly between 0 and 1.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    target = primary[["listing_id", TARGET]]
    data = features.merge(target, on="listing_id", how="inner", validate="one_to_one")
    data = data[data["comuna_code"].notna()].reset_index(drop=True)
    data[LOG_TARGET] = np.log(data[TARGET])

    order = np.random.default_rng(seed).permutation(len(data))
    n_test = round(len(data) * test_size)
    test = data.iloc[np.sort(order[:n_test])].reset_index(drop=True)
    train = data.iloc[np.sort(order[n_test:])].reset_index(drop=True)
    return train, test
