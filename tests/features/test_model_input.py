import numpy as np
import pandas as pd
import pytest

from medellin_rent.features.model_input import LOG_TARGET, TARGET, make_model_input


def _inputs(n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = [f"l{i}" for i in range(n)]
    comuna = pd.Series(["11" if i % 10 else None for i in range(n)], dtype="str")
    features = pd.DataFrame({"listing_id": ids, "comuna_code": comuna, "area_m2": 70.0})
    primary = pd.DataFrame({"listing_id": ids[::-1], TARGET: np.linspace(1e6, 3e6, n)[::-1]})
    return features, primary


def test_keeps_listings_with_comuna_and_adds_targets() -> None:
    train, test = make_model_input(*_inputs(), test_size=0.2, seed=42)
    both = pd.concat([train, test])
    assert len(both) == 90  # every 10th listing has no comuna
    assert both["comuna_code"].notna().all()
    assert np.allclose(both[LOG_TARGET], np.log(both[TARGET]))
    row = both.set_index("listing_id").loc["l1"]
    assert row[TARGET] == pytest.approx(1e6 + (3e6 - 1e6) / 99)  # joined by id, not position


def test_split_sizes_disjoint_and_reproducible() -> None:
    train, test = make_model_input(*_inputs(), test_size=0.2, seed=42)
    assert (len(train), len(test)) == (72, 18)
    assert set(train["listing_id"]).isdisjoint(test["listing_id"])
    again_train, _ = make_model_input(*_inputs(), test_size=0.2, seed=42)
    assert train["listing_id"].tolist() == again_train["listing_id"].tolist()
    other_train, _ = make_model_input(*_inputs(), test_size=0.2, seed=7)
    assert train["listing_id"].tolist() != other_train["listing_id"].tolist()


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_rejects_invalid_test_size(test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size"):
        make_model_input(*_inputs(), test_size=test_size, seed=42)
