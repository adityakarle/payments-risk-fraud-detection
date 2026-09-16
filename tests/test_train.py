import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import (  # noqa: E402
    compute_scale_pos_weight,
    get_feature_columns,
    time_based_split,
)


def make_toy_df(n=20):
    data = {"Time": list(range(n)), "Amount": [10.0] * n, "Class": [0] * (n - 2) + [1, 1]}
    for i in range(1, 29):
        data[f"V{i}"] = [0.0] * n
    return pd.DataFrame(data)


def test_time_based_split_rejects_unsorted_input():
    df = make_toy_df()
    df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
    with pytest.raises(ValueError):
        time_based_split(df_shuffled)


def test_time_based_split_respects_train_fraction():
    df = make_toy_df(n=20)
    split = time_based_split(df, train_fraction=0.7)
    assert len(split.X_train) == 14
    assert len(split.X_test) == 6


def test_time_based_split_train_is_strictly_earlier_than_test():
    df = make_toy_df(n=20)
    split = time_based_split(df, train_fraction=0.5)
    # X_train/X_test don't carry Time (it's not a feature column), so we
    # check via the original index instead: train indices must all be
    # smaller than test indices for a valid time-ordered split.
    assert split.X_train.index.max() < split.X_test.index.min()


def test_time_based_split_rejects_invalid_fraction():
    df = make_toy_df()
    with pytest.raises(ValueError):
        time_based_split(df, train_fraction=0.0)
    with pytest.raises(ValueError):
        time_based_split(df, train_fraction=1.0)


def test_compute_scale_pos_weight():
    y_train = pd.Series([0, 0, 0, 0, 1])
    assert compute_scale_pos_weight(y_train) == 4.0


def test_compute_scale_pos_weight_raises_when_no_fraud_in_training():
    y_train = pd.Series([0, 0, 0, 0])
    with pytest.raises(ValueError):
        compute_scale_pos_weight(y_train)


def test_get_feature_columns_includes_engineered_by_default():
    cols = get_feature_columns()
    assert "V1" in cols
    assert "amount_log" in cols
    assert "rolling_txn_count" in cols


def test_get_feature_columns_can_exclude_engineered():
    cols = get_feature_columns(include_engineered=False)
    assert "V1" in cols
    assert "amount_log" not in cols
