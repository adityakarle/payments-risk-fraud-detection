import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import (  # noqa: E402
    add_amount_features,
    add_rolling_density_features,
    add_temporal_features,
    engineer_features,
)


def make_toy_df():
    return pd.DataFrame(
        {
            "Time": [0, 10, 20, 30, 3600 * 6, 3600 * 23 + 1800],
            "Amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def test_hour_of_day_computed_correctly():
    df = make_toy_df()
    out = add_temporal_features(df)
    # Row 4 is exactly 6 hours in -> hour_of_day == 6.0
    assert out.loc[4, "hour_of_day"] == pytest.approx(6.0)
    # Row 5 is 23.5 hours in
    assert out.loc[5, "hour_of_day"] == pytest.approx(23.5)


def test_is_night_flags_correct_hours():
    df = make_toy_df()
    out = add_temporal_features(df)
    # Row 0-3 are all within the first minute -> hour_of_day ~0 -> night
    assert out.loc[0, "is_night"] == 1
    # Row 4 at hour 6 -> not night (night defined as <6 or >=23)
    assert out.loc[4, "is_night"] == 0
    # Row 5 at hour 23.5 -> night
    assert out.loc[5, "is_night"] == 1


def test_amount_log_matches_log1p():
    df = make_toy_df()
    out = add_amount_features(df)
    assert np.allclose(out["amount_log"], np.log1p(df["Amount"]))


def test_rolling_density_requires_sorted_time():
    df = pd.DataFrame({"Time": [10, 0, 20], "Amount": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        add_rolling_density_features(df)


def test_rolling_density_counts_within_window():
    # Four transactions all within a 60s window, one far outside it.
    df = pd.DataFrame(
        {"Time": [0, 10, 20, 30, 500], "Amount": [10.0, 10.0, 10.0, 10.0, 999.0]}
    )
    out = add_rolling_density_features(df, window_seconds=60)
    # By the 4th transaction (Time=30), all of the first 4 are within 60s.
    assert out.loc[3, "rolling_txn_count"] == 4
    # The 5th transaction at Time=500 has no neighbors within 60s before it.
    assert out.loc[4, "rolling_txn_count"] == 1


def test_engineer_features_produces_no_nans_on_sorted_input():
    df = make_toy_df()
    out = engineer_features(df)
    engineered_cols = [
        "hour_of_day", "is_night", "amount_log",
        "rolling_txn_count", "rolling_avg_amount", "amount_vs_rolling_avg",
    ]
    assert out[engineered_cols].isna().sum().sum() == 0
