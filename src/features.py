"""
Feature engineering for the credit card fraud dataset.

Important limitation, stated up front rather than glossed over: this
dataset (Pozzolo et al., ULB/Worldline, via Kaggle) has NO account or
customer identifier. Every column except Time, Amount, and Class is a
PCA-transformed, fully anonymized component -- there is no way to know
which transactions belong to the same card or customer.

That means true per-customer velocity features (the kind a real
payments-risk team would build first: "how many transactions has THIS
account made in the last hour") are not possible here. What IS possible,
and what this module builds, are *population-level* temporal and
amount-context features: how dense is transaction volume right now
across the whole stream, and how unusual is this amount relative to
recent activity. These are weaker than per-account velocity but still
real and defensible -- fraud attacks (e.g. bot-driven card testing waves)
often show up as bursts in overall transaction density even without
knowing individual accounts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day-style cyclical features derived from the dataset's
    `Time` column (seconds elapsed since the first transaction in the
    dataset, spanning roughly two days)."""
    out = df.copy()
    seconds_into_day = out["Time"] % SECONDS_PER_DAY
    hour_of_day = seconds_into_day / 3600.0
    out["hour_of_day"] = hour_of_day
    out["is_night"] = ((hour_of_day < 6) | (hour_of_day >= 23)).astype(int)
    return out


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transformed/contextualized versions of `Amount`. Raw
    transaction amounts are heavily right-skewed (a few very large
    transactions dominate the raw scale), which tends to destabilize
    tree-based splits less than linear models, but the log transform
    still gives a more evenly-spread feature and is cheap to add."""
    out = df.copy()
    out["amount_log"] = np.log1p(out["Amount"])
    return out


def add_rolling_density_features(
    df: pd.DataFrame, window_seconds: int = 60
) -> pd.DataFrame:
    """Add population-level rolling features: how many transactions
    occurred in the preceding `window_seconds`, and how this
    transaction's amount compares to the rolling average amount over
    that same window. Requires `df` to already be sorted by `Time`
    ascending -- raises if it isn't, since a rolling window computed on
    unsorted time data would be meaningless.
    """
    if not df["Time"].is_monotonic_increasing:
        raise ValueError(
            "add_rolling_density_features requires the DataFrame to be "
            "sorted by 'Time' ascending. Call df.sort_values('Time') first."
        )

    out = df.copy()
    time_indexed = pd.Series(out["Amount"].values, index=pd.to_timedelta(out["Time"], unit="s"))
    window = f"{window_seconds}s"

    rolling_count = time_indexed.rolling(window).count()
    rolling_mean = time_indexed.rolling(window).mean()

    out["rolling_txn_count"] = rolling_count.values
    out["rolling_avg_amount"] = rolling_mean.values
    # Avoid divide-by-zero when the rolling mean is ~0
    out["amount_vs_rolling_avg"] = out["Amount"] / (out["rolling_avg_amount"] + 1.0)
    return out


def engineer_features(df: pd.DataFrame, window_seconds: int = 60) -> pd.DataFrame:
    """Full feature engineering pipeline. Expects `df` already sorted by
    Time ascending (the caller is responsible for the sort, since sorting
    also matters for the time-based train/test split done at training
    time -- doing it in one place avoids sorting twice)."""
    out = add_temporal_features(df)
    out = add_amount_features(out)
    out = add_rolling_density_features(out, window_seconds=window_seconds)
    return out
