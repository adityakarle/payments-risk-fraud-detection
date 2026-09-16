"""
Model training for fraud detection.

Two deliberate choices worth calling out (both are things a payments-risk
interviewer would specifically probe for):

1. Time-based train/test split, not a random shuffle. Fraud is a
   temporal problem in production -- you only ever have the past to
   predict the future. A random split lets information from "future"
   transactions leak into training (e.g. via the rolling-window features
   in src/features.py, which are computed using neighboring transactions
   in time). Splitting by Time avoids that leakage and gives a more
   honest estimate of real-world performance.

2. Class imbalance is handled via algorithm-level weighting
   (scale_pos_weight for XGBoost, class_weight='balanced' for Random
   Forest) rather than resampling (SMOTE/undersampling). This avoids the
   common mistake of resampling before splitting (which leaks synthetic
   copies of minority-class points across the train/test boundary) and
   keeps the evaluation set's class balance representative of reality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS_BASE = [f"V{i}" for i in range(1, 29)]
ENGINEERED_COLUMNS = [
    "amount_log",
    "hour_of_day",
    "is_night",
    "rolling_txn_count",
    "rolling_avg_amount",
    "amount_vs_rolling_avg",
]
TARGET_COLUMN = "Class"


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def get_feature_columns(include_engineered: bool = True) -> list[str]:
    cols = list(FEATURE_COLUMNS_BASE)
    if include_engineered:
        cols += ENGINEERED_COLUMNS
    return cols


def time_based_split(df: pd.DataFrame, train_fraction: float = 0.7) -> SplitData:
    """Split a dataframe already sorted by Time ascending into a train
    set (earliest `train_fraction` of rows) and a test set (the rest).
    Raises if the input isn't actually sorted, since a time-based split
    on unsorted data would silently produce a meaningless split."""
    if not df["Time"].is_monotonic_increasing:
        raise ValueError(
            "time_based_split requires the DataFrame sorted by 'Time' ascending."
        )
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")

    split_idx = int(len(df) * train_fraction)
    feature_cols = get_feature_columns(
        include_engineered=all(c in df.columns for c in ENGINEERED_COLUMNS)
    )

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return SplitData(
        X_train=train_df[feature_cols],
        X_test=test_df[feature_cols],
        y_train=train_df[TARGET_COLUMN],
        y_test=test_df[TARGET_COLUMN],
    )


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """XGBoost's scale_pos_weight: ratio of negative to positive class
    counts in the TRAINING set only (never the test set -- using test-set
    class balance to tune a training hyperparameter is itself a subtle
    form of leakage)."""
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    if pos == 0:
        raise ValueError("No positive (fraud) examples in training set.")
    return neg / pos


def train_xgboost(split: SplitData, random_state: int = 42):
    import xgboost as xgb

    spw = compute_scale_pos_weight(split.y_train)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(split.X_train, split.y_train)
    return model


def train_random_forest(split: SplitData, random_state: int = 42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(split.X_train, split.y_train)
    return model


def predict_proba_fraud(model, X: pd.DataFrame) -> np.ndarray:
    """Return P(fraud) for each row, regardless of which model type."""
    return model.predict_proba(X)[:, 1]
