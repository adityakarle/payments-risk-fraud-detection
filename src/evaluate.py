"""
Evaluation for fraud detection models.

Accuracy is meaningless on data this imbalanced (a model that predicts
"never fraud" scores ~99.8% accuracy and catches zero fraud) -- this
module leads with PR-AUC and ROC-AUC instead, and adds a cost-based
threshold optimizer, because "what threshold should we actually operate
at" is the real business question a risk team asks, not "what's the
model's F1 score."

The cost model is intentionally simple and its assumptions are named
explicitly as parameters (never hardcoded silently) so anyone using this
can swap in real cost figures instead of trusting made-up ones.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class ClassificationSummary:
    pr_auc: float
    roc_auc: float
    precision_at_threshold: float
    recall_at_threshold: float
    threshold: float
    confusion: np.ndarray  # [[TN, FP], [FN, TP]]


def summarize(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> ClassificationSummary:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return ClassificationSummary(
        pr_auc=average_precision_score(y_true, y_proba),
        roc_auc=roc_auc_score(y_true, y_proba),
        precision_at_threshold=precision,
        recall_at_threshold=recall,
        threshold=threshold,
        confusion=cm,
    )


@dataclass
class CostSweepResult:
    thresholds: np.ndarray
    total_costs: np.ndarray
    best_threshold: float
    best_cost: float


def cost_based_threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_false_negative: float,
    cost_false_positive: float,
    n_thresholds: int = 199,
) -> CostSweepResult:
    """Sweep candidate thresholds and compute expected total cost at
    each one, where:

        total_cost(t) = (# false negatives at t) * cost_false_negative
                      + (# false positives at t) * cost_false_positive

    `cost_false_negative` should represent the average financial loss
    from letting a fraudulent transaction through undetected.
    `cost_false_positive` should represent the operational cost of
    flagging a legitimate transaction for review (analyst time, customer
    friction). Both are required arguments -- there is no silent default,
    because picking a threshold without stating these assumptions is
    exactly the kind of unexamined number this function exists to avoid.

    Returns the threshold that minimizes total expected cost over the
    given data.
    """
    if cost_false_negative <= 0 or cost_false_positive <= 0:
        raise ValueError("Both costs must be positive.")

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    total_costs = np.empty_like(thresholds)

    y_true = np.asarray(y_true)
    for i, t in enumerate(thresholds):
        y_pred = (y_proba >= t).astype(int)
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        total_costs[i] = fn * cost_false_negative + fp * cost_false_positive

    best_idx = int(np.argmin(total_costs))
    return CostSweepResult(
        thresholds=thresholds,
        total_costs=total_costs,
        best_threshold=float(thresholds[best_idx]),
        best_cost=float(total_costs[best_idx]),
    )


def precision_recall_points(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for plotting a PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds


def feature_importance_table(model, feature_names: list[str]) -> list[tuple[str, float]]:
    """Return (feature_name, importance) pairs sorted descending, for
    whichever model type was passed (both XGBoost and RandomForest
    expose `.feature_importances_`)."""
    importances = model.feature_importances_
    pairs = list(zip(feature_names, importances))
    return sorted(pairs, key=lambda p: p[1], reverse=True)
