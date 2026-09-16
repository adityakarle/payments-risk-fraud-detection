import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import cost_based_threshold_sweep, summarize  # noqa: E402


def test_summarize_perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    result = summarize(y_true, y_proba, threshold=0.5)
    assert result.precision_at_threshold == 1.0
    assert result.recall_at_threshold == 1.0
    assert result.pr_auc == 1.0
    assert result.roc_auc == 1.0


def test_summarize_confusion_matrix_shape_and_values():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])  # one FP, one FN at threshold 0.5
    result = summarize(y_true, y_proba, threshold=0.5)
    tn, fp, fn, tp = result.confusion.ravel()
    assert (tn, fp, fn, tp) == (1, 1, 1, 1)


def test_cost_sweep_rejects_nonpositive_costs():
    y_true = np.array([0, 1])
    y_proba = np.array([0.1, 0.9])
    with pytest.raises(ValueError):
        cost_based_threshold_sweep(y_true, y_proba, cost_false_negative=0, cost_false_positive=5)
    with pytest.raises(ValueError):
        cost_based_threshold_sweep(y_true, y_proba, cost_false_negative=500, cost_false_positive=-1)


def test_cost_sweep_prefers_low_threshold_when_missing_fraud_is_expensive():
    # One fraud case scored 0.4, one legit case scored 0.6.
    # If missing fraud is very expensive relative to a false positive,
    # the optimal threshold should drop low enough to catch the fraud
    # case even at the cost of flagging the legit one.
    y_true = np.array([1, 0])
    y_proba = np.array([0.4, 0.6])
    result = cost_based_threshold_sweep(
        y_true, y_proba, cost_false_negative=10_000, cost_false_positive=5
    )
    # At the best threshold, the fraud case (0.4) must be caught.
    assert result.best_threshold <= 0.4


def test_cost_sweep_prefers_high_threshold_when_false_positives_are_expensive():
    # Same scores, but now flagging a false positive is astronomically
    # costly relative to missing fraud -- optimal threshold should rise
    # above 0.6 so the legit transaction is never flagged.
    y_true = np.array([1, 0])
    y_proba = np.array([0.4, 0.6])
    result = cost_based_threshold_sweep(
        y_true, y_proba, cost_false_negative=1, cost_false_positive=10_000
    )
    assert result.best_threshold > 0.6


def test_cost_sweep_total_costs_array_matches_thresholds_array_length():
    y_true = np.array([0, 1, 0, 1, 0])
    y_proba = np.array([0.2, 0.8, 0.3, 0.7, 0.1])
    result = cost_based_threshold_sweep(
        y_true, y_proba, cost_false_negative=500, cost_false_positive=5, n_thresholds=50
    )
    assert len(result.thresholds) == 50
    assert len(result.total_costs) == 50
    assert result.best_cost == result.total_costs.min()
