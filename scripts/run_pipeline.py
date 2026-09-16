"""
End-to-end pipeline: load the real transaction data, engineer features,
split by time, train both models, evaluate, run the cost-based threshold
sweep, and persist everything the Streamlit app needs.

Run with: python scripts/run_pipeline.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import cost_based_threshold_sweep, feature_importance_table, summarize  # noqa: E402
from src.features import engineer_features  # noqa: E402
from src.train import get_feature_columns, predict_proba_fraud, time_based_split, train_random_forest, train_xgboost  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard_sample.csv"
MODELS_DIR = BASE_DIR / "models"

# Business assumptions for the cost-based threshold sweep. These are
# stated explicitly and can be changed to real figures -- they are NOT
# meant to be taken as ground truth, only as a worked example of the
# methodology.
COST_FALSE_NEGATIVE = 500.0  # assumed average $ loss per missed fraud
COST_FALSE_POSITIVE = 5.0    # assumed cost of a manual review


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH).sort_values("Time").reset_index(drop=True)
    print(f"  {len(df)} transactions, {df['Class'].sum()} fraud ({df['Class'].mean()*100:.3f}%)")

    print("Engineering features ...")
    df = engineer_features(df)

    print("Splitting by time (70/30) ...")
    split = time_based_split(df, train_fraction=0.7)
    print(f"  train={len(split.X_train)}  test={len(split.X_test)}  "
          f"train_fraud={int(split.y_train.sum())}  test_fraud={int(split.y_test.sum())}")

    results = {}

    for name, trainer in [("xgboost", train_xgboost), ("random_forest", train_random_forest)]:
        print(f"\nTraining {name} ...")
        model = trainer(split)
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

        y_proba = predict_proba_fraud(model, split.X_test)
        summary = summarize(split.y_test.values, y_proba, threshold=0.5)
        print(f"  PR-AUC={summary.pr_auc:.4f}  ROC-AUC={summary.roc_auc:.4f}  "
              f"precision@0.5={summary.precision_at_threshold:.3f}  recall@0.5={summary.recall_at_threshold:.3f}")

        sweep = cost_based_threshold_sweep(
            split.y_test.values, y_proba,
            cost_false_negative=COST_FALSE_NEGATIVE,
            cost_false_positive=COST_FALSE_POSITIVE,
        )
        print(f"  Cost-optimal threshold: {sweep.best_threshold:.3f} "
              f"(vs default 0.5), expected cost=${sweep.best_cost:,.2f}")

        default_summary = summarize(split.y_test.values, y_proba, threshold=0.5)
        optimal_summary = summarize(split.y_test.values, y_proba, threshold=sweep.best_threshold)

        importances = feature_importance_table(model, get_feature_columns())

        results[name] = {
            "pr_auc": summary.pr_auc,
            "roc_auc": summary.roc_auc,
            "default_threshold": {
                "threshold": 0.5,
                "precision": default_summary.precision_at_threshold,
                "recall": default_summary.recall_at_threshold,
                "confusion": default_summary.confusion.tolist(),
            },
            "cost_optimal_threshold": {
                "threshold": sweep.best_threshold,
                "precision": optimal_summary.precision_at_threshold,
                "recall": optimal_summary.recall_at_threshold,
                "confusion": optimal_summary.confusion.tolist(),
                "expected_cost": sweep.best_cost,
            },
            "top_features": importances[:10],
        }

    (MODELS_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved models and results.json to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
