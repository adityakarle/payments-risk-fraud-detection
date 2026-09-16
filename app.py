"""
Streamlit dashboard for the payments fraud detection model.

Run with: streamlit run app.py

Assumes scripts/run_pipeline.py has already been run once (to train the
models and save models/*.joblib + models/results.json).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluate import cost_based_threshold_sweep, precision_recall_points, summarize
from src.features import engineer_features
from src.train import (
    get_feature_columns,
    predict_proba_fraud,
    time_based_split,
    train_random_forest,
    train_xgboost,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard_sample.csv"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Payments Risk & Fraud Detection", page_icon="🛡️", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).sort_values("Time").reset_index(drop=True)
    df = engineer_features(df)
    return df


@st.cache_resource
def load_models():
    """Load trained models, training them first if this is a fresh
    deployment with no committed .joblib files. Cached so this only
    happens once per app instance, not on every page interaction."""
    trainers = {"xgboost": train_xgboost, "random_forest": train_random_forest}
    missing = [name for name in trainers if not (MODELS_DIR / f"{name}.joblib").exists()]

    if missing:
        with st.spinner(f"First run: training {', '.join(missing)} (happens once, ~30s)..."):
            MODELS_DIR.mkdir(exist_ok=True)
            df = load_data()
            split = time_based_split(df, train_fraction=0.7)
            for name in missing:
                model = trainers[name](split)
                joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    return {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in trainers}


st.title("🛡️ Payments Risk & Fraud Detection")
st.caption(
    "Trained on a real, licensed 20% sample of the ULB/Worldline credit card "
    "fraud dataset (Pozzolo et al., 2015) — 56,874 transactions, 102 confirmed fraud."
)

df = load_data()
models = load_models()
model_name = st.sidebar.selectbox("Model", list(models.keys()), format_func=lambda s: s.replace("_", " ").title())
model = models[model_name]

split = time_based_split(df, train_fraction=0.7)
y_proba_test = predict_proba_fraud(model, split.X_test)
y_test = split.y_test.values

tab_overview, tab_performance, tab_cost, tab_score = st.tabs(
    ["Dataset Overview", "Model Performance", "Cost-Based Threshold", "Score a Transaction"]
)

with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total transactions", f"{len(df):,}")
    col2.metric("Confirmed fraud", int(df["Class"].sum()))
    col3.metric("Fraud rate", f"{df['Class'].mean()*100:.3f}%")

    st.subheader("Amount distribution by class")
    st.caption("Log-scaled amounts, since raw amounts are heavily right-skewed.")
    chart_df = pd.DataFrame({
        "amount_log": df["amount_log"],
        "class": df["Class"].map({0: "Legitimate", 1: "Fraud"}),
    })
    st.bar_chart(chart_df.groupby("class")["amount_log"].mean())

    st.subheader("Transaction density over time (population-level)")
    st.caption(
        "This dataset has no account/customer ID, so this is overall stream "
        "density, not per-customer velocity -- still useful for spotting bursts."
    )
    st.line_chart(df.set_index("Time")["rolling_txn_count"])

with tab_performance:
    st.caption(f"Evaluated on the time-based holdout test set ({len(split.X_test):,} transactions, {int(y_test.sum())} fraud).")
    summary = summarize(y_test, y_proba_test, threshold=0.5)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PR-AUC", f"{summary.pr_auc:.3f}")
    col2.metric("ROC-AUC", f"{summary.roc_auc:.3f}")
    col3.metric("Precision @ 0.5", f"{summary.precision_at_threshold:.3f}")
    col4.metric("Recall @ 0.5", f"{summary.recall_at_threshold:.3f}")

    st.caption(
        "PR-AUC is the primary metric here, not accuracy -- with 0.18% fraud "
        "prevalence, a model that predicts 'never fraud' would score ~99.8% "
        "accuracy while catching zero fraud."
    )

    st.subheader("Precision-Recall curve")
    precision, recall, _ = precision_recall_points(y_test, y_proba_test)
    st.line_chart(pd.DataFrame({"precision": precision, "recall": recall}).set_index("recall"))

    st.subheader("Confusion matrix @ threshold 0.5")
    tn, fp, fn, tp = summary.confusion.ravel()
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual: Legitimate", "Actual: Fraud"],
        columns=["Predicted: Legitimate", "Predicted: Fraud"],
    )
    st.dataframe(cm_df, width='stretch')

    st.subheader("Top feature importances")
    from src.evaluate import feature_importance_table
    importances = feature_importance_table(model, get_feature_columns())[:15]
    imp_df = pd.DataFrame(importances, columns=["feature", "importance"]).set_index("feature")
    st.bar_chart(imp_df)

with tab_cost:
    st.caption(
        "The real business question isn't 'what's the F1 score' -- it's "
        "'what threshold should we actually operate at, given what a missed "
        "fraud costs versus what a manual review costs.'"
    )
    col1, col2 = st.columns(2)
    with col1:
        cost_fn = st.number_input("Assumed cost per missed fraud ($)", min_value=1.0, value=500.0, step=50.0)
    with col2:
        cost_fp = st.number_input("Assumed cost per false-positive review ($)", min_value=0.1, value=5.0, step=1.0)

    sweep = cost_based_threshold_sweep(y_test, y_proba_test, cost_false_negative=cost_fn, cost_false_positive=cost_fp)

    st.metric(
        "Cost-optimal threshold",
        f"{sweep.best_threshold:.3f}",
        help="The threshold that minimizes total expected cost on the test set, given the costs above.",
    )
    st.line_chart(pd.DataFrame({"threshold": sweep.thresholds, "total_cost": sweep.total_costs}).set_index("threshold"))

    optimal_summary = summarize(y_test, y_proba_test, threshold=sweep.best_threshold)
    default_summary = summarize(y_test, y_proba_test, threshold=0.5)

    st.subheader("Default (0.5) vs. cost-optimal threshold")
    compare_df = pd.DataFrame({
        "Default (0.5)": [default_summary.precision_at_threshold, default_summary.recall_at_threshold],
        f"Cost-optimal ({sweep.best_threshold:.3f})": [optimal_summary.precision_at_threshold, optimal_summary.recall_at_threshold],
    }, index=["Precision", "Recall"])
    st.dataframe(compare_df, width='stretch')

with tab_score:
    st.caption(
        "Features V1-V28 are PCA-anonymized and not human-interpretable, so "
        "rather than a manual entry form, pick a real transaction from the "
        "test set to see how the model scores it."
    )
    sample_idx = st.selectbox(
        "Pick a test-set transaction",
        options=list(range(len(split.X_test))),
        format_func=lambda i: f"#{i} — actual: {'FRAUD' if y_test[i] == 1 else 'legitimate'}",
    )
    proba = y_proba_test[sample_idx]
    st.metric("Predicted fraud probability", f"{proba*100:.2f}%")
    st.write(f"**Actual label:** {'🚨 Fraud' if y_test[sample_idx] == 1 else '✅ Legitimate'}")
