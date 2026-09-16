# Payments Risk & Fraud Detection Platform

A fraud detection pipeline built on real, licensed transaction data —
feature engineering, imbalance-aware model training with a proper
time-based validation split, and a cost-based threshold optimizer that
answers the actual business question a payments-risk team asks
("what threshold should we operate at"), not just a model accuracy number.

## Why this exists

Built specifically to close a gap: solid data science fundamentals
(Python, SQL, XGBoost, Random Forest, statistical analysis) with zero
payments/fraud-domain application on the resume. This project applies
that exact toolkit to the exact domain — fraud/risk classification on
real transaction data — rather than adding another generic ML demo.

## The data

`data/creditcard_sample.csv` is a real, licensed 20% sample (56,874
transactions, 102 confirmed fraud, 0.179% fraud rate) of the well-known
credit card fraud dataset from:

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
> Bontempi. *Calibrating Probability with Undersampling for Unbalanced
> Classification.* Symposium on Computational Intelligence and Data
> Mining (CIDM), IEEE, 2015. Collected by Worldline and the Machine
> Learning Group of Université Libre de Bruxelles (ULB).

This sample is redistributed via IBM's open-source
[`xgboost-smote-detect-fraud`](https://github.com/IBM/xgboost-smote-detect-fraud)
code pattern repository (Apache License 2.0), not scraped or fabricated.

**An honest limitation, stated up front rather than glossed over:** this
dataset has no account or customer identifier — every feature except
`Time`, `Amount`, and `Class` is PCA-anonymized, and there's no way to
tell which transactions belong to the same card. That rules out true
per-customer velocity features (the first thing a real payments-risk
team would build). What this project builds instead are
**population-level temporal/amount-context features** — see
"Feature engineering" below for what that means and why it's a real,
if weaker, substitute.

## What it does

1. **Feature engineering** (`src/features.py`) — adds hour-of-day /
   night-time flags, log-transformed amount, and rolling transaction
   density + amount-context features computed over the whole stream
   (not per-account, per the limitation above).
2. **Time-based train/test split** (`src/train.py`) — sorts by `Time`
   and splits chronologically rather than randomly shuffling, because a
   random split lets "future" transactions leak into training via the
   rolling-window features. Real fraud models only ever have the past
   to predict the future; the evaluation should reflect that.
3. **Two models, imbalance handled at the algorithm level** — XGBoost
   (`scale_pos_weight`) and Random Forest (`class_weight="balanced"`),
   computed from the training set only, never resampling before the
   split (a common and subtle leakage bug).
4. **Cost-based threshold optimization** (`src/evaluate.py`) — instead
   of reporting F1 at the default 0.5 cutoff, sweeps thresholds and
   picks the one minimizing `(missed fraud × assumed loss) + (false
   positives × assumed review cost)`. Both costs are explicit,
   named arguments — never hardcoded silently — because picking an
   operating threshold without stating the cost assumptions behind it
   is exactly the kind of unexamined number a risk team should push back on.
5. **Streamlit dashboard** (`app.py`) — dataset overview, PR/ROC curves
   and confusion matrix, an interactive cost-threshold explorer, and a
   transaction scorer.

## Results (time-based holdout test set, 17,063 transactions, 24 fraud)

| Model | PR-AUC | ROC-AUC | Precision @ 0.5 | Recall @ 0.5 |
|---|---|---|---|---|
| XGBoost | 0.726 | 0.936 | 0.654 | 0.708 |
| Random Forest | 0.787 | 0.923 | 1.000 | 0.625 |

PR-AUC is the headline metric here, not accuracy — with 0.18% fraud
prevalence, a model that predicts "never fraud" scores ~99.8% accuracy
while catching zero fraud. These PR-AUC figures are in line with
published benchmarks on this dataset.

At the assumed cost ratio ($500 per missed fraud vs. $5 per manual
review — both adjustable in the app), the cost-optimal threshold drops
to **0.025** for XGBoost and **0.193** for Random Forest — far below the
default 0.5 — because missing fraud is priced far more expensive than a
false-positive review. This is the actual output a risk team would want:
not "the model is 93% accurate," but "given these costs, operate here."

**On the engineered features specifically:** they rank low in feature
importance (10th–34th out of 34 features, both models) — the
anonymized `V1`–`V28` PCA components already carry most of the
predictive signal, since PCA was applied specifically to preserve
variance. This is worth stating plainly rather than overselling: the
methodology (time-aware, population-level context features) is the
valuable part to demonstrate, and it would matter more on data that
hasn't already been through dense feature extraction.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline (trains and saves both models)

```bash
python scripts/run_pipeline.py
```

## Run the dashboard

```bash
streamlit run app.py
```

## Run the tests

```bash
pytest tests/ -v
```

20 tests covering feature engineering correctness (rolling windows,
temporal calculations), the time-based split (rejects unsorted input,
rejects invalid fractions), and the cost-based threshold optimizer
(verified against contrived cases where the correct threshold is known
analytically — e.g. when missing fraud is priced far higher than a false
positive, the optimizer must pick a threshold low enough to catch it).

## Suggested resume bullet points

- *Built a fraud detection pipeline on real transaction data (XGBoost,
  Random Forest) with a time-based validation split to prevent temporal
  leakage, achieving 0.73–0.79 PR-AUC on a severely imbalanced (0.18%
  positive) classification task.*
- *Designed a cost-based threshold optimization framework translating
  model output probabilities into a business-facing operating
  recommendation, given explicit assumptions about false-negative and
  false-positive costs.*
- *Engineered population-level temporal and amount-context features
  from transaction timestamps, and evaluated their marginal contribution
  against PCA-anonymized baseline features via feature importance analysis.*

## Extending it

- Swap the cost assumptions in `app.py`'s Cost-Based Threshold tab for
  real figures if you have them from a specific business context.
- If you get access to data with an account/customer ID, add true
  per-customer velocity features (transactions in last 1h/24h,
  time-since-last-transaction) in `src/features.py` — the module is
  structured so that's a self-contained addition.
- Add SHAP values for per-transaction explanations (useful for actually
  explaining *why* a given transaction was flagged, not just that it was).

## License

Code: MIT — see `LICENSE`. Data: see the citation and provenance note
above and in `LICENSE`.
