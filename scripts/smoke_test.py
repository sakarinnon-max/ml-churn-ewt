"""End-to-end smoke test of the canonical pipeline on data/sample.

Asserts:
  1. contracts validate for every sample table
  2. build_features_monthly runs and passes the features contract
  3. LogisticRegression pipeline beats the base rate clearly (AP >= 2x base)
  4. the planted leakage trap gives near-perfect AP (so ch04's lesson works)

Run:  ./venv/bin/python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import churn_utils, contracts
from src.config import PROJECT_ROOT

SAMPLE = PROJECT_ROOT / "data" / "sample"


def load(name, dates=()):
    df = pd.read_csv(SAMPLE / f"{name}.csv", parse_dates=list(dates))
    contracts.validate_df(name, df)
    return df


students = load("students", ["signup_date"])
attendance = load("attendance_long", ["ep_final_date", "week_start"])
exams = load("exams")
attempts = load("exam_attempts", ["submitted_at"])
events = load("enrollment_events", ["event_date"])
labels = load("labels_monthly", ["churn_date"])
weekly = load("weekly_metrics", ["week_start", "week_end"])
print("1) contracts ok")

features = churn_utils.build_features_monthly(labels, attendance, attempts, weekly, students)
contracts.validate_df("features_monthly", features)
print(f"2) features ok: {features.shape}")

df = features.dropna(subset=["churned_next_month"]).reset_index(drop=True)
base = df["churned_next_month"].mean()
num, cat = churn_utils.NUMERIC_FEATURES, churn_utils.CATEGORICAL_FEATURES
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])
model = Pipeline([("prep", prep),
                  ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])

train = df[df["year"] == 2568]
test = df[df["year"] == 2569]
model.fit(train[num + cat], train["churned_next_month"])
ap = average_precision_score(test["churned_next_month"],
                             model.predict_proba(test[num + cat])[:, 1])
print(f"3) AP train68->test69 = {ap:.3f}  (base rate {test['churned_next_month'].mean():.3f})")
assert ap >= 2 * test["churned_next_month"].mean(), "signal too weak — model barely beats base rate"

trap = pd.read_csv(SAMPLE / "extra_features.csv")
leaky = df.merge(trap, on=["student_key", "month"], how="left")
leaky = leaky.dropna(subset=["att_next_month_pct"])
ap_leak = average_precision_score(
    leaky["churned_next_month"], -leaky["att_next_month_pct"])
print(f"4) leakage-trap AP = {ap_leak:.3f} (should be suspiciously high, >0.9)")
assert ap_leak > 0.9, "trap not leaky enough for the ch04 lesson"

p30 = churn_utils.precision_at_k(test["churned_next_month"],
                                 model.predict_proba(test[num + cat])[:, 1], 30)
tier = churn_utils.tier_baseline_score(test, weekly)
p30_tier = churn_utils.precision_at_k(test["churned_next_month"], tier, 30)
print(f"5) precision@30 model={p30:.2f} vs tier-heuristic={p30_tier:.2f}")

print("\nSMOKE TEST PASSED")
