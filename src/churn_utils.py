"""Canonical implementations shared across chapters.

Rule of the course: exercises have the CEO re-implement pieces of this module;
later chapters import THIS module (the canonical solution), never the CEO's
notebook variables — so an imperfect exercise never corrupts a later chapter.

All functions are pure pandas/sklearn; no DB access here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SEASONS, month_seq, next_month

# ---------------------------------------------------------------- plotting

def plot_style():
    """Thai-capable matplotlib style. Call once per notebook before plotting."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Sukhumvit Set", "Thonburi", "Tahoma", "Arial Unicode MS"]:
        if cand in available:
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.figsize"] = (9, 5)
    matplotlib.rcParams["axes.spines.top"] = False
    matplotlib.rcParams["axes.spines.right"] = False
    return plt


TIER_ORDER = ["green", "yellow", "orange", "red"]
TIER_SCORE = {"green": 0, "yellow": 1, "orange": 2, "red": 3}

# ---------------------------------------------------------------- labels (ch01)

def build_labels_monthly(
    events: pd.DataFrame,
    year: int,
    known_through: str | None = None,
) -> pd.DataFrame:
    """Canonical label builder (ch01 solution).

    Contract of enrollment_events:
      - enroll / re_enroll: event_month = first active month of an interval
      - cancel:             event_month = LAST active month (left at month end)
      - subject_drop is ignored here (it is a feature, not churn)

    known_through: last month whose cancels are fully observed.
      2568 (season over)  -> season end month
      2569 (in progress)  -> last fully finished month, e.g. "2026-07"

    Label per active (student, month t):
      churned_next_month = 1   cancel effective end of t (and t != season end)
                         = 0   stayed into t+1
                         = NaN censored: t == season end (Sep) OR t > known_through
    """
    season = SEASONS[year]
    start_m, end_m = season["start_month"], season["end_month"]
    months = month_seq(start_m, end_m)
    if known_through is None:
        known_through = end_m

    ev = events[(events["year"] == year)].copy()
    ev = ev[ev["event_type"].isin(["enroll", "re_enroll", "cancel"])]
    rows = []
    for student_key, g in ev.sort_values("event_date").groupby("student_key"):
        intervals: list[dict] = []  # {"start": m, "end": m, "churned": bool, ...}
        for _, e in g.iterrows():
            if e["event_type"] in ("enroll", "re_enroll"):
                m = max(e["event_month"], start_m)
                intervals.append({
                    "start": m, "end": end_m, "churned": False,
                    "churn_date": pd.NaT, "source": e["source"],
                })
            elif e["event_type"] == "cancel" and intervals:
                last = intervals[-1]
                if not last["churned"]:
                    last["end"] = min(e["event_month"], end_m)
                    last["churned"] = e["event_month"] < end_m
                    last["churn_date"] = e["event_date"]
                    last["source"] = e["source"]
        for iv in intervals:
            for t in months:
                if not (iv["start"] <= t <= iv["end"]):
                    continue
                if t == end_m or t > known_through:
                    label: float = np.nan
                elif t == iv["end"] and iv["churned"]:
                    label = 1.0
                else:
                    label = 0.0
                rows.append({
                    "student_key": student_key,
                    "year": year,
                    "month": t,
                    "active": 1,
                    "churned_next_month": label,
                    "churn_date": iv["churn_date"] if (t == iv["end"] and iv["churned"]) else pd.NaT,
                    "label_source": iv["source"],
                })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.drop_duplicates(["student_key", "month"], keep="last").reset_index(drop=True)
        out["churn_date"] = pd.to_datetime(out["churn_date"])
    return out

# ---------------------------------------------------------------- time helpers

def month_end_cutoff(month: str) -> pd.Timestamp:
    """First instant AFTER month t (exclusive upper bound for 'data up to end of t')."""
    return pd.Timestamp(next_month(month) + "-01")


def cutoff(df: pd.DataFrame, month: str, date_col: str) -> pd.DataFrame:
    """Rows with date_col strictly before the end of month t. The anti-leakage tool."""
    return df[df[date_col] < month_end_cutoff(month)]

# ---------------------------------------------------------------- weekly metrics

def compute_weekly_metrics(
    attendance_long: pd.DataFrame,
    exam_attempts: pd.DataFrame,
    exams: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Simplified weekly metrics matching the weekly_metrics contract.

    Used for synthetic data and as a 2568 fallback. For real 2569 data the
    richer pm-webapp report_engine replay (src/eduwise_extract.py) is preferred.
    Exam denominators: exams with available_from_week <= week (column optional;
    default = available all season).
    """
    att = attendance_long[attendance_long["year"] == year].copy()
    atm = exam_attempts[exam_attempts["year"] == year].copy()
    exs = exams[exams["year"] == year].copy()
    if "available_from_week" not in exs.columns:
        exs = exs.assign(available_from_week=att["week_start"].min())

    weeks = sorted(att["week_start"].unique())
    subj_of = att.groupby("student_key")["subject_id"].agg(lambda s: sorted(set(s)))
    atm["week_start"] = atm["submitted_at"].dt.normalize() - pd.to_timedelta(
        atm["submitted_at"].dt.weekday, unit="D")

    rows = []
    state: dict[str, dict] = {}
    for wk in weeks:
        wk = pd.Timestamp(wk)
        week_att = att[att["week_start"] == wk]
        cum_att = att[att["week_start"] <= wk]
        cum_atm = atm[atm["week_start"] <= wk]
        week_atm_count = atm[atm["week_start"] == wk].groupby("student_key").size()
        avail = exs[exs["available_from_week"] <= wk]
        for student_key, subjects in subj_of.items():
            st = state.setdefault(student_key, {"silent": 0, "streak": 0})
            w = week_att[week_att["student_key"] == student_key]
            week_total = len(w)
            week_present = int((w["status"] == "present").sum())
            week_pct = 100.0 * week_present / week_total if week_total else np.nan
            c = cum_att[cum_att["student_key"] == student_key]
            cum_pct = 100.0 * (c["status"] == "present").sum() / len(c) if len(c) else np.nan

            new_attempts = int(week_atm_count.get(student_key, 0))
            if week_total > 0 and week_present == 0 and new_attempts == 0:
                st["silent"] += 1
            elif week_total > 0:
                st["silent"] = 0
            if week_total > 0 and week_pct >= 75:
                st["streak"] += 1
            elif week_total > 0:
                st["streak"] = 0

            row = {"student_key": student_key, "year": year,
                   "week_start": wk, "week_end": wk + pd.Timedelta(days=6),
                   "att_week_pct": week_pct, "att_cum_pct": cum_pct,
                   "silent_weeks": st["silent"], "streak_weeks": st["streak"]}
            my_attempts = cum_atm[cum_atm["student_key"] == student_key]
            for etype in ("practice", "checkpoint"):
                total = int((avail["exam_type"].eq(etype)
                             & avail["subject_id"].isin(subjects)).sum())
                mine = my_attempts[my_attempts["exam_type"] == etype]
                done = mine["exam_id"].nunique()
                row[f"{etype}_done"] = int(done)
                row[f"{etype}_total"] = total
                row[f"{etype}_pct"] = 100.0 * done / total if total else np.nan
                row[f"{etype}_avg_score"] = mine["percentage"].mean() if done else np.nan

            if row["silent_weeks"] >= 2:
                tier = "red"
            elif np.isnan(week_pct):
                tier = "green"
            elif week_pct >= 75:
                tier = "green"
            elif week_pct >= 50:
                tier = "yellow"
            elif week_pct >= 25:
                tier = "orange"
            else:
                tier = "red"
            row["tier"] = tier
            rows.append(row)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- features (ch04)

STATIC_FEATURES = ["grade", "live_or_replay", "old_new", "signup_lateness", "n_subjects"]
NUMERIC_FEATURES = [
    "att_month_pct", "att_cum_pct", "att_delta",
    "practice_pct", "checkpoint_pct", "exam_avg_score", "new_attempts_month",
    "max_silent_weeks", "streak_weeks",
    "months_enrolled", "month_index", "signup_lateness", "n_subjects",
]
CATEGORICAL_FEATURES = ["grade", "live_or_replay", "old_new"]


def build_features_monthly(
    labels: pd.DataFrame,
    attendance_long: pd.DataFrame,
    exam_attempts: pd.DataFrame,
    weekly_metrics: pd.DataFrame,
    students: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical feature builder (ch04 solution). One row per active (student, month).

    Every feature uses data from <= month t only (enforced via cutoff())."""
    att = attendance_long.copy()
    att["month"] = att["ep_final_date"].dt.strftime("%Y-%m")
    atm = exam_attempts.copy()
    wm = weekly_metrics.copy()
    wm["month"] = wm["week_start"].dt.strftime("%Y-%m")

    rows = []
    for (year, month), lab in labels.groupby(["year", "month"]):
        att_cut = cutoff(att[att["year"] == year], month, "ep_final_date")
        atm_cut = cutoff(atm[atm["year"] == year], month, "submitted_at")
        wm_cut = wm[(wm["year"] == year) & (wm["month"] <= month)]
        season_start = SEASONS[year]["start_month"]
        month_index = month_seq(season_start, month).__len__()
        prev_m = None
        s_months = month_seq(season_start, month)
        if len(s_months) >= 2:
            prev_m = s_months[-2]

        att_m = att_cut[att_cut["month"] == month]
        att_p = att_cut[att_cut["month"] == prev_m] if prev_m else att_cut.iloc[0:0]
        wm_m = wm_cut[wm_cut["month"] == month]

        def pct(g):
            return 100.0 * (g["status"] == "present").mean() if len(g) else np.nan

        for _, r in lab.iterrows():
            sk = r["student_key"]
            a_m = att_m[att_m["student_key"] == sk]
            a_p = att_p[att_p["student_key"] == sk]
            a_c = att_cut[att_cut["student_key"] == sk]
            m_pct, p_pct = pct(a_m), pct(a_p)
            my_atm = atm_cut[atm_cut["student_key"] == sk]
            my_wm = wm_m[wm_m["student_key"] == sk].sort_values("week_start")
            last_wm = my_wm.iloc[-1] if len(my_wm) else None
            rows.append({
                "student_key": sk, "year": year, "month": month,
                "att_month_pct": m_pct,
                "att_cum_pct": pct(a_c),
                "att_delta": (m_pct - p_pct) if not (np.isnan(m_pct) or np.isnan(p_pct)) else np.nan,
                "practice_pct": last_wm["practice_pct"] if last_wm is not None else np.nan,
                "checkpoint_pct": last_wm["checkpoint_pct"] if last_wm is not None else np.nan,
                "exam_avg_score": my_atm["percentage"].mean() if len(my_atm) else np.nan,
                "new_attempts_month": int((my_atm["submitted_at"] >= pd.Timestamp(month + "-01")).sum()),
                "max_silent_weeks": int(my_wm["silent_weeks"].max()) if len(my_wm) else 0,
                "streak_weeks": int(last_wm["streak_weeks"]) if last_wm is not None else 0,
                "month_index": month_index,
                "churned_next_month": r["churned_next_month"],
            })
    feats = pd.DataFrame(rows)
    static = students[["student_key"] + STATIC_FEATURES].drop_duplicates("student_key")
    feats = feats.merge(static, on="student_key", how="left")
    feats["months_enrolled"] = feats["month_index"] - feats["signup_lateness"]
    feats.loc[feats["months_enrolled"] < 1, "months_enrolled"] = 1
    return feats

# ---------------------------------------------------------------- evaluation

def precision_at_k(y_true, y_score, k: int) -> float:
    """Precision among the k highest-scored rows (mentor can call ~k students)."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)[:k]
    return float(np.nanmean(y_true[order]))


def expanding_window_splits(df: pd.DataFrame, month_col: str = "month",
                            min_train_months: int = 2):
    """Yield (train_idx, test_idx, test_month): train on all months < m, test on m."""
    months = sorted(df[month_col].unique())
    for m in months[min_train_months:]:
        train_idx = df.index[df[month_col] < m]
        test_idx = df.index[df[month_col] == m]
        if len(test_idx):
            yield train_idx, test_idx, m


def bootstrap_ci(y_true, y_score, metric_fn, n_boot: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap CI for metric_fn(y_true, y_score)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    point = metric_fn(y_true, y_score)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric_fn(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def tier_baseline_score(features: pd.DataFrame, weekly_metrics: pd.DataFrame) -> pd.Series:
    """Existing heuristic as a baseline: tier at last week of month t -> 0..3 score."""
    wm = weekly_metrics.copy()
    wm["month"] = wm["week_start"].dt.strftime("%Y-%m")
    last = (wm.sort_values("week_start")
              .groupby(["student_key", "year", "month"], as_index=False).last())
    last["tier_score"] = last["tier"].map(TIER_SCORE)
    merged = features.merge(
        last[["student_key", "year", "month", "tier_score"]],
        on=["student_key", "year", "month"], how="left")
    return merged["tier_score"].fillna(0)

# ---------------------------------------------------------------- explain (ch07)

def logreg_contributions(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Per-row feature contributions (coef * transformed value) for a fitted
    Pipeline(prep, LogisticRegression). Returns one column per transformed feature."""
    prep = pipeline[:-1]
    clf = pipeline[-1]
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = prep.get_feature_names_out()
    contrib = Xt * clf.coef_[0]
    return pd.DataFrame(contrib, columns=names, index=X.index)
