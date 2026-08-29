"""Checks for ch06 — Validation ที่เชื่อถือได้ + threshold ธุรกิจ.

Every expected value is recomputed live from DATA_DIR with the canonical
churn_utils helpers (expanding_window_splits / bootstrap_ci / precision_at_k /
tier_baseline_score) — no hardcoded magic numbers. The model recipes below are
byte-for-byte the same `make_logreg()` / `make_hgb()` the notebook defines, so
the learner's numbers and the canonical numbers must line up.

Heavy pieces (features → 9-fold duel → OOF) are computed once and cached.
If input files are missing, or when running on real data, value comparisons
degrade to structure/range checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks import CheckFail, expect_cols, expect_df, register, sample_path
from src.config import IS_SAMPLE, MODELS_DIR

_CACHE: dict = {}

_INPUT_FILES = ["labels_monthly.csv", "attendance_long.csv", "exam_attempts.csv",
                "weekly_metrics.csv", "students.csv"]

_MONTH_COL = "month"
_MENTOR_CAPACITY = 30          # สายที่ mentor โทรไหวต่อเดือน (สมมติฐานธุรกิจของบทนี้)
_SEASON_69_START = "2026-03"


# ---------------------------------------------------------------- small helpers

def _need(obj, name: str):
    """กันเคส skeleton ที่ยังไม่ได้เติม (____ = None)"""
    if obj is None:
        raise CheckFail(f"`{name}` ยังเป็น None อยู่ — เติมช่องว่าง ____ ก่อนนะครับ")


def _as_float(v, name: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise CheckFail(f"{name} ควรเป็นตัวเลข แต่ได้ {v!r}")


def _close(got, exp, tol: float, name: str, hint: str = ""):
    g, e = _as_float(got, name), float(exp)
    if np.isnan(g) or abs(g - e) > tol:
        raise CheckFail(f"{name} = {g:.3f} แต่ค่าที่ถูกคือประมาณ {e:.3f}"
                        + (f" · {hint}" if hint else ""))


def _to_dict(obj, name: str) -> dict:
    """รับได้ทั้ง dict และ pandas Series (เผื่อคนลืม .to_dict())"""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    raise CheckFail(f"{name} ต้องเป็น dict แต่ได้ {type(obj).__name__} — "
                    f"ถ้าเป็นแถวของ DataFrame ให้ปิดท้ายด้วย .to_dict()")


# ---------------------------------------------------------------- canonical data

def _tables():
    """โหลดวัตถุดิบทั้ง 5 (cache) — None ถ้าไฟล์ไม่ครบ"""
    if "tables" in _CACHE:
        return _CACHE["tables"]
    if not all(sample_path(f).exists() for f in _INPUT_FILES):
        _CACHE["tables"] = None
        return None
    labels = pd.read_csv(sample_path("labels_monthly.csv"), dtype={"month": str},
                         parse_dates=["churn_date"])
    att = pd.read_csv(sample_path("attendance_long.csv"),
                      parse_dates=["ep_final_date", "week_start"])
    atm = pd.read_csv(sample_path("exam_attempts.csv"), parse_dates=["submitted_at"])
    wm = pd.read_csv(sample_path("weekly_metrics.csv"),
                     parse_dates=["week_start", "week_end"])
    stu = pd.read_csv(sample_path("students.csv"), parse_dates=["signup_date"])
    _CACHE["tables"] = (labels, att, atm, wm, stu)
    return _CACHE["tables"]


def _labeled() -> pd.DataFrame | None:
    """ตาราง features เฉพาะแถวที่มี label — ก้อนเดียวกับตัวแปร `labeled` ในสมุด"""
    if "labeled" in _CACHE:
        return _CACHE["labeled"]
    t = _tables()
    if t is None:
        _CACHE["labeled"] = None
        return None
    from src import churn_utils
    feats = churn_utils.build_features_monthly(*t)
    labeled = feats.dropna(subset=["churned_next_month"])
    if not IS_SAMPLE:
        # ข้อมูลจริง: ใช้ exclusions เดียวกับสมุด (การตัดสินใจ CEO 11 & 29 ส.ค. 2026)
        from src.config import DATA_DIR
        _lab_dir = DATA_DIR.parent / "raw" / "labels"
        excl_path = _lab_dir / "exclude_pairs_2569.csv"
        if excl_path.exists():
            excl = pd.read_csv(excl_path, dtype={"month": str})
            labeled = (labeled.merge(excl[["student_key", "month"]],
                                     on=["student_key", "month"], how="left", indicator=True)
                              .query('_merge == "left_only"').drop(columns="_merge"))
        replay_path = _lab_dir / "replay_students_2569.csv"
        if replay_path.exists():
            replay = set(pd.read_csv(replay_path)["student_key"])
            labeled = labeled[~labeled["student_key"].isin(replay)]
    _CACHE["labeled"] = labeled.reset_index(drop=True)
    return _CACHE["labeled"]


def _make_logreg():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from src import churn_utils
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), churn_utils.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), churn_utils.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", prep),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def _make_hgb():
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    from src import churn_utils
    from sklearn.impute import SimpleImputer
    prep = ColumnTransformer([
        # ข้อมูลจริง: ปี 68 มีคอลัมน์ที่ว่างทั้งปี (practice_pct) — HGB binning พังถ้า passthrough
        ("num", SimpleImputer(strategy="median", keep_empty_features=True),
         churn_utils.NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         churn_utils.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", prep),
                     ("clf", HistGradientBoostingClassifier(random_state=42,
                                                            class_weight="balanced"))])


def _x_cols() -> list[str]:
    from src import churn_utils
    return churn_utils.NUMERIC_FEATURES + churn_utils.CATEGORICAL_FEATURES


def _folds() -> pd.DataFrame | None:
    """ตารางขนาดของแต่ละยก (test_month, n_train, n_test) — กรรมการของข้อ 6.1"""
    if "folds" in _CACHE:
        return _CACHE["folds"]
    lab = _labeled()
    if lab is None:
        _CACHE["folds"] = None
        return None
    from src import churn_utils
    _CACHE["folds"] = pd.DataFrame([
        {"test_month": m, "n_train": len(tr), "n_test": len(te)}
        for tr, te, m in churn_utils.expanding_window_splits(
            lab, _MONTH_COL, min_train_months=2)
    ])
    return _CACHE["folds"]


def _duel():
    """(duel_df, oof, oof69) ฉบับ canonical — fit 2 โมเดล × 9 ยก (~3 วิ ครั้งเดียว)"""
    if "duel" in _CACHE:
        return _CACHE["duel"]
    lab = _labeled()
    if lab is None:
        _CACHE["duel"] = None
        return None
    from sklearn.metrics import average_precision_score

    from src import churn_utils
    xc = _x_cols()
    rows, parts = [], []
    for tr_idx, te_idx, m in churn_utils.expanding_window_splits(
            lab, _MONTH_COL, min_train_months=2):
        X_tr, y_tr = lab.loc[tr_idx, xc], lab.loc[tr_idx, "churned_next_month"]
        X_te, y_te = lab.loc[te_idx, xc], lab.loc[te_idx, "churned_next_month"]
        p_lr = _make_logreg().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        p_hgb = _make_hgb().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        rows.append({"test_month": m, "n_test": len(te_idx),
                     "churn_test": int(y_te.sum()),
                     "ap_logreg": average_precision_score(y_te, p_lr),
                     "ap_hgb": average_precision_score(y_te, p_hgb)})
        parts.append(pd.DataFrame({
            "student_key": lab.loc[te_idx, "student_key"].values, "month": m,
            "y_true": y_te.values, "p_logreg": p_lr, "p_hgb": p_hgb}))
    duel = pd.DataFrame(rows)
    oof = pd.concat(parts, ignore_index=True)
    oof["p_model"] = oof["p_logreg"]
    oof69 = oof[oof["month"] >= _SEASON_69_START].reset_index(drop=True)
    _CACHE["duel"] = (duel, oof, oof69)
    return _CACHE["duel"]


def _sweep() -> pd.DataFrame | None:
    """ตาราง threshold sweep ฉบับ canonical (0.30–0.90 ทีละ 0.05)"""
    if "sweep" in _CACHE:
        return _CACHE["sweep"]
    d = _duel()
    if d is None:
        _CACHE["sweep"] = None
        return None
    _, _, oof69 = d
    n_months = oof69["month"].nunique()
    rows = []
    for thr in np.round(np.arange(0.30, 0.91, 0.05), 2):
        flag = oof69["p_model"] >= thr
        n_flag = int(flag.sum())
        rows.append({"threshold": float(thr),
                     "calls_per_month": n_flag / n_months,
                     "precision": oof69.loc[flag, "y_true"].mean() if n_flag else np.nan,
                     "recall": oof69.loc[flag, "y_true"].sum() / oof69["y_true"].sum()})
    _CACHE["sweep"] = pd.DataFrame(rows)
    return _CACHE["sweep"]


# ---------------------------------------------------------------- ex_06_01

@register("ex_06_01")
def _(obj):
    _need(obj, "fold_table")
    expect_cols(obj, ["test_month", "n_train", "n_test"], "fold_table")
    exp = _folds()
    if exp is None:
        if len(obj) < 2:
            raise CheckFail("fold_table มีไม่ถึง 2 ยก — expanding window ต้องได้หลายยก")
        return
    if len(obj) != len(exp):
        raise CheckFail(
            f"fold_table มี {len(obj)} ยก แต่ควรได้ {len(exp)} ยก — "
            f"ข้าม 2 เดือนแรกแล้วหรือยัง (months[2:]) และวนครบทุกเดือนที่เหลือไหม?")
    got = obj.reset_index(drop=True).sort_values("test_month").reset_index(drop=True)
    exp = exp.sort_values("test_month").reset_index(drop=True)
    if list(got["test_month"].astype(str)) != list(exp["test_month"].astype(str)):
        raise CheckFail(
            f"เดือนที่สอบไม่ตรง — ได้ {list(got['test_month'].astype(str))[:3]}... "
            f"ควรเริ่มที่ {exp['test_month'].iloc[0]} และจบที่ {exp['test_month'].iloc[-1]}")
    for col, hint in [("n_train", "train คือทุกแถวที่ month < m (อดีตทั้งหมด ไม่ใช่เดือนก่อนหน้าเดือนเดียว)"),
                      ("n_test", "test คือแถวที่ month == m เท่านั้น")]:
        bad = got.index[pd.to_numeric(got[col], errors="coerce") != exp[col]]
        if len(bad):
            i = bad[0]
            raise CheckFail(
                f"`{col}` ของเดือน {exp['test_month'].iloc[i]} ได้ {got[col].iloc[i]} "
                f"ควรได้ {exp[col].iloc[i]} · {hint}")


# ---------------------------------------------------------------- ex_06_02

@register("ex_06_02")
def _(obj):
    _need(obj, "duel")
    expect_cols(obj, ["test_month", "n_test", "churn_test", "ap_logreg", "ap_hgb"], "duel")
    d = _duel()
    if d is None:
        if not obj["ap_logreg"].between(0, 1).all():
            raise CheckFail("ค่า AP ต้องอยู่ระหว่าง 0–1 — ส่ง predict_proba คอลัมน์ที่ 1 ใช่ไหม?")
        return
    exp = d[0].sort_values("test_month").reset_index(drop=True)
    if len(obj) != len(exp):
        raise CheckFail(f"duel มี {len(obj)} แถว แต่ควรได้ {len(exp)} แถว (1 แถว = 1 ยก) — "
                        f"ใช้ churn_utils.expanding_window_splits(..., min_train_months=2) หรือเปล่า?")
    got = obj.sort_values("test_month").reset_index(drop=True)
    if list(got["test_month"].astype(str)) != list(exp["test_month"].astype(str)):
        raise CheckFail("เดือนที่สอบไม่ตรงกับยกของ expanding window — "
                        "ตัวแปรที่ 3 ที่ generator คืนมาคือ test_month ครับ")
    for col in ["n_test", "churn_test"]:
        bad = got.index[pd.to_numeric(got[col], errors="coerce") != exp[col]]
        if len(bad):
            i = bad[0]
            raise CheckFail(
                f"`{col}` เดือน {exp['test_month'].iloc[i]} ได้ {got[col].iloc[i]} "
                f"ควรได้ {exp[col].iloc[i]} — n_test = จำนวนแถว test, "
                f"churn_test = ผลรวมของ label ในเดือนนั้น")
    for col, tol, hint in [
        ("ap_logreg", 0.03, "fit ด้วย train ของยกนั้น แล้ววัด AP บน test ของยกเดียวกัน (make_logreg)"),
        # HGB เพี้ยนได้เล็กน้อยตาม environment (OpenMP/BLAS) — เลยผ่อนช่วงให้กว้างกว่า
        ("ap_hgb", 0.08, "ใช้ make_hgb() ตัวเดียวกับ cell ตัวอย่าง (random_state=42)"),
    ]:
        g = pd.to_numeric(got[col], errors="coerce")
        diff = (g - exp[col]).abs()
        if diff.isna().any() or (diff > tol).any():
            i = int(diff.fillna(9).argmax())
            raise CheckFail(
                f"`{col}` เดือน {exp['test_month'].iloc[i]} ได้ {g.iloc[i]:.3f} "
                f"ควรได้ประมาณ {exp[col].iloc[i]:.3f} · {hint}")


# ---------------------------------------------------------------- ex_06_03

@register("ex_06_03")
def _(obj):
    _need(obj, "ci_ap")
    if isinstance(obj, (tuple, list, np.ndarray)):
        vals = list(obj)
    else:
        raise CheckFail(f"ci_ap ควรเป็น tuple 3 ตัว (point, lo, hi) แต่ได้ "
                        f"{type(obj).__name__} — เก็บผลจาก bootstrap_ci ทั้งก้อนเลยครับ")
    if len(vals) != 3:
        raise CheckFail(f"ci_ap มี {len(vals)} ค่า แต่ bootstrap_ci คืน 3 ค่า (point, lo, hi)")
    point, lo, hi = (_as_float(v, "ค่าใน ci_ap") for v in vals)
    if not (lo <= point <= hi):
        raise CheckFail(f"ลำดับค่าเพี้ยน: ได้ point={point:.3f}, lo={lo:.3f}, hi={hi:.3f} — "
                        f"ต้องเป็น (point, lo, hi) ตามที่ฟังก์ชันคืนมา อย่าสลับตำแหน่ง")
    d = _duel()
    if d is None:
        if not (0 <= lo <= hi <= 1):
            raise CheckFail("ค่า AP ต้องอยู่ในช่วง 0–1")
        return
    from sklearn.metrics import average_precision_score

    from src import churn_utils
    _, _, oof69 = d
    exp = churn_utils.bootstrap_ci(oof69["y_true"], oof69["p_model"],
                                   average_precision_score)
    _close(point, exp[0], 0.02, "จุดประมาณ AP",
           "ใช้คอลัมน์ p_model (ผู้ชนะ) บน oof69 นะครับ ไม่ใช่ p_hgb")
    _close(lo, exp[1], 0.03, "ขอบล่างของ CI",
           "อย่าเปลี่ยน n_boot/seed — ปล่อย default (1000 รอบ, seed 42)")
    _close(hi, exp[2], 0.03, "ขอบบนของ CI",
           "อย่าเปลี่ยน n_boot/seed — ปล่อย default (1000 รอบ, seed 42)")


# ---------------------------------------------------------------- ex_06_04

@register("ex_06_04")
def _(obj):
    _need(obj, "op_point")
    got = _to_dict(obj, "op_point")
    need = ["threshold", "calls_per_month", "precision", "recall"]
    missing = [k for k in need if k not in got]
    if missing:
        raise CheckFail(f"op_point ขาด key {missing} — ต้องมีครบ 4 ช่อง "
                        f"(มาจากแถวของ sweep ที่เลือกไว้)")
    exp_tab = _sweep()
    if exp_tab is None:
        for k in ["precision", "recall"]:
            v = _as_float(got[k], k)
            if not (0 <= v <= 1):
                raise CheckFail(f"{k} = {v:.3f} ต้องอยู่ในช่วง 0–1")
        return
    exp = exp_tab.iloc[(exp_tab["calls_per_month"] - _MENTOR_CAPACITY).abs().argmin()]
    calls = _as_float(got["calls_per_month"], "calls_per_month")
    if abs(calls - exp["calls_per_month"]) > 0.5:
        raise CheckFail(
            f"จุดที่เลือกโทร {calls:.1f} สาย/เดือน แต่จุดที่ใกล้กำลังของ mentor "
            f"({_MENTOR_CAPACITY} สาย) ที่สุดคือ {exp['calls_per_month']:.1f} สาย/เดือน — "
            f"หารจำนวนคนที่ติดลิสต์ด้วยจำนวนเดือน (5 เดือนของปี 69) แล้วหาแถวที่ "
            f"|calls_per_month - 30| น้อยที่สุดหรือยัง?")
    _close(got["threshold"], exp["threshold"], 0.011, "threshold ของจุดที่เลือก")
    _close(got["precision"], exp["precision"], 0.02, "precision ที่จุดนั้น",
           "precision = สัดส่วน churn จริงในกลุ่มที่ติดลิสต์ (y_true ของแถวที่ flag)")
    _close(got["recall"], exp["recall"], 0.02, "recall ที่จุดนั้น",
           "recall = churn ที่จับได้ ÷ churn ทั้งหมดใน oof69")


# ---------------------------------------------------------------- ex_06_05

@register("ex_06_05")
def _(obj):
    _need(obj, "sensitivity")
    got = _to_dict(obj, "sensitivity")
    need = ["ap_all", "ap_high_conf", "n_dropped_rows"]
    missing = [k for k in need if k not in got]
    if missing:
        raise CheckFail(f"sensitivity ขาด key {missing} — ต้องมีครบ 3 ช่องตามโจทย์")
    d = _duel()
    ev_path = sample_path("enrollment_events.csv")
    if d is None or not ev_path.exists():
        for k in ["ap_all", "ap_high_conf"]:
            v = _as_float(got[k], k)
            if not (0 <= v <= 1):
                raise CheckFail(f"{k} = {v:.3f} ต้องอยู่ในช่วง 0–1")
        return
    from sklearn.metrics import average_precision_score
    _, _, oof69 = d
    events = pd.read_csv(ev_path)
    shaky = set(events.loc[events["confidence"].isin(["medium", "low"]), "student_key"])
    hi = oof69[~oof69["student_key"].isin(shaky)]

    n_drop = _as_float(got["n_dropped_rows"], "n_dropped_rows")
    exp_drop = len(oof69) - len(hi)
    if abs(n_drop - exp_drop) > 0.5:
        raise CheckFail(
            f"n_dropped_rows = {n_drop:.0f} แต่ควรได้ {exp_drop} — "
            f"ตัดออก**ทั้งคน** (ทุกแถวของนักเรียนที่มี event confidence medium/low) "
            f"ไม่ใช่ตัดเฉพาะ event ที่ไม่ชัวร์")
    _close(got["ap_all"], average_precision_score(oof69["y_true"], oof69["p_model"]),
           0.02, "ap_all", "วัดจาก oof69 เต็มก้อน คอลัมน์ p_model")
    _close(got["ap_high_conf"], average_precision_score(hi["y_true"], hi["p_model"]),
           0.02, "ap_high_conf", "วัดจากก้อนที่กรองเด็ก label ไม่ชัวร์ออกแล้ว")


# ---------------------------------------------------------------- ex_06_06

_BUNDLE_KEYS = ["pipeline", "features", "trained_through", "metrics"]
_METRIC_KEYS = ["ap_oof_2569", "p30_oof_2569", "p30_tier_2569"]


def _validate_bundle(b, where: str):
    """โครงสร้าง bundle ที่บท 07–08 จะเปิดใช้ — key ต้องครบและใช้งานได้จริง"""
    if not isinstance(b, dict):
        raise CheckFail(f"{where} ต้องเป็น dict แต่ได้ {type(b).__name__}")
    missing = [k for k in _BUNDLE_KEYS if k not in b]
    if missing:
        raise CheckFail(f"{where} ขาด key {missing} — บท 08 เปิดกล่องตามชื่อ key พวกนี้ "
                        f"ต้องครบทั้ง 4: {_BUNDLE_KEYS}")
    feats = b["features"]
    if not isinstance(feats, dict) or "numeric" not in feats or "categorical" not in feats:
        raise CheckFail(f"{where}['features'] ต้องเป็น dict ที่มี key "
                        f"'numeric' และ 'categorical'")
    from src import churn_utils
    if list(feats["numeric"]) != list(churn_utils.NUMERIC_FEATURES):
        raise CheckFail("features['numeric'] ไม่ตรงกับ churn_utils.NUMERIC_FEATURES — "
                        "ต้องใช้รายชื่อจาก churn_utils ตรงๆ (serve ต้องใช้ชุดเดียวกับ train)")
    if list(feats["categorical"]) != list(churn_utils.CATEGORICAL_FEATURES):
        raise CheckFail("features['categorical'] ไม่ตรงกับ churn_utils.CATEGORICAL_FEATURES")
    if not isinstance(b["trained_through"], str):
        raise CheckFail("trained_through ต้องเป็น string เดือน เช่น \"2026-06\"")
    if IS_SAMPLE and b["trained_through"] != "2026-06":
        raise CheckFail(f"trained_through = {b['trained_through']!r} — บทนี้ train ถึง "
                        f"\"2026-06\" (เว้น 2026-07 ไว้เป็น holdout ของบท 08)")
    m = b["metrics"]
    if not isinstance(m, dict):
        raise CheckFail("metrics ต้องเป็น dict")
    missing_m = [k for k in _METRIC_KEYS if k not in m]
    if missing_m:
        raise CheckFail(f"metrics ขาด key {missing_m} — ใส่ให้ครบ {_METRIC_KEYS} "
                        f"(ตัวเลขจาก cell GATE)")
    for k in _METRIC_KEYS:
        v = _as_float(m[k], f"metrics['{k}']")
        if not (0 <= v <= 1):
            raise CheckFail(f"metrics['{k}'] = {v} ควรอยู่ในช่วง 0–1")


def _predict(pipe, X, where: str) -> np.ndarray:
    if not hasattr(pipe, "predict_proba"):
        raise CheckFail(f"{where} ไม่ใช่โมเดลที่ predict_proba ได้ — "
                        f"ใส่ pipeline ที่ fit แล้วลงช่อง 'pipeline'")
    try:
        p = pipe.predict_proba(X)[:, 1]
    except Exception as e:
        raise CheckFail(f"{where} เรียก predict_proba ไม่ผ่าน ({type(e).__name__}: {e}) — "
                        f"fit ด้วย train_final[X_COLS] แล้วหรือยัง?")
    if np.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise CheckFail(f"{where} คืนค่าความน่าจะเป็นแปลกๆ (ต้องอยู่ 0–1 ไม่มี NaN)")
    return p


@register("ex_06_06")
def _(obj):
    _need(obj, "bundle")
    _validate_bundle(obj, "bundle")

    lab = _labeled()
    if lab is not None:
        xc = _x_cols()
        holdout = lab[lab["month"] == lab["month"].max()]
        p_got = _predict(obj["pipeline"], holdout[xc], "bundle['pipeline']")
        if IS_SAMPLE:
            train_final = lab[lab["month"] <= "2026-06"]
            canon = _make_logreg().fit(train_final[xc],
                                       train_final["churned_next_month"])
            p_exp = canon.predict_proba(holdout[xc])[:, 1]
            if float(np.abs(p_got - p_exp).mean()) > 0.03:
                raise CheckFail(
                    "คำทำนายของโมเดลในกล่องต่างจากที่ควรได้พอสมควร — เช็ค 2 จุด: "
                    "(1) fit ด้วย make_logreg() ผู้ชนะ ไม่ใช่ make_hgb() "
                    "(2) train ด้วยแถวที่ month <= \"2026-06\" ของ labeled (1,227 แถว)")

    # ไฟล์จริงต้องมี เพราะบท 07–08 เปิดจากดิสก์ ไม่ได้ใช้ตัวแปรในสมุด
    path = MODELS_DIR / "churn_model.joblib"
    if not path.exists():
        raise CheckFail("ยังไม่เจอไฟล์ models/churn_model.joblib — "
                        "อย่าลืม joblib.dump(bundle, MODELS_DIR / \"churn_model.joblib\")")
    import joblib
    try:
        back = joblib.load(path)
    except Exception as e:
        raise CheckFail(f"เปิดไฟล์ churn_model.joblib ไม่ได้ ({type(e).__name__}: {e})")
    _validate_bundle(back, "กล่องที่โหลดกลับมาจากไฟล์")
    if lab is not None:
        _predict(back["pipeline"], lab[_x_cols()].head(20),
                 "โมเดลในไฟล์ churn_model.joblib")
