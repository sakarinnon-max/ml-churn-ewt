"""Checks for ch08 — Deploy: batch scoring รายเดือน + overlap กับ tier + monitoring plan

Expected values are recomputed live from DATA_DIR + the model saved at
models/churn_model.joblib (the same one the notebook's setup cell loads/fits),
so the checks stay correct no matter which model ch06 saved.
When IS_SAMPLE is False (real data), only structure/range checks run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks import CheckFail, expect_cols, expect_df, register

REASON_COLS = ["reason_1", "reason_2", "reason_3"]

# cache: (month, model_mtime) -> canonical risk DataFrame (sorted desc)
_RISK_CACHE: dict[tuple, pd.DataFrame] = {}


def _expect_filled(name: str, value):
    if value is None:
        raise CheckFail(f"`{name}` ยังเป็น None อยู่ — เติมช่องว่าง ____ ก่อนนะครับ")


def _load(name: str, dates=()) -> pd.DataFrame | None:
    from src.config import DATA_DIR
    p = DATA_DIR / f"{name}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=list(dates))


def _model_path():
    from src.config import MODELS_DIR
    return MODELS_DIR / "churn_model.joblib"


def _canonical_risk(month: str) -> pd.DataFrame | None:
    """Recompute the canonical risk table for one month with the saved model."""
    mp = _model_path()
    if not mp.exists():
        raise CheckFail("ยังไม่มี models/churn_model.joblib — กลับไปรันเซลล์ setup "
                        "ต้นบทก่อน (มัน fit + save ให้อัตโนมัติ)")
    key = (month, mp.stat().st_mtime_ns)
    if key in _RISK_CACHE:
        return _RISK_CACHE[key]

    labels = _load("labels_monthly", ["churn_date"])
    students = _load("students", ["signup_date"])
    attendance = _load("attendance_long", ["ep_final_date", "week_start"])
    attempts = _load("exam_attempts", ["submitted_at"])
    weekly = _load("weekly_metrics", ["week_start", "week_end"])
    if any(t is None for t in (labels, students, attendance, attempts, weekly)):
        return None

    import joblib
    from src import churn_utils
    lab_m = labels[labels["month"] == month].copy()
    feats = churn_utils.build_features_monthly(lab_m, attendance, attempts, weekly, students)
    try:
        bundle = joblib.load(mp)
        pipeline = bundle["pipeline"]
        num = list(bundle["features"]["numeric"])
        cat = list(bundle["features"]["categorical"])
        feats["risk_score"] = pipeline.predict_proba(feats[num + cat])[:, 1]
    except Exception as e:
        raise CheckFail(f"ใช้โมเดล models/churn_model.joblib ไม่ได้ ({type(e).__name__}) — "
                        "รันเซลล์ setup ต้นบทอีกครั้ง (ถ้ากล่องเก่าใช้ไม่ได้ มันจะ fit ใหม่ให้)")
    out = feats.sort_values("risk_score", ascending=False).reset_index(drop=True)
    _RISK_CACHE.clear()          # keep at most a couple of months around
    _RISK_CACHE[key] = out
    return out


def _basic_risk_table(obj, name: str, month: str):
    """Shared structure/range checks for a risk table (8.1 in-notebook, 8.2 CSV)."""
    expect_df(obj, name)
    expect_cols(obj, ["student_key", "risk_score"] + REASON_COLS, name)
    rs = pd.to_numeric(obj["risk_score"], errors="coerce")
    if rs.isna().any():
        raise CheckFail("risk_score มีค่า NaN/ไม่ใช่ตัวเลข — predict_proba กับคอลัมน์ "
                        "feature ครบชุด (num + cat) หรือยัง?")
    if not ((rs >= 0) & (rs <= 1)).all():
        raise CheckFail("risk_score ต้องอยู่ในช่วง 0–1 — ใช้ predict_proba แล้วเอา "
                        "คอลัมน์หลัง [:, 1] (ไม่ใช่ predict หรือ decision_function)")
    for c in REASON_COLS:
        col = obj[c]
        if col.isna().any() or (col.astype(str).str.strip() == "").any():
            raise CheckFail(f"คอลัมน์ {c} มีช่องว่าง — เหตุผลต้องครบทุกคน "
                            f"(ดู helper top3_reasons_thai ในเซลล์ตัวอย่าง)")
    if obj["student_key"].duplicated().any():
        raise CheckFail(f"{name} มี student_key ซ้ำ — 1 คนต้องมีแถวเดียวต่อเดือน "
                        f"(เช็คว่ากรอง labels เฉพาะเดือน {month} เดือนเดียว)")
    diffs = rs.to_numpy()
    if len(diffs) > 1 and np.any(np.diff(diffs) > 1e-9):
        raise CheckFail("ยังไม่ได้เรียงจากเสี่ยงมาก → น้อย — "
                        "sort_values('risk_score', ascending=False) ก่อนส่งนะครับ")
    return rs


def _compare_scores(obj, month: str, name: str):
    """Sample mode: learner's risk_score must match canonical recomputation."""
    exp = _canonical_risk(month)
    if exp is None:
        return
    want = exp.set_index("student_key")["risk_score"]
    got = obj.set_index("student_key")["risk_score"].astype(float)
    if set(got.index) != set(want.index):
        raise CheckFail(f"รายชื่อนักเรียนใน {name} ไม่ตรงกับเด็ก active เดือน {month} — "
                        f"กรอง labels ด้วย month == '{month}' แล้วส่งเข้า "
                        f"build_features_monthly ใช่ไหม?")
    diff = (got - want.reindex(got.index)).abs().max()
    if diff > 1e-6:
        raise CheckFail("คะแนนความเสี่ยงไม่ตรงกับสูตร canonical (ต่างสูงสุด "
                        f"{diff:.4f}) — feature ต้องมาจาก churn_utils.build_features_monthly "
                        "แล้ว predict ด้วยโมเดลจาก models/churn_model.joblib ตัวเดิม")


# ---------------------------------------------------------------- ex_08_01

@register("ex_08_01")
def _ex_08_01(obj):
    _expect_filled("risk_jul", obj)
    _basic_risk_table(obj, "risk_jul", "2026-07")

    from src.config import IS_SAMPLE
    if not IS_SAMPLE:
        return
    labels = _load("labels_monthly", ["churn_date"])
    if labels is None:
        return
    n_exp = int((labels["month"] == "2026-07").sum())
    if len(obj) != n_exp:
        n_year = int((labels["year"] == 2569).sum())
        if len(obj) == n_year:
            raise CheckFail(f"ได้ {len(obj):,} แถว = ทุกเดือนของปี 2569 — "
                            f"ต้องกรอง labels เอาเฉพาะเดือน SCORE_MONTH ก่อนสร้าง feature")
        raise CheckFail(f"ได้ {len(obj):,} แถว แต่เด็ก active เดือน 2026-07 มี {n_exp} คน — "
                        f"เช็คเงื่อนไขกรอง month == SCORE_MONTH อีกที")
    _compare_scores(obj, "2026-07", "risk_jul")


# ---------------------------------------------------------------- ex_08_02

@register("ex_08_02")
def _ex_08_02(obj):
    if obj is None:
        raise CheckFail("risk_report ยังเป็น None — เติม month + script ในเซลล์ก่อน "
                        "(ถ้าเลือกสคริปต์ของคุณเอง ต้องเติม TODO 3 จุดใน scoring/score_month.py "
                        "ให้รันผ่านด้วยนะครับ)")
    _basic_risk_table(obj, "risk_report", "2026-07")

    from src.config import IS_SAMPLE
    if not IS_SAMPLE:
        return
    labels = _load("labels_monthly", ["churn_date"])
    if labels is None:
        return
    n_jun = int((labels["month"] == "2026-06").sum())
    n_jul = int((labels["month"] == "2026-07").sum())
    if len(obj) != n_jul:
        if len(obj) == n_jun:
            raise CheckFail(f"ได้ {len(obj)} แถว = จำนวนของเดือน มิ.ย. — ข้อนี้ให้รันสคริปต์"
                            f"กับเดือน ก.ค. 69 (เดือน holdout เดียวกับข้อ 8.1) นะครับ "
                            f"(month เขียนแบบ YYYY-MM)")
        raise CheckFail(f"report มี {len(obj)} แถว แต่เด็ก active เดือน 2026-07 มี {n_jul} คน — "
                        f"เช็คว่าในสคริปต์กรอง labels ด้วย args.month")
    _compare_scores(obj, "2026-07", "risk_report")


# ---------------------------------------------------------------- ex_08_03

@register("ex_08_03")
def _ex_08_03(obj):
    _expect_filled("overlap_jul", obj)
    if not isinstance(obj, dict):
        raise CheckFail(f"overlap_jul ควรเป็น dict แต่ได้ {type(obj).__name__} — "
                        "อย่าแก้โครง dict ที่ให้ไว้ในเซลล์นะครับ")
    need = ["n_model", "n_tier", "n_overlap", "jaccard"]
    missing = [k for k in need if k not in obj]
    if missing:
        raise CheckFail(f"dict ขาด key {missing} — ใช้โครงที่ให้ไว้ในเซลล์")
    n_model, n_tier = int(obj["n_model"]), int(obj["n_tier"])
    n_overlap, jac = int(obj["n_overlap"]), float(obj["jaccard"])

    if n_overlap > min(n_model, n_tier):
        raise CheckFail("n_overlap มากกว่าขนาดเซ็ตที่เล็กกว่า — เป็นไปไม่ได้ "
                        "เช็คว่าใช้ & (intersection) ไม่ใช่ | (union)")
    union = n_model + n_tier - n_overlap
    if union > 0 and abs(jac - n_overlap / union) > 1e-6:
        raise CheckFail("jaccard ไม่ตรงสูตร ซ้อน ÷ union — union = รวมสองเซ็ตแบบไม่นับซ้ำ "
                        "(len(top30_jul | tier_jul))")
    if not (0 <= jac <= 1):
        raise CheckFail("jaccard ต้องอยู่ในช่วง 0–1")

    from src.config import IS_SAMPLE
    if not IS_SAMPLE:
        return

    weekly = _load("weekly_metrics", ["week_start", "week_end"])
    labels = _load("labels_monthly", ["churn_date"])
    if weekly is None or labels is None:
        return
    month = "2026-07"
    wm = weekly[(weekly["year"] == 2569)
                & (weekly["week_start"].dt.strftime("%Y-%m") == month)]
    last_week = wm.sort_values("week_start").groupby("student_key").tail(1)
    tier_all = set(last_week.loc[last_week["tier"].isin(["red", "orange"]), "student_key"])
    active = set(labels.loc[labels["month"] == month, "student_key"])
    tier_exp = tier_all & active

    if n_tier != len(tier_exp):
        if n_tier == len(tier_all):
            raise CheckFail(f"n_tier = {n_tier} คือเด็กแดง/ส้มทั้งหมด — "
                            f"ต้อง intersect กับเด็กที่ยัง active เดือน ก.ค. ด้วย (& active)")
        raise CheckFail(f"n_tier = {n_tier} แต่เด็กแดง/ส้มที่ active เดือน ก.ค. มี "
                        f"{len(tier_exp)} คน — เช็คว่าใช้ 'สัปดาห์สุดท้ายของเดือน' ต่อคน "
                        f"(sort_values → groupby → tail(1)) และ isin(['red','orange'])")

    exp_risk = _canonical_risk(month)
    if exp_risk is None:
        return
    top30_exp = set(exp_risk.head(30)["student_key"])
    if n_model != len(top30_exp):
        raise CheckFail(f"n_model = {n_model} — top30_jul ควรมี {len(top30_exp)} คน "
                        f"(set ของ student_key จาก risk_jul.head(30))")
    n_overlap_exp = len(top30_exp & tier_exp)
    if n_overlap != n_overlap_exp:
        raise CheckFail(f"n_overlap = {n_overlap} แต่คำนวณจากโมเดลในเครื่องคุณได้ "
                        f"{n_overlap_exp} — เช็คว่า top30_jul มาจาก risk_jul ของข้อ 8.1 "
                        f"(เรียง risk มาก→น้อยแล้วค่อย head(30))")


# ---------------------------------------------------------------- ex_08_04

@register("ex_08_04")
def _ex_08_04(obj):
    _expect_filled("my_checklist", obj)
    if not isinstance(obj, str):
        raise CheckFail(f"my_checklist ควรเป็น string แต่ได้ {type(obj).__name__}")
    if "____" in obj:
        raise CheckFail("ยังมี ____ ค้างอยู่ — เติมให้ครบทุกช่องก่อนนะครับ")
    bullets = [ln for ln in obj.splitlines()
               if ln.strip().startswith("-") and len(ln.strip()) >= 12]
    if len(bullets) < 4:
        raise CheckFail(f"ตอนนี้มี bullet จริงจัง {len(bullets)} ข้อ — เอาอย่างน้อย 4 ข้อ "
                        f"(หมวดละอย่างน้อย 1) เขียนแบบที่สั่งงานตัวเองได้จริงนะครับ")
    if len(obj.strip()) < 250:
        raise CheckFail("ยังสั้นไปหน่อยครับ — ขยายแต่ละข้อให้บอก 'ทำอะไร ดูตรงไหน เกณฑ์เท่าไหร่' "
                        "แล้วค่อยเปิดเฉลยเทียบ")
