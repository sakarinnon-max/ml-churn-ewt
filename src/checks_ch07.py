"""Checks for ch07 — อ่านโมเดล: ทำไมน้องคนนี้เสี่ยง

Expected values are recomputed live: build canonical features from DATA_DIR,
then fit the canonical LogReg on train (2568) — exactly what the notebook
interprets. The notebook only reuses models/churn_model.joblib when that bundle
IS this same canonical LogReg (coefficients match), so these expectations stay
valid no matter what another chapter last saved into models/. When IS_SAMPLE is
False, or the state can't be built, fall back to structure/range checks only.

Heavy work (feature build + permutation importance) runs at most once per
kernel via the _state() cache, and only when an exercise object is not None.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks import CheckFail, expect_cols, expect_df, register
from src.config import DATA_DIR, IS_SAMPLE

NUM_FEATURES = [
    "att_month_pct", "att_cum_pct", "att_delta",
    "practice_pct", "checkpoint_pct", "exam_avg_score", "new_attempts_month",
    "max_silent_weeks", "streak_weeks",
    "months_enrolled", "month_index", "signup_lateness", "n_subjects",
]
CAT_FEATURES = ["grade", "live_or_replay", "old_new"]
ATT_FAMILY = {"att_month_pct", "att_cum_pct", "att_delta",
              "max_silent_weeks", "streak_weeks"}

_STATE: dict | None = None


def _build_state() -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from src import churn_utils

    def read(name, dates=()):
        return pd.read_csv(DATA_DIR / f"{name}.csv", parse_dates=list(dates))

    students = read("students", ["signup_date"])
    attendance = read("attendance_long", ["ep_final_date", "week_start"])
    attempts = read("exam_attempts", ["submitted_at"])
    labels = read("labels_monthly", ["churn_date"])
    weekly = read("weekly_metrics", ["week_start", "week_end"])

    features = churn_utils.build_features_monthly(
        labels, attendance, attempts, weekly, students)
    df = features.dropna(subset=["churned_next_month"]).reset_index(drop=True)
    num, cat = NUM_FEATURES, CAT_FEATURES
    train = df[df["year"] == 2568]
    test = df[df["year"] == 2569]

    # --- the model the notebook interprets: canonical LogReg on train only ---
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    logreg = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    logreg.fit(train[num + cat], train["churned_next_month"])

    feat_names = list(logreg[:-1].get_feature_names_out())
    coefs = dict(zip(feat_names, logreg[-1].coef_[0]))

    perm = permutation_importance(
        logreg, test[num + cat], test["churned_next_month"],
        scoring="average_precision", n_repeats=10, random_state=42)
    perm_imp = dict(zip(num + cat, perm.importances_mean))
    top5 = [f for f, _ in sorted(perm_imp.items(), key=lambda kv: -kv[1])[:5]]

    # --- 7.3: เด็กเสี่ยงสุดของ มิ.ย. (เดือนเดียวกับตัวอย่างในสมุด) ------------------
    month_ex = "2026-06"
    june = features[(features["year"] == 2569)
                    & (features["month"] == month_ex)].copy()
    kid_key, top3_vals = None, []
    if len(june):
        june["risk"] = logreg.predict_proba(june[num + cat])[:, 1]
        kid = june.sort_values("risk", ascending=False).head(1)
        kid_key = kid["student_key"].iloc[0]
        contrib = churn_utils.logreg_contributions(logreg, kid[num + cat]).iloc[0]
        top3_vals = contrib.sort_values(ascending=False).head(3).tolist()

    # --- 7.4: false negatives เดือนล่าสุดที่มี label --------------------------
    last_month = test["month"].max()
    mm = test[test["month"] == last_month].copy()
    mm["risk_score"] = logreg.predict_proba(mm[num + cat])[:, 1]
    mm = mm.sort_values("risk_score", ascending=False)
    top30_keys = set(mm.head(30)["student_key"])
    churn_keys = set(mm[mm["churned_next_month"] == 1]["student_key"])
    fn_keys = churn_keys - top30_keys

    return {
        "ok": True,
        "feat_names": feat_names, "coefs": coefs,
        "perm_imp": perm_imp, "top5": top5,
        "kid_key": kid_key, "kid_month": month_ex, "top3_vals": top3_vals,
        "last_month": last_month, "top30_keys": top30_keys,
        "churn_keys": churn_keys, "fn_keys": fn_keys,
    }


def _state() -> dict:
    global _STATE
    if _STATE is None:
        if not IS_SAMPLE:
            _STATE = {"ok": False}
        else:
            try:
                _STATE = _build_state()
            except Exception:
                _STATE = {"ok": False}
    return _STATE


def _expect_filled(name: str, value):
    if value is None:
        raise CheckFail(f"`{name}` ยังเป็น None อยู่ — เติมช่องว่าง ____ ก่อนนะครับ")


# ---------------------------------------------------------------- ex_07_01

@register("ex_07_01")
def _ex_07_01(obj):
    _expect_filled("coef_df", obj)
    expect_df(obj, "coef_df")
    expect_cols(obj, ["feature", "coef"], "coef_df")
    if "feature_th" not in obj.columns:
        raise CheckFail("ขาดคอลัมน์ feature_th — อย่าลบบรรทัด .map(thai_name) "
                        "ที่ให้ไว้ใน skeleton นะครับ")
    if not pd.api.types.is_numeric_dtype(obj["coef"]):
        raise CheckFail("คอลัมน์ coef ควรเป็นตัวเลข — ดึงจาก coef_ ของตัว LogReg")
    if obj["coef"].isna().any():
        raise CheckFail("มีค่า NaN ใน coef — เช็คว่า feat_names กับ coef_values ยาวเท่ากัน")

    a = obj["coef"].abs().to_numpy(dtype=float)
    if np.any(a[:-1] < a[1:] - 1e-9):
        raw = obj["coef"].to_numpy(dtype=float)
        if np.all(raw[:-1] >= raw[1:] - 1e-9):
            raise CheckFail("ตอนนี้เรียงตามค่า coef ดิบอยู่ — ค่าติดลบแรงๆ "
                            "(feature สำคัญ!) เลยไปกองท้ายตาราง ต้องเรียงตามค่าสัมบูรณ์ "
                            "(.abs()) จากมากไปน้อย")
        raise CheckFail("ยังไม่ได้เรียงตาม |coef| จากมากไปน้อย — สร้าง order จาก "
                        "coef_df['coef'].abs() แล้วค่อย reindex")

    st = _state()
    if not st.get("ok"):
        if len(obj) < 5:
            raise CheckFail(f"ได้แค่ {len(obj)} แถว — น้อยผิดปกติ เช็คว่าดึงครบทุก feature")
        return
    exp_names = st["feat_names"]
    got_names = set(obj["feature"].astype(str))
    if got_names != set(exp_names):
        missing = sorted(set(exp_names) - got_names)[:3]
        raise CheckFail(f"ชื่อ feature ไม่ตรงกับโมเดล (เช่น ขาด {missing}) — "
                        f"ใช้ logreg[:-1].get_feature_names_out() ดึงชื่อหลังแปลงนะครับ")
    got = dict(zip(obj["feature"].astype(str), obj["coef"].astype(float)))
    for f, e in st["coefs"].items():
        if abs(got[f] - e) > 0.02:
            raise CheckFail(f"ค่า coef ของ '{f}' ไม่ตรงกับโมเดลในสมุด "
                            f"— ดึงจาก logreg[-1].coef_[0] (แถวแรกของ coef_) ใช่ไหม?")


# ---------------------------------------------------------------- ex_07_02

@register("ex_07_02")
def _ex_07_02(obj):
    _expect_filled("perm_df", obj)
    expect_df(obj, "perm_df")
    expect_cols(obj, ["feature", "importance"], "perm_df")

    got_names = set(obj["feature"].astype(str))
    exp_cols_set = set(NUM_FEATURES + CAT_FEATURES)
    if got_names != exp_cols_set:
        if any("__" in f for f in got_names):
            raise CheckFail("ได้ชื่อ feature หลังแปลง (num__/cat__) มา — permutation ทำบน"
                            "คอลัมน์ต้นฉบับ 16 ตัว: ส่ง X ดิบ (test[num + cat]) ให้ทั้ง pipeline "
                            "ไม่ต้อง transform เอง")
        raise CheckFail(f"รายชื่อ feature ไม่ตรง (ได้ {len(got_names)} ตัว ควรได้ "
                        f"{len(exp_cols_set)} ตัว = num + cat) — เช็คว่าใช้คอลัมน์ต้นฉบับ")

    imp = obj["importance"].to_numpy(dtype=float)
    if np.any(imp[:-1] < imp[1:] - 1e-9):
        raise CheckFail("ยังไม่ได้เรียง importance จากมากไปน้อย — "
                        "ใช้ .sort_values('importance', ascending=False)")

    st = _state()
    if not st.get("ok"):
        return
    got = dict(zip(obj["feature"].astype(str), imp))
    for f, e in st["perm_imp"].items():
        if abs(got[f] - e) > 5e-3:
            raise CheckFail(f"ค่า importance ของ '{f}' ไม่ตรง — เช็ค arguments: "
                            f"วัดบน test set (ไม่ใช่ train), scoring='average_precision', "
                            f"n_repeats=10, random_state=42")


# ---------------------------------------------------------------- ex_07_03

@register("ex_07_03")
def _ex_07_03(obj):
    _expect_filled("explain_student", obj)
    if not callable(obj):
        raise CheckFail(f"ต้องส่งตัวฟังก์ชัน explain_student (ไม่ใส่วงเล็บเรียก) "
                        f"แต่ได้ {type(obj).__name__}")

    st = _state()
    if st.get("ok") and st["kid_key"] is not None:
        kid_key, month = st["kid_key"], st["kid_month"]
    else:  # real mode / state build failed — เช็คได้แค่ว่าคืน string
        try:
            out = obj("__ไม่มีเด็กคนนี้__", "2026-06")
        except Exception as e:
            raise CheckFail(f"เรียกฟังก์ชันแล้ว error ({type(e).__name__}: {e}) — "
                            f"เคสหาไม่เจอต้องคืนข้อความ ไม่ใช่พัง")
        if not isinstance(out, str):
            raise CheckFail("ฟังก์ชันต้องคืน string เสมอ")
        return

    try:
        out = obj(kid_key, month)
    except Exception as e:
        raise CheckFail(f"เรียกกับเด็ก high-risk แล้ว error ({type(e).__name__}: {e}) — "
                        f"ไล่ดูใน function ว่าใช้ row[num + cat] ส่งเข้า "
                        f"logreg_contributions หรือยัง")
    if not isinstance(out, str):
        raise CheckFail(f"ฟังก์ชันควรคืน string แต่ได้ {type(out).__name__}")
    if "ยังไม่ได้เติม" in out:
        raise CheckFail("ฟังก์ชันยังคืนข้อความ placeholder อยู่ — เติม ____ สองจุด "
                        "(contrib และ top3) ก่อนนะครับ")
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) != 3:
        raise CheckFail(f"ได้ {len(lines)} บรรทัด แต่ต้องเป็น 3 บรรทัด (top-3 เหตุผล) — "
                        f"ใช้ .head(3) แล้ว join ด้วย '\\n'")
    for i, v in enumerate(st["top3_vals"]):
        if f"{v:+.2f}" not in lines[i]:
            raise CheckFail(f"บรรทัด {i + 1} ยังไม่ใช่เหตุผลอันดับ {i + 1} — "
                            f"ต้องเรียง contributions จากค่าบวกมากสุดไปน้อย "
                            f"(sort_values(ascending=False)) แล้วหยิบ 3 ตัวแรก")


# ---------------------------------------------------------------- ex_07_04

@register("ex_07_04")
def _ex_07_04(obj):
    _expect_filled("fn_df", obj)
    expect_df(obj, "fn_df")
    expect_cols(obj, ["student_key", "month", "churned_next_month", "risk_score"], "fn_df")
    if len(obj) == 0:
        raise CheckFail("ได้ 0 แถว — เดือนล่าสุดบน sample มีเด็กที่หลุดมือแน่ๆ "
                        "เช็คเงื่อนไข & กับ ~ อีกที")
    if (obj["churned_next_month"] != 1).any():
        raise CheckFail("มีแถวที่ไม่ได้ churn ปนอยู่ — เงื่อนไขแรกคือ "
                        "churned_next_month == 1 (คนที่หายไปจริง)")
    if len(obj) == 30:
        raise CheckFail("ได้ 30 แถวพอดี — นั่นคือ top-30 ไม่ใช่คนที่ 'หลุดจาก' top-30 นะครับ")

    st = _state()
    if not st.get("ok"):
        if obj["month"].nunique() != 1:
            raise CheckFail("ควรมีเดือนเดียว (เดือนล่าสุดของ test)")
        return
    if obj["month"].nunique() != 1 or obj["month"].iloc[0] != st["last_month"]:
        raise CheckFail(f"ต้องเป็นเดือนล่าสุดที่มี label ({st['last_month']}) เดือนเดียว — "
                        f"ใช้ mm2 จากตัวอย่างด้านบนได้เลย")
    got_keys = set(obj["student_key"].astype(str))
    if got_keys and got_keys <= st["top30_keys"]:
        raise CheckFail("ได้เด็กที่โมเดล 'จับได้' (อยู่ใน top-30) มาแทน — "
                        "ลืมกลับด้านเงื่อนไขด้วย ~ หน้า .isin(...) หรือเปล่า?")
    if got_keys != st["fn_keys"]:
        raise CheckFail(f"รายชื่อยังไม่ตรง (ได้ {len(got_keys)} คน ควรได้ "
                        f"{len(st['fn_keys'])} คน) — top-30 ต้องนับจาก mm2 "
                        f"ที่เรียง risk_score มาก→น้อยแล้ว (head(30))")


# ---------------------------------------------------------------- ex_07_05

@register("ex_07_05")
def _ex_07_05(obj):
    _expect_filled("top5_features", obj)
    if isinstance(obj, pd.DataFrame):
        raise CheckFail("ได้ DataFrame มา — เอาเฉพาะคอลัมน์ 'feature' แล้ว "
                        ".head(5).tolist() ให้เป็น list ของชื่อ")
    if isinstance(obj, (pd.Series, np.ndarray)):
        obj = list(obj)
    if not isinstance(obj, (list, tuple)):
        raise CheckFail(f"top5_features ควรเป็น list แต่ได้ {type(obj).__name__}")
    if len(obj) != 5:
        raise CheckFail(f"ได้ {len(obj)} ตัว แต่ต้องเป็น 5 อันดับแรกพอดี — ใช้ .head(5)")
    if not all(isinstance(f, str) for f in obj):
        raise CheckFail("ข้างในต้องเป็น 'ชื่อ' feature (string) — "
                        "หยิบคอลัมน์ feature ไม่ใช่ importance")
    valid = set(NUM_FEATURES + CAT_FEATURES)
    bad = [f for f in obj if f not in valid]
    if bad:
        raise CheckFail(f"เจอชื่อแปลกปลอม {bad[:3]} — ต้องเป็นชื่อคอลัมน์ต้นฉบับจาก "
                        f"perm_df ของข้อ 7.2")
    if len(set(obj)) != 5:
        raise CheckFail("มีชื่อซ้ำกันใน 5 ตัว — หยิบ 5 แถวแรกจาก perm_df ตรงๆ พอครับ")
    hits = [f for f in obj if f in ATT_FAMILY]
    if not hits:
        raise CheckFail("ไม่มีครอบครัวพฤติกรรม (attendance/silent) ติด top-5 เลย — "
                        "ถ้าหยิบจาก perm_df ข้อ 7.2 ถูกต้องแล้วยังเป็นแบบนี้ = ธงแดงของโมเดล "
                        "แต่บน sample ต้องมีนะครับ เช็คว่าเรียงมาก→น้อยก่อน head(5)")

    st = _state()
    if not st.get("ok"):
        return
    if set(obj) != set(st["top5"]):
        raise CheckFail("5 ชื่อยังไม่ตรงกับ top-5 ของ perm_df (n_repeats=10, "
                        "random_state=42) — หยิบ 5 แถวแรกหลังเรียง importance มาก→น้อย")
