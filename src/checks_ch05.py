"""Checks for ch05 — โมเดลแรก: Logistic Regression

Expected values are recomputed live from DATA_DIR (never hardcoded):
canonical features via churn_utils.build_features_monthly → drop NaN labels →
train on 2568 / test on 2569 → canonical ColumnTransformer + LogisticRegression
(class_weight="balanced", max_iter=2000) → proba, AP, precision@30, tier baseline.

The heavy part (feature build ~5s + fit) runs at most once per kernel via the
_state() cache, and only when the learner's answer is not None — so a fresh
notebook run (every skeleton still `None`) stays fast.
When IS_SAMPLE is False, or state can't be built, fall back to structure/range
checks only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks import CheckFail, expect_between, expect_cols, expect_df, register
from src.churn_utils import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.config import DATA_DIR, IS_SAMPLE

NUM_COLS = list(NUMERIC_FEATURES)
CAT_COLS = list(CATEGORICAL_FEATURES)

_STATE: dict | None = None


# ---------------------------------------------------------------- state

def _build_state() -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from src import churn_utils

    labels = pd.read_csv(DATA_DIR / "labels_monthly.csv", dtype={"month": str},
                         parse_dates=["churn_date"])
    att = pd.read_csv(DATA_DIR / "attendance_long.csv",
                      parse_dates=["ep_final_date", "week_start"])
    atm = pd.read_csv(DATA_DIR / "exam_attempts.csv", parse_dates=["submitted_at"])
    wm = pd.read_csv(DATA_DIR / "weekly_metrics.csv",
                     parse_dates=["week_start", "week_end"])
    stu = pd.read_csv(DATA_DIR / "students.csv", parse_dates=["signup_date"])

    feats = churn_utils.build_features_monthly(labels, att, atm, wm, stu)
    labeled = feats.dropna(subset=["churned_next_month"]).copy()
    train = labeled[labeled["year"] == 2568].copy()
    test = labeled[labeled["year"] == 2569].copy()

    X_train, y_train = train[NUM_COLS + CAT_COLS], train["churned_next_month"]
    X_test, y_test = test[NUM_COLS + CAT_COLS], test["churned_next_month"]

    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    model = Pipeline([
        ("prep", prep),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    proba_train = model.predict_proba(X_train)[:, 1]

    n_cols = model[:-1].transform(X_train.head(5)).shape[1]

    tier_test = np.asarray(churn_utils.tier_baseline_score(test, wm), dtype=float)
    tier_train = np.asarray(churn_utils.tier_baseline_score(train, wm), dtype=float)

    yt = np.asarray(y_test, dtype=float)
    ytr = np.asarray(y_train, dtype=float)
    return {
        "ok": True,
        "X_train": X_train, "y_test": yt, "n_cols": int(n_cols),
        "proba": proba,
        "base": float(yt.mean()),
        "base_train": float(ytr.mean()),
        "ap": float(average_precision_score(yt, proba)),
        "ap_train": float(average_precision_score(ytr, proba_train)),
        "ap_tier": float(average_precision_score(yt, tier_test)),
        "ap_tier_train": float(average_precision_score(ytr, tier_train)),
        "p30": float(churn_utils.precision_at_k(yt, proba, 30)),
        "p30_tier": float(churn_utils.precision_at_k(yt, tier_test, 30)),
        "p10": float(churn_utils.precision_at_k(yt, proba, 10)),
        "p50": float(churn_utils.precision_at_k(yt, proba, 50)),
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


def _as_float(obj, name: str) -> float:
    if isinstance(obj, (list, tuple, set, dict)):
        raise CheckFail(f"{name} ควรเป็นตัวเลขตัวเดียว แต่ได้ {type(obj).__name__} "
                        f"— ส่งเฉพาะค่าที่โจทย์ขอนะครับ")
    if isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
        raise CheckFail(f"{name} ควรเป็นตัวเลขตัวเดียว แต่ได้ {type(obj).__name__} "
                        f"ขนาด {getattr(obj, 'shape', '?')} — ถ้าเป็น array แสดงว่าส่ง "
                        f"precision/recall ทั้งเส้นมาแทนค่าสรุป")
    try:
        v = float(obj)
    except (TypeError, ValueError):
        raise CheckFail(f"{name} แปลงเป็นตัวเลขไม่ได้ (ได้ {type(obj).__name__}) "
                        f"— ส่งค่าที่คำนวณได้ ไม่ใช่ตัวฟังก์ชัน")
    if np.isnan(v):
        raise CheckFail(f"{name} เป็น NaN — เช็คว่า y/score ที่ป้อนเข้าไปมีค่าครบไหม")
    return v


def _unwrap(trans, kls):
    """คืน instance ของ kls ที่อยู่ใน transformer (รองรับกรณีห่อไว้ใน Pipeline)"""
    from sklearn.pipeline import Pipeline
    if isinstance(trans, kls):
        return trans
    if isinstance(trans, Pipeline):
        for _, step in trans.steps:
            if isinstance(step, kls):
                return step
    return None


# ---------------------------------------------------------------- ex_05_01

@register("ex_05_01")
def _ex_05_01(obj):
    _expect_filled("prep", obj)
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if isinstance(obj, Pipeline):
        raise CheckFail("ข้อนี้ขอแค่ ColumnTransformer ตัวเดียว (ครัวเตรียมวัตถุดิบ) "
                        "— การต่อเข้ากับโมเดลเป็นงานของข้อ 5.2 ครับ")
    if not isinstance(obj, ColumnTransformer):
        raise CheckFail(f"prep ต้องเป็น ColumnTransformer แต่ได้ "
                        f"{type(obj).__name__} — ดูโครงจาก toy_prep ใน cell ตัวอย่าง")

    num_t = cat_t = None
    for _name, trans, cols in obj.transformers:
        cols = [cols] if isinstance(cols, str) else list(cols)
        s = set(map(str, cols))
        if s == set(NUM_COLS):
            num_t = trans
        elif s == set(CAT_COLS):
            cat_t = trans
    if num_t is None:
        raise CheckFail(f"ไม่เจอเส้นทางที่รับคอลัมน์ตัวเลขครบ {len(NUM_COLS)} ตัว — "
                        f"เส้นทาง num ต้องส่ง NUM_COLS ทั้งลิสต์ (ไม่ใช่บางคอลัมน์)")
    if cat_t is None:
        raise CheckFail(f"ไม่เจอเส้นทางที่รับคอลัมน์หมวดหมู่ {CAT_COLS} — "
                        f"เส้นทาง cat ต้องส่ง CAT_COLS ทั้งลิสต์")

    if isinstance(num_t, str):
        raise CheckFail(f"เส้นทาง num ตอนนี้เป็น '{num_t}' — ต้องเป็น Pipeline ที่ "
                        f"เติม NaN ก่อน แล้วค่อยสเกล")
    imp = _unwrap(num_t, SimpleImputer)
    sc = _unwrap(num_t, StandardScaler)
    if imp is None:
        raise CheckFail("เส้นทาง num ยังไม่มี SimpleImputer — ตาราง features มี NaN "
                        "อยู่จริง (เด็กที่ยังไม่เคยทำข้อสอบ ฯลฯ) โมเดลจะกินไม่ได้")
    if getattr(imp, "strategy", None) != "median":
        raise CheckFail(f"SimpleImputer ใช้ strategy='{imp.strategy}' — โจทย์ขอ "
                        f"'median' (ทนค่าโดดกว่า mean)")
    if sc is None:
        raise CheckFail("เส้นทาง num ยังไม่มี StandardScaler — ไม่งั้น att_month_pct "
                        "(0–100) จะเสียงดังกลบ n_subjects (1–6)")
    if isinstance(num_t, Pipeline):
        steps = [s for _, s in num_t.steps]
        i_imp = next(i for i, s in enumerate(steps) if isinstance(s, SimpleImputer))
        i_sc = next(i for i, s in enumerate(steps) if isinstance(s, StandardScaler))
        if i_imp > i_sc:
            raise CheckFail("ลำดับใน Pipeline สลับกัน — ต้องเติมรู (imputer) ก่อน "
                            "แล้วค่อยสเกล ไม่งั้น scaler จะเจอ NaN")

    ohe = _unwrap(cat_t, OneHotEncoder)
    if ohe is None:
        raise CheckFail(f"เส้นทาง cat ต้องเป็น OneHotEncoder แต่ได้ "
                        f"{type(cat_t).__name__} — คอลัมน์ตัวหนังสือต้องแตกเป็น 0/1")
    if getattr(ohe, "handle_unknown", None) != "ignore":
        raise CheckFail(f"OneHotEncoder ตั้ง handle_unknown='"
                        f"{getattr(ohe, 'handle_unknown', None)}' — โจทย์ขอ 'ignore' "
                        f"เพื่อกันข้อมูลปีใหม่มีค่าที่ปี 68 ไม่เคยเห็นแล้ว crash")

    st = _state()
    if not st.get("ok"):
        return
    from sklearn.base import clone
    try:
        Xt = clone(obj).fit_transform(st["X_train"])
    except Exception as e:
        raise CheckFail(f"ลอง fit_transform กับ X_train แล้วพัง "
                        f"({type(e).__name__}: {e}) — เช็คชื่อคอลัมน์ที่ส่งให้แต่ละเส้นทาง")
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=float)
    if Xt.shape[1] != st["n_cols"]:
        raise CheckFail(f"แปลงแล้วได้ {Xt.shape[1]} คอลัมน์ แต่ควรได้ {st['n_cols']} "
                        f"({len(NUM_COLS)} ตัวเลข + one-hot 7) — เช็ค remainder / "
                        f"drop ของ OneHotEncoder ว่าเผลอตั้งไว้หรือเปล่า")
    if np.isnan(Xt).any():
        raise CheckFail("แปลงแล้วยังมี NaN เหลือ — imputer ครอบไม่ครบทุกคอลัมน์ตัวเลข")


# ---------------------------------------------------------------- ex_05_02

@register("ex_05_02")
def _ex_05_02(obj):
    _expect_filled("model / proba", obj)
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        raise CheckFail("ข้อนี้ส่งเป็นคู่ `(model, proba)` — บรรทัด "
                        "checks.check(\"ex_05_02\", (model, proba)) จัดรูปไว้ให้แล้ว")
    model, proba = obj
    _expect_filled("model", model)
    _expect_filled("proba", proba)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    if not isinstance(model, Pipeline):
        raise CheckFail(f"model ต้องเป็น Pipeline (prep → clf) แต่ได้ "
                        f"{type(model).__name__} — ประโยชน์ของ Pipeline คือกัน leakage "
                        f"ให้อัตโนมัติ อย่าข้ามนะครับ")
    clf = model[-1]
    if not isinstance(clf, LogisticRegression):
        raise CheckFail(f"ขั้นสุดท้ายของ Pipeline ต้องเป็น LogisticRegression แต่ได้ "
                        f"{type(clf).__name__}")
    if getattr(clf, "class_weight", None) != "balanced":
        raise CheckFail("LogisticRegression ยังไม่ได้ตั้ง class_weight='balanced' — "
                        "ไม่งั้นโมเดลจะกลายเป็นหมอดูขี้เกียจ (ทายว่าไม่มีใคร churn)")
    if int(getattr(clf, "max_iter", 100)) < 1000:
        raise CheckFail(f"max_iter = {clf.max_iter} น้อยไป (lbfgs ยังลู่เข้าไม่ทัน) "
                        f"— โจทย์ให้ใช้ 2000")
    if not hasattr(clf, "coef_"):
        raise CheckFail("โมเดลยังไม่ได้ fit — เรียก model.fit(X_train, y_train) "
                        "ก่อนนะครับ (ปี 2568 เท่านั้น)")

    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 2:
        raise CheckFail(f"proba มี {arr.shape[1]} คอลัมน์ — predict_proba คืน "
                        f"[P(อยู่ต่อ), P(churn)] ต้องหยิบเฉพาะคอลัมน์ที่สอง [:, 1]")
    p = arr.ravel()
    if p.size == 0:
        raise CheckFail("proba ว่างเปล่า — predict_proba บน X_test หรือยัง?")
    if ((p < 0) | (p > 1)).any():
        raise CheckFail("ค่าใน proba หลุดช่วง 0–1 — ใช้ predict_proba ไม่ใช่ "
                        "decision_function นะครับ")
    if set(np.unique(p)) <= {0.0, 1.0}:
        raise CheckFail("proba มีแต่ 0 กับ 1 — นั่นคือผลจาก predict() (คำตอบฟันธง) "
                        "แต่เราต้องการ 'คะแนนความเสี่ยง' ไว้เรียงลำดับลิสต์โทร")

    st = _state()
    if not st.get("ok"):
        return
    exp = st["proba"]
    if len(p) != len(exp):
        raise CheckFail(f"proba มี {len(p):,} ค่า แต่ข้อสอบปี 2569 มี {len(exp):,} แถว "
                        f"— predict_proba บน X_test (ไม่ใช่ X_train) หรือเปล่า?")
    diff = float(np.abs(p - exp).mean())
    if diff <= 0.05:
        return
    if float(np.abs(p - (1.0 - exp)).mean()) <= 0.05:
        raise CheckFail("คะแนนกลับหัวกลับหาง — ดูเหมือนหยิบคอลัมน์แรก [:, 0] ซึ่งคือ "
                        "P(อยู่ต่อ) · ความเสี่ยง churn อยู่คอลัมน์ที่สอง [:, 1]")
    raise CheckFail(f"คะแนนความเสี่ยงต่างจากที่ควรได้เฉลี่ย {diff:.3f} — เช็ค 2 จุด: "
                    f"(1) fit ด้วย X_train, y_train ปี 2568 ล้วนๆ ไหม "
                    f"(2) prep ในท่อตรงสเปกข้อ 5.1 ไหม")


# ---------------------------------------------------------------- ex_05_03

@register("ex_05_03")
def _ex_05_03(obj):
    _expect_filled("ap", obj)
    v = _as_float(obj, "ap")
    expect_between(v, 0.0, 1.0, "AP")

    st = _state()
    if not st.get("ok"):
        return
    if abs(v - st["ap"]) <= 0.01:
        return
    if abs(v - st["ap_train"]) <= 0.01:
        raise CheckFail(f"ค่านี้คือ AP บน 'ข้อสอน' ปี 2568 — ข้อนี้วัดบนข้อสอบปี 2569 "
                        f"คือ average_precision_score(y_test, proba_ok)")
    if abs(v - st["base"]) <= 0.005:
        raise CheckFail("AP ออกมาเท่ากับ base rate พอดี — แปลว่าคะแนนที่ป้อนเรียงลำดับ "
                        "ไม่ได้เลย (สลับ argument หรือส่งค่าคงที่ไปหรือเปล่า? ลำดับคือ "
                        "(เฉลยจริง, คะแนน))")
    if abs(v - st["ap_tier"]) <= 0.01:
        raise CheckFail("ค่านี้คือ AP ของ tier heuristic ไม่ใช่ของโมเดล — "
                        "ข้อนี้ใช้ proba_ok (คะแนนจากโมเดล)")
    raise CheckFail(f"AP = {v:.3f} แต่ควรได้ ≈ {st['ap']:.3f} — เรียก "
                    f"average_precision_score(y_test, proba_ok) เฉลยจริงมาก่อน "
                    f"คะแนนตามหลังเสมอ")


# ---------------------------------------------------------------- ex_05_04

@register("ex_05_04")
def _ex_05_04(obj):
    _expect_filled("p_at_30", obj)
    v = _as_float(obj, "p_at_30")
    if v > 1.0:
        raise CheckFail(f"ได้ {v:.1f} ซึ่งมากกว่า 1 — ดูเหมือนเป็น 'จำนวนคน' "
                        f"ข้อนี้ขอ precision@30 ที่เป็นสัดส่วน 0–1 (จำนวนคน = สัดส่วน × 30)")
    expect_between(v, 0.0, 1.0, "precision@30")

    st = _state()
    if not st.get("ok"):
        return
    if abs(v - st["p30"]) <= 0.02:
        return
    if abs(v - st["p10"]) <= 1e-9 or abs(v - st["p50"]) <= 1e-9:
        raise CheckFail(f"ค่านี้ใช้ k ตัวอื่น ไม่ใช่ 30 — งบโทรของ mentor คือ 30 สาย/เดือน")
    if abs(v - st["p30_tier"]) <= 1e-9:
        raise CheckFail("ค่านี้คือ precision@30 ของ tier heuristic — ข้อนี้วัดของโมเดล "
                        "(proba_ok) ส่วน tier เก็บไว้ประลองกันในข้อ 5.5")
    if abs(v - st["base"]) <= 0.01:
        raise CheckFail("precision@30 ออกมาเท่ากับ base rate — เหมือนลิสต์ไม่ได้เรียง "
                        "ตามคะแนนเสี่ยง (ส่ง proba_ok เป็น y_score ถูกตำแหน่งไหม?)")
    raise CheckFail(f"precision@30 = {v:.3f} แต่ควรได้ ≈ {st['p30']:.3f} — เรียก "
                    f"churn_utils.precision_at_k(y_test, proba_ok, 30)")


# ---------------------------------------------------------------- ex_05_05

@register("ex_05_05")
def _ex_05_05(obj):
    _expect_filled("baseline_table", obj)
    expect_df(obj, "baseline_table")
    expect_cols(obj, ["baseline", "ap", "precision_at_30"], "baseline_table")

    names = [str(x) for x in obj["baseline"]]
    need = ["base_rate", "tier", "model"]
    missing = [n for n in need if n not in names]
    if missing:
        raise CheckFail(f"คอลัมน์ baseline ขาดแถว {missing} (ตอนนี้มี {names}) — "
                        f"ชื่อแถวต้องเป็น 'base_rate', 'tier', 'model' เป๊ะๆ")
    if len(obj) != 3:
        raise CheckFail(f"ตารางมี {len(obj)} แถว แต่ควรมี 3 แถว (เดามั่ว / tier / โมเดล)")
    for col in ("ap", "precision_at_30"):
        if not pd.api.types.is_numeric_dtype(obj[col]):
            raise CheckFail(f"คอลัมน์ {col} ต้องเป็นตัวเลข — ตอนนี้เป็น {obj[col].dtype} "
                            f"(เผลอใส่เป็นข้อความหรือ tuple หรือเปล่า?)")
        if obj[col].isna().any():
            raise CheckFail(f"คอลัมน์ {col} มี NaN — เติมค่าให้ครบทั้ง 3 แถว")

    got = {str(r["baseline"]): (float(r["ap"]), float(r["precision_at_30"]))
           for _, r in obj.iterrows()}

    st = _state()
    if not st.get("ok"):
        for name, (ap, p30) in got.items():
            expect_between(ap, 0.0, 1.0, f"ap ของแถว {name}")
            expect_between(p30, 0.0, 1.0, f"precision_at_30 ของแถว {name}")
        return

    exp = {
        "base_rate": (st["base"], st["base"]),
        "tier": (st["ap_tier"], st["p30_tier"]),
        "model": (st["ap"], st["p30"]),
    }
    hints = {
        "base_rate": "แถวนี้ใช้ base rate ของ y_test ทั้งสองช่อง (เดามั่ว = ค่าคาดหวัง)",
        "tier": "แถวนี้คำนวณจาก churn_utils.tier_baseline_score(test_df, weekly) "
                "บนปี 2569 (ระวังเผลอใช้ train_df)",
        "model": "แถวนี้คำนวณจาก proba_ok บน y_test",
    }
    for name, (e_ap, e_p30) in exp.items():
        g_ap, g_p30 = got[name]
        if abs(g_ap - e_p30) <= 0.001 and abs(g_p30 - e_ap) <= 0.001 and \
                abs(e_ap - e_p30) > 0.01:
            raise CheckFail(f"แถว {name}: ค่า ap กับ precision_at_30 สลับคอลัมน์กัน")
        if abs(g_ap - e_ap) > 0.01:
            if name == "tier" and abs(g_ap - st["ap_tier_train"]) <= 0.01:
                raise CheckFail("แถว tier: ค่า ap ตรงกับที่วัดบนปี 2568 — สังเวียนนี้ "
                                "แข่งกันบนข้อสอบปี 2569 (test_df) ครับ")
            raise CheckFail(f"แถว {name}: ap = {g_ap:.3f} แต่ควรได้ ≈ {e_ap:.3f} · "
                            f"{hints[name]}")
        if abs(g_p30 - e_p30) > 0.02:
            raise CheckFail(f"แถว {name}: precision_at_30 = {g_p30:.3f} แต่ควรได้ ≈ "
                            f"{e_p30:.3f} · {hints[name]}")
