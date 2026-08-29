"""เฉลยเต็มของ scoring/score_month.py — เปิดเทียบบรรทัดต่อบรรทัดกับไฟล์ skeleton ได้เลย

Batch scoring: จัดอันดับความเสี่ยง churn ของเดือนที่ขอ -> reports/risk_YYYY-MM.csv
กติกาเหล็ก (กัน train/serve skew): สร้าง feature ด้วย src.churn_utils ตัวเดียวกับ
ตอน train เสมอ ห้ามเขียน logic ใหม่ในไฟล์นี้

ใช้:  ML_CHURN_DATA=sample ./venv/bin/python scoring/score_month_solution.py --month 2026-07
      (default: ใช้ข้อมูลจริงจาก data/processed — ตั้ง ML_CHURN_DATA=sample เพื่อซ้อม)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from src import churn_utils
from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR


def _drop_replay(labels: pd.DataFrame) -> pd.DataFrame:
    """ตัดนักเรียนสายเทปออกจาก scope โมเดล — CEO ยืนยันรายชื่อ 29 ส.ค. 2026
    (attendance คาบสดไม่สื่อความหมายกับสายเทป · กลุ่มนี้ดูแลแยกเลนด้วยสัญญาณอื่น)"""
    path = DATA_DIR.parent / "raw" / "labels" / "replay_students_2569.csv"
    if not path.exists():
        return labels
    replay = set(pd.read_csv(path)["student_key"])
    kept = labels[~labels["student_key"].isin(replay)]
    if len(kept) < len(labels):
        print(f"ตัดสายเทปออกจากการให้คะแนน {labels['student_key'].isin(replay).sum()} แถว "
              f"({len(replay)} คน — ดูแลแยกเลน)")
    return kept.reset_index(drop=True)


def load_tables():
    read = lambda n, d=(): pd.read_csv(DATA_DIR / f"{n}.csv", parse_dates=list(d))
    return {
        "students": read("students", ["signup_date"]),
        "attendance": read("attendance_long", ["ep_final_date", "week_start"]),
        "attempts": read("exam_attempts", ["submitted_at"]),
        "labels": _drop_replay(read("labels_monthly", ["churn_date"])),
        "weekly": read("weekly_metrics", ["week_start", "week_end"]),
    }


def load_bundle(model_path: Path, t: dict) -> dict:
    """เปิดกล่องโมเดล — ถ้าเปิดไม่ได้ค่อย fit baseline ในหน่วยความจำให้คอร์สเดินต่อได้

    ไฟล์ .joblib ผูกกับเวอร์ชัน scikit-learn ที่ใช้ตอน fit: รันสคริปต์ด้วย Python คนละตัว
    กับที่เทรน (เช่น kernel ของ Jupyter vs ./venv) กล่องจะเปิดไม่ออก
    ของจริงต้อง **pin เวอร์ชัน** แล้วให้ serve โหลดโมเดลที่ผ่าน validate เท่านั้น —
    ห้าม fit ตอน serve เด็ดขาด (ที่นี่ทำเพื่อไม่ให้บทเรียนสะดุด และไม่ทับไฟล์โมเดลเดิม)
    """
    if model_path.exists():
        try:
            bundle = joblib.load(model_path)
            f = bundle["features"]
            probe = pd.DataFrame([{**{c: 0.0 for c in f["numeric"]},
                                   **{c: "?" for c in f["categorical"]}}])
            bundle["pipeline"].predict_proba(probe)   # โหลดได้ยังไม่พอ — ต้องใช้ได้ด้วย
            return bundle
        except Exception as e:
            print(f"⚠️  ใช้ {model_path.name} ไม่ได้ ({type(e).__name__}) — "
                  f"น่าจะ save มาจาก scikit-learn คนละเวอร์ชันกับ Python ตัวที่รันสคริปต์นี้")
    else:
        print(f"⚠️  ไม่พบไฟล์โมเดล {model_path}")
    print("   fit LogReg สูตรมาตรฐานของคอร์สให้ชั่วคราว (ไม่ทับไฟล์เดิม) — "
          "ของจริงห้ามทำแบบนี้: serve ต้องโหลดโมเดลที่ validate แล้วเท่านั้น\n")

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num, cat = churn_utils.NUMERIC_FEATURES, churn_utils.CATEGORICAL_FEATURES
    feats = churn_utils.build_features_monthly(
        t["labels"], t["attendance"], t["attempts"], t["weekly"], t["students"])
    train = feats.dropna(subset=["churned_next_month"])
    train = train[train["month"] <= "2026-06"]   # กัน 2026-07 ไว้เป็น holdout เหมือนบท 06
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    pipe = Pipeline([("prep", prep),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(train[num + cat], train["churned_next_month"])
    return {"pipeline": pipe,
            "features": {"numeric": num, "categorical": cat},
            "trained_through": "2026-06"}   # key เดียวกับ bundle ของบท 06


# ---------- helper แปลเหตุผลเป็นภาษาไทย (ตัวเดียวกับใน notebook บท 08) ----------

REASON_TH = {
    "att_month_pct": "การเข้าเรียนเดือนนี้", "att_cum_pct": "การเข้าเรียนสะสมทั้งซีซัน",
    "att_delta": "การเข้าเรียนเปลี่ยนจากเดือนก่อน", "practice_pct": "การทำ practice",
    "checkpoint_pct": "การทำ checkpoint", "exam_avg_score": "คะแนนสอบเฉลี่ย",
    "new_attempts_month": "จำนวนข้อสอบที่ทำเดือนนี้", "max_silent_weeks": "สัปดาห์ที่หายเงียบ",
    "streak_weeks": "ความต่อเนื่องรายสัปดาห์", "months_enrolled": "จำนวนเดือนที่เรียนมา",
    "month_index": "ช่วงเวลาของซีซัน", "signup_lateness": "สมัครช้ากว่ารุ่น",
    "n_subjects": "จำนวนวิชาที่ลง", "grade": "ระดับชั้น",
    "live_or_replay": "เรียนสด/เทป", "old_new": "เด็กเก่า/เด็กใหม่",
}


def thai_feature_name(feat_name):
    # "num__att_month_pct" -> "การเข้าเรียนเดือนนี้" | "cat__live_or_replay_เทป" -> "เรียนสด/เทป=เทป"
    raw = feat_name.split("__", 1)[1] if "__" in feat_name else feat_name
    if feat_name.startswith("cat__"):
        for base in ["grade", "live_or_replay", "old_new"]:
            if raw.startswith(base + "_"):
                return f"{REASON_TH[base]}={raw[len(base) + 1:]}"
    return REASON_TH.get(raw, raw)


def top3_reasons_thai(pipeline, X):
    """คืน DataFrame 3 คอลัมน์ reason_1..3 — แรงที่ "ดันความเสี่ยงขึ้น" มากสุด 3 อันดับต่อคน

    โมเดลเชิงเส้น (มี coef_) ใช้ contributions จากบท 07 — ถ้าวันหนึ่งสลับเป็นตระกูลต้นไม้
    จะ fallback เป็น "feature ที่เบี่ยงจากค่าเฉลี่ยรุ่นแรงสุด" แทน (สคริปต์ไม่ล้ม)"""
    if hasattr(pipeline[-1], "coef_"):
        contrib = churn_utils.logreg_contributions(pipeline, X)   # canonical จากบท 07
        rows = []
        for _, row in contrib.iterrows():
            top = row.sort_values(ascending=False).head(3)
            rows.append([f"{thai_feature_name(c)} (+{v:.2f})" if v > 0 else "-"
                         for c, v in top.items()])
        return pd.DataFrame(rows, columns=["reason_1", "reason_2", "reason_3"], index=X.index)

    # fallback: ค่า feature เด่น ๆ เทียบค่าเฉลี่ยรุ่น (z-score กลับทิศให้ "สูง = น่าห่วง")
    num_only = X.select_dtypes("number")
    z = (num_only - num_only.mean()) / num_only.std(ddof=0)
    bad_high = {"max_silent_weeks", "signup_lateness", "month_index"}
    for c in z.columns:
        if c not in bad_high:
            z[c] = -z[c]
    rows = []
    for _, row in z.iterrows():
        top = row.dropna().sort_values(ascending=False).head(3)
        rows.append([f"{REASON_TH.get(c, c)} (ผิดจากรุ่น {v:+.1f} SD)" if v > 0 else "-"
                     for c, v in top.items()] + ["-"] * (3 - len(top)))
    return pd.DataFrame(rows, columns=["reason_1", "reason_2", "reason_3"], index=X.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="เดือนที่จะให้คะแนน เช่น 2026-07")
    ap.add_argument("--model", default=str(MODELS_DIR / "churn_model.joblib"))
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    t = load_tables()
    bundle = load_bundle(Path(args.model), t)   # {"pipeline", "features", "trained_through", ...}
    pipeline = bundle["pipeline"]
    num = list(bundle["features"]["numeric"])
    cat = list(bundle["features"]["categorical"])

    # กติกาประจำระบบ: ประกาศทุกครั้งก่อน score ว่าโมเดลเห็น label ถึงไหน
    trained_through = bundle.get("trained_through", "?")
    print(f"โมเดลเห็น label ถึง: {trained_through} | เดือนที่จะให้คะแนน: {args.month}")
    if trained_through != "?" and args.month <= trained_through:
        print("⚠️  in-sample! เดือนนี้โมเดลเห็นเฉลยไปแล้วตอน train — "
              "ดูกลไกได้ แต่ห้ามใช้ตัวเลขนี้วัดความแม่น")

    # ========== เฉลย TODO 3 จุด (แบบฝึกหัด 8.1 / 8.2) ==========
    # 1) เด็ก active เดือนที่ขอ + feature จาก "ครัวกลาง" churn_utils — ตัวเดียวกับตอน train
    labels_m = t["labels"][t["labels"]["month"] == args.month].copy()
    if len(labels_m) == 0:
        sys.exit(f"ไม่มีเด็ก active เดือน {args.month} ใน labels_monthly — "
                 "เช็ครูปแบบเดือน (YYYY-MM) อีกที")
    features = churn_utils.build_features_monthly(
        labels_m, t["attendance"], t["attempts"], t["weekly"], t["students"])

    # 2) คะแนนเสี่ยง = ความน่าจะเป็นของ class 1 (คอลัมน์ที่สองของ predict_proba)
    features["risk_score"] = pipeline.predict_proba(features[num + cat])[:, 1]

    # 3) เหตุผล top-3 ภาษาไทยต่อคน — mentor อ่านแล้วรู้ว่าจะเปิดบทสนทนาว่าอะไร
    features[["reason_1", "reason_2", "reason_3"]] = top3_reasons_thai(
        pipeline, features[num + cat])
    # ============================================================

    out = (features.sort_values("risk_score", ascending=False)
           .merge(t["students"][["student_key", "student_code", "display_name"]],
                  on="student_key", how="left"))
    cols = ["student_key", "student_code", "display_name", "risk_score",
            "att_month_pct", "att_cum_pct", "max_silent_weeks",
            "reason_1", "reason_2", "reason_3"]
    out = out[[c for c in cols if c in out.columns]]

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"risk_{args.month}.csv"
    out.to_csv(path, index=False)
    print(f"เขียน {path} ({len(out)} คน)\n")
    print(f"Top {args.top} เสี่ยงสุด:")
    print(out.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
