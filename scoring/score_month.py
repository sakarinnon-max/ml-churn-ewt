"""Batch scoring: จัดอันดับความเสี่ยง churn ของเดือนล่าสุด -> reports/risk_YYYY-MM.csv

โครงนี้คือ "ของจริง" ที่จะรันทุกเดือน — บทที่ 08 ให้ CEO เติมส่วนที่เป็น ____
กติกาสำคัญ (กัน train/serve skew): สร้าง feature ด้วย src.churn_utils ตัวเดียวกับ
ตอน train เสมอ ห้ามเขียน logic ใหม่ในไฟล์นี้

ใช้:  ./venv/bin/python scoring/score_month.py --month 2026-07
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


# ---------------------------------------------------------------- helpers (เติมตามแบบฝึกหัด 8.2)

def _drop_replay(labels: pd.DataFrame) -> pd.DataFrame:
    """ตัดนักเรียนสายเทปออกจาก scope โมเดล — CEO ยืนยันรายชื่อ 29 ส.ค. 2026"""
    path = DATA_DIR.parent / "raw" / "labels" / "replay_students_2569.csv"
    if not path.exists():
        return labels
    replay = set(pd.read_csv(path)["student_key"])
    kept = labels[~labels["student_key"].isin(replay)]
    if len(kept) < len(labels):
        print(f"ตัดสายเทปออกจากการให้คะแนน {labels['student_key'].isin(replay).sum()} แถว "
              f"({len(replay)} คน — ดูแลแยกเลน)")
    return kept.reset_index(drop=True)


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


def thai_feature_name(col: str) -> str:
    base = col.split("__", 1)[1] if "__" in col else col
    for feat, th in REASON_TH.items():
        if base == feat:
            return th
        if base.startswith(feat + "_"):          # one-hot เช่น grade_ม.3
            return f"{th}={base[len(feat) + 1:]}"
    return base


def top3_reasons_thai(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """คืน DataFrame 3 คอลัมน์ reason_1..3 — แรงที่ดันความเสี่ยงขึ้นมากสุด 3 อันดับต่อคน"""
    contrib = churn_utils.logreg_contributions(pipeline, X)
    rows = []
    for _, row in contrib.iterrows():
        top = row.sort_values(ascending=False).head(3)
        rows.append([f"{thai_feature_name(c)} (+{v:.2f})" if v > 0 else "-"
                     for c, v in top.items()])
    return pd.DataFrame(rows, columns=["reason_1", "reason_2", "reason_3"], index=X.index)


def load_tables():
    read = lambda n, d=(): pd.read_csv(DATA_DIR / f"{n}.csv", parse_dates=list(d))
    return {
        "students": read("students", ["signup_date"]),
        "attendance": read("attendance_long", ["ep_final_date", "week_start"]),
        "attempts": read("exam_attempts", ["submitted_at"]),
        "labels": _drop_replay(read("labels_monthly", ["churn_date"])),
        "weekly": read("weekly_metrics", ["week_start", "week_end"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", required=True, help="เดือนที่จะให้คะแนน เช่น 2026-07")
    ap.add_argument("--model", default=str(MODELS_DIR / "churn_model.joblib"))
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    t = load_tables()
    bundle = joblib.load(args.model)          # {"pipeline": ..., "features": {...}}
    pipeline = bundle["pipeline"]
    num = bundle["features"]["numeric"]
    cat = bundle["features"]["categorical"]

    trained_through = bundle.get("trained_through", "?")
    print(f"โมเดลเห็น label ถึง: {trained_through} | เดือนที่จะให้คะแนน: {args.month}")
    if str(args.month) <= str(trained_through):
        print(f"⚠️ เดือน {args.month} อยู่ในช่วงที่โมเดลเคยเห็นตอนเทรน (in-sample) — "
              "คะแนนจะดูดีเกินจริง ใช้ซ้อมได้ แต่ห้ามเอาไปวัดผล")

    # 1) เด็ก active เดือนที่ขอ + features จากครัวกลาง (สูตรเดียวกับตอน train)
    labels_m = t["labels"][t["labels"]["month"] == args.month].copy()
    if len(labels_m) == 0:
        sys.exit(f"ไม่มีเด็ก active เดือน {args.month} ใน labels_monthly")
    features = churn_utils.build_features_monthly(
        labels_m, t["attendance"], t["attempts"], t["weekly"], t["students"])

    # 2) คะแนนเสี่ยง = ความน่าจะเป็นของ class 1
    features["risk_score"] = pipeline.predict_proba(features[num + cat])[:, 1]

    # 3) เหตุผล top-3 ภาษาไทยรายคน
    features[["reason_1", "reason_2", "reason_3"]] = top3_reasons_thai(
        pipeline, features[num + cat])

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
