# เฉลย 8.1 — risk report เดือน ก.ค. 69 (โครงเดียวกับตัวอย่างเดือน พ.ค. เปลี่ยนแค่เดือน)
SCORE_MONTH = "2026-07"
announce_scoring(SCORE_MONTH)   # ก.ค. > trained_through (2026-06) → ไม่มีคำเตือน in-sample

# TODO 1: เด็ก active เดือน ก.ค. + feature จากครัวกลาง (churn_utils = ตัวเดียวกับตอน train)
labels_jul = labels[labels["month"] == SCORE_MONTH].copy()
features_jul = churn_utils.build_features_monthly(labels_jul, attendance, attempts, weekly, students)

risk_jul = features_jul.copy()
# TODO 2: คะแนนเสี่ยง = ความน่าจะเป็นของ class 1 (คอลัมน์หลังของ predict_proba)
risk_jul["risk_score"] = pipeline.predict_proba(risk_jul[num_cols + cat_cols])[:, 1]
# TODO 3: เหตุผล top-3 ภาษาไทย (helper จากเซลล์ตัวอย่าง)
risk_jul[["reason_1", "reason_2", "reason_3"]] = top3_reasons_thai(pipeline, risk_jul[num_cols + cat_cols])
risk_jul = risk_jul.sort_values("risk_score", ascending=False)

print(risk_jul[["student_key", "risk_score", "att_month_pct",
                "max_silent_weeks", "reason_1"]].head(10).to_string(index=False))
checks.check("ex_08_01", risk_jul)
