# เฉลย 4.3 — exam features: คะแนนเฉลี่ยสะสม + จำนวน attempt ใหม่ในเดือน t
parts = []
for (year, month), g in panel_base.groupby(["year", "month"]):
    atm_year = exam_attempts[exam_attempts["year"] == year]
    atm_cut = churn_utils.cutoff(atm_year, month, "submitted_at")    # กรองด้วย submitted_at เท่านั้น!
    month_start = pd.Timestamp(month + "-01")

    avg = atm_cut.groupby("student_key")["percentage"].mean()        # สะสมถึงสิ้นเดือน t
    new_cnt = (atm_cut[atm_cut["submitted_at"] >= month_start]
               .groupby("student_key").size())                       # ส่งใหม่เฉพาะเดือน t

    out = g.copy()
    out["exam_avg_score"] = out["student_key"].map(avg)              # ไม่มี attempt → NaN (ไม่รู้ ≠ ศูนย์)
    out["new_attempts_month"] = out["student_key"].map(new_cnt).fillna(0).astype(int)
    parts.append(out)

panel_exam = pd.concat(parts, ignore_index=True)
print(panel_exam.shape)     # (1610, 7)
checks.check("ex_04_03", panel_exam)
