# เฉลย 7.4 — false negatives: churn จริง แต่หลุดจาก top-30 ที่ mentor โทร
mm2 = mm.copy()   # มี risk_score และเรียงเสี่ยงมาก -> น้อยแล้ว

in_top30 = set(mm2.head(30)["student_key"])   # 30 อันดับแรก = รายชื่อที่ mentor ได้โทร

fn_df = mm2[(mm2["churned_next_month"] == 1) & (~mm2["student_key"].isin(in_top30))]
fn_df = fn_df[["student_key", "month", "churned_next_month", "risk_score",
               "att_month_pct", "att_cum_pct", "max_silent_weeks", "streak_weeks"]]
print(f"โมเดลพลาด (churn จริงแต่คะแนนต่ำ): {len(fn_df)} คน")
print(fn_df.round(2).to_string(index=False))

checks.check("ex_07_04", fn_df)
