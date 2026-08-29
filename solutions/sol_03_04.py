# เฉลย 3.4 — แจกแจง silent_weeks: churner vs stayer
# .max() ไม่ใช่ .last() — "เดือนนี้เคยเงียบยาวสุดกี่สัปดาห์" (กลับมาสัปดาห์ท้ายก็ไม่ลบประวัติ)
silent_month = wm.groupby(["student_key", "month"], as_index=False)["silent_weeks"].max()

silent_vs_label = (labels.merge(silent_month, on=["student_key", "month"], how="left")
                         .dropna(subset=["churned_next_month"]))

# normalize="columns" → แต่ละกลุ่ม (label 0 / 1) รวมได้ 1 เทียบกันได้แม้ขนาดกลุ่มต่างกัน 10 เท่า
silent_dist = pd.crosstab(silent_vs_label["silent_weeks"],
                          silent_vs_label["churned_next_month"],
                          normalize="columns")

print((silent_dist * 100).round(1))
print("\nเคยเงียบ ≥1 สัปดาห์ (%):")
print((100 * silent_dist.loc[silent_dist.index >= 1].sum()).round(1))
checks.check("ex_03_04", silent_dist)
