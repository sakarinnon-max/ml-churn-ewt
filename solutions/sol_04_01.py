# เฉลย 4.1 — โครง panel จาก labels_monthly (กรอง active) + month_index แยกตามปี
panel = labels[labels["active"] == 1].copy()
panel = panel[["student_key", "year", "month", "churned_next_month"]]

# mapping {(year, month) -> ลำดับเดือนของซีซัน} จากทั้ง 2 ปี
idx_of = {}
for y, season in SEASONS.items():
    for i, m in enumerate(month_seq(season["start_month"], season["end_month"]), start=1):
        idx_of[(y, m)] = i
panel["month_index"] = [idx_of[(y, m)] for y, m in zip(panel["year"], panel["month"])]

print(panel.shape)          # (1610, 5) — แถว NaN (censored) ยังอยู่ครบ
checks.check("ex_04_01", panel)
