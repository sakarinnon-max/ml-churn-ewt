# เฉลย ex_00_02 — tier ล่าสุดของนักเรียนแต่ละคน ซีซัน 2569
wm = pd.read_csv(DATA_DIR / "weekly_metrics.csv", parse_dates=["week_start", "week_end"])

wm69 = wm[wm["year"] == 2569]                       # กรองปีก่อนเสมอ

last_week = (wm69.sort_values("week_start")         # 1) เรียงเวลา
                  .groupby("student_key")           # 2) แยกกองรายคน
                  .last())                          # 3) หยิบสัปดาห์ล่าสุด

tier_counts = last_week["tier"].value_counts()

print(tier_counts)
checks.check("ex_00_02", tier_counts)
