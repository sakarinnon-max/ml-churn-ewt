# เฉลย 1.5 — churn rate ต่อเดือน (สร้าง labels ใหม่ในตัว เผื่อข้อ 1.4 ยังไม่ได้รัน)
events = pd.read_csv(DATA_DIR / "enrollment_events.csv", parse_dates=["event_date"])
labels = pd.concat([
    churn_utils.build_labels_monthly(events, year=2568),
    churn_utils.build_labels_monthly(events, year=2569, known_through="2026-07"),
], ignore_index=True)

churn_by_month = labels.groupby("month")["churned_next_month"].mean()
print((churn_by_month * 100).round(1))
checks.check("ex_01_05", churn_by_month)
