# เฉลย 1.4 — สร้าง labels จริงจาก enrollment_events ทั้ง 2 ปี
events = pd.read_csv(DATA_DIR / "enrollment_events.csv", parse_dates=["event_date"])
labels_2568 = churn_utils.build_labels_monthly(events, year=2568)                       # ซีซันจบแล้ว → default
labels_2569 = churn_utils.build_labels_monthly(events, year=2569, known_through="2026-07")
labels = pd.concat([labels_2568, labels_2569], ignore_index=True)

contracts.validate_df("labels_monthly", labels)   # ผ่านเงียบๆ = โครงสร้างตรง contract
print(f"{len(labels):,} แถว | churn rate รวม (เฉพาะเดือนที่มี label) = "
      f"{labels['churned_next_month'].mean():.1%}")
checks.check("ex_01_04", labels)
