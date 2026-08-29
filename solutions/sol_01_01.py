# เฉลย 1.1 — เดือนที่ active ของแต่ละคน (interval enroll → cancel)
SEASON_END = "2026-09"
rows = []
for sk, g in toy_events.groupby("student_key"):
    start = g.loc[g["event_type"] == "enroll", "event_month"].iloc[0]
    cancels = g.loc[g["event_type"] == "cancel", "event_month"]
    end = cancels.iloc[0] if len(cancels) else SEASON_END   # subject_drop ไม่เกี่ยว — mask ข้ามให้เอง
    for m in month_seq(start, end):
        rows.append({"student_key": sk, "month": m})

active_months = pd.DataFrame(rows, columns=["student_key", "month"])
print(f"ได้ {len(active_months)} แถว (เป้าหมาย: 33)")
checks.check("ex_01_01", active_months)
