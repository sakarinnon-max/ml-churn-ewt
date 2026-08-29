# เฉลย 1.2 — label ฉบับ naive: 1 เมื่อเดือนนั้นคือเดือน cancel (เดือนสุดท้ายที่ active)
cancel_month = (toy_events[toy_events["event_type"] == "cancel"]
                .set_index("student_key")["event_month"])

toy_labels_naive = active_months.copy()
toy_labels_naive["churned_next_month"] = (
    toy_labels_naive["month"] == toy_labels_naive["student_key"].map(cancel_month)
).astype(int)

print(toy_labels_naive[toy_labels_naive["churned_next_month"] == 1])
checks.check("ex_01_02", toy_labels_naive)
