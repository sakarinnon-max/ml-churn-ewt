# เฉลย 1.3 — censor: ก.ย. (จบซีซัน) + เดือนที่ยังไม่ปิดยอด (> known_through) → NaN
KNOWN_THROUGH = "2026-07"

toy_labels = toy_labels_naive.copy()
toy_labels["churned_next_month"] = toy_labels["churned_next_month"].astype("float64")

censored = (toy_labels["month"] == "2026-09") | (toy_labels["month"] > KNOWN_THROUGH)
toy_labels.loc[censored, "churned_next_month"] = float("nan")

# ดูเป็น pivot ตาราง (คน × เดือน) — อ่านง่ายสุด
print(toy_labels.pivot(index="student_key", columns="month", values="churned_next_month"))
checks.check("ex_01_03", toy_labels)
