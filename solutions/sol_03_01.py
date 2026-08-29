# เฉลย 3.1 — churn rate รายเดือน แยกปี
labels["mon"] = labels["month"].str[5:]        # "2025-07" → "07" ให้สองซีซันขึ้นแกนเดียวกัน

churn_pivot = labels.pivot_table(
    index="mon",                    # แถว = เดือนของซีซัน
    columns="year",                 # คอลัมน์ = ปี (2568 / 2569)
    values="churned_next_month",    # ค่า 0/1
    aggfunc="mean",                 # mean ของ 0/1 = สัดส่วนที่เป็น 1 = churn rate
)
# เดือน ก.ย. หลุดออกไปเอง เพราะ censored ทั้งแถว (NaN ล้วน) — pivot_table ตัดให้อัตโนมัติ
# ปี 69 เดือน ส.ค. เป็น NaN ด้วยเหตุผลเดียวกัน (ยังปิดยอดไม่ครบ) — ไม่ใช่ churn = 0

print((churn_pivot * 100).round(1))
checks.check("ex_03_01", churn_pivot)
