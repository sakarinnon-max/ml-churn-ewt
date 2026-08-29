# เฉลย 4.6 — สืบสวนแล้วประหาร feature จากอนาคต
# ขั้นสืบ: เด็กที่ churned_next_month = 1 ค่า att_next_month_pct = 0 "ทุกคน"
# → มันคือ % เข้าเรียนของเดือน t+1 — เด็กที่หายเดือน t+1 ย่อมเรียนเดือน t+1 เป็น 0 แน่นอน
print(features_v1.groupby("churned_next_month", dropna=False)["att_next_month_pct"]
      .agg(["mean", "count"]))

# คำตัดสิน: ณ สิ้นเดือน t ค่านี้ยังไม่เกิดขึ้นจริง = คำตอบปลอมตัวมา ไม่ใช่ตัวทำนาย → ทิ้ง
features_clean = features_v1.drop(columns=["att_next_month_pct"])

print("เหลือ:", features_clean.shape)   # (1610, 20)
checks.check("ex_04_06", features_clean)
