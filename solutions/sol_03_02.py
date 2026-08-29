# เฉลย 3.2 — survival curve = retention คูณทบลงมา
retention = 1 - churn_pivot          # เดือนนั้น "อยู่ต่อ" กี่ % (pandas ลบทั้งตารางให้เลย)
survival = retention.cumprod()       # ต้องรอดทุกเดือนก่อนหน้าด้วย → คูณทบ (NaN ติดต่อไปเอง)

print((survival * 100).round(1))
print(f"\nปี 68 สิ้น ส.ค. เหลือ {survival.loc['08', 2568] * 100:.1f}% "
      f"· ปี 69 สิ้น ก.ค. เหลือ {survival.loc['07', 2569] * 100:.1f}%")
checks.check("ex_03_02", survival)
