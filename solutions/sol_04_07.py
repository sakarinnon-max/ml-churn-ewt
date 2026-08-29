# เฉลย 4.7 — validate ก่อน save เสมอ แล้วค่อยส่งมอบ
features_final = features_clean_ok.copy()   # หรือ features_clean ของคุณถ้าข้อ 4.6 ผ่านแล้ว

contracts.validate_df("features_monthly", features_final)          # ผ่านเงียบๆ = โครงสร้างตรง
features_final.to_csv(DATA_DIR / "features_monthly.csv", index=False)

print("บันทึกแล้ว:", DATA_DIR / "features_monthly.csv", features_final.shape)
checks.check("ex_04_07", features_final)
