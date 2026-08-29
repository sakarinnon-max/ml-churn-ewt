# เฉลย 2.1 — audit ไฟล์รับสมัครปี 68 (อ่านด้วย dtype=str เสมอ — อย่าให้ pandas เดา)
raw68 = pd.read_csv(MESSY_DIR / "messy_2568_students.csv", dtype=str)

print(raw68["ระดับชั้น"].value_counts(dropna=False))   # "ม.3" ปน "3" ปน "ม3" ปน "ม.3 " — 4 หน้าตา ค่าเดียวกัน
print(raw68["สด/เทป"].value_counts(dropna=False))      # มี "สด " (ช่องว่างท้าย) และ "ผสม" โผล่มาด้วย

audit = {
    "n_rows": len(raw68),                                # 30 แถวข้อมูล (header ไม่นับ)
    "n_junk": int(raw68["รหัส"].isna().sum()),           # แถว test/แถวเปล่า ไม่มีรหัสนักเรียน
    "n_missing_school": int(raw68["โรงเรียน"].isna().sum()),  # นับทั้งไฟล์ รวมแถวขยะ
}
print(audit)
checks.check("ex_02_01", audit)
