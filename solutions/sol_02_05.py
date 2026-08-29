# เฉลย 2.5 — guard เช็ค schema 2 ปีก่อน concat (กัน concat "ใจดีเกินไป")
def assert_same_schema(df_a, df_b):
    # 1) ชุดคอลัมน์ต้องเท่ากัน (set — ลำดับไม่สำคัญ) · ^ คือ symmetric difference: ตัวที่หาย/เกิน
    assert set(df_a.columns) == set(df_b.columns), \
        f"คอลัมน์ไม่ตรงกัน: {set(df_a.columns) ^ set(df_b.columns)}"
    # 2) dtype ทุกคอลัมน์ต้อง "เข้ากัน": ตัวเลขทั้งคู่ (int64+float64 โอเค) หรือ dtype เดียวกันเป๊ะ
    for c in df_a.columns:
        both_numeric = (pd.api.types.is_numeric_dtype(df_a[c])
                        and pd.api.types.is_numeric_dtype(df_b[c]))
        assert both_numeric or df_a[c].dtype == df_b[c].dtype, \
            f"คอลัมน์ '{c}' dtype ไม่เข้ากัน: {df_a[c].dtype} vs {df_b[c].dtype}"
    return True

# ลองกับ students จริง 2 ปี — guard ผ่านแล้วค่อย concat
stu = pd.read_csv(DATA_DIR / "students.csv")
s68 = stu[stu["year"] == 2568]
s69 = stu[stu["year"] == 2569]
assert_same_schema(s68, s69)
combined = pd.concat([s68, s69], ignore_index=True)
print("guard ผ่าน — concat 2 ปีได้:", combined.shape)
checks.check("ex_02_05", assert_same_schema)
