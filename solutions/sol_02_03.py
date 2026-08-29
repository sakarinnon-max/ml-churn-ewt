# เฉลย 2.3 — melt attendance ปี 68: wide (W1..W10) -> long ตามภาษา contract
SUBJ = {"เลข": 1, "ฟิสิกส์": 4, "เคมี": 5, "ชีวะ": 6}   # เผื่อยังไม่ได้รันข้อ 2.2
wide68 = pd.read_csv(MESSY_DIR / "messy_2568_attendance_wide.csv", dtype=str)

long68 = wide68.melt(id_vars=["รหัส", "วิชา"], var_name="week", value_name="status_th")
long68 = long68.dropna(subset=["status_th"])                   # ช่องว่าง = ยังไม่บันทึก ไม่ใช่ "ขาด"!

long68["status"] = long68["status_th"].str.strip().map(        # strip ก่อน map — "มา " != "มา"
    {"มา": "present", "ขาด": "absent", "ลา": "leave"})
long68["subject_id"] = long68["วิชา"].map(SUBJ)
long68["week_no"] = long68["week"].str.removeprefix("W").astype(int)
long68["student_key"] = "2568-" + long68["รหัส"]

att68 = long68[["student_key", "subject_id", "week_no", "status"]].reset_index(drop=True)
print(att68.shape)
print(att68["status"].value_counts())                          # present ควรนำขาดลอย
checks.check("ex_02_03", att68)
