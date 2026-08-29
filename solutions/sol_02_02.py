# เฉลย 2.2 — clean ปี 68 เข้า contract students (บางส่วน — ยังไม่แตะคอลัมน์วันที่)
raw68 = pd.read_csv(MESSY_DIR / "messy_2568_students.csv", dtype=str)
SUBJ = {"เลข": 1, "ฟิสิกส์": 4, "เคมี": 5, "ชีวะ": 6}

df = raw68[raw68["รหัส"].notna()].copy()                       # 1) ตัดแถวขยะ (test/แถวเปล่า)
df = df.rename(columns={"รหัส": "student_code",                # 2) ภาษาชีต -> ภาษาโมเดล
                        "ชื่อเล่น": "display_name",
                        "ระดับชั้น": "grade",
                        "สด/เทป": "live_or_replay",
                        "old / new": "old_new"})

ids = df.apply(lambda r: sorted(SUBJ[c] for c in SUBJ if r[c] == "TRUE"), axis=1)
df["subject_ids"] = ids.map(lambda x: ";".join(map(str, x)))   # 3) TRUE/FALSE -> "1;4;5" (sort แล้ว)
df["n_subjects"] = ids.map(len)

df["grade"] = df["grade"].str.strip()                          # 4) ล้างช่องว่างท้ายจากชีต
df["live_or_replay"] = df["live_or_replay"].str.strip()
df["old_new"] = df["old_new"].str.strip().str.lower()          #    "New"/"old " -> new/old

df["year"] = 2568                                              # 5) int ตาม contract
df["student_key"] = "2568-" + df["student_code"]               #    key กลางของทั้งโปรเจกต์

students68 = df[["student_key", "year", "student_code", "display_name", "grade",
                 "live_or_replay", "old_new", "n_subjects", "subject_ids"]].reset_index(drop=True)
print(students68.shape)
print(students68.head(3).to_string())
checks.check("ex_02_02", students68)
