# เฉลย 2.4 — ให้ contract พิสูจน์ว่าตารางสะอาด + เช็ค assumption กับดักโดเมนเองอีกชั้น
att = pd.read_csv(DATA_DIR / "attendance_long.csv", parse_dates=["ep_final_date", "week_start"])
atm = pd.read_csv(DATA_DIR / "exam_attempts.csv", parse_dates=["submitted_at"])

validation_report = {
    "attendance_ok": contracts.validate_df("attendance_long", att),
    "attempts_ok": contracts.validate_df("exam_attempts", atm),
    # กับดัก 3: ไม่มี retake -> (student, exam_id) ห้ามซ้ำ
    "dup_exam_pairs": int(atm.duplicated(["student_key", "exam_id"]).sum()),
    # กับดัก 1: EP สอน 2 รอบต้อง merge แล้ว -> (student, subject, EP) ห้ามซ้ำ
    "dup_ep_rows": int(att.duplicated(["student_key", "subject_id", "episode_number", "year"]).sum()),
}
print(validation_report)
checks.check("ex_02_04", validation_report)
