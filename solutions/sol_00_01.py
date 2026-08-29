# เฉลย ex_00_01 — โหลด weekly_metrics + นับนักเรียน/สัปดาห์แบบไม่ซ้ำ
wm = pd.read_csv(DATA_DIR / "weekly_metrics.csv", parse_dates=["week_start", "week_end"])

n_students = wm["student_key"].nunique()   # นับค่าไม่ซ้ำ (1 คนมีหลายแถว)
n_weeks = wm["week_start"].nunique()

print("นักเรียน:", n_students, "คน · สัปดาห์:", n_weeks, "สัปดาห์ ·", f"{len(wm):,} แถว")
checks.check("ex_00_01", (wm, n_students, n_weeks))
