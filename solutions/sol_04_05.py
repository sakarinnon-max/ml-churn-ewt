# เฉลย 4.5 — ประกอบ features_monthly: 3 panel + static + months_enrolled
ID = ["student_key", "year", "month"]

features = panel_att_ok.merge(
    panel_exam_ok.drop(columns=["month_index", "churned_next_month"]), on=ID)
features = features.merge(
    panel_eng_ok.drop(columns=["month_index", "churned_next_month"]), on=ID)

static_cols = ["grade", "live_or_replay", "old_new", "signup_lateness", "n_subjects"]
static = students[["student_key"] + static_cols].drop_duplicates("student_key")
features = features.merge(static, on="student_key", how="left")

features["months_enrolled"] = (features["month_index"]
                               - features["signup_lateness"]).clip(lower=1)

print(features.shape)       # (1610, 20) — แถวต้องไม่บวมไม่หาย
checks.check("ex_04_05", features)
