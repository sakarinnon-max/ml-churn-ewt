# เฉลย 4.4 — engagement features: max ของเดือน + snapshot สัปดาห์สุดท้ายของเดือน
wm = weekly.copy()
wm["month"] = wm["week_start"].dt.strftime("%Y-%m")

parts = []
for (year, month), g in panel_base.groupby(["year", "month"]):
    wm_m = wm[(wm["year"] == year) & (wm["month"] == month)].sort_values("week_start")

    max_silent = wm_m.groupby("student_key")["silent_weeks"].max()
    last = wm_m.drop_duplicates("student_key", keep="last").set_index("student_key")

    out = g.copy()
    out["max_silent_weeks"] = out["student_key"].map(max_silent).fillna(0).astype(int)
    out["streak_weeks"] = out["student_key"].map(last["streak_weeks"]).fillna(0).astype(int)
    out["practice_pct"] = out["student_key"].map(last["practice_pct"])      # NaN ได้
    out["checkpoint_pct"] = out["student_key"].map(last["checkpoint_pct"])  # NaN ได้
    parts.append(out)

panel_eng = pd.concat(parts, ignore_index=True)
print(panel_eng.shape)      # (1610, 9)
checks.check("ex_04_04", panel_eng)
