# เฉลย 7.5 — smell test: top-5 importance ต้องมีครอบครัวพฤติกรรม (attendance/silent)
ATT_FAMILY = {"att_month_pct", "att_cum_pct", "att_delta", "max_silent_weeks", "streak_weeks"}

top5_features = perm_df["feature"].head(5).tolist()   # perm_df เรียงมาก -> น้อยแล้ว (ข้อ 7.2)

hits = [f for f in top5_features if f in ATT_FAMILY]
print("top-5 features:", top5_features)
print("ครอบครัวพฤติกรรมที่ติด top-5:", hits if hits else "ไม่มีเลย — ธงแดง! 🚩")

checks.check("ex_07_05", top5_features)
