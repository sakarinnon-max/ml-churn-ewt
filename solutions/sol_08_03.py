# เฉลย 8.3 — top-30 ของโมเดล vs tier แดง/ส้ม เดือน ก.ค.
MONTH = "2026-07"

# TODO 1: weekly ปี 2569 เฉพาะสัปดาห์ของเดือน ก.ค. → แถวสัปดาห์สุดท้ายต่อคน
wm_jul = weekly[(weekly["year"] == 2569)
                & (weekly["week_start"].dt.strftime("%Y-%m") == MONTH)]
last_week_jul = wm_jul.sort_values("week_start").groupby("student_key").tail(1)

# TODO 2: set เด็ก tier แดง/ส้ม ที่ยัง active เดือน ก.ค.
active_jul = set(labels.loc[labels["month"] == MONTH, "student_key"])
tier_jul = set(last_week_jul.loc[last_week_jul["tier"].isin(["red", "orange"]),
                                 "student_key"]) & active_jul

# TODO 3: set ของ 30 อันดับเสี่ยงสุดจาก risk_jul (เรียงไว้แล้วจากข้อ 8.1)
top30_jul = set(risk_jul.head(30)["student_key"])

overlap_jul = {
    "n_model": len(top30_jul),
    "n_tier": len(tier_jul),
    "n_overlap": len(top30_jul & tier_jul),
    "jaccard": len(top30_jul & tier_jul) / len(top30_jul | tier_jul),
}
print(overlap_jul)
print(f"ชื่อใหม่ที่ tier มองไม่เห็น: {len(top30_jul - tier_jul)} คน — "
      "ไปดูเหตุผลใน risk_jul ได้เลยว่าโมเดลเห็นอะไร")
checks.check("ex_08_03", overlap_jul)
