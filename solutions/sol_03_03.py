# เฉลย 3.3 — พฤติกรรมเดือน t ประกบ label ของเดือน t+1
wm["month"] = wm["week_start"].dt.strftime("%Y-%m")     # datetime → "YYYY-MM" ให้ join กับ label ได้

# สัปดาห์สุดท้ายของแต่ละ (นักเรียน, เดือน) = สถานะ ณ สิ้นเดือน t
month_snap = (wm.sort_values("week_start")
                .groupby(["student_key", "month"], as_index=False)
                .last())

att_vs_label = (labels.merge(month_snap[["student_key", "month", "att_cum_pct"]],
                             on=["student_key", "month"],   # กุญแจ 2 ดอก: คน + เดือน
                             how="left")
                      .dropna(subset=["churned_next_month"]))  # censored เทียบไม่ได้ ตัดทิ้ง

print(att_vs_label.shape)
print(att_vs_label.groupby("churned_next_month")["att_cum_pct"].median().round(1))
checks.check("ex_03_03", att_vs_label)
