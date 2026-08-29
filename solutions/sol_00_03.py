# เฉลย ex_00_03 — % เข้าเรียนเฉลี่ยรายสัปดาห์ ซีซัน 2569
wm = pd.read_csv(DATA_DIR / "weekly_metrics.csv", parse_dates=["week_start", "week_end"])

wm69 = wm[wm["year"] == 2569]
weekly_avg = wm69.groupby("week_start")["att_week_pct"].mean()   # NaN = ไม่มีคลาส, mean ข้ามให้เอง

ax = weekly_avg.plot(marker="o", linewidth=2)
ax.set_title("เข้าเรียนเฉลี่ยรายสัปดาห์ — ซีซัน 2569 (sample)")
ax.set_xlabel("สัปดาห์")
ax.set_ylabel("% เข้าเรียน")
ax.set_ylim(0, 100)                  # แกนเริ่มที่ 0 — ไม่หลอกตา
ax.grid(axis="y", alpha=0.3)
plt.show()

checks.check("ex_00_03", weekly_avg)
