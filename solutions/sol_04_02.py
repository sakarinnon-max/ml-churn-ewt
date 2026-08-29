# เฉลย 4.2 — attendance features ทั้ง panel: cutoff ก่อน แล้วค่อยสรุปรายคน
att = att_long.copy()
att["month"] = att["ep_final_date"].dt.strftime("%Y-%m")

def _pct(s):
    return 100.0 * (s == "present").mean()

parts = []
for (year, month), g in panel_base.groupby(["year", "month"]):
    att_year = att[att["year"] == year]
    att_cut = churn_utils.cutoff(att_year, month, "ep_final_date")   # ยืนที่สิ้นเดือน t!
    prev = str(pd.Period(month, freq="M") - 1)

    cur = att_cut[att_cut["month"] == month].groupby("student_key")["status"].apply(_pct)
    cum = att_cut.groupby("student_key")["status"].apply(_pct)       # สะสมทั้งก้อนที่ cutoff แล้ว
    prv = att_cut[att_cut["month"] == prev].groupby("student_key")["status"].apply(_pct)

    out = g.copy()
    out["att_month_pct"] = out["student_key"].map(cur)
    out["att_cum_pct"] = out["student_key"].map(cum)
    out["att_delta"] = out["att_month_pct"] - out["student_key"].map(prv)  # NaN ลบอะไรก็ NaN
    parts.append(out)

panel_att = pd.concat(parts, ignore_index=True)
print(panel_att.shape)      # (1610, 8)
checks.check("ex_04_02", panel_att)
