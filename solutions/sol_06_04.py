# เฉลย 6.4 — threshold sweep แล้วปักหมุดจุดที่ mentor โทรไหว (~30 สาย/เดือน)
rows = []
for thr in thresholds:
    flag = oof69["p_model"] >= thr           # ใครติดลิสต์โทรบ้าง
    n_flag = int(flag.sum())
    rows.append({
        "threshold": thr,
        "calls_per_month": n_flag / n_months_69,
        "precision": oof69.loc[flag, "y_true"].mean() if n_flag else np.nan,
        "recall": oof69.loc[flag, "y_true"].sum() / oof69["y_true"].sum(),
    })

sweep = pd.DataFrame(rows)

# จุดปฏิบัติการ = แถวที่ภาระงานใกล้กำลังของ mentor (30 สาย/เดือน) ที่สุด
idx = (sweep["calls_per_month"] - 30).abs().argmin()
op_point = sweep.iloc[idx].to_dict()

print(sweep.round(3).to_string(index=False))
print(f"\nจุดที่เลือก: threshold {op_point['threshold']:.2f} | "
      f"{op_point['calls_per_month']:.1f} สาย/เดือน | "
      f"precision {op_point['precision']:.2f} | recall {op_point['recall']:.2f}")
print("คำตอบให้ผู้ปกครอง: ระบบไม่ได้ทำนายอนาคตแม่นๆ — มันช่วยจัดลำดับว่า mentor "
      "ควรโทรหาใครก่อน ให้ทุกสายมีค่ามากขึ้นเกือบเท่าตัวเทียบกับการสุ่มโทร")
checks.check("ex_06_04", op_point)
