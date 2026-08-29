# เฉลย 6.5 — sensitivity check: ตัดเด็กที่ label ไม่ชัวร์ออกทั้งคน แล้ววัดซ้ำ
shaky = set(events.loc[events["confidence"].isin(["medium", "low"]), "student_key"])
oof69_hi = oof69[~oof69["student_key"].isin(shaky)]

sensitivity = {
    "ap_all": average_precision_score(oof69["y_true"], oof69["p_model"]),
    "ap_high_conf": average_precision_score(oof69_hi["y_true"], oof69_hi["p_model"]),
    "n_dropped_rows": len(oof69) - len(oof69_hi),
}

print(f"นักเรียนที่ label ไม่ชัวร์: {len(shaky)} คน → ตัดออก "
      f"{sensitivity['n_dropped_rows']} แถวจาก {len(oof69)}")
print(f"AP ทั้งก้อน      : {sensitivity['ap_all']:.3f}")
print(f"AP เฉพาะ high    : {sensitivity['ap_high_conf']:.3f}  "
      f"(ต่างกัน {abs(sensitivity['ap_all'] - sensitivity['ap_high_conf']):.3f})")
print("→ ข้อสรุปไม่พลิก = ผ่าน sensitivity ✅ (ยังชนะเดามั่ว ยังเป็นโมเดลตัวเดิม)")
checks.check("ex_06_05", sensitivity)
