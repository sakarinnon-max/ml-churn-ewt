# เฉลย 7.3 — explain_student: top-3 เหตุผลรายคน เป็นภาษาไทย (โบนัส: โชว์ค่าจริงต่อท้าย)
def explain_student(student_key, month):
    # คืน string 3 บรรทัด: top-3 เหตุผลที่ดันความเสี่ยงของเด็กคนนี้ในเดือนนั้น
    row = features[(features["student_key"] == student_key) & (features["month"] == month)]
    if len(row) == 0:
        return f"ไม่พบ {student_key} ในเดือน {month}"
    contrib = churn_utils.logreg_contributions(logreg, row[num + cat]).iloc[0]
    top3 = contrib.sort_values(ascending=False).head(3)   # ค่าบวกสูงสุด = ดันความเสี่ยงแรงสุด
    lines = []
    for i, (f, v) in enumerate(top3.items(), start=1):
        base = f.split("__", 1)[1] if "__" in f else f
        extra = ""
        if base in row.columns:                            # โบนัส: แนบค่าจริงของ feature
            actual = row.iloc[0][base]
            if isinstance(actual, (int, float, np.integer, np.floating)) and pd.notna(actual):
                extra = f" (ค่าจริง {float(actual):.1f})"
        lines.append(f"{i}) {thai_name(f)} — ดันความเสี่ยงขึ้น {v:+.2f}{extra}")
    return "\n".join(lines)

# ลองกับเด็กเสี่ยงสุดของเดือน มิ.ย. (จากตัวอย่างด้านบน)
print(explain_student(kid_key, month_ex))
checks.check("ex_07_03", explain_student)
