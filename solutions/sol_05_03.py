# เฉลย 5.3 — PR curve + Average Precision บนข้อสอบปี 2569
precision, recall, thresholds = precision_recall_curve(y_test, proba_ok)
ap = average_precision_score(y_test, proba_ok)
base = y_test.mean()

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(recall, precision, color="#2a78d6", lw=2, label=f"โมเดล LogReg (AP = {ap:.3f})")
ax.axhline(base, color="#999999", ls="--", label=f"เดามั่ว (base rate = {base:.3f})")
ax.set_xlabel("Recall — เก็บเด็กเสี่ยงได้กี่ส่วนของตัวจริงทั้งหมด")
ax.set_ylabel("Precision — ในลิสต์ที่ชี้มา เป็นตัวจริงกี่ส่วน")
ax.set_title("PR curve — ทำนายเด็กหลุดคอร์ส ปี 2569")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.show()

print(f"AP = {ap:.3f} | base rate = {base:.3f} → ดีกว่าเดามั่ว {ap / base:.1f} เท่า")
checks.check("ex_05_03", ap)
