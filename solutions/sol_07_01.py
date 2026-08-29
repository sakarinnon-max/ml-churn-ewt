# เฉลย 7.1 — coefficients เรียงตามขนาด + ชื่อไทย
feat_names = logreg[:-1].get_feature_names_out()   # ส่วน prep ของ pipeline → ชื่อหลังแปลง
coef_values = logreg[-1].coef_[0]                  # ตัว LogReg → coef ของคลาส churn (แถวแรก)

coef_df = pd.DataFrame({"feature": list(feat_names), "coef": list(coef_values)})
coef_df["feature_th"] = coef_df["feature"].map(thai_name)
order = coef_df["coef"].abs().sort_values(ascending=False).index   # เรียงตาม |coef| ไม่ใช่ค่าดิบ!
coef_df = coef_df.reindex(order)

from matplotlib.patches import Patch
d = coef_df.iloc[::-1]
colors = ["#D55E00" if c > 0 else "#0072B2" for c in d["coef"]]
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(d["feature_th"], d["coef"], color=colors, height=0.6)
ax.axvline(0, color="#999999", lw=1)
ax.set_xlabel("coefficient (− กดความเสี่ยงลง · + ดันความเสี่ยงขึ้น)")
ax.set_title("โมเดลดูอะไร — coefficients (เรียงตามขนาด)")
ax.legend(handles=[Patch(color="#D55E00", label="ดันความเสี่ยงขึ้น"),
                   Patch(color="#0072B2", label="กดความเสี่ยงลง")],
          loc="lower right")
plt.tight_layout()

print(coef_df.head(5).round(3).to_string(index=False))
checks.check("ex_07_01", coef_df)
