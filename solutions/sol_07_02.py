# เฉลย 7.2 — permutation importance บน test set (ส่ง X ดิบให้ทั้ง pipeline)
perm_result = permutation_importance(
    logreg, test[num + cat], test["churned_next_month"],
    scoring="average_precision",   # metric เดียวกับตอนประเมิน
    n_repeats=10, random_state=42)

perm_df = (pd.DataFrame({"feature": num + cat,
                         "importance": perm_result.importances_mean})
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))

d = perm_df.iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh([THAI_NAMES.get(f, f) for f in d["feature"]], d["importance"],
        color="#0072B2", height=0.6)
ax.axvline(0, color="#999999", lw=1)
ax.set_xlabel("AP ที่หายไปเมื่อสลับคอลัมน์นั้น (มาก = โมเดลพึ่งจริง)")
ax.set_title("Permutation importance บน test set")
plt.tight_layout()

print(perm_df.round(4).to_string(index=False))
checks.check("ex_07_02", perm_df)
