# เฉลย 5.5 — ตารางประลอง: เดามั่ว vs tier heuristic vs โมเดล (วัดบนข้อสอบปี 2569)
base = y_test.mean()
tier_score = churn_utils.tier_baseline_score(test_df, weekly)   # 0=เขียว … 3=แดง

baseline_table = pd.DataFrame({
    "baseline": ["base_rate", "tier", "model"],
    "ap": [base,
           average_precision_score(y_test, tier_score),
           average_precision_score(y_test, proba_ok)],
    "precision_at_30": [base,
                        churn_utils.precision_at_k(y_test, tier_score, 30),
                        churn_utils.precision_at_k(y_test, proba_ok, 30)],
})

print(baseline_table.round(3).to_string(index=False))
print()
for _, r in baseline_table.iterrows():
    print(f"  {r['baseline']:10s} → โทร 30 สาย เจอตัวจริง ~{round(r['precision_at_30'] * 30)} คน")
checks.check("ex_05_05", baseline_table)
