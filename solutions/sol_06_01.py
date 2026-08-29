# เฉลย 6.1 — expanding window เขียนเองกับมือ แล้วเทียบกับฟังก์ชันกลาง
rows = []
for m in months[2:]:                          # ข้าม 2 เดือนแรก (min_train_months=2)
    train = labeled[labeled["month"] < m]     # อดีตทั้งหมด ณ ตอนนั้น
    test = labeled[labeled["month"] == m]     # เดือนที่กำลังจะทำนาย
    rows.append({"test_month": m, "n_train": len(train), "n_test": len(test)})

fold_table = pd.DataFrame(rows)

# พิสูจน์ว่าตรงกับของกลางเป๊ะ (generator คืน train_idx, test_idx, test_month)
canon = pd.DataFrame([
    {"test_month": m, "n_train": len(tr_idx), "n_test": len(te_idx)}
    for tr_idx, te_idx, m in churn_utils.expanding_window_splits(
        labeled, "month", min_train_months=2)
])

print(fold_table.to_string(index=False))
print(f"\n{len(fold_table)} ยก | ตรงกับ churn_utils.expanding_window_splits:",
      fold_table.equals(canon))
checks.check("ex_06_01", fold_table)
