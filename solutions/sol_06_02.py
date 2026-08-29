# เฉลย 6.2 — ศึก 9 ยกใต้กติกา expanding window (ยกโค้ดจาก cell ตัวอย่างมาใส่ลูป)
rows = []
for tr_idx, te_idx, m in churn_utils.expanding_window_splits(
        labeled, "month", min_train_months=2):
    X_tr, y_tr = labeled.loc[tr_idx, X_COLS], labeled.loc[tr_idx, "churned_next_month"]
    X_te, y_te = labeled.loc[te_idx, X_COLS], labeled.loc[te_idx, "churned_next_month"]

    p_lr = make_logreg().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    p_hgb = make_hgb().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

    rows.append({
        "test_month": m,
        "n_test": len(te_idx),
        "churn_test": int(y_te.sum()),
        "ap_logreg": average_precision_score(y_te, p_lr),
        "ap_hgb": average_precision_score(y_te, p_hgb),
    })

duel = pd.DataFrame(rows)

print(duel.round(3).to_string(index=False))
print(f"\nAP เฉลี่ย: LogReg {duel['ap_logreg'].mean():.3f} | HGB {duel['ap_hgb'].mean():.3f}")
print("→ แชมป์เก่าชนะ: data เล็กแบบเรา โมเดลซับซ้อนไม่ได้แปลว่าดีกว่า")
checks.check("ex_06_02", duel)
