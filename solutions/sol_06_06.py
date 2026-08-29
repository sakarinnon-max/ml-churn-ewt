# เฉลย 6.6 — fit ผู้ชนะบนข้อมูล ≤ 2026-06 แล้วเซฟเป็น bundle ให้บท 07–08 ใช้ต่อ
train_final = labeled[labeled["month"] <= "2026-06"]       # เว้น 2026-07 ไว้เป็น holdout
final_pipe = make_logreg().fit(train_final[X_COLS],
                               train_final["churned_next_month"])

bundle = {
    "pipeline": final_pipe,
    "features": {"numeric": churn_utils.NUMERIC_FEATURES,
                 "categorical": churn_utils.CATEGORICAL_FEATURES},
    "trained_through": "2026-06",
    "metrics": {"ap_oof_2569": float(ap_oof_69),
                "p30_oof_2569": float(p30_model_69),
                "p30_tier_2569": float(p30_tier_69)},
}

MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(bundle, MODELS_DIR / "churn_model.joblib")

print(f"fit ด้วย {len(train_final):,} แถว (ถึงเดือน {bundle['trained_through']})")
print("เซฟแล้ว:", MODELS_DIR / "churn_model.joblib")
print("เปิดกล่องเช็ค:", sorted(joblib.load(MODELS_DIR / "churn_model.joblib")))
checks.check("ex_06_06", bundle)
