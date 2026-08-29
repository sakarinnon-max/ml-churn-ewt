# เฉลย 6.3 — bootstrap CI ของ AP ผู้ชนะ บน OOF ปี 69
ci_ap = churn_utils.bootstrap_ci(oof69["y_true"], oof69["p_model"],
                                 average_precision_score)
point, lo, hi = ci_ap

print(f"AP (LogReg, OOF ปี 69) = {point:.3f}")
print(f"95% CI = [{lo:.3f}, {hi:.3f}]  → กว้าง {hi - lo:.3f} "
      f"(ขอบบนเกือบเท่าตัวของขอบล่าง)")
print(f"เทียบเดามั่ว = {oof69['y_true'].mean():.3f} → ขอบล่างยังสูงกว่า แปลว่ามีของจริง")
checks.check("ex_06_03", ci_ap)
