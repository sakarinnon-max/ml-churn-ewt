# เฉลย 5.4 — precision@30 = งบโทรจริงของ mentor 1 คนต่อเดือน
k = 30
p_at_30 = churn_utils.precision_at_k(y_test, proba_ok, k)
base = y_test.mean()

print(f"โทรตามลิสต์โมเดล {k} สาย → เจอตัวจริง {round(p_at_30 * k)} คน "
      f"(precision@{k} = {p_at_30:.3f})")
print(f"โทรสุ่ม {k} สาย        → เจอตัวจริง ~{round(base * k)} คน (= base rate {base:.1%})")
print(f"→ ลิสต์ของโมเดลพา mentor ไปเจอตัวจริงมากกว่าสุ่ม {p_at_30 / base:.1f} เท่า "
      f"ด้วยแรงเท่าเดิม 30 สาย")
checks.check("ex_05_04", p_at_30)
