# เฉลย 5.2 — Pipeline (prep + LogReg balanced) → สอนด้วยปี 2568 → ให้คะแนนเสี่ยงปี 2569
model = Pipeline([
    ("prep", prep_ok),                                            # หรือ prep ของคุณเองจากข้อ 5.1
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
])

model.fit(X_train, y_train)                   # fit ด้วยข้อสอนล้วนๆ — test ห้ามโดน .fit เด็ดขาด
proba = model.predict_proba(X_test)[:, 1]     # คอลัมน์ที่ 2 = P(churn) = คะแนนความเสี่ยง

print(f"proba: {len(proba)} ค่า | ต่ำสุด {proba.min():.3f} | สูงสุด {proba.max():.3f} "
      f"| เฉลี่ย {proba.mean():.3f}")
checks.check("ex_05_02", (model, proba))
