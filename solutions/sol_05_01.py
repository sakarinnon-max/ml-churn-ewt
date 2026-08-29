# เฉลย 5.1 — ColumnTransformer: สถานีตัวเลข (เติมรู → สเกล) + สถานีหมวดหมู่ (one-hot)
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   # NaN = "ไม่รู้" ไม่ใช่ 0 → เติมด้วยค่ากลาง
    ("scaler", StandardScaler()),                    # ให้ทุก feature เสียงดังเท่ากัน
])

prep = ColumnTransformer([
    ("num", num_pipe, NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),   # เจอค่าใหม่ก็ไม่ crash
])

# ยังไม่ fit — ตอนนี้ prep เป็นแค่ "ผัง" ว่าคอลัมน์ไหนเข้าสถานีไหน
print(prep)
checks.check("ex_05_01", prep)
