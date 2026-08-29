# Chapter 08 — Deploy: risk report รายเดือน → Eduwise/LINE
# build: ./venv/bin/python -m src.nb_build notebooks_src/ch08_deploy_cells.py

TITLE = "08 — Deploy: risk report รายเดือน → Eduwise/LINE"

CELLS = [

# ══════════════════════════════════════════════ header
("md", r"""# 08 — Deploy: risk report รายเดือน → Eduwise/LINE

บท 07 เราแกะสมองโมเดลจนตอบได้ว่า "ทำไมน้องคนนี้เสี่ยง" เป็นภาษาไทยแล้ว — มาถึงบทสุดท้ายครับ 🎉
โมเดลที่ดีแต่ไม่มีใครได้ใช้ = ไม่มีอยู่จริง บทนี้ผมจะพาคุณเปลี่ยนทุกอย่างที่สร้างมา 7 บท
ให้กลายเป็น "ระบบรายเดือน" ที่ mentor ใช้ได้จริง

**สิ่งที่จะได้ (~75 นาที):**
- รัน **batch scoring** จริงบนเดือน holdout ที่โมเดลไม่เคยเห็น → รายชื่อเด็กเสี่ยง + เหตุผลภาษาไทย
- กติกาเหล็กกัน **train/serve skew** — ทำไม feature ตอนใช้จริงต้องมาจาก "ครัวกลาง" เดียวกับตอน train
- นิสัยกันพลาด: ประกาศ **trained_through** ทุกครั้งก่อน score — แยกให้ออกว่าตัวเลขไหน
  in-sample (โมเดลเห็นเฉลยแล้ว) ตัวเลขไหน out-of-sample (เชื่อได้จริง)
- เช็คว่าโมเดล **ทับ/เสริม** ระบบ tier แดง-ส้ม ที่ mentor ใช้อยู่แค่ไหน
- วางแผน **monitoring + retrain** และปิดจุดอ่อนข้อมูลถาวร (ประวัติ enrollment ถูกทับ)

> เส้นทางที่ผ่านมา: บท 04 สร้าง features → บท 05–06 เทรน + validate + save โมเดล → บท 07 อ่านเหตุผลรายคน
> บทนี้: ต่อทุกอย่างเป็นสายพานเดียว แล้ววางแผนให้มันหมุนเองทุกสิ้นเดือน"""),

("code", r"""import sys; sys.path.insert(0, "..")
import pandas as pd
from src import checks, churn_utils, contracts
from src.config import DATA_DIR, IS_SAMPLE
plt = churn_utils.plot_style()
print("โหมดข้อมูล:", "SAMPLE (ข้อมูลจำลอง)" if IS_SAMPLE else f"REAL ({DATA_DIR})")"""),

# ══════════════════════════════════════════════ load tables + model guard
("md", r"""## เตรียมของ: ตาราง 5 ตัว + โมเดลจากบท 06

เซลล์ล่างโหลดข้อมูล แล้วเปิด "กล่องโมเดล" `models/churn_model.joblib` ที่ save ไว้ตอนจบบท 06
ของสำคัญในกล่องมี 3 อย่าง: `pipeline` (โมเดลพร้อม preprocess ในตัว), `features` (รายชื่อคอลัมน์
ที่ใช้ตอน train — ตอน serve จะได้หยิบ **ชุดเดียวกับตอน train** เสมอ ไม่ต้องเดา) และ
`trained_through` (โมเดลเห็น label ถึงเดือนไหน) — ตัวสุดท้ายนี้เราจะ **print ทุกครั้งก่อน score**
เพราะมันคือเส้นแบ่งระหว่างตัวเลขที่เชื่อได้ (out-of-sample) กับตัวเลขหลอกตา (in-sample)

ถ้ายังไม่มีไฟล์โมเดล (เช่น ข้ามมาจากบทอื่น) **หรือเปิดกล่องเก่าไม่ได้** เซลล์นี้จะ fit LogReg สูตรมาตรฐาน
ของคอร์สแบบเดียวกับบท 06 เป๊ะ — ใช้ label ถึงแค่ `2026-06` และกันเดือน `2026-07` ไว้เป็น holdout
เหมือนเดิม — แล้ว save ให้ก่อน จะได้เรียนบทนี้ได้เลยครับ

> ⚠️ **กับดัก!** ไฟล์โมเดลผูกกับเวอร์ชัน scikit-learn ที่ใช้ตอน fit — เปลี่ยนเครื่อง/อัปเดต library
> เมื่อไหร่ กล่องเก่าอาจ **"เปิดได้แต่ใช้ไม่ได้"** (พังตอนเรียก predict ไม่ใช่ตอนเปิด) เซลล์นี้จึง
> ลองยิง 1 แถวหลังเปิดกล่องทุกครั้ง ของจริงต้อง **pin เวอร์ชัน** ใน requirements.txt
> และเก็บ "วันที่เทรน + เวอร์ชัน" ติดกล่องไว้เสมอ"""),

("code", r"""import joblib
from src.config import PROJECT_ROOT, MODELS_DIR, REPORTS_DIR

read = lambda name, dates=(): pd.read_csv(DATA_DIR / f"{name}.csv", parse_dates=list(dates))
students   = read("students", ["signup_date"])
attendance = read("attendance_long", ["ep_final_date", "week_start"])
attempts   = read("exam_attempts", ["submitted_at"])
labels     = read("labels_monthly", ["churn_date"])
weekly     = read("weekly_metrics", ["week_start", "week_end"])
print(f"นักเรียน {students['student_key'].nunique()} คน | labels {len(labels):,} แถว (นักเรียน-เดือน)")

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
bundle = None
if MODEL_PATH.exists():
    try:
        bundle = joblib.load(MODEL_PATH)
        feat_spec = bundle["features"]
        probe = pd.DataFrame([{**{c: 0.0 for c in feat_spec["numeric"]},
                               **{c: "?" for c in feat_spec["categorical"]}}])
        bundle["pipeline"].predict_proba(probe)   # โหลดได้ยังไม่พอ — ต้องลองใช้จริง 1 แถวด้วย
    except Exception as e:
        print(f"กล่องโมเดลเดิมใช้ไม่ได้ ({type(e).__name__}) — น่าจะคนละเวอร์ชัน sklearn, fit ใหม่ให้เลย")
        bundle = None

if bundle is None:
    print("fit LogReg สูตรมาตรฐานของคอร์สให้ก่อน (แบบเดียวกับบท 06: train ถึง 2026-06 เท่านั้น)")
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num, cat = churn_utils.NUMERIC_FEATURES, churn_utils.CATEGORICAL_FEATURES
    feats = churn_utils.build_features_monthly(labels, attendance, attempts, weekly, students)
    train = feats.dropna(subset=["churned_next_month"])
    train = train[train["month"] <= "2026-06"]   # กัน 2026-07 ไว้เป็น holdout เหมือนบท 06
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    pipe = Pipeline([("prep", prep),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(train[num + cat], train["churned_next_month"])
    MODELS_DIR.mkdir(exist_ok=True)
    import sklearn
    joblib.dump({"pipeline": pipe,
                 "features": {"numeric": num, "categorical": cat},
                 "trained_through": "2026-06",   # key เดียวกับ bundle ของบท 06
                 "sklearn_version": sklearn.__version__,
                 "trained_at": pd.Timestamp.now().isoformat()}, MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)

pipeline = bundle["pipeline"]
num_cols = list(bundle["features"]["numeric"])
cat_cols = list(bundle["features"]["categorical"])
TRAINED_THROUGH = bundle.get("trained_through", "?")

def announce_scoring(month):
    # กติกาประจำบท: ประกาศทุกครั้งก่อน score ว่าโมเดลเห็น label ถึงไหน + กำลังให้คะแนนเดือนไหน
    print(f"โมเดลเห็น label ถึง: {TRAINED_THROUGH} | เดือนที่จะให้คะแนน: {month}")
    if TRAINED_THROUGH != "?" and month <= TRAINED_THROUGH:
        print("⚠️ in-sample! เดือนนี้โมเดลเห็นเฉลยไปแล้วตอน train — "
              "ดูกลไกได้ แต่ห้ามใช้ตัวเลขนี้วัดความแม่น")

pd.set_option("display.max_colwidth", 42)
print(f"โมเดลพร้อม: {MODEL_PATH.name} | เห็น label ถึง {TRAINED_THROUGH} | "
      f"ใช้ {len(num_cols)} numeric + {len(cat_cols)} categorical features")"""),

# ══════════════════════════════════════════════ unit 8.1
("md", r"""## [แนวคิด] 8.1 สายพานให้คะแนน: ข้อมูลเดือน t → รายชื่อเสี่ยง + เหตุผล

ทุกสิ้นเดือน คำถามของ mentor มีข้อเดียว: **"เดือนหน้า ควรโทรหาใครก่อน?"**
(โควตาโทรมี ~30 สาย/เดือน สายละ 15–20 นาที — โทรทุกคนไม่ไหว ต้องเรียงคิวให้ถูก)
batch scoring ตอบใน 3 ขั้น: ① เด็ก active เดือนนั้น (แถวใน `labels` ของเดือนนั้น) →
② สร้าง feature ด้วย `churn_utils.build_features_monthly` → ③ `pipeline.predict_proba` แล้วเรียงคะแนน

ขั้น ② คือหัวใจของบทนี้: **ต้องเรียกฟังก์ชันตัวเดียวกับตอน train เป๊ะๆ** เพราะโมเดลจำ "ความหมาย"
ของแต่ละ feature ตอน train ไว้ ถ้าตอน serve นิยามเพี้ยนไปนิดเดียว คะแนนจะเพี้ยนทั้งรายงาน
ปิดท้ายด้วยเหตุผล top-3 ภาษาไทยต่อคน (เครื่องมือจากบท 07) เพื่อให้ mentor รู้ว่าจะเปิดบทสนทนาว่าอะไร

ตัวอย่างล่างซ้อมกลไกกับเดือน **พ.ค.** — แต่โมเดลเรา train ด้วย label ถึง มิ.ย.
แปลว่า พ.ค. โมเดล **เห็นเฉลยไปแล้ว** (in-sample) คะแนนที่เห็นใช้ดูกลไกได้อย่างเดียว
สังเกตคำเตือนจาก `announce_scoring` ที่โผล่ก่อน score — ส่วนการให้คะแนน
"เดือนที่โมเดลไม่เคยเห็น" แบบใช้จริง เป็นหน้าที่ของแบบฝึกหัดครับ

> ⚠️ **กับดัก! train/serve skew** — เขียนโค้ดสร้าง feature "ใหม่" ตอนใช้งานจริง แม้ตั้งใจให้เหมือนเดิม
> (เช่น นับ attendance ต่างจากตอน train แค่บรรทัดเดียว) โมเดลจะได้ตัวเลขคนละความหมาย
> **โดยไม่มี error ฟ้องสักตัว** — คะแนนออกมาหน้าตาปกติ แต่ผิดหมด นี่คือบั๊กที่เจ็บสุดใน ML production"""),

("code", r"""# ── ตัวอย่าง: ให้คะแนนความเสี่ยง "เดือน พ.ค. 69" ทีละขั้น ─────────────────────
# ขั้น 1: เด็ก active เดือน พ.ค. = แถวใน labels ของเดือนนั้น
labels_may = labels[labels["month"] == "2026-05"].copy()

# ขั้น 2: สร้าง feature ด้วยครัวกลางตัวเดียวกับตอน train — กติกาเหล็ก!
features_may = churn_utils.build_features_monthly(labels_may, attendance, attempts, weekly, students)

# ขั้น 3: คะแนนเสี่ยง = ความน่าจะเป็นของ class 1 (คอลัมน์ที่สองของ predict_proba)
# — แต่ก่อน score ต้องประกาศก่อนเสมอว่าโมเดลเห็นข้อมูลถึงไหน
announce_scoring("2026-05")   # พ.ค. ≤ trained_through → เห็นคำเตือน in-sample (ตั้งใจโชว์ครับ)
risk_may = features_may.copy()
risk_may["risk_score"] = pipeline.predict_proba(risk_may[num_cols + cat_cols])[:, 1]

# ── helper: แปลง "เหตุผลเชิงตัวเลข" ของบท 07 เป็นภาษาไทยสั้นๆ ให้ mentor อ่านรู้เรื่อง ──
REASON_TH = {
    "att_month_pct": "การเข้าเรียนเดือนนี้", "att_cum_pct": "การเข้าเรียนสะสมทั้งซีซัน",
    "att_delta": "การเข้าเรียนเปลี่ยนจากเดือนก่อน", "practice_pct": "การทำ practice",
    "checkpoint_pct": "การทำ checkpoint", "exam_avg_score": "คะแนนสอบเฉลี่ย",
    "new_attempts_month": "จำนวนข้อสอบที่ทำเดือนนี้", "max_silent_weeks": "สัปดาห์ที่หายเงียบ",
    "streak_weeks": "ความต่อเนื่องรายสัปดาห์", "months_enrolled": "จำนวนเดือนที่เรียนมา",
    "month_index": "ช่วงเวลาของซีซัน", "signup_lateness": "สมัครช้ากว่ารุ่น",
    "n_subjects": "จำนวนวิชาที่ลง", "grade": "ระดับชั้น",
    "live_or_replay": "เรียนสด/เทป", "old_new": "เด็กเก่า/เด็กใหม่",
}

def thai_feature_name(feat_name):
    # "num__att_month_pct" -> "การเข้าเรียนเดือนนี้" | "cat__live_or_replay_เทป" -> "เรียนสด/เทป=เทป"
    raw = feat_name.split("__", 1)[1]
    if feat_name.startswith("cat__"):
        for base in ["grade", "live_or_replay", "old_new"]:
            if raw.startswith(base + "_"):
                return f"{REASON_TH[base]}={raw[len(base) + 1:]}"
    return REASON_TH.get(raw, raw)

def top3_reasons_thai(pipeline, X):
    # คืน DataFrame 3 คอลัมน์ reason_1..3 — แรงที่ "ดันความเสี่ยงขึ้น" มากสุด 3 อันดับต่อคน
    contrib = churn_utils.logreg_contributions(pipeline, X)   # จากบท 07 (canonical)
    rows = []
    for _, row in contrib.iterrows():
        top = row.sort_values(ascending=False).head(3)
        rows.append([f"{thai_feature_name(c)} (+{v:.2f})" if v > 0 else "-"
                     for c, v in top.items()])
    return pd.DataFrame(rows, columns=["reason_1", "reason_2", "reason_3"], index=X.index)

risk_may[["reason_1", "reason_2", "reason_3"]] = top3_reasons_thai(pipeline, risk_may[num_cols + cat_cols])
risk_may = risk_may.sort_values("risk_score", ascending=False)

# (+x.xx) = ขนาดของแรงที่ดันความเสี่ยงขึ้น เทียบกับเด็กเฉลี่ยของรุ่น (อ่านวิธีคิดละเอียดในบท 07)
print(f"เดือน 2026-05: active {len(risk_may)} คน | risk เฉลี่ย {risk_may['risk_score'].mean():.2f}")
risk_may[["student_key", "risk_score", "att_month_pct", "max_silent_weeks", "reason_1"]].head()"""),

("md", r"""### [แบบฝึกหัด 8.1] risk report เดือน ก.ค. 69 — เดือน holdout ที่โมเดลไม่เคยเห็น

ถึงตาคุณครับ: สร้างรายงานความเสี่ยงของ **`2026-07`** — เดือนที่บท 06 ตั้งใจกันไว้เป็น holdout
(โมเดล train ถึง มิ.ย. จึงไม่เคยเห็นเดือนนี้ = out-of-sample สถานการณ์เดียวกับเดือนใหม่จริงๆ
ตอน deploy) เด็ก active 106 คน ทำตามลำดับ — โครงเดียวกับตัวอย่างเดือน พ.ค. เป๊ะ เปลี่ยนแค่เดือน:

1. `labels_jul` — กรอง `labels` เอาเฉพาะแถวที่ `month == SCORE_MONTH`
2. `features_jul` — ส่ง `labels_jul` + ตารางดิบทั้ง 4 เข้า `churn_utils.build_features_monthly(...)`
3. `risk_score` — ความน่าจะเป็นของ class 1 จาก `pipeline` (ใช้เฉพาะคอลัมน์ `num_cols + cat_cols`)
4. `reason_1..3` — จาก `top3_reasons_thai(...)` (helper ที่เพิ่งสร้างในเซลล์ตัวอย่าง)

**ผลที่คาด:** บรรทัดประกาศจาก `announce_scoring` ต้อง **ไม่มีคำเตือน in-sample** ·
ตาราง 106 แถว เรียง risk มาก→น้อย มีคอลัมน์ `risk_score` (ค่า 0–1) และ `reason_1..3`
พร้อมตาราง 10 อันดับเสี่ยงสุดโผล่ใต้เซลล์ — นี่แหละคือ list ที่ mentor จะใช้โทรเดือน ส.ค."""),

("code", r"""# แบบฝึกหัด 8.1 — เติม ____ ทั้ง 3 TODO (โครงเดียวกับตัวอย่างเดือน พ.ค.)
SCORE_MONTH = "2026-07"
announce_scoring(SCORE_MONTH)   # นิสัยประจำบท: ประกาศก่อน score ทุกครั้ง

____ = None   # กันเซลล์พัง — แก้ ____ ในบรรทัด TODO ด้านล่างได้เลย

# TODO 1: เด็ก active เดือน ก.ค. + feature จากครัวกลาง (2 บรรทัด)
labels_jul = ____
features_jul = ____

risk_jul = None
if features_jul is not None:
    risk_jul = features_jul.copy()
    # TODO 2: คะแนนเสี่ยง = ความน่าจะเป็นของ class 1
    risk_jul["risk_score"] = ____
    # TODO 3: เหตุผล top-3 ภาษาไทย (helper จากเซลล์ตัวอย่าง)
    reasons = ____
    if reasons is not None:
        risk_jul[["reason_1", "reason_2", "reason_3"]] = reasons
        risk_jul = risk_jul.sort_values("risk_score", ascending=False)
        print(risk_jul[["student_key", "risk_score", "att_month_pct",
                        "max_silent_weeks", "reason_1"]].head(10).to_string(index=False))

checks.check("ex_08_01", risk_jul)   # ยังไม่ผ่านจนกว่าจะเติมถูก — ปกติ!"""),

("md", r"""<details><summary>คำใบ้ 1 (แนวทาง)</summary>

- เลื่อนขึ้นไปดูเซลล์ตัวอย่างเดือน พ.ค. — โครงเดียวกัน 100% แค่เปลี่ยนเดือนเป็นค่าที่อยู่ใน SCORE_MONTH
- ขั้นแรกกรองตาราง label ให้เหลือเฉพาะแถวของเดือนที่ต้องการ (ทำสำเนาหลังกรองกัน warning)
  แล้วส่งเข้าครัวกลางพร้อมตารางดิบอีก 4 ตัว เรียงลำดับเหมือนในเซลล์ตัวอย่างทุกตัว
- ความน่าจะเป็นที่โมเดลคืนมี 2 คอลัมน์ [โอกาสอยู่ต่อ, โอกาสหาย] — เราต้องการคอลัมน์หลัง
- เหตุผลรายคนไม่ต้องเขียนใหม่ — helper ภาษาไทยจากเซลล์ตัวอย่างรับโมเดลกับตาราง feature ชุดเดียวกัน
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- กรองเดือน: boolean mask เทียบคอลัมน์ `month` แล้วต่อท้ายด้วย `.copy()`
- `churn_utils.build_features_monthly(...)` — รับ 5 ตาราง เรียง: label ที่กรองแล้ว,
  attendance, attempts, weekly, students
- `.predict_proba(...)[:, 1]` — ใส่เฉพาะคอลัมน์ feature ครบชุด (numeric + categorical)
- `top3_reasons_thai(...)` แล้วปิดท้ายด้วย `.sort_values("risk_score", ascending=False)`
</details>"""),

("code", r"""# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_08_01.py"""),

# ══════════════════════════════════════════════ unit 8.2
("md", r"""## [แนวคิด] 8.2 จาก notebook → สคริปต์ production

notebook เหมาะกับการเรียนรู้ แต่ของจริงต้อง **รันเองได้โดยไม่มีใครเปิด notebook** —
ทุกสิ้นเดือน n8n/cron จะสั่ง `python scoring/score_month.py --month 2026-XX`
แล้วได้ไฟล์ `reports/risk_2026-XX.csv` ไปเข้า LINE/Eduwise ต่อ

เปิดไฟล์ `scoring/score_month.py` ดูคู่กันเลยครับ — โครงมันคือข้อ 8.1 ยกไปใส่ไฟล์:
โหลดตาราง → เปิดกล่องโมเดล → **TODO 3 จุด (ตรงกับที่คุณเพิ่งทำเป๊ะ)** → เขียน CSV + โชว์ top-30
สังเกตว่าสคริปต์อ่านรายชื่อ feature จาก "กล่องโมเดล" — serve ใช้ชุดเดียวกับ train เสมอโดยอัตโนมัติ

> ⚠️ **กับดัก!** อย่า copy โค้ดสร้าง feature มาแปะในสคริปต์เด็ดขาด — ต้อง `import churn_utils` เท่านั้น
> สูตรอยู่ที่เดียว แก้ครั้งเดียวได้ผลทั้ง train และ serve (สองสำเนาเมื่อไหร่ วันหนึ่งเหลื่อมกันแน่นอน)

*(ไฟล์เฉลย `score_month_solution.py` มีของเกินจาก skeleton สองจุด: ฟังก์ชัน `load_bundle()`
กันเคส "ไม่มีไฟล์โมเดล/กล่องเปิดไม่ได้" และบรรทัดประกาศ `trained_through` ก่อน score —
ไม่ใช่ส่วนของ TODO ข้ามได้เลยตอนเทียบ)*"""),

("code", r"""# ── ตัวอย่าง: เรียกสคริปต์ "เฉลยเต็ม" ให้คะแนนเดือน ก.ค. — พิสูจน์ว่ารันนอก notebook ได้จริง ──
import subprocess

r = subprocess.run(
    [sys.executable, "scoring/score_month_solution.py", "--month", "2026-07"],
    cwd=str(PROJECT_ROOT), capture_output=True, text=True,
)
print("\n".join(ln for ln in r.stdout.splitlines()
                if "⚠️" in ln or ln.startswith(("เขียน", "โมเดลเห็น"))))
if r.returncode != 0:
    print("สคริปต์ล้ม — stderr ท้ายๆ:")
    print("\n".join(r.stderr.strip().splitlines()[-5:]))

demo_report = pd.read_csv(REPORTS_DIR / "risk_2026-07.csv")
print(f"ไฟล์ report มี {len(demo_report)} คน — top-10 เสี่ยงสุด:")
demo_report.head(10)"""),

("md", r"""### [แบบฝึกหัด 8.2] สั่งสคริปต์ให้คะแนนเดือน ก.ค. 69 — แล้วเทียบกับมือคุณ

ข้อนี้ซ้อม "ท่าที่ n8n จะทำแทนคุณทุกสิ้นเดือน" คือ **เรียกสคริปต์จากข้างนอก แล้วอ่านไฟล์ผลลัพธ์**
ใช้เดือนเดียวกับข้อ 8.1 (**ก.ค. 69** — เดือน holdout) จะได้พิสูจน์ไปในตัวว่าสายพานในไฟล์
ให้คะแนน **ตรงกับที่คุณคำนวณเองใน notebook เป๊ะ** (= ไม่มี train/serve skew)
ในเซลล์ล่างเติม 2 ช่อง:

1. `month` — เดือนที่จะให้คะแนน เขียนแบบ `"YYYY-MM"`
2. `script` — สคริปต์ที่จะเรียก เลือกได้ 2 ทาง
   - `"scoring/score_month_solution.py"` — ฉบับเฉลย ใช้ได้เลยตอนนี้
   - `"scoring/score_month.py"` — **ของคุณเอง** ถ้าเปิดไฟล์ไปเติม `____` ทั้ง 3 จุดแล้ว (ดูการบ้านล่าง)

**การบ้านตัวจริงของข้อนี้ (ทำนอก notebook):** เปิด `scoring/score_month.py` ใน editor →
เติม TODO 3 จุด ด้วยคำตอบข้อ 8.1 (ในไฟล์ ตารางอยู่ใน dict `t` เช่น `t["labels"]`, `t["attendance"]`
รายชื่อ feature คือ `num`, `cat`, เดือนคือ `args.month` และต้องมีฟังก์ชัน `top3_reasons_thai` ในไฟล์ด้วย)
→ กลับมารันเซลล์นี้ด้วย `script = "scoring/score_month.py"` → **ต้องได้ CSV เหมือนเฉลยเป๊ะ**
→ ปิดท้ายด้วยการเปิดสองไฟล์เทียบบรรทัดต่อบรรทัด จุดที่ต่าง = บทเรียนของคุณ
(ลองเองก่อนอย่างน้อย 10 นาทีค่อยแอบดูเฉลยนะครับ 😄)

**ผลที่คาด:** ไฟล์ `reports/risk_2026-07.csv` — เด็ก active 106 คน เรียง risk มาก→น้อย
คะแนนตรงกับ `risk_jul` ของข้อ 8.1 ทุกคน พร้อมตาราง top-10 โผล่ใต้เซลล์"""),

("code", r"""# แบบฝึกหัด 8.2 — เรียกสคริปต์ production ให้คะแนนเดือน ก.ค. 69
import subprocess

____ = None
# TODO 1: เดือนที่จะให้คะแนน (สตริง "YYYY-MM" — เดือน holdout เดียวกับข้อ 8.1)
month = ____
# TODO 2: สคริปต์ที่จะเรียก — ของคุณเอง หรือฉบับเฉลย (สตริง path จาก root ของโปรเจกต์)
script = ____

risk_report = None
if month is None or script is None:
    print("ยังไม่ได้เติม month/script — แก้ ____ ด้านบนก่อนนะครับ")
else:
    report_path = REPORTS_DIR / f"risk_{month}.csv"
    report_path.unlink(missing_ok=True)          # ลบไฟล์เก่า กัน check ผ่านด้วยของค้าง
    r = subprocess.run([sys.executable, script, "--month", month],
                       cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    print("\n".join(ln for ln in r.stdout.splitlines()
                    if "⚠️" in ln or ln.startswith(("เขียน", "โมเดลเห็น"))))
    if r.returncode != 0:
        print("สคริปต์ยัง error — อ่าน stderr ท้ายๆ แล้วกลับไปแก้ TODO ในไฟล์:")
        print("\n".join(r.stderr.strip().splitlines()[-4:]))
    if report_path.exists():
        risk_report = pd.read_csv(report_path)
        show = [c for c in ["display_name", "risk_score", "att_month_pct", "reason_1"]
                if c in risk_report.columns]
        print(f"อ่าน {report_path.name} ได้ {len(risk_report)} คน — 10 อันดับที่ mentor ควรโทรก่อน:")
        print(risk_report.head(10)[show].to_string(index=False))

checks.check("ex_08_02", risk_report)   # ยังไม่ผ่านจนกว่าสคริปต์จะทำงานจริง — ปกติ!"""),

("md", r"""<details><summary>คำใบ้ 1 (แนวทาง)</summary>

- ในเซลล์นี้เติมแค่ 2 สตริง: เดือน (ตัวเดียวกับที่ข้อ 8.1 เพิ่งให้คะแนน) กับ path ของสคริปต์
  ตามที่โจทย์ระบุ (path เขียนเทียบจาก root โปรเจกต์ เพราะ subprocess ถูกตั้ง working directory ให้แล้ว)
- ถ้าเซลล์บอก "สคริปต์ยัง error" ให้อ่าน stderr บรรทัดสุดท้าย — มันบอกเลขบรรทัดที่พังในไฟล์
- TODO ในไฟล์คือ 3 ขั้นเดียวกับข้อ 8.1 (กรองเดือน → ครัวกลาง → คะแนน + เหตุผล)
  แค่เปลี่ยนไปหยิบของจากตัวแปรของสคริปต์: ตารางทั้งห้าอยู่ในตัวแปร dict
  ส่วนเดือนมาจาก argument ที่ผู้ใช้ส่งเข้ามา
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- ในเซลล์: เดือนรูปแบบ `"YYYY-MM"` (ปี ค.ศ.) · path สคริปต์ขึ้นต้นด้วยโฟลเดอร์ `scoring/`
- ในไฟล์ skeleton: กรองด้วย boolean mask เทียบกับ `args.month` แล้ว `.copy()` ·
  `churn_utils.build_features_monthly(...)` (ตารางทั้ง 5 หยิบจาก dict `t` ตามชื่อ) ·
  `.predict_proba(...)[:, 1]` · `top3_reasons_thai(...)` — ฟังก์ชันนี้ลอกทั้งก้อนจาก
  `score_month_solution.py` มาวางในไฟล์ได้เลย
</details>"""),

("code", r"""# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_08_02.py"""),

# ══════════════════════════════════════════════ unit 8.3
("md", r"""## [แนวคิด] 8.3 โมเดลใหม่ vs tier เดิม — ควรทับกันแค่ไหน?

ทีมเรามีระบบ tier (เขียว/เหลือง/ส้ม/แดง) ที่ mentor ใช้อยู่แล้ว ก่อนสลับไปเชื่อโมเดล
ต้องตอบให้ได้ว่า top-30 ของโมเดล (= โควตาโทร ~30 สาย/เดือนของ mentor)
สัมพันธ์กับเด็ก tier แดง/ส้ม ยังไง:

- **เหมือนกันเป๊ะ** → โมเดลไม่ได้เพิ่มอะไร ใช้ tier ต่อฟรีๆ ดีกว่า (ไม่ต้อง maintain โมเดล)
- **คนละโลกเลย** → น่าสงสัยมาก เพราะทั้งคู่กินสัญญาณเดียวกัน (การเข้าเรียน/ความเงียบ)
  มักแปลว่า feature เพี้ยนหรือโมเดลพัง ไม่ใช่โมเดลอัจฉริยะ
- **ที่อยากเห็น: ทับส่วนใหญ่ + เพิ่มชื่อใหม่บางคน** ที่ tier มองไม่เห็น — เช่น เด็กเข้าเรียนโอเค
  แต่สมัครช้า + เงียบเรื่องข้อสอบ นี่คือ "signal ใหม่" ที่ทำให้โมเดลคุ้มค่าการดูแล

ตัววัดง่ายๆ: นับรายชื่อที่ซ้อนกัน + **Jaccard** = ซ้อน ÷ ขนาด union (รวมสองเซ็ตแบบไม่นับซ้ำ)

> อ่าน Jaccard อย่างระวัง: สองลิสต์ขนาดไม่เท่ากัน (top-30 vs เด็กแดง/ส้มสิบกว่าคน) Jaccard จะต่ำ
> โดยธรรมชาติ ให้ดูคู่กับ **coverage = ซ้อน ÷ จำนวนเด็กแดง/ส้ม** ("โมเดลเก็บเด็กที่ tier ห่วงไว้กี่ %")"""),

("code", r"""# ── ตัวอย่าง: เดือน พ.ค. — top-30 ของโมเดล vs เด็ก tier แดง/ส้ม ─────────────────
# (เดือน พ.ค. เป็น in-sample — ใช้ซ้อมกลไกการเทียบ ไม่ได้อวดความแม่น)
# tier ของเดือน = tier ในแถว "สัปดาห์สุดท้าย" ของเดือนนั้น ต่อคน
wm_may = weekly[(weekly["year"] == 2569)
                & (weekly["week_start"].dt.strftime("%Y-%m") == "2026-05")]
last_week_may = wm_may.sort_values("week_start").groupby("student_key").tail(1)

active_may = set(labels.loc[labels["month"] == "2026-05", "student_key"])
tier_may = set(last_week_may.loc[last_week_may["tier"].isin(["red", "orange"]),
                                 "student_key"]) & active_may
top30_may = set(risk_may.head(30)["student_key"])   # risk_may จากเซลล์ตัวอย่าง 8.1 (เรียงแล้ว)

both = top30_may & tier_may
print(f"tier แดง/ส้ม พ.ค.: {len(tier_may)} คน | ทับกับ top-30 ของโมเดล: {len(both)} คน")
print(f"Jaccard = {len(both) / len(top30_may | tier_may):.3f}")
print(f"ชื่อใหม่ที่โมเดลเห็นแต่ tier ไม่เห็น: {len(top30_may - tier_may)} คน")"""),

("md", r"""### [แบบฝึกหัด 8.3] เดือน ก.ค.: โมเดลทับ tier แดง/ส้ม แค่ไหน?

ทำแบบเดียวกับตัวอย่าง แต่ใช้ **เดือน ก.ค. (`2026-07`)** และ `risk_jul` จากข้อ 8.1
(ถ้าข้อ 8.1 ยังไม่ผ่าน กลับไปทำก่อน หรือ `%load` เฉลยได้):

1. `last_week_jul` — `weekly` ปี 2569 เฉพาะสัปดาห์ที่ `week_start` อยู่ในเดือน ก.ค.
   → เรียงตาม `week_start` → เอาแถวสุดท้ายต่อคน (`groupby` + `tail(1)`)
2. `tier_jul` — **set** ของ student_key ที่ `tier` เป็น red หรือ orange **และ** ยัง active เดือน ก.ค.
   (เด็ก active ดูจาก `labels`)
3. `top30_jul` — **set** ของ student_key จาก 30 แถวแรกของ `risk_jul` (เรียงไว้แล้วจากข้อ 8.1)
4. dict `overlap_jul` ประกอบให้แล้ว — แค่เติม 3 ตัวบนให้ถูก

**ผลที่คาด:** ตอนผมเตรียมบทด้วยโมเดลจากบท 06 ได้ tier แดง/ส้ม 11 คน ทับกับ top-30 อยู่ 7 คน
Jaccard ≈ 0.21 → coverage = 7/11 ≈ 64% (เก็บเด็กที่ tier ห่วงไว้เกินครึ่ง) + เพิ่มชื่อใหม่อีก 23 คน
= ภาพ "ทับส่วนใหญ่ + เพิ่ม signal ใหม่" ที่เราอยากเห็นพอดี
— ตัวเลขของคุณอาจต่างเล็กน้อยถ้าโมเดลใน `models/` ไม่ใช่ตัวเดียวกับของผม
(เครื่องตรวจคำนวณตามโมเดลจริงในเครื่องคุณ ไม่ต้องกังวลครับ)"""),

("code", r"""# แบบฝึกหัด 8.3 — เติม ____ ทั้ง 3 TODO
MONTH = "2026-07"
____ = None

# TODO 1: weekly ปี 2569 เฉพาะสัปดาห์ของเดือน ก.ค. → แถวสัปดาห์สุดท้ายต่อคน
last_week_jul = ____

# TODO 2: set เด็ก tier แดง/ส้ม ที่ยัง active เดือน ก.ค. (intersect กับ set จาก labels)
tier_jul = ____

# TODO 3: set ของ 30 อันดับเสี่ยงสุดจาก risk_jul (ข้อ 8.1)
top30_jul = ____

overlap_jul = None
if tier_jul is not None and top30_jul is not None:
    overlap_jul = {
        "n_model": len(top30_jul),
        "n_tier": len(tier_jul),
        "n_overlap": len(top30_jul & tier_jul),
        "jaccard": len(top30_jul & tier_jul) / len(top30_jul | tier_jul),
    }
    print(overlap_jul)
    print(f"ชื่อใหม่ที่ tier มองไม่เห็น: {len(top30_jul - tier_jul)} คน — "
          "ไปดูเหตุผลใน risk_jul ได้เลยว่าโมเดลเห็นอะไร")

checks.check("ex_08_03", overlap_jul)"""),

("md", r"""<details><summary>คำใบ้ 1 (แนวทาง)</summary>

- ดูเซลล์ตัวอย่างเดือน พ.ค. เป็นแม่แบบ — โครงเดียวกันทุกขั้น เปลี่ยนแค่เดือน
- "สัปดาห์สุดท้ายของเดือนต่อคน": กรองตาราง weekly ให้เหลือปี 2569 เฉพาะสัปดาห์ของเดือนนั้น
  (แปลงคอลัมน์วันเริ่มสัปดาห์เป็นสตริงรูปแบบปี-เดือนก่อน ค่อยเทียบ)
  แล้วเรียงตามวันเริ่มสัปดาห์ ก่อนเก็บแถวท้ายสุดของแต่ละคน
- เด็ก active ของเดือน ดูจากตาราง label ของเดือนนั้น — แปลงเป็น set
  แล้วเอาไป intersect กับ set เด็กที่ tier เป็นแดงหรือส้ม
- top-30 ของโมเดล: ตารางผลข้อ 8.1 เรียงมาก→น้อยไว้แล้ว หยิบหัวตาราง 30 แถว
  มาทำเป็น set ของรหัสนักเรียน
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.dt.strftime("%Y-%m")` เทียบกับเดือนเป้าหมาย
- `sort_values(...)` → `groupby(...)` → `.tail(1)`
- `.isin(["red", "orange"])` · ประกอบ set ด้วย `set(...)` · intersect ด้วยตัวดำเนินการ `&`
- `.head(30)` ก่อนแปลงคอลัมน์รหัสนักเรียนเป็น set
</details>"""),

("code", r"""# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_08_03.py"""),

# ══════════════════════════════════════════════ unit 8.4
("md", r"""## [แนวคิด] 8.4 Monitoring — โมเดลไม่ใช่หม้อหุงข้าว ตั้งแล้วลืมไม่ได้

โมเดลจะค่อยๆ เพี้ยนแบบเงียบๆ เพราะโลกเปลี่ยน: ซีซันใหม่ เด็กรุ่นใหม่ รูปแบบคอร์สใหม่
ทางแก้คือ "ปิด loop" ทุกเดือน: เทียบ **ทำนายไว้ vs เกิดจริง** เหมือนที่คุณเทียบ budget vs actual ในงบ

ตัวอย่างล่างย้อนดู report เดือน ก.ค. (ไฟล์จากข้อ 8.2) ว่า top-30 จับเด็กที่หายจริงได้กี่คน
(precision@30) — รายงาน 1 บรรทัดแบบนี้แหละที่ควรจดลง log ทุกเดือน ตกเมื่อไหร่ = สัญญาณเตือน

> ⚠️ **กับดัก! in-sample vs out-of-sample** — ตัวเลข monitoring ต้องวัดจากเดือนที่โมเดล
> **ไม่เคยเห็นตอน train เท่านั้น** (out-of-sample เช่น ก.ค. ที่โมเดลเรา train ถึงแค่ มิ.ย.)
> ถ้าเผลอวัดเดือนที่โมเดลเห็นเฉลยแล้ว (in-sample เช่น พ.ค.) ตัวเลขจะสวยเกินจริง —
> เหมือนให้เด็กสอบข้อสอบชุดที่เคยเฉลยให้ดู คะแนนวัดความจำ ไม่ได้วัดความเก่ง
> บรรทัดประกาศ `trained_through` ก่อน score ทุกครั้ง มีไว้กันพลาดจุดนี้แหละครับ

> จำเกณฑ์ของทีมไว้ (จาก README): ถ้าโมเดลแพ้ tier heuristic ติดต่อกัน → หยุดเชื่อโมเดล
> กลับไปใช้ tier ชั่วคราว แล้วสืบสาเหตุ — อย่าฝืนใช้ของที่วัดแล้วแพ้ของเดิม"""),

("code", r"""# ── ตัวอย่าง: ปิด loop เดือน ก.ค. (out-of-sample) — ทำนายไว้ (report) vs เกิดจริง (labels) ──
jul_actual = labels.loc[labels["month"] == "2026-07", ["student_key", "churned_next_month"]]
chk = demo_report.merge(jul_actual, on="student_key", how="left")

p30 = churn_utils.precision_at_k(chk["churned_next_month"], chk["risk_score"], 30)
caught = int(chk.head(30)["churned_next_month"].sum())     # report เรียง risk มาก→น้อยแล้ว
total = int(chk["churned_next_month"].sum())
print(f"เดือน 2026-07: เด็กหายจริง {total} คน | top-30 จับได้ {caught} คน | precision@30 = {p30:.2f}")
print("เดือนนี้โมเดลไม่เคยเห็นตอน train → ตัวเลขนี้เชื่อได้จริง — "
      "บรรทัดนี้แหละ = log ประจำเดือนที่ต้องจดเก็บไว้ดู trend")"""),

("md", r"""### [แบบฝึกหัด 8.4] เขียน monitoring checklist ของคุณเอง (โจทย์เปิด)

ข้อสุดท้ายไม่มีโค้ดครับ 😄 เขียน checklist ที่ **คุณกับ mentor จะใช้จริงทุกเดือน** ลงในตัวแปร
`my_checklist` (string) ให้ครอบ 4 หมวด หมวดละอย่างน้อย 1–2 bullet:

1. **ทุกต้นเดือน** — เทียบ predicted vs actual ยังไง จดอะไรลง log
2. **สัญญาณว่าโมเดล/ข้อมูลเริ่มเพี้ยน** — ดูอะไรถึงรู้ว่า drift (คะแนน/จำนวนข้อมูล)
3. **Retrain เมื่อไหร่** — จังหวะไหน เงื่อนไขอะไร
4. **ข้อมูลที่ต้องเก็บต่อเนื่อง** — โดยเฉพาะ label ที่ระบบทับประวัติ

เขียนเป็นภาษาคน สั่งงานตัวเองได้จริง — เครื่องตรวจดูความยาว+จำนวน bullet (อย่างน้อย 4 ข้อ)
แล้วค่อยเปิดเฉลยเทียบว่าของผมมีอะไรที่คุณยังไม่ได้คิดถึง"""),

("code", r"""# แบบฝึกหัด 8.4 — เติมข้อความแทน ____ ทุกจุด (เพิ่ม bullet ได้ตามใจ)
my_checklist = '''
## Monitoring plan — ระบบเตือนเด็กเสี่ยงหลุด

### ทุกต้นเดือน
- ____

### สัญญาณว่าโมเดล/ข้อมูลเริ่มเพี้ยน
- ____

### Retrain เมื่อไหร่
- ____

### ข้อมูลที่ต้องเก็บต่อเนื่อง
- ____
'''
print(my_checklist)
checks.check("ex_08_04", my_checklist)   # เขียนจริงจังหน่อยนะ — เดี๋ยวได้ใช้จริงเดือนหน้าเลย"""),

("md", r"""<details><summary>คำใบ้ 1 (แนวทาง)</summary>

คำถามช่วยคิดต่อหมวด:
- ต้นเดือน: report เดือนก่อนบอกใครเสี่ยง แล้วใครหายจริง? จับได้กี่คนใน top-30? จดตัวเลขไว้ที่ไหน?
- สัญญาณเพี้ยน: risk เฉลี่ยกระโดดผิดปกติ? จำนวนแถว attendance/exam ตกฮวบ (ข้อมูลขาด ไม่ใช่เด็กหาย)?
  เด็กที่ mentor รู้ว่าปกติดีโผล่ top-10 หลายคน?
- retrain: ต้นซีซันใหม่ (มี.ค.) ดีสุดเพราะอะไร? retrain แล้วต้องเทียบกับอะไรก่อนสลับใช้?
- ข้อมูล: label sheet ต้องกรอกตอนไหน (ทำไมย้อนหลังไม่ได้)? snapshot enrollment ใครดูแล?
</details>

<details><summary>คำใบ้ 2 (ตัวอย่างบรรทัด)</summary>

```text
- เปิด risk report เดือนก่อน เทียบกับเด็กที่หายจริง → จด precision@30 ลงชีต log
- ถ้า precision@30 แพ้ tier heuristic 2 เดือนติด → หยุดเชื่อโมเดล กลับไปใช้ tier แล้วสืบสาเหตุ
```
</details>"""),

("code", r"""# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_08_04.py"""),

# ══════════════════════════════════════════════ closing
("md", r"""---

## แผนใช้จริง — จากบทเรียนสู่ระบบที่หมุนเองทุกเดือน

**1) Live test เดือน ส.ค. นี้ (รอบเดียวที่เหลือของซีซัน 69)**
ซีซันนี้เหลือรอบให้ยิงจริงแค่ครั้งเดียว: สิ้นเดือน ส.ค. รัน
`ML_CHURN_DATA=real ./venv/bin/python scoring/score_month.py --month 2026-08`
แล้วส่ง list ให้ mentor โทรจริง — อย่าคาดหวังปาฏิหาริย์จากรอบเดียว **เป้าหมายจริงคือระบบที่พร้อม
เต็มตัวตั้งแต่วันแรกของซีซัน 2570 (มี.ค.)** โดยมีข้อมูลปี 68+69 เป็นทุน

**2) Snapshot enrollment รายเดือน — แก้ปัญหาข้อมูลถูกทับ "ถาวร" (ผมตั้งให้)**
ปัญหาใหญ่สุดของโปรเจกต์นี้คือ Eduwise ทับประวัติการสมัคร/ยกเลิก ทำให้สร้าง label ย้อนหลังไม่ได้
ทางแก้เชิงระบบ: **n8n (หรือ cron) dump ตาราง `student_enrollments_tb` ทุกสิ้นเดือน →
`data/raw/2569_supabase/enrollments_snapshot_YYYY-MM.csv`** — มี snapshot รายเดือนเมื่อไหร่
เราเทียบเดือนต่อเดือนได้เองว่าใครหาย ไม่ต้องพึ่งความจำอีกต่อไป (ผมจะตั้ง workflow ให้ ไม่ต้องทำเองครับ)

**3) ขั้นถัดไป — ผมทำหลังคอร์สจบ**
- wire `reports/risk_YYYY-MM.csv` → n8n → **LINE แจ้ง mentor** ทุกสิ้นเดือน (top-10 + เหตุผล)
- หน้า risk dashboard ใน **pm-webapp** ให้เปิดดู list เต็ม + ติ๊กตามผลการโทร

**4) การบ้านเก็บข้อมูลเพิ่ม ปี 2570 (สำคัญกับความแม่นมากกว่าเปลี่ยนโมเดล)**

| ข้อมูล | ทำไมถึงสำคัญ |
|--------|--------------|
| วันชำระเงินรายเดือนต่อคน | จ่ายช้า = leading indicator ที่แรงที่สุดที่เรายังไม่มี — มักมาก่อนการหายตัว |
| สด/เทป + old/new ใน Eduwise ให้ครบทุกคน | ปีนี้กรอกไม่ครบ ต้องเดา/เติมมือ — feature กลุ่มนี้มีแรงทำนายจริง |
| Parent-meeting attendance | ผู้ปกครองที่มางาน 1-on-1 = ครอบครัวที่ engage — น่าจะกันหลุดได้แรง |

---

## สรุปสิ่งที่ได้จากบทนี้

บทนี้คุณรัน batch scoring จริงบนเดือน holdout ที่โมเดลไม่เคยเห็น เข้าใจกติกาเหล็กกัน
train/serve skew (feature จากครัวกลางเดียว) ติดนิสัยประกาศ `trained_through` ก่อน score
ทุกครั้งเพื่อแยก in-sample/out-of-sample พิสูจน์ว่าโมเดล "ทับส่วนใหญ่ + เพิ่มชื่อใหม่"
เหนือ tier เดิม และมี monitoring plan ที่ใช้ได้ทุกเดือน

มองย้อนทั้งคอร์ส: จากนิยาม churn → label → wrangling 2 ปี → EDA → features (และบทเรียน leakage)
→ โมเดล + validation → คำอธิบายภาษาไทย → ระบบ deploy ครบวงจร
โมเดลจะเก่งขึ้นทุกปีตามข้อมูลที่เราเก็บ และเด็กที่ "เกือบหลุดแต่ไม่หลุด" เพราะ mentor โทรถูกคน
คือกำไรที่แท้จริงของโปรเจกต์นี้ — ทั้งหมดนี้คุณเขียนเองทุกบรรทัด เก่งมากครับ 🎉💚"""),

]
