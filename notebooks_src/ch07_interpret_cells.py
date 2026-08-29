# Chapter 07 — อ่านโมเดล: ทำไมน้องคนนี้เสี่ยง
# สร้างด้วย: ./venv/bin/python -m src.nb_build notebooks_src/ch07_interpret_cells.py

TITLE = "07 — อ่านโมเดล: ทำไมน้องคนนี้เสี่ยง"

CELLS = [

# ============================================================ หัวบท
("md", r"""# 07 — อ่านโมเดล: ทำไมน้องคนนี้เสี่ยง

บท 06 เราวัดผลจนมั่นใจแล้วว่าโมเดลทายเก่งกว่าเดา แต่ mentor ไม่ได้โทรหาน้องเพราะเห็นเลข 0.87 —
เขาโทรเพราะรู้ว่า "น้องขาดเรียน 3 สัปดาห์ติด แถมไม่แตะข้อสอบเลย" บทนี้เราจะเปิดฝากล่องโมเดล
แปลงตัวเลขข้างในให้กลายเป็นประโยคภาษาไทยที่ทีมเอาไปใช้ได้จริง

**เรียนจบบทนี้ คุณจะ:**
- อ่าน coefficients ของ Logistic Regression แล้วเล่าเป็นภาษาคนได้
- cross-check ด้วย permutation importance — กันโดน coef หลอก
- เขียน `explain_student()` สรุป top-3 เหตุผลรายคนเป็นภาษาไทย (ของจริงที่จะส่งให้ mentor)
- ทำ error analysis: เดือนล่าสุดโมเดลพลาดใคร แล้วเราเรียนรู้อะไรจากมัน
- smell test: รู้ว่าโมเดลแบบไหน "อย่าเพิ่งใช้"

⏱ ~75 นาที · รันบน sample data ได้ทั้งเล่ม"""),

("code", r"""import sys; sys.path.insert(0, "..")
import pandas as pd
from src import checks, churn_utils, contracts
from src.config import DATA_DIR, IS_SAMPLE
plt = churn_utils.plot_style()
print("โหมดข้อมูล:", "SAMPLE (ข้อมูลจำลอง)" if IS_SAMPLE else f"REAL ({DATA_DIR})")"""),

("md", r"""## เตรียมของ: ข้อมูล + โมเดล

เราสร้าง features ด้วยฟังก์ชัน canonical ตัวเดิมจาก `src/churn_utils` (กฎเหล็กของโปรเจกต์:
train กับตอนใช้จริงต้องสร้าง feature ด้วยโค้ดตัวเดียวกัน) แล้ว fit **LogReg สูตร canonical
บน train ปี 2568 เท่านั้น** ไว้เป็น "ล่าม" ประจำบท — สมุดเล่มนี้จึงรันเดี่ยว ๆ ได้เสมอ
ไม่ต้องรอบทไหนมาก่อน

แล้วโมเดลที่บทก่อนเซฟไว้ใน `models/churn_model.joblib` ล่ะ? เราโหลดมาดูด้วย —
แต่จะหยิบมาใช้ตีความ **ก็ต่อเมื่อมันคือ LogReg ตัวเดียวกับ canonical จริง ๆ** เหตุผลสำคัญ:

> ทั้งบทนี้เราส่องโมเดล **บน test set (ปี 2569)** ถ้าไฟล์ที่เจอดันเป็นโมเดลที่ fit รวมปี 2569
> ไปแล้ว (เช่น เซฟตอนเทรนด้วยข้อมูลทั้งหมด) การอ่าน importance บน test จะกลายเป็น
> "ถามนักเรียนด้วยข้อสอบที่เขาเคยเห็นเฉลย" — ตัวเลขสวยแต่เชื่อไม่ได้
> ส่วนกรณีโมเดลหลักเป็นตระกูลต้นไม้ (เช่น HGB) ก็ใช้ LogReg ตัวนี้เป็นล่ามเหมือนกัน:
> ต้นไม้ทายเก่งแต่พูดไม่เก่ง ส่วน LogReg เล่าเรื่องได้"""),

("code", r"""import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR

read = lambda name, dates=(): pd.read_csv(DATA_DIR / f"{name}.csv", parse_dates=list(dates))
students   = read("students", ["signup_date"])
attendance = read("attendance_long", ["ep_final_date", "week_start"])
attempts   = read("exam_attempts", ["submitted_at"])
labels     = read("labels_monthly", ["churn_date"])
weekly     = read("weekly_metrics", ["week_start", "week_end"])

# features ชุด canonical เดียวกับตอน train (กัน train/serve skew)
features = churn_utils.build_features_monthly(labels, attendance, attempts, weekly, students)
df = features.dropna(subset=["churned_next_month"]).reset_index(drop=True)
num, cat = churn_utils.NUMERIC_FEATURES, churn_utils.CATEGORICAL_FEATURES
train, test = df[df["year"] == 2568], df[df["year"] == 2569]

def fit_canonical_logreg():
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    pipe = Pipeline([("prep", prep),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(train[num + cat], train["churned_next_month"])
    return pipe

# ล่ามประจำบท: fit บน train (2568) เท่านั้น — ตัวนี้แหละที่เราจะผ่าออกอ่านทั้งบท
logreg = fit_canonical_logreg()

# แล้วโมเดลที่บทก่อนเซฟไว้ล่ะ? เปิดดู — ใช้ก็ต่อเมื่อเป็น LogReg ตัวเดียวกับ canonical
bundle_path = MODELS_DIR / "churn_model.joblib"
if not bundle_path.exists():
    note = "ยังไม่มี models/churn_model.joblib — บทนี้ใช้ LogReg ที่ fit ในสมุดนี้"
else:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # กัน warning เวอร์ชัน sklearn ไม่ตรง
            b = joblib.load(bundle_path)
            saved = b["pipeline"] if isinstance(b, dict) else b
            saved.predict_proba(train[num + cat].head(1))   # เช็คสุขภาพก่อนใช้จริง
        same = (isinstance(saved[-1], LogisticRegression)
                and saved[-1].coef_.shape == logreg[-1].coef_.shape
                and np.allclose(saved[-1].coef_[0], logreg[-1].coef_[0], atol=1e-6))
        if same:
            logreg = saved
            note = "bundle ที่เซฟไว้ = LogReg canonical ตัวเดียวกัน → ใช้ตัวนั้นเลย"
        else:
            note = (f"bundle ที่เซฟไว้เป็น {type(saved[-1]).__name__} และไม่ตรงกับ canonical "
                    f"(คนละสูตร/คนละชุด train) → บทนี้อ่าน LogReg ที่ fit ในสมุดแทน "
                    f"เพราะจะส่องกันบน test set")
    except Exception as e:   # ไฟล์เสีย/คนละเวอร์ชัน sklearn → ไม่เป็นไร เรามีของเราแล้ว
        note = f"เปิด bundle ไม่สำเร็จ ({type(e).__name__}) — ใช้ LogReg ที่ fit ในสมุดแทน"
print("โมเดลที่จะอ่าน:", note)

ap = average_precision_score(test["churned_next_month"],
                             logreg.predict_proba(test[num + cat])[:, 1])
print(f"เช็คความจำ: AP บน test (ปี 2569) = {ap:.3f}  (base rate {test['churned_next_month'].mean():.3f})")"""),

# ============================================================ Unit 1: coefficients
("md", r"""## [แนวคิด] 7.1 โมเดลเชิงเส้นคือ "สูตรบวกคะแนนความเสี่ยง"

LogReg ให้คะแนนเด็กแต่ละคนแบบตรงไปตรงมา: เอาค่า feature (หลัง scale) คูณ coef แล้วบวกกันหมด
ผลรวมยิ่งสูง → ความเสี่ยงยิ่งสูง ดังนั้นอ่าน coef ได้เลยว่า:

- **coef เป็นบวก** → ค่า feature ยิ่งสูง ความเสี่ยงยิ่ง**ขึ้น**
- **coef เป็นลบ** → ค่า feature ยิ่งสูง ความเสี่ยงยิ่ง**ลง**
- ตัวเลขทุกตัวผ่าน StandardScaler ตอน fit → ขนาดของ coef เทียบข้ามตัวกันได้ (ผลต่อ 1 SD)

ทำไมต้องอ่าน? เพราะก่อนปล่อยให้ mentor ใช้ เราต้องตอบให้ได้ว่า "โมเดลดูอะไร" —
ถ้าเหตุผลข้างในฟังไม่ขึ้น อย่าเพิ่งเชื่อคะแนนข้างนอก"""),

("code", r"""# ตัวอย่าง (เคสคู่ขนาน): โมเดลจิ๋ว 2 features — ฝึกดึง "ชื่อ + coef" ออกจาก pipeline
mini_features = ["att_cum_pct", "streak_weeks"]
mini_prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), mini_features),
])
mini = Pipeline([("prep", mini_prep),
                 ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
mini.fit(train[mini_features], train["churned_next_month"])

mini_names = mini[:-1].get_feature_names_out()   # pipeline หั่นได้เหมือน list: [:-1] = ส่วน prep
mini_coefs = mini[-1].coef_[0]                   # [-1] = ตัว LogisticRegression · แถว 0 = คลาส churn
mini_df = pd.DataFrame({"feature": mini_names, "coef": mini_coefs}).round(3)
print(mini_df.to_string(index=False))
# อ่าน: att_cum_pct coef ติดลบชัด → เข้าเรียนสะสมยิ่งสูง ความเสี่ยงยิ่ง "ลด"
#       streak_weeks เกือบศูนย์ → ในโมเดลจิ๋วนี้แทบไม่มีผล

# ---- ของแถมให้เลย (ใช้ทั้งบท): dict ชื่อไทย + ตัวแปลชื่อ feature ----
THAI_NAMES = {
    "att_month_pct": "% เข้าเรียนเดือนนี้",
    "att_cum_pct": "% เข้าเรียนสะสม",
    "att_delta": "% เข้าเรียนเปลี่ยนจากเดือนก่อน",
    "practice_pct": "% ทำข้อสอบ practice",
    "checkpoint_pct": "% ทำข้อสอบ checkpoint",
    "exam_avg_score": "คะแนนสอบเฉลี่ย",
    "new_attempts_month": "จำนวนข้อสอบที่ทำเดือนนี้",
    "max_silent_weeks": "สัปดาห์ที่เงียบหาย (สูงสุด)",
    "streak_weeks": "สัปดาห์เข้าเรียนต่อเนื่อง",
    "months_enrolled": "เรียนมาแล้วกี่เดือน",
    "month_index": "เดือนที่เท่าไรของซีซัน",
    "signup_lateness": "สมัครช้ากี่เดือน",
    "n_subjects": "จำนวนวิชาที่ลง",
    "grade": "ระดับชั้น",
    "live_or_replay": "เรียนสด/เทป",
    "old_new": "เด็กเก่า/ใหม่",
}

def thai_name(feature):
    # "num__att_month_pct" -> "% เข้าเรียนเดือนนี้" · "cat__grade_ม.2" -> "ระดับชั้น = ม.2"
    base = feature.split("__", 1)[1] if "__" in feature else feature
    if base in THAI_NAMES:
        return THAI_NAMES[base]
    for key, th in THAI_NAMES.items():
        if base.startswith(key + "_"):
            return f"{th} = {base[len(key) + 1:]}"
    return base

print("\nลองตัวแปลชื่อ:", thai_name("num__att_month_pct"), "|", thai_name("cat__grade_ม.2"))"""),

("md", r"""### [แบบฝึกหัด 7.1] กราฟ coefficients ฉบับเล่าให้ทีมฟัง

ดึง coefficients ของโมเดลจริง (`logreg`) มาเรียงตามขนาด แล้ววาดเป็นกราฟชื่อไทย
(`THAI_NAMES` + `thai_name` จาก cell ตัวอย่างด้านบน และโค้ดกราฟ ให้ไว้แล้ว — งานคุณคือ 3 บรรทัด TODO)

1. `feat_names` — ชื่อ features หลังแปลง จากส่วน prep ของ pipeline (`logreg[:-1]`)
2. `coef_values` — ค่า coef แถวแรก จากตัว LogReg (`logreg[-1]`)
3. `order` — ลำดับ index ที่เรียงตาม **|coef| มาก → น้อย** (ระวัง: เรียง `coef` ดิบ ๆ
   จะได้ค่าติดลบไปกองท้ายตาราง ซึ่งผิด — ตัวลบแรง ๆ คือ feature สำคัญ!)

**ผลที่คาด (sample):** ตาราง 20 แถว · อันดับ 1 คือ "% เข้าเรียนเดือนนี้" coef ≈ −0.53
(เข้าเรียนเยอะ → เสี่ยงลด — สมเหตุสมผล) และกราฟแท่งแดง=ดันเสี่ยงขึ้น น้ำเงิน=กดเสี่ยงลง"""),

("code", r"""# ==== แบบฝึกหัด 7.1: coefficients + ชื่อไทย ====
# (THAI_NAMES และ thai_name มาจาก cell ตัวอย่างด้านบน — รันมาแล้วใช้ได้เลย)
____ = None  # กันสมุดพัง — อย่าลบบรรทัดนี้
feat_names = ____    # TODO: ชื่อ features หลังแปลง จากส่วน prep ของ logreg
coef_values = ____   # TODO: ค่า coef ของคลาส churn (แถวแรก)

if feat_names is None or coef_values is None:
    coef_df = None   # ยังไม่ได้เติมด้านบน — เติมแล้วรันใหม่
else:
    coef_df = pd.DataFrame({"feature": list(feat_names), "coef": list(coef_values)})
    coef_df["feature_th"] = coef_df["feature"].map(thai_name)
    order = ____     # TODO: ลำดับ index เรียงตาม |coef| มาก -> น้อย
    coef_df = coef_df.reindex(order)

    # ---- กราฟ (ให้แล้ว ไม่ต้องแก้) ----
    from matplotlib.patches import Patch
    d = coef_df.iloc[::-1]
    colors = ["#D55E00" if c > 0 else "#0072B2" for c in d["coef"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(d["feature_th"], d["coef"], color=colors, height=0.6)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_xlabel("coefficient (− กดความเสี่ยงลง · + ดันความเสี่ยงขึ้น)")
    ax.set_title("โมเดลดูอะไร — coefficients (เรียงตามขนาด)")
    ax.legend(handles=[Patch(color="#D55E00", label="ดันความเสี่ยงขึ้น"),
                       Patch(color="#0072B2", label="กดความเสี่ยงลง")],
              loc="lower right")
    plt.tight_layout()

checks.check("ex_07_01", coef_df)"""),

("md", r"""<details><summary>คำใบ้ 1 (แนวทาง)</summary>

pipeline หั่นได้เหมือน list — ทุกขั้นยกเว้นตัวสุดท้ายคือส่วนเตรียมข้อมูล (ถามชื่อ features
หลังแปลงจากมันได้) ส่วนขั้นสุดท้ายคือตัว LogisticRegression ที่เก็บ coef ไว้ในตัว
ลอกลายวิธีดึงจากโมเดลจิ๋วในตัวอย่างด้านบนได้เกือบตรง ๆ เลย ส่วนการเรียง:
สร้างลำดับ index จากค่า **สัมบูรณ์** ของ coef (ไม่ใช่ค่าดิบ) แล้วค่อยจัดตารางใหม่ตามลำดับนั้น
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.get_feature_names_out()` — เรียกจากส่วน prep ของ pipeline
- `.coef_[0]` — attribute ของตัว LogReg (แถวแรก = คลาส churn)
- `.abs()` + `.sort_values(ascending=False)` + `.index` — สร้างลำดับ แล้วส่งให้ `.reindex(...)`
</details>"""),

("code", r"""# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_07_01.py"""),

# ============================================================ Unit 2: permutation importance
("md", r"""## [แนวคิด] 7.2 Permutation importance — เครื่องจับโกหก coef

วิธีคิดแบบบ้าน ๆ: อยากรู้ว่าโมเดลพึ่งคอลัมน์ไหน "จริง" → เอาคอลัมน์นั้นมา**สลับค่ามั่ว ๆ**
(ข้อมูลพัง แต่คอลัมน์อื่นอยู่ครบ) แล้วดูว่า AP บน test ร่วงแค่ไหน — ร่วงมาก = พึ่งจริง

ข้อดีที่ทำให้มันเป็นเครื่องมือติดตัวระยะยาว:
- วัดจาก **ผลงานจริงบน test set** ไม่ใช่จากสูตรข้างในโมเดล
- ใช้ได้กับโมเดล**ทุกชนิด** — LogReg วันนี้, ต้นไม้พรุ่งนี้, อะไรก็ตามปีหน้า
- ให้ความสำคัญเป็น "ต่อคอลัมน์ต้นฉบับ" (16 ตัว) ไม่ใช่ต่อ dummy ย่อย ๆ อ่านง่ายกว่า

เดี๋ยวเราจะเห็นว่าอันดับจาก permutation กับอันดับจาก |coef| **ไม่ตรงกัน** ตรงไหน —
แล้วนั่นแหละคือบทเรียนสำคัญที่สุดของบทนี้"""),

("code", r"""# ตัวอย่าง (เคสคู่ขนาน): permutation importance ของโมเดลจิ๋ว
from sklearn.inspection import permutation_importance

r_mini = permutation_importance(
    mini, test[mini_features], test["churned_next_month"],
    scoring="average_precision",   # ใช้ metric เดียวกับตอนประเมินเสมอ
    n_repeats=10, random_state=42)

print(pd.DataFrame({"feature": mini_features,
                    "importance": r_mini.importances_mean}).round(4).to_string(index=False))
# อ่าน: att_cum_pct ≈ 0.13 → สลับคอลัมน์นี้แล้ว AP ร่วงเยอะ = โมเดลพึ่งมันจริง
#       streak_weeks ≈ 0.00 → สลับแล้วเฉย ๆ = แทบไม่ได้ใช้"""),

("md", r"""### [แบบฝึกหัด 7.2] permutation importance ของโมเดลจริง บน test set

1. เรียก `permutation_importance` กับ `logreg` **ทั้ง pipeline** (ส่ง X ดิบ ไม่ต้องแปลงเอง)
   บน `test[num + cat]` กับ label ของ test · ใช้ `scoring="average_precision"`,
   `n_repeats=10`, `random_state=42`
2. ตาราง + เรียง + กราฟ — โค้ดให้แล้ว
3. ผ่านแล้วอย่าเพิ่งเลื่อน: เทียบกับกราฟ 7.1 — **ตัวไหน coef ใหญ่แต่ importance จิ๋ว?**

**ผลที่คาด (sample):** ตาราง 16 แถว · อันดับ 1 = % เข้าเรียนเดือนนี้ (~0.09)
· ส่วน practice_pct ที่ coef สวย ๆ (+0.46) ร่วงไปเกือบท้ายตาราง (~0 หรือติดลบ)"""),

("code", r"""# ==== แบบฝึกหัด 7.2: permutation importance บน test ====
____ = None  # กันสมุดพัง — อย่าลบบรรทัดนี้
perm_result = ____   # TODO: permutation_importance(...) — ดู args ในโจทย์

if perm_result is None:
    perm_df = None   # ยังไม่ได้เติมด้านบน — เติมแล้วรันใหม่
else:
    perm_df = (pd.DataFrame({"feature": num + cat,
                             "importance": perm_result.importances_mean})
               .sort_values("importance", ascending=False)
               .reset_index(drop=True))

    # ---- กราฟ (ให้แล้ว ไม่ต้องแก้) ----
    d = perm_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([THAI_NAMES.get(f, f) for f in d["feature"]], d["importance"],
            color="#0072B2", height=0.6)
    ax.axvline(0, color="#999999", lw=1)
    ax.set_xlabel("AP ที่หายไปเมื่อสลับคอลัมน์นั้น (มาก = โมเดลพึ่งจริง)")
    ax.set_title("Permutation importance บน test set")
    plt.tight_layout()

checks.check("ex_07_02", perm_df)"""),

("md", r"""<details><summary>คำใบ้ 1 (แนวทาง)</summary>

ฟังก์ชันเดียวจบ อยู่ในโมดูล inspection ของ sklearn (import ให้แล้วในตัวอย่าง) — ส่งของ 4 อย่าง:
โมเดลทั้ง pipeline, ตาราง X ของ test แบบคอลัมน์ดิบทั้ง 16 ตัว, label ของ test
แล้วก็ keyword สามตัวตามโจทย์ ผลลัพธ์ที่คืนมามี attribute เก็บค่าเฉลี่ย importance
ต่อคอลัมน์ (ตัวเดียวกับที่ใช้ในตัวอย่าง)
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `permutation_importance(..., scoring="average_precision", n_repeats=10, random_state=42)`
- `.importances_mean` — attribute ของผลลัพธ์
</details>"""),

("code", r"""# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_07_02.py"""),

("md", r"""> ⚠️ **กับดัก! features ที่ correlate กัน ทำให้อ่าน coef ตรง ๆ ไม่ได้**
>
> บน sample ของเรา `att_month_pct` กับ `att_cum_pct` correlate กันถึง **~0.91** —
> ข้อมูลเกือบซ้ำกัน โมเดลเลย "แบ่งเครดิต" ระหว่างคู่แฝดยังไงก็ได้โดยคำทำนายแทบไม่เปลี่ยน
> ขนาดและทิศของ coef รายตัวจึงแกว่งได้ตามอารมณ์ของ solver
>
> ตัวอย่างคาตา: `practice_pct` ได้ coef **+0.46** — อ่านตรง ๆ คือ "ยิ่งทำ practice ยิ่งเสี่ยง"?!
> แต่ permutation บอก ~0: พอข้อมูลของมันซ้ำกับเพื่อนร่วมแก๊ง (checkpoint_pct, การเข้าเรียน)
> โมเดลใช้ coef บวกตัวนี้แค่ "ถ่วงดุล" เพื่อนตัวอื่น ไม่ใช่สัญญาณจริงจากข้อมูล
>
> **วิธีอยู่กับมัน:** อ่าน coef เป็นโครงเรื่องเบื้องต้น → cross-check ด้วย permutation เสมอ →
> และจำไว้ว่า permutation เองก็แบ่งความสำคัญระหว่างคู่แฝดเหมือนกัน (ทั้งคู่โดนกดต่ำลง)
> ทางที่ชัวร์คืออ่านเป็น **"ครอบครัว feature"** เช่น ครอบครัวการเข้าเรียนทั้งยวง ไม่ใช่รายตัว"""),

# ============================================================ Unit 3: explain_student
("md", r"""## [แนวคิด] 7.3 จาก "score 0.81" → "โทรหาน้องเพราะอะไร"

mentor ไม่ได้อยากรู้เลขทศนิยม — อยากรู้ว่า **ทำไม** โชคดีที่ LogReg ตอบได้เป็นรายคน:
คะแนนของเด็กหนึ่งคน = ผลบวกของ `coef × ค่า feature (หลัง scale) ของเด็กคนนั้น`
แต่ละก้อนเรียกว่า **contribution**

- contribution **บวกมาก** = ปัจจัยที่ดันความเสี่ยงของ "คนนี้" (ไม่ใช่ค่าเฉลี่ยรุ่น)
- เอา top-3 มาเรียบเรียงเป็นภาษาไทย → วางใน LINE ให้ mentor ใช้เปิดบทสนทนากับผู้ปกครองได้เลย

`churn_utils.logreg_contributions` คำนวณให้แล้ว (คืนตาราง 1 คอลัมน์ต่อ 1 transformed feature)
งานของเราคือห่อมันเป็นฟังก์ชัน `explain_student` ที่ใครก็เรียกใช้ได้"""),

("code", r"""# ตัวอย่าง (เคสคู่ขนาน): ดู contributions ดิบ ๆ ของเด็กเสี่ยงสุดเดือน มิ.ย.
month_ex = "2026-06"
june = features[(features["year"] == 2569) & (features["month"] == month_ex)].copy()
june["risk_score"] = logreg.predict_proba(june[num + cat])[:, 1]
kid = june.sort_values("risk_score", ascending=False).head(1)

kid_key = kid["student_key"].iloc[0]
kid_name = students.set_index("student_key")["display_name"].get(kid_key, "?")
print(f"เสี่ยงสุดของ {month_ex}: {kid_name} ({kid_key})  score = {kid['risk_score'].iloc[0]:.2f}")

contrib_ex = churn_utils.logreg_contributions(logreg, kid[num + cat]).iloc[0]
print("\nตัวที่ดันความเสี่ยงขึ้นมากสุด (ค่าบวก = ผลักไปทาง churn):")
print(contrib_ex.sort_values(ascending=False).head(5).round(2).to_string())"""),

("md", r"""### [แบบฝึกหัด 7.3] เขียน explain_student(student_key, month)

เขียนฟังก์ชันคืน **string 3 บรรทัด** = top-3 เหตุผลที่ดันความเสี่ยงของเด็กคนนั้นในเดือนนั้น
(โครงฟังก์ชัน + บรรทัด format ให้แล้ว — เติม 2 จุด)

1. `contrib` — contributions ของ row นี้ ผ่าน `churn_utils.logreg_contributions`
   (ผลเป็นตาราง 1 แถว → เอา `.iloc[0]` ให้เป็น Series)
2. `top3` — 3 อันดับที่ค่า**บวกสูงสุด** (เรียงจากมากไปน้อย แล้วหยิบหัวตาราง)

**ผลที่คาด (sample):** เรียกกับเด็กเสี่ยงสุดของ มิ.ย. จากตัวอย่างด้านบน จะได้ประมาณ

```
1) % เข้าเรียนเดือนนี้ — ดันความเสี่ยงขึ้น +0.66
2) สมัครช้ากี่เดือน — ดันความเสี่ยงขึ้น +0.37
3) % ทำข้อสอบ checkpoint — ดันความเสี่ยงขึ้น +0.33
```

(อยากโชว์ค่าจริงต่อท้าย เช่น "(ค่าจริง 35.7)" ก็เพิ่มได้ — ในเฉลยมีให้ดู)"""),

("code", r"""# ==== แบบฝึกหัด 7.3: อธิบายรายคนเป็นภาษาไทย ====
____ = None  # กันสมุดพัง — อย่าลบบรรทัดนี้

def explain_student(student_key, month):
    # คืน string 3 บรรทัด: top-3 เหตุผลที่ดันความเสี่ยงของเด็กคนนี้ในเดือนนั้น
    row = features[(features["student_key"] == student_key) & (features["month"] == month)]
    if len(row) == 0:
        return f"ไม่พบ {student_key} ในเดือน {month}"
    contrib = ____   # TODO: contributions ของ row นี้ (DataFrame 1 แถว -> .iloc[0] เป็น Series)
    top3 = ____      # TODO: 3 อันดับที่ค่าบวกสูงสุดจาก contrib
    if contrib is None or top3 is None:
        return "ยังไม่ได้เติมโค้ด — เติม ____ สองจุดด้านบนก่อน"
    lines = [f"{i}) {thai_name(f)} — ดันความเสี่ยงขึ้น {v:+.2f}"
             for i, (f, v) in enumerate(top3.items(), start=1)]
    return "\n".join(lines)

# ลองกับเด็กเสี่ยงสุดของเดือน มิ.ย. (จากตัวอย่างด้านบน)
print(explain_student(kid_key, month_ex))
checks.check("ex_07_03", explain_student)"""),

("md", r"""<details><summary>คำใบ้ 1 (แนวทาง)</summary>

ตัวคำนวณ contributions เป็นฟังก์ชันสำเร็จรูปในครัวกลาง (โจทย์บอกชื่อไว้แล้ว) —
ส่งโมเดลกับแถวของเด็กคนนี้เฉพาะคอลัมน์ features เข้าไป จะได้ตารางหนึ่งแถวกลับมา
หยิบแถวแรกออกมาให้กลายเป็น Series ก่อน จากนั้นเรียงจากมากไปน้อยแล้วหยิบสามตัวแรก —
เหมือนที่ cell ตัวอย่างทำกับเด็กเสี่ยงสุดเป๊ะ
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `churn_utils.logreg_contributions(...)` ต่อท้ายด้วย `.iloc[0]`
- `.sort_values(ascending=False)` + `.head(3)`
</details>"""),

("code", r"""# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_07_03.py"""),

# ============================================================ Unit 4: error analysis
("md", r"""## [แนวคิด] 7.4 Error analysis — โมเดลพลาดใคร แล้วเจ็บแบบไหน

โมเดลพลาดได้ 2 แบบ และสองแบบนี้ **ราคาไม่เท่ากัน**:

- **False positive** — โมเดลว่าเสี่ยง แต่น้องอยู่ต่อ → mentor โทรไป 1 สาย (15–20 นาที)
  กลายเป็น care call เสียเวลานิดหน่อย แถมผู้ปกครองมักประทับใจว่าเราใส่ใจ (ต้นทุนต่ำ)
- **False negative** — โมเดลว่าปลอดภัย แต่น้องหายไปจริง → เสีย subscription ทั้งก้อน
  โดยไม่มีใครได้โทรหาเลยสักสาย (แผลจริงของธุรกิจ)

เพราะงั้นเวลาส่องความผิดพลาด เราให้น้ำหนักกับ FN เป็นพิเศษ: ดูรายคนเลยว่าใครหลุดมือ
feature ของเขาหน้าตาเป็นยังไง — โมเดลตาบอดเรื่องอะไร"""),

("code", r"""# ตัวอย่าง (เคสคู่ขนาน): ส่อง "false positives" ของเดือนล่าสุดที่มี label ก่อน
last_month = test["month"].max()
mm = test[test["month"] == last_month].copy()
mm["risk_score"] = logreg.predict_proba(mm[num + cat])[:, 1]
mm = mm.sort_values("risk_score", ascending=False)

top30 = mm.head(30)                        # mentor โทรไหว ~30 สาย/เดือน (สายละ 15-20 นาที)
fp = top30[top30["churned_next_month"] == 0]
caught = int(top30["churned_next_month"].sum())
total_churn = int(mm["churned_next_month"].sum())
print(f"เดือน {last_month}: เด็ก active {len(mm)} คน · churn จริง {total_churn} คน")
print(f"top-30 จับถูก {caught} คน (precision@30 = {caught / 30:.2f}) · อีก {len(fp)} คนเป็น FP")
fp[["student_key", "risk_score", "att_month_pct", "att_cum_pct", "max_silent_weeks"]].head()"""),

("md", r"""### [แบบฝึกหัด 7.4] ตามหา false negatives

หาเด็กที่ **churn จริงแต่โมเดลไม่จัดให้อยู่ใน top-30** ของเดือนล่าสุด (คนที่หลุดมือ mentor)

1. `in_top30` — set ของ `student_key` ใน 30 อันดับแรก (`mm2` มี risk_score
   และเรียงจากเสี่ยงมาก → น้อยไว้แล้ว)
2. กรอง `mm2` เอาเฉพาะแถวที่ `churned_next_month == 1` **และ** ไม่อยู่ใน `in_top30`

คอลัมน์ผลลัพธ์ (เลือกให้แล้ว): student_key, month, churned_next_month, risk_score
+ features การเข้าเรียนไว้ประกอบการวินิจฉัย

**ผลที่คาด (sample):** ~6 คน · และจะเจอเรื่องน่าคิด — บางคนเข้าเรียน 80–90% ก็ยังหายไป!"""),

("code", r"""# ==== แบบฝึกหัด 7.4: false negatives เดือนล่าสุด ====
____ = None  # กันสมุดพัง — อย่าลบบรรทัดนี้
mm2 = mm.copy()   # จากตัวอย่างด้านบน: มี risk_score และเรียงมาก -> น้อยแล้ว

in_top30 = ____   # TODO: set ของ student_key ใน 30 อันดับแรกของ mm2

if in_top30 is None:
    fn_df = None  # ยังไม่ได้เติมด้านบน — เติมแล้วรันใหม่
else:
    fn_df = mm2[____]   # TODO: เงื่อนไข "churn จริง" และ "ไม่อยู่ใน in_top30"
    fn_df = fn_df[["student_key", "month", "churned_next_month", "risk_score",
                   "att_month_pct", "att_cum_pct", "max_silent_weeks", "streak_weeks"]]
    print(f"โมเดลพลาด (churn จริงแต่คะแนนต่ำ): {len(fn_df)} คน")
    print(fn_df.round(2).to_string(index=False))

checks.check("ex_07_04", fn_df)"""),

("md", r"""<details><summary>คำใบ้ 1 (แนวทาง)</summary>

ตารางถูกเรียงตามความเสี่ยงจากมากไปน้อยไว้แล้ว — สามสิบแถวแรกก็คือรายชื่อที่ mentor ได้โทร
เอาคอลัมน์รหัสนักเรียนของกลุ่มนั้นมาแปลงเป็นเซ็ตไว้เช็คสมาชิกภาพ ส่วนตัวกรอง:
ต้องจริงพร้อมกันสองเงื่อนไข — churn จริง **และ** "ไม่" อยู่ในเซ็ตนั้น
(อย่าลืมกลับด้านเงื่อนไขสมาชิกภาพ ไม่งั้นจะได้กลุ่มที่โมเดลจับได้แทน)
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.head(30)` + `set(...)` — สร้างเซ็ตรายชื่อ 30 อันดับแรก
- `.isin(...)` กลับด้านด้วย `~` · เชื่อมสองเงื่อนไขด้วย `&` (ครอบวงเล็บทีละก้อน)
</details>"""),

("code", r"""# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_07_04.py"""),

("md", r"""### ชวนคิด: ทำไมโมเดลพลาด?

ไล่ดูรายคนที่หลุดมือ จะเจอเด็กแบบนี้: เข้าเรียน 87% streak 9 สัปดาห์ ทำข้อสอบครบ —
สัญญาณพฤติกรรมไม่มีเลย แล้วอยู่ ๆ ก็ยกเลิก คำถามชวนคิด (ไม่มีเฉลยตายตัว):

- **โมเดลขาดข้อมูลมิติไหน?** เหตุผลฝั่งผู้ปกครอง เช่น การเงินที่บ้าน ย้ายไปติวที่อื่น
  ตารางเรียนพิเศษชน หรือความรู้สึกไม่คลิกกับพี่หมอ — ทั้งหมดนี้ไม่อยู่ใน Eduwise เลย
  → action ที่ถูกไม่ใช่ "เพิ่ม feature ไปเรื่อย ๆ" แต่คือ **เก็บเหตุผลตอนยกเลิกทุกครั้ง**
  (ต้นทุนต่ำมาก มูลค่าสูงมาก — ทั้งกับโมเดลรุ่นหน้าและกับการปรับ product)
- **หรือ label ผิด?** บางคน "หาย" เพราะจ่ายช้า/ขอพักชั่วคราว ไม่ใช่ churn จริง —
  ถ้าเจอเคสแบบนี้ กลับไปแก้ที่ enrollment_events ไม่ใช่แก้ที่โมเดล
- ML ไม่ได้อ่านใจคนได้ทุกคน — เป้าเราไม่ใช่จับ 100% แต่คือ **จับได้มากกว่าระบบ tier เดิม
  ด้วยแรง mentor เท่าเดิม** ซึ่งวัดไปแล้วในบทที่ 6"""),

# ============================================================ Unit 5: smell test
("md", r"""## [แนวคิด] 7.5 Smell test — โมเดลที่เชื่อได้ ต้องเล่าเรื่องแล้วเราพยักหน้าตาม

ก่อนปล่อยโมเดลไปทำงานจริง ถามคำถามเดียว: **"top features คือกลุ่มพฤติกรรมใช่ไหม?"**

- ถ้าใช่ (เข้าเรียน / เงียบหาย / ทำข้อสอบ) → ตรง common sense ธุรกิจติวเตอร์
  (เด็กที่จะหายมักแผ่วให้เห็นก่อน) และ **actionable** — mentor โทรตามได้ ช่วยทัน
- ถ้า top กลายเป็น grade/school/โปรไฟล์ตอนสมัคร → น่าสงสัย: อาจมี label ผิด,
  กลุ่มตัวอย่างเบี้ยว, leakage แบบเนียน ๆ หรือโมเดลกำลัง "เหมารวมตามโปรไฟล์"
  ซึ่งเอาไปทำอะไรไม่ได้ — โทรไปบอกว่า "น้องเสี่ยงเพราะอยู่ ม.3" เนี่ยนะ?

โมเดลที่ดูโปรไฟล์อย่างเดียวก็ "พอทายได้" ระดับหนึ่ง (เดี๋ยวเห็นในตัวอย่าง) —
แต่มันคือหมอดู ไม่ใช่ผู้ช่วย mentor"""),

("code", r"""# ตัวอย่าง (เคสคู่ขนาน): โมเดล "หมอดูโปรไฟล์" — ใช้เฉพาะข้อมูลตอนสมัคร ไม่มีพฤติกรรมเลย
profile_num = ["signup_lateness", "n_subjects"]
profile_cat = ["grade", "live_or_replay", "old_new"]
profile_prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), profile_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), profile_cat),
])
profile_model = Pipeline([("prep", profile_prep),
                          ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
profile_model.fit(train[profile_num + profile_cat], train["churned_next_month"])

ap_profile = average_precision_score(
    test["churned_next_month"],
    profile_model.predict_proba(test[profile_num + profile_cat])[:, 1])
print(f"AP หมอดูโปรไฟล์ = {ap_profile:.3f}  vs โมเดลจริง = {ap:.3f}")

r_prof = permutation_importance(
    profile_model, test[profile_num + profile_cat], test["churned_next_month"],
    scoring="average_precision", n_repeats=10, random_state=42)
print(pd.DataFrame({"feature": profile_num + profile_cat,
                    "importance": r_prof.importances_mean})
      .sort_values("importance", ascending=False).round(4).to_string(index=False))
# top เป็นจำนวนวิชา/ระดับชั้น/สมัครช้า ล้วน ๆ — ทายได้ระดับหนึ่ง แต่บอกอะไร mentor ไม่ได้เลย
# ถ้าโมเดลจริงของเราหน้าตาแบบนี้ = ธงแดง อย่าเพิ่งใช้"""),

("md", r"""### [แบบฝึกหัด 7.5] smell test โมเดลจริงของเรา

1. `top5_features` — ชื่อ feature 5 อันดับแรกจากคอลัมน์ `feature` ของ `perm_df` (ข้อ 7.2)
   เป็น list ของชื่อคอลัมน์ต้นฉบับ
2. cell จะพิมพ์ให้ดูว่า attendance/silent family ติด top-5 กี่ตัว — ตัวตรวจจะเช็คว่ามี
   อย่างน้อย 1 ตัว (ถ้าไม่มีเลย = โมเดลน่าสงสัย อย่าเพิ่งเอาไปใช้!)

**ผลที่คาด (sample):** list 5 ชื่อ มี `att_month_pct` และ `att_cum_pct` โผล่แน่นอน → ผ่าน smell test"""),

("code", r"""# ==== แบบฝึกหัด 7.5: smell test ====
ATT_FAMILY = {"att_month_pct", "att_cum_pct", "att_delta", "max_silent_weeks", "streak_weeks"}

____ = None  # กันสมุดพัง — อย่าลบบรรทัดนี้
top5_features = ____   # TODO: 5 ชื่อแรกจาก perm_df (list ของชื่อคอลัมน์ต้นฉบับ)

if top5_features is not None:
    hits = [f for f in top5_features if f in ATT_FAMILY]
    print("ครอบครัวพฤติกรรมที่ติด top-5:", hits if hits else "ไม่มีเลย — ธงแดง! 🚩")

checks.check("ex_07_05", top5_features)"""),

("md", r"""<details><summary>คำใบ้ 1 (แนวทาง)</summary>

ตารางผลของข้อ 7.2 ถูกเรียงจาก importance มากไปน้อยไว้แล้ว — แค่หยิบเฉพาะคอลัมน์ชื่อ
feature มาห้าแถวแรก แล้วแปลงให้เป็น list ธรรมดา
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.head(5)` + `.tolist()`
</details>"""),

("code", r"""# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_07_05.py"""),

# ============================================================ สรุป
("md", r"""## สรุปสิ่งที่ได้จากบทนี้

- **coefficients**: อ่านทิศ (+ ดันเสี่ยงขึ้น / − กดเสี่ยงลง) และขนาด (ผลต่อ 1 SD) ได้ —
  แต่รู้แล้วว่าอ่านเดี่ยว ๆ ไม่ได้เมื่อ features correlate กัน (att_month × att_cum ~0.91)
- **permutation importance**: เครื่อง cross-check ที่ใช้ได้กับโมเดลทุกชนิด ตลอดชีพ
- **explain_student()**: แปลงคะแนนเป็น 3 เหตุผลภาษาไทย — ของจริงที่ส่งให้ mentor ใช้เปิดสาย
- **error analysis**: FN สอนเราว่าข้อมูลขาดมิติไหน → เริ่มเก็บ "เหตุผลตอนยกเลิก" ตั้งแต่วันนี้
- **smell test**: พฤติกรรมต้องนำโปรไฟล์ — ไม่งั้นโมเดลเป็นหมอดู อย่าเพิ่งใช้

**บทต่อไป (08):** เอาทั้งหมดนี้เข้าเครื่องจริง — เติม `scoring/score_month.py` ให้รันได้ทุกเดือน:
สร้าง features → ให้คะแนน → top-30 พร้อมเหตุผล 3 ข้อต่อคน → ส่งรายชื่อให้ mentor โทร 🚀"""),

]
