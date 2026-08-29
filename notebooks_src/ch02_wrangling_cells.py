# -*- coding: utf-8 -*-
"""บท 02 — Data wrangling: รวม 2 ปีเข้า schema เดียว (cell source)

build: ./venv/bin/python -m src.nb_build notebooks_src/ch02_wrangling_cells.py
"""

TITLE = "02 — Data wrangling: รวม 2 ปีเข้า schema เดียว"

CELLS = [

# ============================================================ หัวบท
("md", """# 02 — Data wrangling: รวม 2 ปีเข้า schema เดียว

⏱ ~90 นาที · ต่อจากบท 01 (เรามี label แล้ว — คราวนี้จัดการ "ข้อมูลพฤติกรรม" ให้เข้าที่)

**จบบทนี้คุณจะ:**
1. **Audit** ไฟล์ sheet เลอะๆ ของปี 68 ได้อย่างเป็นระบบ (`.info()`, `.value_counts()`, `.isna()`)
2. **Clean** ข้อมูลจาก Google Form เข้า contract กลางของโปรเจกต์
3. **Melt** ตาราง attendance แบบ wide (W1..W10) ให้เป็น long format ที่โมเดลใช้ได้
4. เข้าใจ**กับดักโดเมน 4 ข้อ**ของข้อมูล Eduwise — และรู้ว่าทำไม *ไม่ต้อง* (และห้าม!) เขียนแก้เอง
5. **Validate** ทุกตารางด้วย `contracts.validate_df` ก่อนส่งต่อให้บทถัดไป"""),

# ============================================================ setup
("code", """import sys; sys.path.insert(0, "..")
import pandas as pd
from src import checks, churn_utils, contracts
from src.config import DATA_DIR, IS_SAMPLE
plt = churn_utils.plot_style()
print("โหมดข้อมูล:", "SAMPLE (ข้อมูลจำลอง)" if IS_SAMPLE else f"REAL ({DATA_DIR})")

# ไฟล์จำลอง "sheet ปี 68" ใช้ฝึกในบทนี้ — อยู่ใน data/sample เสมอ (ทุกโหมด)
from pathlib import Path
from src.config import PROJECT_ROOT
MESSY_DIR = PROJECT_ROOT / "data" / "sample"
print("ไฟล์ฝึกบทนี้:", sorted(p.name for p in MESSY_DIR.glob("messy_*.csv")))"""),

# ============================================================ intro story
("md", """## สองปี สองโลก

ข้อมูลของเราตอนนี้อยู่กันคนละโลกจริงๆ ครับ:

| | ปี 2568 | ปี 2569 |
|---|---|---|
| ที่อยู่ | Google Form + Sheets (ทีมกรอกมือ) | Supabase (Eduwise) |
| หัวคอลัมน์ | ภาษาไทย "รหัส", "สด/เทป", TRUE/FALSE | ภาษาอังกฤษ, มี id เป็นระบบ |
| ความสะอาด | มีแถว test, ช่องว่าง, วันที่ พ.ศ. ปน ค.ศ. | สะอาดกว่า แต่มีกับดักโดเมนซ่อนอยู่ |

ทำไมต้องเหนื่อยรวมให้เป็น schema เดียว? เพราะแผนของเราคือ **train โมเดลด้วยปี 68 แล้วทดสอบกับปี 69**
(บท 06) — ถ้าคอลัมน์ชื่อไม่ตรง ค่าไม่ตรง โมเดลจะพังแบบเงียบๆ เช่น feature "สด/เทป" ของปี 68
หายไปเฉยๆ ตอน predict ปี 69 แล้วเราไม่รู้ตัว

งาน wrangling นี่แหละครับคือ 80% ของงาน ML จริง — ไม่หวือหวา แต่ผลตอบแทนสูงสุด:
**ถ้าข้อมูลผิด mentor จะโทรหาผิดคน** เด็กที่เสี่ยงจริงไม่มีใครโทรหา แบบนั้นแย่กว่าไม่มีโมเดลอีก

เป้าหมายปลายทางของบทนี้คือ contract ใน `src/contracts.py` — schema กลางที่ทั้ง 2 ปีต้องเข้ารูปเดียวกัน
มี `student_key` เป็นกุญแจกลาง (ปี 69 = uuid จาก Supabase, ปี 68 = `"2568-<รหัส>"`)"""),

# ============================================================ UNIT 1: audit
("md", """## [แนวคิด] 2.1 Audit ก่อน — อย่าเพิ่งแก้อะไรทั้งนั้น

เหมือนพี่หมอตรวจร่างกายก่อนรักษาครับ ถ้าเรารีบ clean โดยไม่ audit เราจะ "แก้สิ่งที่ไม่เห็น"
เช่น ลบแถวขยะไม่ครบ หรือแปลง dtype แล้วข้อมูลหายเงียบๆ อาวุธ audit มี 3 ชิ้น:

1. `df.info()` — เห็นภาพรวม: กี่แถว คอลัมน์อะไร dtype อะไร missing เท่าไหร่
2. `df["คอลัมน์"].value_counts(dropna=False)` — เห็นค่าเพี้ยน เช่น `"ม.3"` ปน `"3"` ปน `"ม3"`
3. `df["คอลัมน์"].isna().sum()` — นับช่องว่างเป็นตัวเลขชัดๆ

> ⚠️ **กับดัก!** ไฟล์ sheet เลอะๆ ให้อ่านด้วย `pd.read_csv(path, dtype=str)` เสมอ —
> ถ้าปล่อยให้ pandas เดา dtype เอง คอลัมน์ TRUE/FALSE จะกลายเป็น bool, "รหัส" กลายเป็นตัวเลข
> (เลขนำหน้าด้วย 0 หาย!) แล้วความเลอะจะถูก "ซ่อน" ก่อนที่เราจะได้เห็นมัน
> และระวังคอลัมน์วันที่จาก Google Form: ปี พ.ศ. (2568) ปน ค.ศ. (2025) ในไฟล์เดียวกัน —
> ห้าม `pd.to_datetime` ตรงๆ เด็ดขาด (บทนี้เราจึงยังไม่แตะคอลัมน์ "เวลากรอก")"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): sheet รับสมัครค่ายสั้นๆ ที่เลอะแบบเดียวกัน
camp = pd.DataFrame({
    "ชื่อ":     ["ไอซ์", "test", "ปุ๊น", None,  "มายมิ้น"],
    "ชั้น":     ["ม.2",  None,   "2",   None,  "ม.2 "],
    "จ่ายเงิน": ["TRUE", "FALSE", "TRUE", None, "TRUE"],
})
camp.info()   # แถวไหน non-null ไม่ครบ = มี missing
print()
print(camp["ชั้น"].value_counts(dropna=False))   # "ม.2" กับ "2" กับ "ม.2 " คือค่าเดียวกันในสายตาคน แต่คนละค่าในสายตา pandas!
print()
print("แถวที่ไม่มีชื่อ (ขยะ):", camp["ชื่อ"].isna().sum())"""),

("md", """### [แบบฝึกหัด 2.1] Audit ไฟล์รับสมัครปี 68

ทีมส่งไฟล์ export จาก Google Form ปี 68 มาให้ (จำลองไว้ที่ `messy_2568_students.csv` —
หัวคอลัมน์อิงของจริง: "รหัส", "ชื่อ", "สกุล", "ระดับชั้น", "สด/เทป", "old / new",
วิชาเป็น TRUE/FALSE, "เวลากรอก" วันที่ปนรูปแบบ) ก่อนแตะอะไร — audit ก่อนครับ

**คำสั่ง:**
1. อ่านไฟล์แล้วดู `raw68.info()` (โค้ดอ่านไฟล์เขียนให้แล้ว — สังเกตว่าใช้ `dtype=str`)
2. ลอง `value_counts(dropna=False)` กับคอลัมน์ `"ระดับชั้น"` และ `"สด/เทป"` — เจอกี่แบบ?
3. เติม dict `audit` 3 ช่อง:
   - `n_rows` = จำนวนแถวทั้งหมดในไฟล์
   - `n_junk` = จำนวนแถวขยะ (นิยาม: แถวที่ `"รหัส"` ว่าง — แถว test/แถวเปล่าไม่มีรหัสนักเรียน)
   - `n_missing_school` = จำนวนแถวที่ `"โรงเรียน"` ว่าง (นับทั้งไฟล์ รวมแถวขยะด้วย)

**ผลที่คาด:** `n_rows` = 30 · อีกสองค่าเป็นเลขหลักเดียว แล้ว `checks.check` จะบอกว่าผ่านไหม"""),

("code", """____ = None  # TODO: แก้ทุก ____ แล้วรันใหม่

raw68 = pd.read_csv(MESSY_DIR / "messy_2568_students.csv", dtype=str)
raw68.info()

# TODO 1: ลองดูค่าเพี้ยนในสองคอลัมน์นี้ (แค่ print ดู ไม่ต้องส่งตรวจ)
# print(raw68["ระดับชั้น"].value_counts(dropna=False))
# print(raw68["สด/เทป"].value_counts(dropna=False))

# TODO 2: เติม 3 ค่านี้
audit = {
    "n_rows": ____,
    "n_junk": ____,
    "n_missing_school": ____,
}
checks.check("ex_02_01", audit)   # ยังไม่ผ่านจนกว่าจะเติมถูก — ปกติ!"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

- จำนวนแถวทั้งหมด: ความยาวของ DataFrame ก็คือจำนวนแถวอยู่แล้ว — ฟังก์ชันวัดความยาวมาตรฐานของ Python ใช้ได้เลย
- "แถวที่คอลัมน์ X ว่าง" = เลือกคอลัมน์นั้นมาถามทีละช่องว่าง ว่างไหม (ได้จริง/เท็จรายแถว) แล้วรวมจำนวนที่เป็นจริง
- ค่าตัวเลขที่ pandas คืนมา ส่งตรวจได้เลย ไม่ต้องแปลงชนิดก่อน
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `len(...)`
- `.isna().sum()` — เรียกบนคอลัมน์เดียว (เลือกคอลัมน์ก่อนค่อยเรียก)
</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_02_01.py"""),

# ============================================================ UNIT 2: clean to contract
("md", """## [แนวคิด] 2.2 Clean เข้า contract — จากภาษาชีตเป็นภาษาโมเดล

Audit เสร็จ เรารู้แล้วว่าไฟล์นี้มีแถวขยะ, ค่ามีช่องว่างท้าย, TRUE/FALSE เป็น string
ทีนี้แปลงเข้า schema `students` ของ contract ทีละขั้น สูตรมาตรฐานคือ:

1. **กรองแถวขยะ** ออกก่อน (boolean mask + `.notna()`)
2. **rename** หัวคอลัมน์ไทย → อังกฤษ (dict mapping — เขียนครั้งเดียว ใช้ได้ทั้งปี)
3. **แปลงค่า**: TRUE/FALSE 4 วิชา → `subject_ids` (เช่น `"1;4;5"` — **sort แล้ว join `";"`**
   ให้ตรงรูปแบบที่ extractor ปี 69 ใช้เป๊ะ) + `n_subjects` · รหัสวิชา: เลข=1, ฟิสิกส์=4, เคมี=5, ชีวะ=6
4. **ทำความสะอาด string**: `.str.strip()` (ตัดช่องว่างท้าย) / `.str.lower()`
5. **สร้าง key กลาง**: `student_key = "2568-" + รหัส` — กุญแจที่เชื่อมกับทุกตารางในโปรเจกต์"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): แปลงคอลัมน์ TRUE/FALSE "ช่องทางที่รู้จัก EWT" เป็น string เดียว
know = pd.DataFrame({
    "รหัส":     ["901", "902", "903"],
    "Facebook": ["TRUE", "FALSE", "TRUE"],
    "Ig":       ["TRUE", "TRUE", "FALSE"],
})
CHANNEL = {"Facebook": "FB", "Ig": "IG"}

# ต่อแถว: เก็บชื่อช่องทางที่ค่าเป็น "TRUE" แล้ว join
know["channels"] = know.apply(
    lambda r: ";".join(v for c, v in CHANNEL.items() if r[c] == "TRUE"), axis=1)
know["n_channels"] = know["channels"].str.split(";").map(lambda x: 0 if x == [""] else len(x))

# rename + สร้าง key
know = know.rename(columns={"รหัส": "student_code"})
know["student_key"] = "2568-" + know["student_code"]
know"""),

("md", """### [แบบฝึกหัด 2.2] Clean ปี 68 เข้า contract students

เอาจริงแล้วครับ: แปลง `messy_2568_students.csv` ให้เป็นตาราง `students68`
ที่เข้า contract `students` (บางส่วน — คอลัมน์วันสมัครเราข้ามไปก่อนเพราะวันที่ พ.ศ./ค.ศ. ปนกัน)

**คำสั่ง:**
1. อ่านไฟล์ด้วย `dtype=str` (เหมือนข้อ 2.1) แล้ว**เก็บเฉพาะแถวที่ `"รหัส"` ไม่ว่าง**
2. rename: `"รหัส"`→`student_code`, `"ชื่อเล่น"`→`display_name`, `"ระดับชั้น"`→`grade`,
   `"สด/เทป"`→`live_or_replay`, `"old / new"`→`old_new` (ระวัง: มี space รอบ `/` ตามชีตจริง!)
3. สร้าง `subject_ids` จาก 4 คอลัมน์วิชา: ค่า `== "TRUE"` → เก็บรหัสวิชา
   (`{"เลข": 1, "ฟิสิกส์": 4, "เคมี": 5, "ชีวะ": 6}`) → **sort** → join ด้วย `";"`
   และ `n_subjects` = จำนวนวิชา
4. ทำความสะอาด: `live_or_replay` และ `grade` → `.str.strip()` ·
   `old_new` → `.str.strip().str.lower()` (ให้เหลือแค่ `old`/`new`)
5. เพิ่ม `year = 2568` (int) และ `student_key = "2568-" + student_code`
6. เลือกคอลัมน์ตามลำดับนี้: `student_key, year, student_code, display_name, grade,
   live_or_replay, old_new, n_subjects, subject_ids`

> **หมายเหตุ:** ตอน audit เราเจอ `grade` เพี้ยน (ม.3 / 3 / ม3) และ `สด/เทป` มีค่า "ผสม" —
> บทนี้**จงใจยังไม่ normalize** สองคอลัมน์นี้ (แค่ strip พอ) เพื่อโฟกัสโครงสร้างก่อน
> ตอนต่อข้อมูลจริงหลังบทนี้เราจะ map ม3/3 → ม.3 และเก็บ "ผสม" เป็นหมวดของมันเอง

**ผลที่คาด:** shape = (30 − จำนวนแถวขยะจากข้อ 2.1, 9) · แถวแรก `student_key` = `"2568-7001"`,
`subject_ids` = `"1;4;5;6"`, `n_subjects` = 4"""),

("code", """____ = None  # TODO: แก้ทุก ____ ไล่ทีละขั้นตามโจทย์ 1–6 แล้วรันใหม่

raw68 = pd.read_csv(MESSY_DIR / "messy_2568_students.csv", dtype=str)
SUBJ = {"เลข": 1, "ฟิสิกส์": 4, "เคมี": 5, "ชีวะ": 6}

try:
    df = ____                          # TODO 1: ตัดแถวขยะ — เก็บเฉพาะแถวที่ "รหัส" ไม่ว่าง (+ .copy())
    df = df.rename(columns=____)       # TODO 2: dict แปลหัวคอลัมน์ 5 คู่ (ระวัง "old / new" มี space รอบ /)
    ids = df.apply(____, axis=1)       # TODO 3: ต่อแถว → ลิสต์รหัสวิชาที่ค่าเป็น "TRUE" (sort แล้ว)
    df["subject_ids"] = ids.map(____)  #         ลิสต์ → string เดียว join ด้วย ";"
    df["n_subjects"] = ids.map(____)   #         ลิสต์ → จำนวนวิชา
    df["grade"] = ____                 # TODO 4: ตัดช่องว่างหัว/ท้าย
    df["live_or_replay"] = ____        #         ตัดช่องว่างหัว/ท้าย
    df["old_new"] = ____               #         strip + lower ให้เหลือแค่ old/new
    df["year"] = ____                  # TODO 5: 2568 (int)
    df["student_key"] = ____           #         "2568-" + student_code
    students68 = df[____]              # TODO 6: เลือก 9 คอลัมน์ตามลำดับ contract
except (AttributeError, TypeError, KeyError) as e:
    students68 = None
    print(f"ยังเติม ____ ไม่ครบ หรือขั้นไหนสักขั้นยังพัง ({type(e).__name__}: {e}) — ไล่แก้ทีละขั้นแล้วรันใหม่")

checks.check("ex_02_02", students68)   # ยังไม่ผ่านจนกว่าจะเติมถูก — ปกติ!"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

- เริ่มจากกรองเอาเฉพาะแถวที่รหัสไม่ว่าง แล้วทำสำเนาตารางเก็บไว้ก่อน — กันคำเตือนตอนแก้ค่าทีหลัง
- สร้างลิสต์รหัสวิชาต่อแถวเก็บไว้ก่อน (ได้ Series ของลิสต์) แล้วค่อยแตกเป็นสองคอลัมน์:
  ต่อลิสต์เป็นข้อความหนึ่งคอลัมน์ กับนับความยาวลิสต์อีกหนึ่งคอลัมน์ — จะได้ไม่ต้องวนซ้ำสองรอบ
- ตอนต่อลิสต์เป็นข้อความ ตัวเชื่อมต้องการสมาชิกที่เป็นข้อความ แต่รหัสวิชาเป็นตัวเลข — แปลงก่อน
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.notna()` · `.copy()`
- `.rename(columns={...})` — dict 5 คู่ คีย์คือหัวคอลัมน์ไทยเดิม
- `.apply(lambda r: ..., axis=1)` · `sorted(...)`
- `";".join(map(str, ...))` · `.map(len)`
- `.str.strip()` · `.str.lower()`
</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_02_02.py"""),

# ============================================================ UNIT 3: melt
("md", """## [แนวคิด] 2.3 Wide → Long — ปลดล็อกตาราง attendance ปี 68

ปี 68 ทีมเช็คชื่อใน Sheets แบบ **wide**: 1 แถวต่อ (นักเรียน, วิชา), คอลัมน์ W1..W10 เป็นรายสัปดาห์
ค่า มา/ขาด/ลา อ่านง่ายสำหรับคน แต่โมเดล (และ contract เรา) ต้องการ **long**:
1 แถวต่อ 1 เหตุการณ์ — เหมือน `attendance_long.csv` ของปี 69

เครื่องมือคือ `pd.melt` (หรือ `df.melt`):
- `id_vars` = คอลัมน์ที่ "ยึดไว้" ไม่ละลาย (รหัส, วิชา)
- คอลัมน์ที่เหลือถูกละลายเป็น 2 คอลัมน์ใหม่: `var_name` (ชื่อคอลัมน์เดิม เช่น "W3")
  กับ `value_name` (ค่าในช่อง เช่น "มา")

สัปดาห์ที่ยังไม่บันทึก (ช่องว่าง) จะกลายเป็นแถว NaN — ต้อง `dropna` ทิ้ง
ไม่ใช่นับเป็น "ขาด"! เด็กที่ทีมลืมจดกับเด็กที่ขาดจริงคือคนละเรื่องกัน"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): melt ตารางคะแนน quiz แบบ wide
quiz = pd.DataFrame({
    "รหัส": ["901", "902"],
    "Q1":   ["ผ่าน", "ไม่ผ่าน"],
    "Q2":   ["ผ่าน", None],       # 902 ยังไม่สอบ Q2
})
quiz_long = quiz.melt(id_vars="รหัส", var_name="quiz", value_name="result_th")
print("ก่อน dropna:", len(quiz_long), "แถว (มีแถว NaN ปนอยู่)")
quiz_long = quiz_long.dropna(subset=["result_th"])
quiz_long["result"] = quiz_long["result_th"].map({"ผ่าน": "pass", "ไม่ผ่าน": "fail"})
quiz_long"""),

("md", """### [แบบฝึกหัด 2.3] Melt attendance ปี 68 เป็น long

ไฟล์ `messy_2568_attendance_wide.csv`: คอลัมน์ `รหัส, วิชา, W1..W10` ค่า มา/ขาด/ลา
(บางช่องว่าง = ยังไม่บันทึก และบางค่ามีช่องว่างท้ายแบบชีตจริง)

**คำสั่ง:**
1. อ่านไฟล์ด้วย `dtype=str` แล้ว melt: `id_vars=["รหัส", "วิชา"]`,
   `var_name="week"`, `value_name="status_th"`
2. `dropna` แถวที่ `status_th` ว่าง (สัปดาห์ที่ยังไม่บันทึก)
3. map เป็นภาษา contract: `.str.strip()` ก่อน แล้ว
   มา→`present`, ขาด→`absent`, ลา→`leave` เก็บในคอลัมน์ `status`
4. `subject_id` จากชื่อวิชาไทย (dict `SUBJ` เดิม) · `week_no` = ตัด "W" ออกเป็น int ·
   `student_key` = `"2568-" + รหัส`
5. เลือกคอลัมน์: `student_key, subject_id, week_no, status`

**ผลที่คาด:** att68 ราวๆ 280–290 แถว × 4 คอลัมน์ · `status` มีแค่ 3 ค่า ·
ลอง `value_counts` ดู — "มา" ควรเป็นค่าส่วนใหญ่แบบขาดลอย"""),

("code", """____ = None  # TODO: แก้ทุก ____ ไล่ทีละขั้นตามโจทย์ 1–5 แล้วรันใหม่

wide68 = pd.read_csv(MESSY_DIR / "messy_2568_attendance_wide.csv", dtype=str)
print(wide68.head(3))

try:
    long68 = ____                        # TODO 1: melt — id_vars=["รหัส", "วิชา"], var_name/value_name ตามโจทย์
    long68 = long68.dropna(subset=____)  # TODO 2: ตัดสัปดาห์ที่ยังไม่บันทึก (ช่องว่าง ≠ ขาด!)
    long68["status"] = ____              # TODO 3: strip ก่อน แล้ว map มา/ขาด/ลา → ภาษา contract
    long68["subject_id"] = ____          # TODO 4: ชื่อวิชาไทย → รหัสวิชา (dict SUBJ เดิม)
    long68["week_no"] = ____             #         ตัด "W" ออก แล้วแปลงเป็น int
    long68["student_key"] = ____         #         "2568-" + รหัส
    att68 = long68[____]                 # TODO 5: เลือก 4 คอลัมน์ตามลำดับโจทย์
except (AttributeError, TypeError, KeyError) as e:
    att68 = None
    print(f"ยังเติม ____ ไม่ครบ หรือขั้นไหนสักขั้นยังพัง ({type(e).__name__}: {e}) — ไล่แก้ทีละขั้นแล้วรันใหม่")

checks.check("ex_02_03", att68)   # ยังไม่ผ่านจนกว่าจะเติมถูก — ปกติ!"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

- melt เสร็จลองดูหัวตารางสักสิบกว่าแถว จะเห็นคอลัมน์รายสัปดาห์ทั้งสิบละลายลงมาเรียงต่อกันแนวตั้งแล้ว
- ถ้า map แล้ว status มีค่าว่างโผล่ทั้งที่ตัดแถวว่างไปแล้ว — นั่นคือค่าที่มีช่องว่างท้าย
  (ค่า มา แบบมีช่องว่างต่อท้าย ไม่ใช่ค่าเดียวกับ มา เฉยๆ ในสายตา pandas) → ตัดช่องว่างก่อนค่อย map
- เมธอดจัดการข้อความของทั้งคอลัมน์รวมอยู่ใต้ accessor เดียวกันเกือบทั้งหมด — ทั้งตัดช่องว่างและตัดตัวนำหน้า
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.melt(id_vars=[...], var_name=..., value_name=...)`
- `.dropna(subset=[...])`
- `.str.strip()` · `.map({...})` — dict 3 คู่ ไทย → อังกฤษ
- `.str.removeprefix("W")` · `.astype(int)`
</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_02_03.py"""),

# ============================================================ extractor + 4 traps
("md", """## ปี 69 ไม่ต้องเหนื่อยเอง — `src/eduwise_extract.py` ทำให้แล้ว

ข่าวดี: ฝั่งปี 69 (Supabase) ผมเขียน extractor ไว้ให้เรียบร้อย มัน **ไม่ได้เขียน logic ใหม่**
แต่ replay `report_engine.py` ของ pm-webapp — ตัวเดียวกับที่ผลิตรายงานให้ mentor ใช้จริง
ทุกสัปดาห์ — ย้อนหลังทีละสัปดาห์ตั้งแต่ มี.ค. 69 แล้ว flatten ออกมาเป็น CSV ตาม contract

รันครั้งเดียวใน terminal (ไม่ต้องรันใน notebook — ใช้ creds Supabase และกินเวลาหลายนาที):

```bash
./venv/bin/python -m src.eduwise_extract            # ดึง+replay ปี 69 → data/raw/2569_supabase/
./venv/bin/python -m src.eduwise_extract --regress  # เทียบผลกับรายงานจริงใน pm-webapp/tasks.db
```

สิ่งสำคัญที่ extractor ป้องกันให้ คือ**กับดักโดเมน 4 ข้อ** — คุณต้อง*เข้าใจ*มัน
(เพื่ออ่านข้อมูลไม่ผิด) แต่**ห้าม implement เอง**:

> ⚠️ **กับดัก 1 — EP เดียว สอน 2 รอบ:** พี่หมอสอน 1 EP ซ้ำ 2 รอบคนละวัน (รอบสด+รอบซ่อม)
> เด็กเข้ารอบไหนก็ถือว่า "มา" EP นั้น — ถ้านับ attendance รายแถว/รายวันตรงๆ
> เด็กที่เข้ารอบสองจะถูกนับ "ขาด" รอบแรกฟรีๆ → attendance ต่ำเกินจริงทั้งระบบ
> extractor merge 2 รอบเหลือ 1 แถวต่อ (นักเรียน, วิชา, EP) ให้แล้ว

> ⚠️ **กับดัก 2 — `submitted_at` ไม่ใช่ `created_at`:** ใน Supabase คอลัมน์ `created_at`
> ของ exam attempts คือเวลาที่ระบบ bulk sync ข้อมูล — ขยะล้วนๆ ถ้ากรองช่วงเวลาด้วยมัน
> จะเห็นเด็ก "ทำข้อสอบ 50 ชุดในคืนเดียว" เวลาจริงที่เด็กส่งข้อสอบคือ `submitted_at` เท่านั้น

> ⚠️ **กับดัก 3 — ไม่มี retake → นับ distinct `exam_id`:** ระบบเราไม่มีสอบซ่อม
> ถ้าข้อมูลมี (นักเรียน, exam_id) ซ้ำ = ข้อมูลผิด ไม่ใช่เด็กขยัน การนับ "ทำไปกี่ชุด"
> ต้องนับ `exam_id` ที่ไม่ซ้ำ ไม่ใช่นับแถว

> ⚠️ **กับดัก 4 — ตัวหาร % = เฉพาะบทที่สอนแล้ว:** เด็กเพิ่งเรียนถึงบท 5 จาก 20 บท
> ทำข้อสอบครบ 5 ชุด = 100% ไม่ใช่ 25%! ตัวหารต้องนับเฉพาะข้อสอบของบทที่สอนไปแล้ว
> (+ ผ่อนผัน 2 สัปดาห์) ไม่งั้นเด็กทุกคนจะดู "ทำข้อสอบน้อย" ผิดๆ ช่วงต้นซีซัน

**ทำไมห้าม reimplement เอง?** สามเหตุผลครับ:
1. `report_engine.py` ผ่านการใช้งานจริงทุกสัปดาห์ + มี regression check เทียบกับ
   รายงานที่ mentor เห็น — เขียนใหม่เมื่อไหร่ เลขไม่ตรงกับรายงานเมื่อนั้น แล้วจะเถียงกันไม่จบ
2. กับดักอยู่ที่เดียว: ถ้าวันหน้ามีกับดักข้อ 5 แก้ที่ report_engine ที่เดียว ทุกระบบได้พร้อมกัน
3. Bonus ที่ extractor แถมให้: Supabase **ทับสถานะ enrollment ตอนยกเลิก** (ไม่มีประวัติ)
   — extractor เลย force ทุกคนเป็น confirmed ตอน replay เพื่อให้เด็กที่ออกไปแล้ว
   (กลุ่มที่โมเดลต้องเรียนรู้!) ยังมี metrics ครบ ส่วนสถานะจริงเก็บแยกใน `enrollments_snapshot.csv`"""),

("code", """# cell นี้แค่บอกวิธีรัน — ไม่รัน extractor จริงใน notebook (ต้องใช้ creds Supabase + ช้า)
if IS_SAMPLE:
    print("โหมด SAMPLE: ใช้ข้อมูลจำลองอยู่ ไม่ต้องรัน extractor")
    print("เมื่อพร้อมใช้ข้อมูลจริง รันใน terminal:")
else:
    print("โหมด REAL: ถ้า data/raw/2569_supabase/ ยังว่าง รันใน terminal:")
print("  ./venv/bin/python -m src.eduwise_extract")
print("  ./venv/bin/python -m src.eduwise_extract --regress   # เช็คเลขตรงกับรายงาน mentor")"""),

# ============================================================ UNIT 4: validate with contracts
("md", """## [แนวคิด] 2.4 Validate — ด่านตรวจสุดท้ายก่อนส่งของ

clean เสร็จไม่ได้แปลว่าถูกครับ ต้อง**พิสูจน์**ว่าถูก `contracts.validate_df(ชื่อตาราง, df)`
คือด่านตรวจของโปรเจกต์นี้: เช็คคอลัมน์ครบ, dtype ถูก, และ**กติกาเฉพาะตาราง**
ที่ฝังกับดักโดเมนไว้แล้ว เช่น

- `attendance_long`: ห้ามมี (student, subject, EP) ซ้ำ ← กับดัก 1 (EP 2 รอบต้อง merge แล้ว)
- `exam_attempts`: ห้ามมี (student, exam_id) ซ้ำ ← กับดัก 3 (ไม่มี retake)

ผ่าน → คืน `True` · ไม่ผ่าน → โยน `AssertionError` พร้อมข้อความไทยบอกจุดผิด
เรียกมันทุกครั้งที่สร้างตารางใหม่ — ถูกจับตอนนี้ ดีกว่าไปเจอตอนโมเดลทายมั่วในบท 05

จุดพลาดยอดฮิต: อ่าน CSV แล้วลืม `parse_dates` — คอลัมน์วันที่เป็น string
contract จะเตือนทันทีว่า "ควรเป็น datetime — ลอง pd.to_datetime(...)\""""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): validate ตาราง exams จิ๋วๆ ที่สร้างเอง
mini_exams = pd.DataFrame({
    "exam_id": ["68-S1-P1", "68-S1-P2"],
    "year": [2568, 2568],
    "subject_id": [1, 1],
    "exam_type": ["practice", "practice"],
    "chapter": ["1", "2"],
    "total_questions": [20, 10],
})
print("ตารางดี:", contracts.validate_df("exams", mini_exams))

# ตารางพัง (ลบคอลัมน์ทิ้ง) — ดูข้อความ error ที่ contract ฟ้อง
broken = mini_exams.drop(columns=["total_questions"])
try:
    contracts.validate_df("exams", broken)
except AssertionError as e:
    print("ตารางพัง:", e)"""),

("md", """### [แบบฝึกหัด 2.4] Validate ตาราง attendance + exam_attempts

ตรวจของจริง: อ่าน 2 ตารางหลักจาก `DATA_DIR` แล้วให้ contract พิสูจน์ว่าสะอาด
พร้อมเช็ค assumption กับดักโดเมนด้วยตัวเองอีกชั้น

**คำสั่ง:**
1. อ่าน `attendance_long.csv` โดย parse วันที่ 2 คอลัมน์: `ep_final_date`, `week_start`
2. อ่าน `exam_attempts.csv` โดย parse `submitted_at`
3. เติม dict `validation_report`:
   - `attendance_ok` = ผลจาก `contracts.validate_df("attendance_long", att)`
   - `attempts_ok` = ผลจาก `contracts.validate_df("exam_attempts", atm)`
   - `dup_exam_pairs` = จำนวนแถวซ้ำของคู่ `["student_key", "exam_id"]` ใน atm (กับดัก 3)
   - `dup_ep_rows` = จำนวนแถวซ้ำของ `["student_key", "subject_id", "episode_number", "year"]`
     ใน att (กับดัก 1)

**ผลที่คาด:** `{"attendance_ok": True, "attempts_ok": True, "dup_exam_pairs": 0, "dup_ep_rows": 0}`
— ศูนย์ทั้งคู่ เพราะ extractor merge/dedup มาให้แล้ว"""),

("code", """____ = None  # TODO: แก้ทุก ____

# TODO 1–2: อ่าน 2 ไฟล์ พร้อม parse_dates
att = ____
atm = ____

validation_report = {
    "attendance_ok": ____,
    "attempts_ok": ____,
    "dup_exam_pairs": ____,
    "dup_ep_rows": ____,
}
print(validation_report)
checks.check("ex_02_04", validation_report)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

- ตอนอ่านไฟล์ บอกให้ pandas แปลงคอลัมน์วันที่ได้เลยตั้งแต่ตอนอ่าน — ส่งรายชื่อคอลัมน์วันที่เข้าไป
- นับแถวซ้ำ: ถามตารางว่าแถวไหนซ้ำตามชุดคอลัมน์ที่สนใจ (ได้จริง/เท็จรายแถว) แล้วรวมเป็นจำนวน —
  ครอบเป็นจำนวนเต็มอีกชั้นให้ dict อ่านง่าย
- ถ้าด่านตรวจโยน error ให้อ่านข้อความของมัน — มันบอกตรงๆ ว่าคอลัมน์ไหนยังไม่เป็น datetime
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `pd.read_csv(..., parse_dates=[...])`
- `contracts.validate_df("attendance_long", ...)` · `contracts.validate_df("exam_attempts", ...)`
- `.duplicated([...]).sum()` · `int(...)`
</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_02_04.py"""),

# ============================================================ UNIT 5: schema assert
("md", """## [แนวคิด] 2.5 ด่านสุดท้าย — 2 ปีต้องต่อกันสนิท

ปลายทางของบทนี้คือ `pd.concat` ข้อมูล 2 ปีเป็นตารางเดียวแล้ว train ข้ามปีได้
แต่ `concat` ใจดีเกินไป — คอลัมน์ไม่ตรงกันมันก็ต่อให้ (เติม NaN เงียบๆ)
dtype ไม่ตรง (เช่น `subject_id` ปีนึงเป็น int อีกปีเป็น string) ก็รวมร่างเป็น object เงียบๆ
แล้วไประเบิดเอาไกลๆ ตอน fit โมเดล

วิธีป้องกันแบบมืออาชีพ: เขียน **guard function** เล็กๆ ของตัวเอง — `assert` เงื่อนไข
ที่เราต้องการ พร้อมข้อความบอกว่าพังตรงไหน เรียกก่อน concat ทุกครั้ง
เกณฑ์ "dtype เข้ากัน" ที่เราใช้: **เป็นตัวเลขทั้งคู่** (int64 กับ float64 ต่อกันได้ ไม่เป็นไร)
หรือไม่ก็ **dtype เดียวกันเป๊ะ** — เครื่องมือเช็คคือ `pd.api.types.is_numeric_dtype`"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): guard function จิ๋ว + เช็ค dtype "เข้ากัน"
def assert_has_col(df, col):
    assert col in df.columns, f"ไม่มีคอลัมน์ '{col}' — เช็คชื่อคอลัมน์อีกที"
    return True

a = pd.Series([1, 2, 3])            # int64
b = a.astype(float)                 # float64
c = pd.Series(["1", "2", "3"])      # string
print("int กับ float เป็นตัวเลขทั้งคู่:",
      pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b))   # เข้ากัน
print("int กับ string:",
      pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(c))   # ไม่เข้ากัน!

demo = pd.DataFrame({"x": [1]})
print(assert_has_col(demo, "x"))
try:
    assert_has_col(demo, "y")
except AssertionError as e:
    print("จับได้:", e)"""),

("md", """### [แบบฝึกหัด 2.5] เขียน guard เช็ค schema 2 ปีก่อน concat

เขียนฟังก์ชัน `assert_same_schema(df_a, df_b)` ที่:

**คำสั่ง:**
1. `assert` ว่า**ชุดคอลัมน์เท่ากัน** (ลำดับไม่สำคัญ — เทียบเป็น `set`)
   พร้อมข้อความบอกว่าคอลัมน์ไหนหาย/เกิน
2. วนทุกคอลัมน์ `assert` ว่า **dtype เข้ากัน**: เป็นตัวเลขทั้งคู่ หรือ dtype เท่ากันเป๊ะ
3. ผ่านหมด → `return True`
4. ทดลองใช้จริง: อ่าน `students.csv` จาก `DATA_DIR` แยกเป็นปี 68 / 69
   เรียก guard แล้วค่อย `pd.concat` — print shape ของตารางรวม

**ผลที่คาด:** guard ผ่านบน students 2 ปี (เพราะ extractor ทำ schema เดียวกันมาแล้ว)
· `checks.check` จะเอาฟังก์ชันคุณไปทดสอบกับเคสหลอกหลายแบบ — ทั้งเคสที่ต้องผ่านและเคสที่ต้องจับได้"""),

("code", """____ = None  # TODO: แก้บรรทัดนี้ แล้วเติม assert ทั้งสองจุด

def assert_same_schema(df_a, df_b):
    # TODO 1: ชุดคอลัมน์ต้องเท่ากัน (เทียบเป็น set — ลำดับไม่สำคัญ)
    assert ____, ("เงื่อนไขข้อ 1 ยังไม่เป็นจริง (ถ้า ____ ยังเป็น None แปลว่ายังไม่ได้เติม) — "
                  f"คอลัมน์ที่หาย/เกิน: {set(df_a.columns) ^ set(df_b.columns)}")
    # TODO 2: dtype ทุกคอลัมน์ต้อง 'เข้ากัน' (ตัวเลขทั้งคู่ หรือ dtype เดียวกัน)
    for c in df_a.columns:
        both_numeric = ____
        assert both_numeric or df_a[c].dtype == df_b[c].dtype, \\
            f"คอลัมน์ '{c}' dtype ไม่เข้ากัน: {df_a[c].dtype} vs {df_b[c].dtype}"
    return True

# TODO 3–4: ลองกับ students.csv จริง — แยกปี 68/69, เรียก guard, แล้ว concat
# stu = pd.read_csv(DATA_DIR / "students.csv")
# ...

checks.check("ex_02_05", assert_same_schema)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

- ข้อ 1: แปลงรายชื่อคอลัมน์ทั้งสองฝั่งเป็นเซ็ต แล้วเทียบว่าเท่ากันไหม — เท่านั้นเลย
- ข้อ 2: "ตัวเลขทั้งคู่" = เรียกตัวเช็คชนิดตัวเลขตัวเดียวกันกับทั้งสองฝั่ง แล้วต้องเป็นจริงทั้งคู่
- ขั้นทดลองใช้จริง: แยกปีด้วยการกรองแถวจากคอลัมน์ปี เรียก guard ให้ผ่านก่อน แล้วค่อยต่อตาราง
</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `set(...)` เทียบด้วย `==` — ส่วนตัวรายงานคอลัมน์ที่หาย/เกิน (`^` = symmetric difference) มีให้แล้วในโครง
- `pd.api.types.is_numeric_dtype(...)`
- `pd.concat([...], ignore_index=True)`
</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_02_05.py"""),

# ============================================================ summary
("md", """## สรุปสิ่งที่ได้จากบทนี้

- **Audit ก่อนแก้เสมอ**: `dtype=str` + `.info()` + `.value_counts(dropna=False)` + `.isna().sum()`
- **สูตร clean เข้า contract**: กรองขยะ → rename → แปลงค่า (TRUE/FALSE → `subject_ids`) →
  strip/lower → สร้าง `student_key` → validate
- **`pd.melt`** เปลี่ยน wide (W1..W10) เป็น long — และช่องว่าง ≠ ขาด ต้อง dropna
- **กับดักโดเมน 4 ข้อ** (EP 2 รอบ · `submitted_at` · distinct `exam_id` · ตัวหารเฉพาะบทที่สอน)
  อยู่ใน extractor ที่เดียว — เข้าใจมัน แต่อย่าเขียนเอง
- **`contracts.validate_df` + guard ของตัวเอง** คือเข็มขัดนิรภัยก่อนส่งข้อมูลให้บทถัดไป

**บทต่อไป (03 — EDA):** ข้อมูลเข้ารูปแล้ว ได้เวลาเปิดดูด้วยตา — churn พีคเดือนไหน?
เด็กที่กำลังจะออก พฤติกรรมต่างจากเด็กที่อยู่ต่อยังไง? (สปอยล์: `silent_weeks` คือพระเอก)
เจอกันครับ 💪"""),

]
