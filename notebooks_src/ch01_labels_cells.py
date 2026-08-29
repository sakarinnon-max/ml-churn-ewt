# -*- coding: utf-8 -*-
"""บท 01 — นิยามปัญหา + สร้าง label (cell source)

build:  ./venv/bin/python -m src.nb_build notebooks_src/ch01_labels_cells.py
"""

TITLE = "01 — นิยามปัญหา + สร้าง label"

CELLS = [

# ----------------------------------------------------------------- หัวบท
("md", """# 01 — นิยามปัญหา + สร้าง label

บท 00 เราเปิดข้อมูลทั้ง 3 แหล่งดูจนคุ้นมือแล้ว — บทนี้คือบท "ปรัชญา"
ที่สำคัญที่สุดของคอร์ส เพราะกฎเหล็กมีข้อเดียว: **ไม่มี label = ไม่มี supervised learning**
และคำถามที่ค้างไว้ท้ายบท 00 ("ตกลงนับยังไงว่าเด็กคนหนึ่ง churn?") จะถูกตอบเป็นโค้ดในบทนี้

เป้าหมายปลายทางของโปรเจกต์นี้ชัดมาก: ทุกต้นเดือน mentor ได้ลิสต์
"น้องที่เสี่ยงหายเดือนหน้า" แล้วโทร/ทักไปดูแล **ก่อน** ที่ผู้ปกครองจะกดยกเลิก
แต่จะ train โมเดลแบบนั้นได้ ต้องมีตัวอย่างในอดีตก่อนว่า *ใครหายไป ตอนไหน*
— นั่นแหละครับคือ **label** และมันไม่ได้ตกจากฟ้า เราต้องสร้างเอง

**สิ่งที่จะได้จากบทนี้ (~90 นาที)**

1. frame churn ให้เป็นโจทย์ classification ราย **(นักเรียน, เดือน)**
2. เข้าใจ **censoring** — เดือนไหน "ตอบไม่ได้" ต้องกล้าตอบว่าไม่รู้ (NaN)
3. เข้าใจ **label noise** — label เรามาจากบันทึกมือ ไม่ใช่ database เทพๆ
4. สร้างตาราง `labels_monthly` จาก `enrollment_events` ได้ด้วยมือตัวเอง
   แล้วเทียบกับ canonical function ของโปรเจกต์"""),

("code", """import sys; sys.path.insert(0, "..")
import pandas as pd
from src import checks, churn_utils, contracts
from src.config import DATA_DIR, IS_SAMPLE
plt = churn_utils.plot_style()
print("โหมดข้อมูล:", "SAMPLE (ข้อมูลจำลอง)" if IS_SAMPLE else f"REAL ({DATA_DIR})")"""),

# ----------------------------------------------------------------- นิยาม churn
("md", """## churn ของ EWT คืออะไรกันแน่

คำว่า "เด็กหลุดคอร์ส" ฟังดูเข้าใจตรงกัน แต่พอต้องเขียนเป็นโค้ด ทุกคำต้องเป๊ะ
นี่คือนิยามที่เราตกลงกัน (ตรงกับ `docs/data-dictionary.md` — จำชุดนี้ให้ขึ้นใจครับ):

| กติกา | ความหมาย |
|---|---|
| หน่วยของโจทย์ | 1 แถว = **(นักเรียน, เดือนที่ active)** · active = enrolled ตอนต้นเดือน |
| `churned_next_month = 1` | ยกเลิก **ทุกวิชา** มีผลสิ้นเดือน t → เดือน t+1 น้องหายไป |
| `churned_next_month = 0` | เดือน t+1 ยังอยู่กับเรา |
| `subject_drop` (ลดบางวิชา) | **ไม่ใช่ churn!** — เก็บไว้เป็นสัญญาณเตือน (feature) ในบทหลัง |
| เดือน ก.ย. | จบซีซันปกติ — ออกตอน ก.ย. ไม่นับ churn → label = **NaN** |
| เดือนที่ยังไม่รู้อนาคต | label = **NaN** (censored — เดี๋ยวเจอในข้อ 1.3) |

**ทำไมต้องราย "เดือน"?** เพราะธุรกิจเราคือ subscription รายเดือน —
จุดตัดสินใจของผู้ปกครองคือรอบจ่ายรายเดือน และ mentor ก็ทำงานเป็นรอบเดือนพอดี
โมเดลที่ตอบว่า "สิ้นเดือนนี้ น้องคนนี้จะหายไหม" จึง match กับจังหวะที่เราลงมือช่วยได้จริง"""),

("md", """## ทำไมต้องสร้าง label เอง — เพราะ Supabase "ทับ" ประวัติ

ข่าวร้ายที่ทำให้บทนี้มีอยู่: Eduwise/Supabase เก็บสถานะ **ปัจจุบัน** ของน้องเท่านั้น
พอมีการยกเลิก ระบบอัปเดตทับค่าเดิม → เราเปิด database วันนี้จะเห็นว่า "ใครยังอยู่"
แต่**ไม่เห็นว่าใครเคยออกไป เมื่อไหร่** ประวัติการยกเลิกจริงๆ กระจายอยู่ใน
แชท LINE ผู้ปกครอง, ชีตการเงิน (เดือนไหนหยุดจ่าย), และความจำของทีม mentor

แปลว่าคนเดียวที่รวบรวม label ได้... คือคุณครับ 😄
โปรเจกต์เตรียม template ไว้ให้แล้ว — มาดูหน้าตากัน"""),

("code", """# template สำหรับกรอกบันทึกยกเลิก — อยู่ที่ data/raw/labels/
from src.config import LABELS_DIR

template = pd.read_csv(LABELS_DIR / "label_sheet_template.csv")
template"""),

("md", """แถวตัวอย่าง 2 แถวในไฟล์โชว์ 2 เคสหลัก: ยกเลิกหมด (full → churn)
กับลดวิชา (partial → subject_drop ไม่ใช่ churn) ความหมายแต่ละคอลัมน์:

| คอลัมน์ | กรอกอะไร |
|---|---|
| `student_code`, `display_name` | รหัส + ชื่อน้อง (ปี 68 ใช้รหัสจากชีตเดิม) |
| `year` | 2568 หรือ 2569 |
| `subjects` | วิชาที่ยกเลิก/ลด คั่นด้วย `;` |
| `cancel_date` | วันที่แจ้ง (ถ้ารู้) |
| `cancel_month` | **เดือนสุดท้ายที่ยัง active** — คอลัมน์สำคัญสุดในไฟล์นี้ |
| `partial_or_full` | `full` = ยกเลิกหมด (→ churn) · `partial` = ลดบางวิชา (→ subject_drop) |
| `source` | `line` / `sheet` / `memory` — บันทึกไว้ว่าไปขุดมาจากไหน |
| `confidence` | `high` / `medium` / `low` — ความมั่นใจในบันทึกแถวนั้น |
| `note` | บริบท เช่น เหตุผลที่ออก |

> 📝 **การบ้านจริง — เริ่มวันนี้เลย (นี่คือคอขวดของทั้งโปรเจกต์)**
>
> **เริ่ม consolidate บันทึกยกเลิกปี 68 + 69 ลงชีตตาม template นี้ตั้งแต่วันนี้ครับ**
>
> - ไล่จาก 3 แหล่งตามลำดับ: แชท LINE OA/ผู้ปกครอง → ชีตการเงิน (เดือนไหนหยุดจ่าย) → ความจำทีม mentor
> - 1 แถว = 1 เหตุการณ์ (ยกเลิกหมด หรือ ลดวิชา) พร้อม `source` + `confidence` ทุกแถว
> - ทำไมด่วน: บทเรียนทุกบทซ้อมกับข้อมูลจำลอง (sample) ไปพลางได้
>   แต่**โมเดลจริงเกิดไม่ได้จนกว่า label จริงจะมี** — และงานนี้ไม่มีใครทำแทนได้
>   เพราะประวัติอยู่ในแชทกับความจำของทีมเราเอง

> ⚠️ **กับดัก label noise!** label ของเราไม่ได้มาจากระบบที่เป๊ะ — มาจากบันทึกมือ
> บางแถวจะจำเดือนคลาดเคลื่อน บางแถวหาหลักฐานไม่เจอ **อย่าทิ้งแถวที่ไม่ชัวร์
> แต่ก็อย่าเนียนว่าชัวร์** — กรอก `confidence` ตามจริง แล้วบท 06 เราจะทำ
> sensitivity analysis: ตัด label ที่ confidence ต่ำออกแล้วดูว่าโมเดลเปลี่ยนไหม"""),

# ----------------------------------------------------------------- toy data
("md", """## สนามซ้อม: toy data 6 คน

ก่อนไปแตะข้อมูลจริง 310 คน มาซ้อมกับ "รุ่นจิ๋ว" 6 คนที่เห็นทะลุด้วยตาเปล่าก่อน
(นิสัยที่ดีมากเวลาเขียน logic ใหม่ — ถ้า 6 คนยังทำไม่ถูก 310 คนก็ไม่มีทางถูกครับ)

สถานการณ์ซีซัน 2569 (มี.ค.–ก.ย. 2026) — วันนี้คือ 3 ส.ค. 2026:

| น้อง | เรื่องราว |
|---|---|
| A | สมัคร มี.ค. อยู่ยาวจนจบซีซัน |
| B | สมัคร มี.ค. **ยกเลิกทุกวิชา มีผลสิ้นเดือน พ.ค.** |
| C | สมัครช้า (มิ.ย.) แล้วยกเลิกสิ้นเดือน ส.ค. |
| D | **ลดวิชา** (subject_drop) เดือน มิ.ย. แต่ไม่ได้ยกเลิก |
| E | เรียนจนจบ แล้วออกตอนสิ้นซีซัน (ก.ย.) |
| F | สมัครช้า 1 เดือน (เม.ย.) อยู่ยาว |"""),

("code", """# toy_events — เหตุการณ์สมัคร/ยกเลิกของน้อง 6 คน (schema ย่อจาก enrollment_events จริง)
_toy_rows = [
    # (student, event_type,   event_date,   event_month, source,        confidence)
    ("A", "enroll",       "2026-03-02", "2026-03", "label_sheet", "high"),
    ("B", "enroll",       "2026-03-05", "2026-03", "label_sheet", "high"),
    ("B", "cancel",       "2026-05-28", "2026-05", "line",        "high"),
    ("C", "enroll",       "2026-06-01", "2026-06", "label_sheet", "high"),
    ("C", "cancel",       "2026-08-30", "2026-08", "memory",      "medium"),
    ("D", "enroll",       "2026-03-01", "2026-03", "label_sheet", "high"),
    ("D", "subject_drop", "2026-06-15", "2026-06", "label_sheet", "high"),
    ("E", "enroll",       "2026-03-03", "2026-03", "label_sheet", "high"),
    ("E", "cancel",       "2026-09-28", "2026-09", "label_sheet", "high"),
    ("F", "enroll",       "2026-04-07", "2026-04", "label_sheet", "high"),
]
toy_events = pd.DataFrame(
    _toy_rows,
    columns=["student_key", "event_type", "event_date", "event_month", "source", "confidence"],
)
toy_events.insert(1, "year", 2569)
toy_events["event_date"] = pd.to_datetime(toy_events["event_date"])
toy_events"""),

# ================================================================= ex_01_01
("md", """## [แนวคิด] 1.1 จาก "เหตุการณ์" เป็น "เดือนที่ active"

โมเดลเราทำนายราย (นักเรียน, เดือน) → คำถามแรกจึงไม่ใช่ "ใคร churn"
แต่คือ **"เดือนไหนบ้างที่น้องคนนี้มีตัวตนให้ทำนาย"**

- `enrollment_events` เก็บเป็นเหตุการณ์: `enroll` = เดือนแรก ·
  `cancel` = **เดือนสุดท้ายที่ยัง active** (ออกมีผลสิ้นเดือน) ·
  `subject_drop` = ลดวิชา — ไม่เกี่ยวกับข้อนี้
- interval ของแต่ละคน = enroll → cancel (ไม่มี cancel → ยาวถึง ก.ย. จบซีซัน)
- ขยาย interval เป็นรายเดือนด้วย `month_seq(start, end)` (helper ของโปรเจกต์ **รวมหัวท้าย**)

ตัวอย่างข้างล่างทำกับกลุ่มทดลองเล็กๆ 2 คน (X, Y) — ดูโครงให้เข้าใจ
แล้วค่อยไปลงมือกับ toy 6 คนของเราครับ"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): กลุ่มทดลอง 2 คน — X ยกเลิกสิ้น เม.ย. / Y สมัคร พ.ค. อยู่ยาว
from src.config import month_seq

demo = pd.DataFrame([
    {"student_key": "X", "event_type": "enroll", "event_month": "2026-03"},
    {"student_key": "X", "event_type": "cancel", "event_month": "2026-04"},  # เดือนสุดท้าย = เม.ย.
    {"student_key": "Y", "event_type": "enroll", "event_month": "2026-05"},
])
SEASON_END = "2026-09"

rows = []
for sk, g in demo.groupby("student_key"):
    start = g.loc[g["event_type"] == "enroll", "event_month"].iloc[0]
    cancels = g.loc[g["event_type"] == "cancel", "event_month"]
    end = cancels.iloc[0] if len(cancels) else SEASON_END
    for m in month_seq(start, end):
        rows.append({"student_key": sk, "month": m})

demo_active = pd.DataFrame(rows)
print(demo_active)
print("\\nX active 2 เดือน (มี.ค.–เม.ย.) · Y active 5 เดือน (พ.ค.–ก.ย.)")"""),

("md", """### [แบบฝึกหัด 1.1] หาเดือนที่ active ของน้องแต่ละคน

จาก `toy_events` สร้างตาราง `active_months` ที่มี 1 แถวต่อ (นักเรียน, เดือนที่ active)

**คำสั่ง**

1. หา interval ของแต่ละคน: `start` = event_month ของแถว `enroll` ·
   `end` = event_month ของแถว `cancel` (ถ้าไม่มี cancel → `"2026-09"`)
2. ระวัง: `subject_drop` **ไม่ใช่** การยกเลิก — ห้ามเอามาปิด interval
3. ขยายเป็นรายเดือนด้วย `month_seq(start, end)` → DataFrame คอลัมน์ `student_key`, `month`

**ผลลัพธ์ที่คาด:** 33 แถว — น้อง B ได้ 3 เดือน (2026-03 → 2026-05) ·
น้อง D ได้ 7 เดือนเต็มซีซัน (subject_drop ไม่ตัด) · น้อง F ได้ 6 เดือน (เม.ย. → ก.ย.)"""),

("code", """____ = None  # TODO: เติมโค้ดแทน ____ ทีละจุด

SEASON_END = "2026-09"
rows = []
for sk, g in toy_events.groupby("student_key"):
    # TODO 1: เดือนแรกที่ active = event_month ของแถวที่ event_type == "enroll"
    start = ____
    # TODO 2: เดือนสุดท้าย = event_month ของแถว "cancel" (ไม่มี cancel → SEASON_END)
    #         ระวัง: subject_drop ไม่ใช่การยกเลิก!
    end = ____
    # TODO 3: วนทุกเดือนใน month_seq(start, end) แล้ว append {"student_key": sk, "month": m}
    # for m in ____:
    #     rows.append(____)

active_months = pd.DataFrame(rows, columns=["student_key", "month"])
print(f"ได้ {len(active_months)} แถว (เป้าหมาย: 33)")
checks.check("ex_01_01", active_months)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

ทำตามตัวอย่างกลุ่ม X, Y ได้เกือบตรงๆ เลยครับ — ต่างแค่ชื่อตาราง
- จุดเริ่ม: กรองเอาเฉพาะแถวประเภทสมัครเรียน แล้วดึงค่าเดือนของแถวแรกที่เจอ
- จุดจบ: กรองเอาเฉพาะแถวประเภทยกเลิกไว้ก่อน ถ้าผลการกรองว่างเปล่า (ไม่เคยยกเลิก)
  ค่อยถอยไปใช้เดือนจบซีซันแทน
- subject_drop ไม่ต้องเขียนอะไรจัดการเลย — การกรองที่เลือกเฉพาะ enroll/cancel
  ข้ามมันให้เองอยู่แล้ว

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- เลือกแถว+คอลัมน์พร้อมกัน: `.loc[mask, "event_month"]` · ดึงตัวแรก: `.iloc[0]`
- เช็คว่ามีแถว cancel ไหม: `len(...)` ใน conditional expression (`... if ... else ...`)
- ขยายเป็นรายเดือน: `month_seq(start, end)` (รวมหัวท้าย) · เก็บทีละแถว: `.append({...})`

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_01_01.py"""),

# ================================================================= ex_01_02
("md", """## [แนวคิด] 1.2 ติด label แบบ naive ก่อน

`churned_next_month` ตอบคำถามของ mentor ตรงๆ: **"สิ้นเดือนนี้ น้องจะหายไหม"**
— เห็นสัญญาณเดือน t → โทรหาก่อนสิ้นเดือน t นั่นคือเหตุผลที่ label มองไปข้างหน้า 1 เดือน

กติกาข้อมูลช่วยเราไว้แล้ว: แถว `cancel` บันทึก `event_month` = เดือนสุดท้ายที่ active
→ label 1 จึงตกที่ (คน, เดือน) ที่ `month == เดือน cancel` เป๊ะๆ · ที่เหลือเป็น 0

เวอร์ชันนี้ตั้งใจให้ "naive" — ยังไม่สนเรื่องจบซีซัน/อนาคต
เดี๋ยวข้อ 1.3 ค่อยแก้ (งาน ML ชีวิตจริงก็ iterate ทีละชั้นแบบนี้แหละครับ)"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): ติด label ให้กลุ่ม X, Y
cancel_month_demo = (demo[demo["event_type"] == "cancel"]
                     .set_index("student_key")["event_month"])   # X → "2026-04"

demo_labels = demo_active.copy()
demo_labels["churned_next_month"] = (
    demo_labels["month"] == demo_labels["student_key"].map(cancel_month_demo)
).astype(int)
print(demo_labels)
print("\\nX ได้ 1 ที่ 2026-04 (เดือนสุดท้ายก่อนหาย) · Y เป็น 0 ทุกเดือน (map แล้วได้ NaN → เทียบเป็น False)")"""),

("md", """### [แบบฝึกหัด 1.2] สร้าง churned_next_month (ฉบับ naive)

ต่อยอดจาก `active_months` ของข้อ 1.1

**คำสั่ง**

1. สร้าง Series `cancel_month`: map จาก `student_key` → เดือน cancel
   (จากแถว `event_type == "cancel"` ใน `toy_events`)
2. copy `active_months` เป็น `toy_labels_naive` แล้วเพิ่มคอลัมน์ `churned_next_month`:
   **1** เมื่อ `month` ตรงกับเดือน cancel ของคนนั้น, นอกนั้น **0** (dtype int)

**ผลลัพธ์ที่คาด:** 33 แถว · มีเลข 1 ทั้งหมด 3 จุด — B@2026-05, C@2026-08, E@2026-09
(ใช่ครับ น้อง E ที่จบซีซันปกติได้ 1 ไปก่อน — เดี๋ยวข้อถัดไปเราแก้ให้ถูกชีวิตจริง)"""),

("code", """____ = None  # TODO: เติมโค้ดแทน ____

# TODO 1: Series แปลง student_key → เดือน cancel (โครงเดียวกับตัวอย่างกลุ่ม X, Y)
cancel_month = ____

toy_labels_naive = active_months.copy()
# TODO 2: 1 เมื่อ month ตรงกับเดือน cancel ของคนนั้น, นอกนั้น 0 (แปลงเป็น int ด้วย)
toy_labels_naive["churned_next_month"] = ____

checks.check("ex_01_02", toy_labels_naive)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

สองจังหวะครับ — (1) ทำ "สมุดโทรศัพท์" แปลงรหัสนักเรียนเป็นเดือน cancel
โดยใช้เฉพาะแถวประเภทยกเลิก (2) map สมุดนี้ลงตาราง แล้วเทียบกับคอลัมน์เดือน
คนที่ไม่มี cancel จะ map ได้ค่าว่าง ซึ่งเทียบแล้วเป็นเท็จ → กลายเป็น 0 พอดี ไม่ต้องจัดการเพิ่ม

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- สมุดโทรศัพท์: กรองแถว cancel → `.set_index("student_key")` → เลือกคอลัมน์เดียว จะได้ Series
- ติด label: `.map(...)` → เทียบด้วย `==` → ปิดท้าย `.astype(int)`

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_01_02.py"""),

# ----------------------------------------------------------------- pitfall censoring
("md", """> ⚠️ **กับดัก censoring!** ตาราง naive ที่เพิ่งทำมีจุดโกหกอยู่ 2 แบบ —
>
> 1. **น้อง E ออกตอน ก.ย. = จบซีซันปกติ** (ปิดคอร์ส แยกย้ายไปสอบ) ถ้าปล่อยให้เป็น 1
>    โมเดลจะเรียนรู้ว่า "เรียนถึงปลายซีซัน = เสี่ยงหาย" ทั้งที่จริงคือเรียนจบอย่างสวยงาม
> 2. **เดือนที่ยังไม่รู้อนาคต** — วันนี้ 3 ส.ค. 69 บันทึกยกเลิกของเดือน ส.ค. ยังเก็บไม่ครบ
>    ถ้าให้ 0 กับทุกคนที่ "ยังไม่เห็นใบยกเลิก" = สอนโมเดลว่าทุกคนปลอดภัย ทั้งที่บางคน
>    กำลังจะกดยกเลิกสิ้นเดือนนี้ · และจะให้ 1 เฉพาะคนที่รู้แล้วก็ไม่ได้ — เดือนนั้นจะมีแต่ 1
>    ไม่มี 0 ที่เชื่อถือได้ (bias หนักกว่าเดิม) → ทางถูกคือ **censor ทั้งเดือน**
>
> ศัพท์ทางการเรียกว่า **censoring** = "สังเกตไม่ครบ" ไม่ใช่ "ไม่ churn" —
> คำตอบที่ซื่อสัตย์ที่สุดคือ NaN แล้วตัดแถวพวกนี้ทิ้งตอน train"""),

# ================================================================= ex_01_03
("md", """## [แนวคิด] 1.3 censor: กล้าตอบว่า "ไม่รู้"

เดือนที่ต้องเป็น NaN มี 2 กรณี (ตามกล่องกับดักด้านบน):

1. **ก.ย.** — จบซีซัน ไม่มี "เดือนถัดไป" ให้วัด
2. **เดือน > `known_through`** — เดือนล่าสุดที่บันทึกยกเลิก "ปิดยอดชัวร์" แล้ว
   วันนี้ 3 ส.ค. 69 → รู้ชัวร์ถึงสิ้น ก.ค. → `known_through = "2026-07"`

trick ที่ใช้บ่อยทั้งคอร์ส: เดือนรูปแบบ `"YYYY-MM"` **เปรียบเทียบเป็น string ได้เลย**
(`"2026-08" > "2026-07"` → True) — เรียงตามตัวอักษร = เรียงตามเวลา พอดีเป๊ะ

สังเกตด้วยว่าน้อง C ที่ยกเลิกสิ้นเดือน ส.ค. จะโดน censor ไปด้วย — ตั้งใจครับ
(เหตุผลอยู่ในกล่องกับดัก ข้อ 2)"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): censor กลุ่ม X, Y — สมมุติบันทึกปิดยอดชัวร์ถึงแค่ มิ.ย.
KNOWN_THROUGH_DEMO = "2026-06"

demo_censored = demo_labels.copy()
demo_censored["churned_next_month"] = demo_censored["churned_next_month"].astype("float64")
mask = (demo_censored["month"] == "2026-09") | (demo_censored["month"] > KNOWN_THROUGH_DEMO)
demo_censored.loc[mask, "churned_next_month"] = float("nan")
print(demo_censored)
print("\\nY เดือน ก.ค.–ก.ย. กลายเป็น NaN (เกิน มิ.ย. + จบซีซัน) · X ไม่โดนเพราะจบก่อน มิ.ย.")"""),

("md", """### [แบบฝึกหัด 1.3] mark เดือน censored เป็น NaN

แก้ตาราง naive ให้ตรงชีวิตจริง ด้วย `known_through = "2026-07"`

**คำสั่ง**

1. copy `toy_labels_naive` เป็น `toy_labels` แล้วแปลง `churned_next_month` เป็น float
   (ทำไว้ให้แล้วใน skeleton — int เก็บ NaN ไม่ได้)
2. สร้าง boolean mask `censored`: เดือนเป็น `"2026-09"` **หรือ** เดือน > `KNOWN_THROUGH`
3. ใช้ `.loc` ใส่ `float("nan")` ลงแถวที่โดน censor

**ผลลัพธ์ที่คาด:** NaN 9 แถว (ส.ค.+ก.ย. ของทุกคนที่ active ช่วงนั้น) ·
เลข 1 เหลือจุดเดียวคือ B@2026-05 · ที่เหลือเป็น 0 อีก 23 แถว"""),

("code", """____ = None  # TODO: เติมโค้ดแทน ____
KNOWN_THROUGH = "2026-07"   # วันนี้ 3 ส.ค. 69 → เดือนล่าสุดที่ปิดยอดชัวร์คือ ก.ค.

toy_labels = toy_labels_naive.copy()
toy_labels["churned_next_month"] = toy_labels["churned_next_month"].astype("float64")

# TODO 1: boolean mask เดือนที่ต้อง censor — เป็น ก.ย. ("2026-09") หรือ เกิน KNOWN_THROUGH
censored = ____
# TODO 2: ใช้ .loc ใส่ float("nan") ลงแถวที่โดน censor
# toy_labels.loc[____, "churned_next_month"] = ____

checks.check("ex_01_03", toy_labels)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

- เงื่อนไข 2 ข้อ (เป็นเดือน ก.ย. / เกินเดือนที่ปิดยอดชัวร์) เชื่อมกันด้วยตัว "หรือ"
  ของ pandas และ**ครอบวงเล็บแต่ละเงื่อนไข**เสมอ (ไม่งั้น pandas งอน)
- เดือนรูปแบบ YYYY-MM เทียบเป็น string ได้ตรงๆ ทั้งแบบเท่ากับและแบบมากกว่า
- ตอนเขียนทับเฉพาะบางแถว: ในวงเล็บเหลี่ยมใส่ mask ของแถวก่อน ตามด้วยชื่อคอลัมน์หลัง comma

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- ประกอบ mask: `(... == ...) | (... > ...)`
- เขียนทับเฉพาะแถวที่โดน: `.loc[mask, ...] = float("nan")`

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_01_03.py"""),

# ================================================================= ex_01_04
("md", """## [แนวคิด] 1.4 ส่งไม้ต่อให้ canonical function

logic ที่เราเพิ่งเขียนมือทั้งหมด ถูกห่อไว้ใน
`churn_utils.build_labels_monthly(events, year, known_through)` เรียบร้อยแล้ว
(บวกเคสที่ toy ไม่มี: `re_enroll` — น้องออกแล้วกลับมาสมัครใหม่กลางซีซัน)

กติกาของคอร์สนี้: แบบฝึกหัดให้เราเขียนเองเพื่อ **เข้าใจ** แต่บทถัดๆ ไปใช้
canonical function เสมอ — ต่อให้เราเขียนพลาด บทหลังก็ไม่พังตาม

- ปี **2568** ซีซันจบแล้ว → `known_through` ปล่อย default
- ปี **2569** → `known_through="2026-07"`

label ผิด = ทุกบทที่เหลือผิดหมด → ก่อนเชื่อผลลัพธ์ ต้อง sanity check เสมอครับ"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): เรียก canonical function กับ toy_events แล้วดูเป็น pivot
toy_canon = churn_utils.build_labels_monthly(toy_events, year=2569, known_through="2026-07")
toy_canon.pivot(index="student_key", columns="month", values="churned_next_month")
# อ่านแนวนอน: ว่าง = เดือนที่ไม่ active · 1.0 = เดือนสุดท้ายก่อนหาย · NaN = censored
# ถ้าข้อ 1.3 ทำถูก ตารางนี้จะตรงกับ toy_labels ของเราเป๊ะ"""),

("md", """### [แบบฝึกหัด 1.4] สร้าง labels จริงจาก sample ทั้ง 2 ปี

ได้เวลาข้อมูลจริง (sample): นักเรียน 310 คน — ปี 2568 จำนวน 160 คน + ปี 2569 จำนวน 150 คน

**คำสั่ง**

1. โหลด `enrollment_events.csv` จาก `DATA_DIR` (อย่าลืม `parse_dates=["event_date"]`)
2. สร้าง labels ปี 2568 (ซีซันจบแล้ว → ไม่ต้องส่ง `known_through`)
3. สร้าง labels ปี 2569 (`known_through="2026-07"`)
4. ต่อสองตารางเป็น `labels` ตารางเดียว

จากนั้น check จะ sanity ให้ 3 ข้อ: ก.ย. เป็น NaN หมด ·
churn rate รายเดือน (เฉพาะเดือนที่มี label) อยู่ช่วง 3–30% · ไม่มี (student, month) ซ้ำ

**ผลลัพธ์ที่คาด (sample):** 1,610 แถว 7 คอลัมน์ ตาม contract `labels_monthly`"""),

("code", """____ = None  # TODO: เติมโค้ดแทน ____

# TODO 1: โหลด enrollment_events.csv จาก DATA_DIR (parse_dates=["event_date"])
events = ____
# TODO 2: labels ปี 2568 — ซีซันจบแล้ว ไม่ต้องส่ง known_through
labels_2568 = ____
# TODO 3: labels ปี 2569 — บันทึกปิดยอดชัวร์ถึง ก.ค. 69
labels_2569 = ____
# TODO 4: ต่อสองตารางเป็นตารางเดียว (แถวต่อแถว, index ไม่ซ้ำ)
labels = ____

checks.check("ex_01_04", labels)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

4 บรรทัด 4 ท่าครับ: อ่านไฟล์ → เรียก canonical function สองครั้ง
(ต่างกันแค่ปี กับการส่ง/ไม่ส่งจุดปิดยอด) → เอาสองตารางมาต่อกันแนวแถว
ส่วน DATA_DIR เป็น Path อยู่แล้ว ต่อชื่อไฟล์ด้วยเครื่องหมายหารได้เลย

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `pd.read_csv(..., parse_dates=["event_date"])`
- `churn_utils.build_labels_monthly(..., year=...)` — เฉพาะปี 2569 เพิ่ม `known_through="2026-07"`
- `pd.concat([...], ignore_index=True)`

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_01_04.py"""),

# ================================================================= ex_01_05
("md", """## [แนวคิด] 1.5 base rate: เลขแรกที่ต้องจำก่อนสร้างโมเดล

churn rate ต่อเดือน = **base rate**: ถ้าไม่มีโมเดลเลยแล้วสุ่มชี้ "คนนี้จะหาย"
จะถูกประมาณเท่านี้ — ทุกโมเดลในบทหลังต้องพิสูจน์ว่าชนะเลขนี้ให้ได้

- mean ของคอลัมน์ 0/1 = สัดส่วนที่เป็น 1 → `mean()` รายเดือนคือ churn rate พอดี
- pandas ข้าม NaN ให้อัตโนมัติ → เดือน censored กลายเป็น NaN ไปเอง ไม่ปนเปื้อนค่าเฉลี่ย
- เดือน peak = เดือนที่ mentor ควรเทแรงมากสุด — insight เชิงธุรกิจฟรีๆ ก่อนมีโมเดลด้วยซ้ำ"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): churn rate ต่อเดือนของกลุ่ม toy 6 คน (จาก toy_canon)
toy_rate = toy_canon.groupby("month")["churned_next_month"].mean()
print((toy_rate * 100).round(1))
print("\\nพ.ค. = 20% (น้อง B หาย 1 จาก 5 คนที่ active) · ส.ค./ก.ย. = NaN (censored ทั้งเดือน)")"""),

("md", """### [แบบฝึกหัด 1.5] churn rate ต่อเดือนของข้อมูลจริง

**คำสั่ง**

1. จาก `labels` (ข้อ 1.4) คำนวณ churn rate ต่อเดือน:
   groupby เดือน แล้วเฉลี่ยคอลัมน์ `churned_next_month` → เก็บเป็น Series `churn_by_month`
2. ดูผลแล้วตอบตัวเองให้ได้: เดือนไหน peak? สองปี pattern เหมือนกันไหม?

**ผลลัพธ์ที่คาด:** Series 14 เดือน (2025-03…09 และ 2026-03…09) —
เดือน censored เป็น NaN · เดือนที่มี label ควรอยู่ช่วงราวๆ 3–15%"""),

("code", """____ = None  # TODO: เติมโค้ดแทน ____

# TODO: churn rate ต่อเดือน = ค่าเฉลี่ยของ churned_next_month แยกตามเดือน (จาก labels ข้อ 1.4)
churn_by_month = ____

if churn_by_month is not None:
    print((churn_by_month * 100).round(1))
checks.check("ex_01_05", churn_by_month)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

จัดกลุ่มตามคอลัมน์เดือน เลือกคอลัมน์ label มาคอลัมน์เดียว แล้วหาค่าเฉลี่ย — จบครับ
(อย่าเพิ่งคูณ 100 — check ต้องการสัดส่วน 0–1)

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.groupby("month")` → เลือกคอลัมน์เดียวด้วย `[...]` → ปิดท้าย `.mean()`

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_01_05.py"""),

# ----------------------------------------------------------------- ภาพรวม + ตีความ
("code", """# ภาพรวมเป็นกราฟ — cell นี้คำนวณจาก canonical function ตรงๆ (รันได้เสมอ ไม่พึ่งคำตอบข้อไหน)
_ev = pd.read_csv(DATA_DIR / "enrollment_events.csv", parse_dates=["event_date"])
_lab = pd.concat([
    churn_utils.build_labels_monthly(_ev, year=2568),
    churn_utils.build_labels_monthly(_ev, year=2569, known_through="2026-07"),
], ignore_index=True)
_rate = _lab.groupby("month")["churned_next_month"].mean().dropna() * 100

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(_rate.index, _rate.values, color="#2a78d6", width=0.62)
_peak = _rate.idxmax()
for b, (m, v) in zip(bars, _rate.items()):
    if m == _peak:  # direct label เฉพาะจุด peak — จุดที่อยากให้สายตาไปหยุด
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Churn rate ต่อเดือน สองซีซัน (เดือน censored ไม่ปรากฏ)", loc="left")
ax.set_ylabel("% ของนักเรียน active ที่หายเดือนถัดไป")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()"""),

("md", """### อ่านกราฟแบบเจ้าของธุรกิจ

(ย้ำอีกครั้ง: นี่คือ sample — ข้อมูลจำลองที่สร้างเลียนแบบ pattern ของเรา
เลขจริงจะได้เห็นหลังการบ้าน label sheet เสร็จครับ)

- **ปี 68 peak ที่ ก.ค. (~14%)** แล้วยังสูงต่อ ส.ค. — กลางซีซันคือช่วงครอบครัว
  ประเมินผล "คุ้มไหม ไปต่อไหม" ก่อนตัดสินใจลงคอร์สโค้งสุดท้าย
- **ปี 69 เดือน มี.ค. ก็สูง (~12%)** — เดือนแรกมีธรรมชาติแบบ "ทดลองเรียน"
  สมัครแล้วถอยเร็วถ้าไม่ใช่
- นัยเชิงปฏิบัติ: mentor ควรเทแรงที่ **เดือนแรกของน้องใหม่ + ช่วง มิ.ย.–ก.ค.**
  — และนี่คือเดือนที่โมเดลของเราต้องแม่นที่สุดด้วย
- ค่าเฉลี่ยรวม ~9% ต่อเดือน แปลว่า class 1 มีน้อยกว่า class 0 ราว 10 เท่า —
  โจทย์ **imbalanced** ซึ่งกระทบทั้งวิธี train และวิธีวัดผล (เจอกันบท 05)"""),

# ----------------------------------------------------------------- สรุป
("md", """## สรุปสิ่งที่ได้จากบทนี้

- frame ปัญหา: 1 แถว = (นักเรียน, เดือนที่ active) · `churned_next_month` ∈ {0, 1, NaN}
- **censoring**: ก.ย. (จบซีซัน) + เดือนที่ยังไม่ปิดยอด = NaN — ไม่ใช่ 0 และไม่ใช่ 1
- **label noise**: label มาจากบันทึกมือ → เก็บ `source` + `confidence` ติดไว้เสมอ
  (บท 06 ใช้ทำ sensitivity analysis)
- สร้าง `labels_monthly` ครบ 2 ปีด้วย `churn_utils.build_labels_monthly`
  — sample ได้ 1,610 แถว churn rate รวม ~8.8% peak กลางซีซัน
- 📝 **การบ้านที่สำคัญกว่าทุกแบบฝึกหัด**: เริ่มกรอก label sheet ปี 68+69 วันนี้

**บทถัดไป → บท 02: audit + extract ข้อมูลจริง** — เอาชีตปี 68 กับ Supabase ปี 69
มาเข้า schema เดียวกัน แล้ว label sheet ที่คุณกำลังกรอก จะกลายเป็นวัตถุดิบหลัก
ของเส้นทางทั้งหมดครับ 💪"""),

]
