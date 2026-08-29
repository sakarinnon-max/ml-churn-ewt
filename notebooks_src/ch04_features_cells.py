TITLE = "04 — Feature Engineering: ตาราง (นักเรียน, เดือน)"

CELLS = [

# ============================================================ หัวบท
("md", """# 04 — Feature Engineering: ตาราง (นักเรียน, เดือน)

บท 03 เราใช้ตาส่อง EDA จนเห็นแล้วว่าสัญญาณไหนน่าจะเกี่ยวกับ churn —
บทนี้ถึงเวลาแปลงสัญญาณเหล่านั้นเป็นตัวเลขจริง ถึงบทหัวใจของคอร์สแล้วครับ 💚
โมเดลจะเก่งแค่ไหนไม่ได้อยู่ที่อัลกอริทึม
แต่อยู่ที่ **ตาราง features** ที่เราสร้างในบทนี้ — และบทนี้มีด่านพิเศษ:
คุณจะโดน **leakage** หลอกต่อหน้าต่อตา 1 ครั้ง (แบบปลอดภัย) แล้วจำมันไปทั้งชีวิต

**สิ่งที่จะได้จากบทนี้ (~90 นาที):**
1. เข้าใจ panel data — 1 แถว = 1 (นักเรียน active, เดือน) พร้อมคำตอบ `churned_next_month`
2. กฎเหล็ก **"ยืนอยู่ที่สิ้นเดือน t แล้วมองย้อนหลังเท่านั้น"** + เครื่องมือ `churn_utils.cutoff`
3. สร้าง features ครบ 4 กลุ่ม: การเข้าเรียน / ข้อสอบ / engagement / ข้อมูลนิ่ง (static)
4. จับ leakage ได้ด้วยจมูกตัวเอง — feature "เทพเกินจริง" ที่แอบมาจากอนาคต
5. ไฟล์ `features_monthly.csv` ที่ผ่าน contract พร้อมเทรนโมเดลในบท 05"""),

("code", """import sys; sys.path.insert(0, "..")
import pandas as pd
from src import checks, churn_utils, contracts
from src.config import DATA_DIR, IS_SAMPLE
plt = churn_utils.plot_style()
print("โหมดข้อมูล:", "SAMPLE (ข้อมูลจำลอง)" if IS_SAMPLE else f"REAL ({DATA_DIR})")"""),

("code", """import numpy as np
from src.config import SEASONS, month_seq

# โหลดวัตถุดิบทั้ง 5 ตาราง (schema ตาม docs/data-dictionary.md)
labels = pd.read_csv(DATA_DIR / "labels_monthly.csv", dtype={"month": str}, parse_dates=["churn_date"])
att_long = pd.read_csv(DATA_DIR / "attendance_long.csv", parse_dates=["ep_final_date", "week_start"])
exam_attempts = pd.read_csv(DATA_DIR / "exam_attempts.csv", parse_dates=["submitted_at"])
weekly = pd.read_csv(DATA_DIR / "weekly_metrics.csv", parse_dates=["week_start", "week_end"])
students = pd.read_csv(DATA_DIR / "students.csv", parse_dates=["signup_date"])

for name, df in [("labels_monthly", labels), ("attendance_long", att_long),
                 ("exam_attempts", exam_attempts), ("weekly_metrics", weekly),
                 ("students", students)]:
    print(f"{name:16s} {df.shape[0]:6,d} แถว × {df.shape[1]:2d} คอลัมน์")
labels.head(3)"""),

("md", """## Panel data — โต๊ะทำงานของ mentor ในรูปตาราง

ลองนึกภาพจริงก่อนครับ: **ต้นเดือน ส.ค. mentor ต้องตัดสินใจว่าจะโทรหาใคร**
ข้อมูลที่มีจริง ณ วันนั้นคือทุกอย่าง *จนถึง 31 ก.ค.* เท่านั้น — คำถามที่ mentor
ถามในใจคือ *"เด็กคนนี้ ณ สิ้นเดือนนี้ เสี่ยงหายเดือนหน้าแค่ไหน?"*

ตาราง **features_monthly** คือคำถามนั้นในรูปตาราง:
- 1 แถว = 1 คู่ **(นักเรียนที่ active, เดือน t)** — sample เรามี 310 คน รวม 1,610 แถว
- คอลัมน์ features = สิ่งที่ "รู้แล้ว ณ สิ้นเดือน t" · คอลัมน์คำตอบ = `churned_next_month`
- แถวที่ label เป็น NaN (censored — เช่น ก.ย. = จบซีซันปกติ) **ห้ามทิ้ง!**
  ใช้เทรนไม่ได้ก็จริง แต่เดือนล่าสุดคือแถวที่เราต้องส่งลิสต์ให้ mentor ทำนายจริง

**กฎเหล็กของบทนี้:** ยืนอยู่ที่ *สิ้นเดือน t* แล้วมองย้อนหลังเท่านั้น —
อะไรที่เกิดหลังเที่ยงคืนคืนสุดท้ายของเดือน t **ไม่มีสิทธิ์** โผล่ในแถวของเดือน t"""),

("code", """# เครื่องมือประจำกายของบทนี้: month_end_cutoff + cutoff
print("month_end_cutoff('2025-06') =", churn_utils.month_end_cutoff("2025-06"))
print("→ ขอบบน (ไม่รวม) ของ 'ข้อมูลจนถึงสิ้นเดือน มิ.ย.' คือ 1 ก.ค. 00:00 พอดี")

# ตัวอย่าง: น้องต้นน้ำ (2568-1007) — ม.3 บดินทรเดชา เป้า TU เราจะใช้เคสนี้เดินเรื่องทั้งบท
sk = "2568-1007"
my_attempts = exam_attempts[exam_attempts["student_key"] == sk]
seen_june = churn_utils.cutoff(my_attempts, "2025-06", "submitted_at")
print(f"attempts ของน้องต้นน้ำทั้งซีซัน: {len(my_attempts)} ครั้ง")
print(f"แต่ยืนที่สิ้นเดือน มิ.ย. จะเห็นแค่: {len(seen_june)} ครั้ง  ← โมเดลก็ต้องเห็นเท่านี้!")"""),

("md", """> ⚠️ **กับดัก! Temporal leakage — ตัวการอันดับ 1 ที่ทำให้โมเดล "เก่งปลอม"**
> ถ้า feature ของเดือน t แอบใช้ข้อมูลหลังสิ้นเดือน t แม้แต่นิดเดียว โมเดลจะดูเทพมาก
> ตอน train/test (เพราะแอบเห็นเฉลย) แต่พอใช้งานจริงต้นเดือน ส.ค. — ข้อมูลอนาคต
> ไม่มีอยู่จริง → โมเดลใบ้กิน mentor โทรผิดคน เสียทั้งเวลาทั้งความเชื่อใจของผู้ปกครอง
> ทางกันพลาด: **ทุกครั้งที่หยิบตารางมาคำนวณ ให้ผ่าน `churn_utils.cutoff(df, month, date_col)` ก่อนเสมอ**
> เดี๋ยวท้ายบทเราจะโดนของจริงกัน 1 ดอก (แบบมีเข็มขัดนิรภัย)"""),

# ============================================================ Unit 1 — ex_04_01
("md", """## [แนวคิด] 4.1 โครง panel — เริ่มจากกระดูกสันหลังก่อน

`labels_monthly` จากบท 01 คือกระดูกสันหลังของ panel อยู่แล้ว: มันบอกว่า
*ใคร active เดือนไหน* พร้อมคำตอบ `churned_next_month` ต่อแถว
งานแรกเลยง่ายมาก — กรองเอาเฉพาะแถว active แล้วเก็บคอลัมน์ที่จำเป็น

แถมอีกหนึ่ง feature ฟรีๆ: **month_index** (มี.ค. = 1 ... ก.ย. = 7)
ทำไมต้องมี? เพราะ churn ของ EWT มีฤดูกาลชัดมาก — sample ของเรา churn
พีคช่วงเดือนที่ 5–6 ของซีซัน (ก.ค.–ส.ค. ≈ 13.4% / 11.5%) เทียบกับต้นซีซัน ~6–9%
ช่วงนั้นแหละที่เด็กเริ่มท้อ ผู้ปกครองเริ่มทบทวนค่าใช้จ่ายรายเดือน — โมเดลควรได้รู้ว่า
"ตอนนี้อยู่ช่วงไหนของซีซัน" · การนับแยกตามปี (2568 เริ่ม 2025-03, 2569 เริ่ม 2026-03)
ทำให้สองปีเทียบกันได้แถวต่อแถว"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): ตาราง labels จิ๋วที่แต่งขึ้นเอง 2 คน
toy = pd.DataFrame({
    "student_key": ["A", "A", "A", "B", "B"],
    "year": [2568] * 5,
    "month": ["2025-03", "2025-04", "2025-05", "2025-04", "2025-05"],
    "active": [1, 1, 1, 1, 0],                       # B เดือน พ.ค. ไม่ active
    "churned_next_month": [0.0, 0.0, np.nan, 1.0, np.nan],
})

mini = toy[toy["active"] == 1].copy()                # เก็บเฉพาะเดือนที่ active
mini = mini[["student_key", "year", "month", "churned_next_month"]]

# month_index: สร้าง dict {เดือน -> ลำดับ} จาก month_seq แล้ว map
idx_of = {m: i for i, m in enumerate(month_seq("2025-03", "2025-09"), start=1)}
mini["month_index"] = mini["month"].map(idx_of)
mini    # 4 แถว — แถว B พ.ค. หายไป และ NaN (censored) ยังอยู่ครบ"""),

("md", """### [แบบฝึกหัด 4.1] โครง panel จาก labels_monthly + month_index

ทำแบบเดียวกับตัวอย่าง แต่ใช้ **ของจริง**: ตาราง `labels` ที่โหลดไว้แล้ว

**คำสั่ง:**
1. กรอง `labels` เอาเฉพาะแถวที่ `active == 1` (ใช้ `.copy()` กันเสียงบ่นจาก pandas)
2. เก็บ 4 คอลัมน์: `student_key`, `year`, `month`, `churned_next_month`
3. เพิ่มคอลัมน์ `month_index` — เดือนที่เท่าไรของซีซัน **แยกตามปี**:
   ปี 2568 เริ่ม 2025-03, ปี 2569 เริ่ม 2026-03 (ใช้ `SEASONS` + `month_seq` ที่ import ไว้แล้ว)
4. เก็บผลใน `panel` — **ห้าม** ทิ้งแถวที่ label เป็น NaN

**ผลลัพธ์ที่คาด:** `(1610, 5)` — แถวแรกคือน้อง 2568-1001 เดือน 2025-06
(สมัครกลางซีซัน) ดังนั้น `month_index` = 4"""),

("code", """# TODO: สร้าง panel ตามคำสั่ง 1-4 ข้างบน
#   เคล็ด: สร้าง dict {(year, month) -> ลำดับ} จากทั้ง 2 ปีก่อน แล้วค่อยเติมคอลัมน์
____ = None   # ← ลบบรรทัดนี้ แล้วเขียนโค้ดของคุณแทน

panel = ____

checks.check("ex_04_01", panel)   # ยังไม่ผ่านจนกว่าจะเติมถูก — ปกติ!"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

ต่างจากตัวอย่างตรงที่มี **2 ปี** — mapping ต้องใช้ key เป็นคู่ (ปี, เดือน)
ไม่ใช่เดือนเดี่ยวๆ: วนลูปซีซันทีละปี ข้างในวนเดือนของซีซันนั้นพร้อมเลขลำดับ
ที่เริ่มนับจากหนึ่ง จากนั้นเติมคอลัมน์ด้วยการไล่จับคู่ปีกับเดือนของแต่ละแถว
ไปเปิดค่าจาก mapping ที่สร้างไว้

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- กรองด้วย boolean mask + `.copy()` · เลือกคอลัมน์ด้วย `[["...", "..."]]`
- `SEASONS[y]["start_month"]` / `SEASONS[y]["end_month"]` · `month_seq(start, end)`
- `enumerate(..., start=1)` · dict comprehension `{(y, m): i for ...}`
- `zip(...)` คู่กับ list comprehension ตอนเติมคอลัมน์

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_04_01.py"""),

("md", """---
**จุด checkpoint (ตาข่ายกันตก)** — กติกาของคอร์สเรา: cell ถัดไปสร้างตัวแปรเวอร์ชัน
**canonical** จาก `src/churn_utils` (เฉลยกลางที่ผ่านการตรวจแล้ว) ไว้ใช้เดินเรื่องต่อ
ข้อไหนยังไม่ผ่านก็เรียนข้อถัดไปได้ ไม่มีอะไรพังต่อกันเป็นโดมิโน — และบทถัดๆ ไป
ก็ใช้ canonical เสมอ ไม่ใช้ตัวแปรในสมุดเรา (ผิดพลาดในแบบฝึกหัดจะได้ไม่ลามไปบทหน้า)

ป.ล. `feats_canonical` มีคอลัมน์เฉลยของข้อถัดๆ ไปครบเลยนะครับ...
อย่าเพิ่งแอบเปิดดูล่ะ 😄 เดี๋ยวไม่ได้ฝึก"""),

("code", """# canonical = เฉลยกลางของบทนี้ทั้งบท (จะเป็นกรรมการเทียบคำตอบใน 4.5 ด้วย)
feats_canonical = churn_utils.build_features_monthly(labels, att_long, exam_attempts, weekly, students)
panel_base = feats_canonical[["student_key", "year", "month", "month_index", "churned_next_month"]].copy()
print("panel_base:", panel_base.shape, "| feats_canonical:", feats_canonical.shape)"""),

# ============================================================ Unit 2 — ex_04_02
("md", """## [แนวคิด] 4.2 Attendance features — สัญญาณที่ mentor เห็นก่อนใคร

การเข้าเรียนคือชีพจรของเด็ก เราจะกลั่นเป็น 3 ตัวเลขต่อ (คน, เดือน):

| feature | ความหมาย | ภาษา mentor |
|---|---|---|
| `att_month_pct` | % เข้าเรียนเฉพาะเดือน t | "เดือนนี้มาเรียนแค่ไหน" |
| `att_cum_pct` | % เข้าเรียนสะสมตั้งแต่ต้นซีซันถึงสิ้นเดือน t | "โดยนิสัยเป็นเด็กแบบไหน" |
| `att_delta` | เดือนนี้ − เดือนก่อน | "กำลังดิ่งหรือกำลังฟื้น" |

ใน sample ของเรา เดือนที่ `att_delta` ร่วงแรง (ต่ำกว่า −20 จุด) churn ~11.8%
สูงกว่าเดือนที่ลงนิดหน่อย (~7.1%) ราว 1.7 เท่า — delta คือสัญญาณ "เพิ่งเกิดอะไรขึ้น"
ที่ค่าสะสมมองไม่เห็น · เดือนของแต่ละคาบเรียนนับจาก `ep_final_date`
(วันสอนรอบสุดท้ายของ EP — จำกับดัก "1 EP สอน 2 รอบ" จากบท 02 ได้ใช่ไหมครับ)"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): คิดผ่านน้องต้นน้ำ *ทีละเดือน* ก่อน — ยังไม่ต้อง groupby ทั้งตาราง
att = att_long.copy()
att["month"] = att["ep_final_date"].dt.strftime("%Y-%m")   # เดือนของแต่ละ EP
mine = att[att["student_key"] == sk]

def present_pct(df):
    return 100.0 * (df["status"] == "present").mean() if len(df) else float("nan")

for month in ["2025-05", "2025-06", "2025-07"]:
    seen = churn_utils.cutoff(mine, month, "ep_final_date")     # ยืนที่สิ้นเดือน t ก่อนเสมอ!
    prev = str(pd.Period(month, freq="M") - 1)                  # "2025-06" -> "2025-05"
    cur_pct = present_pct(seen[seen["month"] == month])
    cum_pct = present_pct(seen)
    delta = cur_pct - present_pct(seen[seen["month"] == prev])
    print(f"{month}: เดือนนี้ {cur_pct:5.1f}% | สะสม {cum_pct:5.1f}% | delta {delta:+6.1f}")

print("\\nสิ้นเดือน ก.ค. เราเห็น delta -31.9 → ควรโทรด่วน... และน้องต้นน้ำ churn จริงสิ้นเดือนนั้น")

# ภาพเดียวจบ: เส้นทาง "เท่าที่เห็น ณ สิ้นเดือน ก.ค." — สังเกตว่าพล็อตก็ต้อง cutoff ก่อน!
# (ไม่งั้นจะแอบเห็น EP สัปดาห์คาบเกี่ยวที่ ep_final_date ตกไปต้นเดือน ส.ค. — ของจากอนาคต)
seen_july = churn_utils.cutoff(mine, "2025-07", "ep_final_date")
story = (seen_july.assign(present=seen_july["status"].eq("present"))
                  .groupby("month")["present"].mean().mul(100.0))
months_x = list(story.index)
colors = ["#9aa5b1"] * len(months_x)                   # เดือนปกติ = เทากลางๆ ไม่แย่งซีน
colors[months_x.index("2025-07")] = "#d03b3b"          # เดือนสุดท้ายก่อน churn = แดงเด่น

fig, ax = plt.subplots(figsize=(7, 3.5))
bars = ax.bar(months_x, story.values, color=colors, width=0.55)
ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=10)
ax.annotate("เดือนสุดท้ายก่อนหาย", xy=("2025-07", story["2025-07"]),
            xytext=(0, 30), textcoords="offset points", ha="center", color="#d03b3b")
ax.set_ylim(0, 108)
ax.set_ylabel("% เข้าเรียน")
ax.set_title("น้องต้นน้ำ (2568-1007): % เข้าเรียนรายเดือน (ข้อมูล ณ สิ้นเดือน ก.ค.)")
plt.show()"""),

("md", """### [แบบฝึกหัด 4.2] Attendance features ทั้ง panel ด้วย cutoff()

สเกลวิธีคิดจาก 1 คน → **ทุกคนทุกเดือน** โครงคือ: วนทีละเดือน → `cutoff` ก่อน →
ค่อยสรุปรายคนด้วย `groupby` → `.map()` กลับเข้า panel

**คำสั่ง:**
1. เตรียม `att` + คอลัมน์ `month` (โค้ดให้แล้วใน skeleton)
2. วนลูป `for (year, month), g in panel_base.groupby(["year", "month"]):`
3. ในลูป: `att_cut = churn_utils.cutoff(att ของปีนั้น, month, "ep_final_date")`
   แล้วคำนวณ % เข้าเรียนรายคน (Series ที่ index เป็น `student_key`) 3 ชุด:
   เฉพาะเดือน t / สะสมทั้ง `att_cut` / เฉพาะเดือนก่อนหน้า
4. เติม 3 คอลัมน์ลง `g.copy()` ด้วย `.map()`: `att_month_pct`, `att_cum_pct`,
   `att_delta` (= เดือนนี้ − เดือนก่อน; ถ้าเดือนใดเดือนหนึ่งไม่มีคลาส → NaN เอง)
5. `pd.concat(parts, ignore_index=True)` → เก็บใน `panel_att`

**ผลลัพธ์ที่คาด:** `(1610, 8)` — spot check: น้องต้นน้ำ เดือน 2025-07
ต้องได้ `att_month_pct` 55.6, `att_delta` −31.9 เหมือนตัวอย่างเป๊ะ"""),

("code", """att = att_long.copy()
att["month"] = att["ep_final_date"].dt.strftime("%Y-%m")   # ให้ฟรี 2 บรรทัดนี้

# TODO: วนทีละ (year, month) ของ panel_base → cutoff → % รายคน 3 ชุด → map ใส่ g
____ = None   # ← ลบบรรทัดนี้ แล้วเขียนโค้ดของคุณแทน

panel_att = ____

checks.check("ex_04_02", panel_att)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

ในลูปแต่ละรอบ "ยืนที่สิ้นเดือน t": cutoff **ก่อน** แล้วทุกอย่างคำนวณจากก้อนที่
cutoff แล้วเท่านั้น — ค่าเดือนนี้กรองเอาเฉพาะแถวของเดือน t, ค่าสะสมใช้ทั้งก้อน
ไม่ต้องกรองเพิ่ม, ค่าเดือนก่อนกรองเอาแถวของเดือนก่อนหน้า · สรุปรายคนแล้วจะได้
Series ที่ index เป็นรหัสนักเรียน เอาไปจับคู่ใส่คอลัมน์ของกลุ่มได้เลย
(คนที่ไม่มีคลาส → NaN อัตโนมัติ ซึ่งถูกแล้ว)

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `churn_utils.cutoff(..., "ep_final_date")`
- `.groupby("student_key")["status"].apply(...)` — ข้างในคิด % จาก `.mean()` ของเงื่อนไข `== "present"`
- เดือนก่อนหน้า: `str(pd.Period(..., freq="M") - 1)`
- `.map(...)` ทีละ Series แล้ว delta ก็แค่เอาสองคอลัมน์ลบกัน
- ปิดท้าย `pd.concat(..., ignore_index=True)`

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_04_02.py"""),

("code", """# checkpoint: เวอร์ชัน canonical ของ 4.2 ไว้ใช้ต่อ
panel_att_ok = feats_canonical[["student_key", "year", "month", "month_index",
                                "churned_next_month", "att_month_pct", "att_cum_pct",
                                "att_delta"]].copy()
print("panel_att_ok:", panel_att_ok.shape)"""),

# ============================================================ Unit 3 — ex_04_03
("md", """## [แนวคิด] 4.3 Exam features — ความขยันวัดเป็นตัวเลข

การบ้าน/ข้อสอบคือกิจกรรม "นอกเวลาบังคับ" ตัวแรกที่ถูกทิ้งเมื่อใจเริ่มถอด เราสกัด 2 ตัว:

- `new_attempts_month` — จำนวนชุดข้อสอบที่ **ส่งใหม่ในเดือน t** ("เดือนนี้ยังขยับอยู่ไหม")
  · ไม่มี attempt = **0** ไม่ใช่ NaN
- `exam_avg_score` — คะแนนเฉลี่ย **สะสม** ของทุก attempt ถึงสิ้นเดือน t ("พื้นฐานแน่นแค่ไหน")

**แต่อย่าเพิ่งเชื่อสมมติฐานตัวเอง** — ลองเทียบกับข้อมูลจริงใน sample ดู: เดือนที่เด็ก
ไม่ส่งข้อสอบเลย churn 23.5% เทียบ base 8.8% (แต่มีแค่ 17 เดือน-คน ตัวอย่างน้อย อย่าเพิ่งฟันธง)
· ส่วนเด็กคะแนนเฉลี่ยต่ำกว่า 60 churn แค่ 7.7% — **ต่ำกว่า base ด้วยซ้ำ!**
คะแนนน้อย ≠ กำลังจะหาย (เด็กที่ยังสู้ทั้งที่คะแนนไม่สวยมีเยอะ) นี่คือเรื่องปกติของงานนี้ครับ:
เราสร้าง feature จากสมมติฐานที่ *สมเหตุสมผล* แล้วปล่อยให้โมเดล (บท 05–06) และบท 07
เป็นคนตัดสินว่าตัวไหนมีของจริง — หน้าที่เราตอนนี้คือสร้างมันให้ **ถูกกติกาเวลา** ก่อน

> ⚠️ **กับดัก!** กรองช่วงเวลา attempt ด้วย `submitted_at` **เท่านั้น** —
> คอลัมน์ `created_at` ใน Supabase คือเวลา bulk sync (ขยะ!) ใช้แล้ว attempt
> ทั้งก้อนจะย้ายเดือนมั่ว นี่คือกับดักโดเมนข้อ 2 จาก data dictionary
> โชคดีที่ตาราง `exam_attempts` ของเราถูก extractor กรองมาให้ถูกแล้ว —
> แต่เวลาคุณไปดึงเองจาก Supabase ต้องจำให้ขึ้นใจ"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): น้องต้นน้ำ ณ สิ้นเดือน มิ.ย.
seen = churn_utils.cutoff(my_attempts, "2025-06", "submitted_at")
in_june = seen[seen["submitted_at"] >= pd.Timestamp("2025-06-01")]
print(f"ยืนที่สิ้นเดือน มิ.ย.: เห็น {len(seen)} attempts (จากทั้งซีซัน {len(my_attempts)})")
print(f"exam_avg_score  = {seen['percentage'].mean():.2f}  (เฉลี่ยสะสมถึงสิ้น มิ.ย.)")
print(f"new_attempts_month = {len(in_june)}  (ส่งใหม่เฉพาะเดือน มิ.ย.)")"""),

("md", """### [แบบฝึกหัด 4.3] Exam features ทั้ง panel

โครงลูปเดียวกับข้อ 4.2 เปลี่ยนวัตถุดิบเป็น `exam_attempts` + `submitted_at`

**คำสั่ง:**
1. วนลูป `(year, month)` ของ `panel_base` เหมือนเดิม
2. `atm_cut = churn_utils.cutoff(exam_attempts ของปีนั้น, month, "submitted_at")`
3. `exam_avg_score` = ค่าเฉลี่ย `percentage` รายคนจาก `atm_cut` ทั้งก้อน (สะสม)
4. `new_attempts_month` = จำนวนแถวของ `atm_cut` ที่ `submitted_at` ≥ วันแรกของเดือน t
   รายคน — คนไม่มี attempt ต้องเป็น **0** (int) ไม่ใช่ NaN
5. รวมเป็น `panel_exam`

**ผลลัพธ์ที่คาด:** `(1610, 7)` — spot check: น้องต้นน้ำ 2025-06 →
`exam_avg_score` 75.68, `new_attempts_month` 3"""),

("code", """# TODO: ลูปเดิม เปลี่ยนวัตถุดิบ — อย่าลืม fillna(0).astype(int) ให้ new_attempts_month
____ = None   # ← ลบบรรทัดนี้ แล้วเขียนโค้ดของคุณแทน

panel_exam = ____

checks.check("ex_04_03", panel_exam)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

สองสถิตินี้มาจากก้อนที่ cutoff แล้วก้อนเดียวกัน: ค่าเฉลี่ยใช้ทั้งก้อน (สะสม)
ส่วนตัวนับกรองเพิ่มว่าเวลาส่งอยู่ในเดือน t ก่อนค่อยนับขนาดกลุ่มรายคน ·
พอจับคู่กลับเข้า panel คนที่ไม่เจอจะเป็น NaN — คะแนนเฉลี่ยปล่อย NaN ไว้
(ถูกแล้ว: ไม่รู้ ≠ ศูนย์) แต่ตัวนับต้องถมศูนย์แล้วแปลงเป็นจำนวนเต็ม (ไม่ส่งเลยจริงๆ)

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `churn_utils.cutoff(..., "submitted_at")`
- `.groupby("student_key")["percentage"].mean()`
- วันแรกของเดือน: `pd.Timestamp(... + "-01")`
- ตัวนับ: กรอง `"submitted_at" >=` วันแรกของเดือน แล้ว `.groupby("student_key").size()`
- `.map(...).fillna(0).astype(int)` เฉพาะตัวนับ

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_04_03.py"""),

("code", """# checkpoint: เวอร์ชัน canonical ของ 4.3
panel_exam_ok = feats_canonical[["student_key", "year", "month", "month_index",
                                 "churned_next_month", "exam_avg_score",
                                 "new_attempts_month"]].copy()
print("panel_exam_ok:", panel_exam_ok.shape)"""),

# ============================================================ Unit 4 — ex_04_04
("md", """## [แนวคิด] 4.4 Engagement features — จาก weekly มาเป็น monthly

`weekly_metrics` คือหน้าปัดรายสัปดาห์ที่ทีมเราใช้อยู่แล้ว (silent, streak, tier...)
ปัญหาเดียว: panel เราเป็นรายเดือน — ต้องเลือกวิธี "ย่อ 4-5 สัปดาห์ให้เหลือ 1 แถว"
ซึ่งแต่ละ feature ย่อไม่เหมือนกัน:

- `max_silent_weeks` — ค่า **สูงสุด** ของ `silent_weeks` ในเดือน t: เงียบจุดไหน
  ของเดือนก็ต้องเห็น (ใน sample เดือนที่เด็ก silent ≥ 2 สัปดาห์ churn ~50%
  เทียบกับปกติ ~8.5% — สัญญาณแรงสุดในบทนี้!) · ไม่มีข้อมูล = 0
- `streak_weeks`, `practice_pct`, `checkpoint_pct` — **snapshot สัปดาห์สุดท้าย**
  ของเดือน: สามตัวนี้เป็นค่าสะสม/สถานะล่าสุดอยู่แล้ว เอาค่า ณ สิ้นเดือนพอ
  (เฉลี่ยทั้งเดือนจะเบลอสถานะจริง)

สัปดาห์นับเป็นของเดือนไหน ดูจากเดือนของ `week_start` (สัปดาห์คาบเกี่ยวเดือน
นับตามวันจันทร์ต้นสัปดาห์ — นิยามเดียวกับที่ mentor ใช้)"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): เดือน ก.ค. ของน้องต้นน้ำ มี 4 สัปดาห์
wm = weekly.copy()
wm["month"] = wm["week_start"].dt.strftime("%Y-%m")
july = wm[(wm["student_key"] == sk) & (wm["month"] == "2025-07")].sort_values("week_start")
display_cols = ["week_start", "att_week_pct", "silent_weeks", "streak_weeks",
                "practice_pct", "checkpoint_pct", "tier"]
print(july[display_cols].to_string(index=False))

last_week = july.iloc[-1]                       # แถวสัปดาห์สุดท้ายของเดือน
print("\\nmax_silent_weeks =", july["silent_weeks"].max(),
      "| streak_weeks =", last_week["streak_weeks"],
      "| practice_pct =", round(last_week["practice_pct"], 1),
      "| checkpoint_pct =", round(last_week["checkpoint_pct"], 1))"""),

("md", """### [แบบฝึกหัด 4.4] Engagement features ทั้ง panel

**คำสั่ง:**
1. เตรียม `wm` + คอลัมน์ `month` จาก `week_start` (ทำแล้วใน cell ตัวอย่าง ใช้ต่อได้เลย)
2. วนลูป `(year, month)` ของ `panel_base` — คราวนี้กรองตรงๆ:
   `wm_m = wm[(wm["year"] == year) & (wm["month"] == month)]` แล้ว **sort ตาม `week_start`**
3. `max_silent_weeks` = max ของ `silent_weeks` รายคน (ไม่มีข้อมูล → 0, int)
4. หาแถวสัปดาห์สุดท้ายรายคนจาก `wm_m` แล้วดึง `streak_weeks` (0 ถ้าไม่มี, int),
   `practice_pct`, `checkpoint_pct` (สองตัวนี้ปล่อย NaN ได้)
5. รวมเป็น `panel_eng`

**ผลลัพธ์ที่คาด:** `(1610, 9)` — spot check: น้องต้นน้ำ 2025-07 →
max_silent 0, streak 1, practice_pct 50.0"""),

("code", """# TODO: ลูปเดิม + เทคนิคหาแถวสุดท้ายรายคน (sort แล้ว drop_duplicates keep="last")
____ = None   # ← ลบบรรทัดนี้ แล้วเขียนโค้ดของคุณแทน

panel_eng = ____

checks.check("ex_04_04", panel_eng)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

สอง pattern ในลูปเดียว: (1) ค่าเงียบสุด = จัดกลุ่มรายคนแล้วเอาค่าสูงสุด
ตรงไปตรงมา (2) snapshot สัปดาห์สุดท้าย = เรียงตามวันเริ่มสัปดาห์ก่อน
แล้วตัดแถวซ้ำของแต่ละคนโดยเก็บแถวท้ายสุดไว้ จากนั้นตั้งรหัสนักเรียนเป็น index
จะได้ตารางที่ดึงทีละคอลัมน์ไปจับคู่ใส่ panel ได้เลย · ระวังชนิดข้อมูล:
silent/streak ต้องจบเป็นจำนวนเต็ม (ถมศูนย์ก่อนแปลงชนิด)

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.sort_values("week_start")`
- `.groupby("student_key")["silent_weeks"].max()`
- `.drop_duplicates("student_key", keep="last")` ต่อด้วย `.set_index("student_key")`
- `.map(...).fillna(0).astype(int)` สำหรับ silent/streak
- practice/checkpoint: `.map(...)` เฉยๆ ไม่ต้อง fillna

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_04_04.py"""),

("code", """# checkpoint: เวอร์ชัน canonical ของ 4.4
panel_eng_ok = feats_canonical[["student_key", "year", "month", "month_index",
                                "churned_next_month", "max_silent_weeks",
                                "streak_weeks", "practice_pct", "checkpoint_pct"]].copy()
print("panel_eng_ok:", panel_eng_ok.shape)"""),

# ============================================================ Unit 5 — ex_04_05
("md", """## [แนวคิด] 4.5 รวมร่าง + static features — แล้วเทียบกับกรรมการ

เหลือกลุ่มสุดท้าย: **ข้อมูลนิ่ง (static)** จากตาราง `students` — ไม่เปลี่ยนรายเดือน
แต่บอก "โปรไฟล์ความเสี่ยงตั้งต้น" ได้ดีมาก ใน sample ของเรา:
เด็กเรียน**เทป** churn ~14.0% vs เรียน**สด** ~6.9% (ต่างกัน 2 เท่า!)

| static | ทำไมน่าจะเกี่ยว |
|---|---|
| `grade` | ม.3 ใกล้สอบจริง แรงจูงใจต่างจาก ม.1 |
| `live_or_replay` | เรียนเทป = ไม่มีเพื่อน ไม่มีจังหวะถามสด หลุดง่าย |
| `old_new` | เด็กเก่ารู้จักระบบแล้ว |
| `signup_lateness` | สมัครช้า = ตามเพื่อนไม่ทัน + ใจร้อนอยากเห็นผล |
| `n_subjects` | ลงหลายวิชา = ผูกพันมาก ถอนยาก |

แถม feature ผสม 1 ตัว: `months_enrolled = month_index − signup_lateness`
(อยู่กับเรามากี่เดือนแล้ว) — เด็กใหม่ 1-2 เดือนแรกคือช่วงเปราะบางสุด
ปิดท้ายด้วย `.clip(lower=1)`: ใน sample มันไม่เปลี่ยนค่าอะไรเลย (ต่ำสุดได้ 1 พอดี)
แต่ข้อมูลจริงมีเด็กที่วันสมัครถูกกรอกผิดเดือน → ค่าติดลบโผล่มาหลอกโมเดลได้
กันไว้ก่อนราคาถูกกว่าตามแก้ทีหลังเสมอครับ
เสร็จแล้วเราจะ**เทียบทั้งตารางกับ canonical** เลขต้องตรงทุกช่อง ✊"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): pattern การ merge แบบไม่ให้คอลัมน์ชนกัน
left = pd.DataFrame({"student_key": ["A", "B"], "month": ["2025-03", "2025-03"],
                     "month_index": [1, 1], "f1": [10, 20]})
right = pd.DataFrame({"student_key": ["A", "B"], "month": ["2025-03", "2025-03"],
                      "month_index": [1, 1], "f2": [7, 8]})
# ถ้า merge ตรงๆ month_index จะโดนแตกเป็น _x/_y — drop ฝั่งขวาก่อนถึงจะสะอาด
merged = left.merge(right.drop(columns=["month_index"]), on=["student_key", "month"])
print(merged)

# static: 1 คนมีแถวเดียว — กัน bug ด้วย drop_duplicates ก่อน merge เสมอ
static_cols = ["grade", "live_or_replay", "old_new", "signup_lateness", "n_subjects"]
students[["student_key"] + static_cols].drop_duplicates("student_key").head(3)"""),

("md", """### [แบบฝึกหัด 4.5] ประกอบ features_monthly แล้วเทียบผลกับ canonical

**คำสั่ง:**
1. ตั้ง `ID = ["student_key", "year", "month"]`
2. merge `panel_att_ok` + `panel_exam_ok` + `panel_eng_ok` บน `ID`
   (สองตัวหลัง drop `month_index`, `churned_next_month` ก่อน กันคอลัมน์ชน)
3. merge static 5 คอลัมน์จาก `students` (`drop_duplicates("student_key")` ก่อน,
   `how="left"`, key แค่ `student_key`)
4. เพิ่ม `months_enrolled` = `month_index − signup_lateness` ขั้นต่ำ 1
5. เก็บใน `features` — check จะเทียบเลขจริง**ทุกคอลัมน์**กับ
   `churn_utils.build_features_monthly` (กรรมการตัวจริง)

**ผลลัพธ์ที่คาด:** `(1610, 20)` — 1 แถวต่อ active (student, month) เป๊ะ"""),

("code", """# TODO: ประกอบร่างตามคำสั่ง 1-4 (ระวัง: อย่าให้แถวเพิ่ม/หายระหว่าง merge!)
____ = None   # ← ลบบรรทัดนี้ แล้วเขียนโค้ดของคุณแทน

features = ____

checks.check("ex_04_05", features)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

รวมสาม panel บนคีย์ประจำตัวครบชุด (inner ได้ เพราะแถวตรงกัน 1,610 เป๊ะ) —
ถ้าแถวบวมเกิน 1,610 แปลว่าคอลัมน์คีย์ไม่ครบหรือมีแถวซ้ำ · ฝั่ง static
ใช้คีย์แค่รหัสนักเรียน (ค่าเดียวใช้ทุกเดือน) และต้องรวมแบบยึดฝั่งซ้าย ·
ขั้นต่ำหนึ่งของ months_enrolled มีเมธอดตัดเพดานล่างให้จบในขั้นตอนเดียว

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.merge(..., on=[...])` สองรอบ — ฝั่งขวา `.drop(columns=[...])` คอลัมน์ซ้ำก่อน
- static: `.merge(..., on="student_key", how="left")` หลัง `.drop_duplicates("student_key")`
- `.clip(lower=1)` ปิดท้ายผลลบของสองคอลัมน์

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_04_05.py"""),

("code", """# checkpoint: ตาราง features เวอร์ชัน canonical (= กรรมการของข้อ 4.5)
features_ok = feats_canonical.copy()
print("features_ok:", features_ok.shape)"""),

# ============================================================ Unit 6 — ex_04_06 (leakage trap!)
# NOTE: trap unit — จงใจใช้เคสจริงของโจทย์ ไม่มีเคสคู่ขนาน (ความช็อคคือบทเรียน)
#       ห้ามเติม toy case ภายหลัง — ดู authoring-guide "ข้อยกเว้น trap unit"
("md", """## [แนวคิด] 4.6 จดหมายลึกลับ: "พี่ครับ ผมเจอ feature เทพมาก"

ระหว่างที่คุณประกอบตารางอยู่ มีน้องฝึกงานส่งไฟล์ `extra_features.csv` มาให้
พร้อมข้อความว่า *"ลองใส่ดูครับพี่ คะแนนพุ่งทะลุเพดาน!"*

เราเป็นทีมที่น่ารัก ก็ลอง merge ให้ตามคำขอ... แล้ววัดผลแบบเร็วๆ ด้วย **AP**
(Average Precision — คะแนนความสามารถ "เรียงเด็กเสี่ยงไว้หัวแถว": 1.0 = เพอร์เฟกต์,
เท่า base rate ≈ เดามั่ว · บท 05-06 จะเจาะลึกตัวนี้กัน)"""),

("code", """from sklearn.metrics import average_precision_score

extra = pd.read_csv(DATA_DIR / "extra_features.csv", dtype={"month": str})
print("extra_features:", extra.shape, "| คอลัมน์:", list(extra.columns))
features_v1 = features_ok.merge(extra, on=["student_key", "month"], how="left")
print("merge แล้ว:", features_v1.shape)

# วัดแบบเร็ว: ใช้ค่า feature เป็นคะแนนความเสี่ยงตรงๆ (เข้าเรียนน้อย = เสี่ยงมาก → ใส่เครื่องหมายลบ)
lab_new = features_v1.dropna(subset=["churned_next_month", "att_next_month_pct"])
ap_new = average_precision_score(lab_new["churned_next_month"], -lab_new["att_next_month_pct"])

lab_old = features_v1.dropna(subset=["churned_next_month", "att_month_pct"])
ap_old = average_precision_score(lab_old["churned_next_month"], -lab_old["att_month_pct"])

print(f"AP ของ feature ใหม่       : {ap_new:.3f}   ← ?!?!")
print(f"AP ของ att_month_pct เดิม : {ap_old:.3f}")
print(f"base rate (เดามั่ว)        : {lab_new['churned_next_month'].mean():.3f}")"""),

("md", """### [แบบฝึกหัด 4.6] สืบสวน feature เทพ — แล้วตัดสินคดี

AP 0.99 ในปัญหาที่มนุษย์ผู้เชี่ยวชาญยังทายยาก... **ดีเกินจริงขนาดนี้ ต้องสงสัยไว้ก่อน**

**คำสั่ง (สืบสวน — ทำใน cell ไหนก็ได้):**
1. อ่านชื่อคอลัมน์ดีๆ: `att_next_month_pct` — คำว่า *next* บอกอะไรเรา?
2. หาหลักฐาน: ลอง `features_v1.groupby("churned_next_month", dropna=False)["att_next_month_pct"].agg(["mean", "count"])`
   — เด็กที่ churn เดือนถัดไป ค่านี้เป็นเท่าไร *ทุกคน*? เพราะอะไร?
3. คำถามตัดสิน: **"ยืนอยู่ที่สิ้นเดือน t เรารู้ค่านี้หรือยัง?"**
   ถ้า mentor ต้องโทรต้นเดือน ส.ค. ค่า "% เข้าเรียนเดือน ส.ค." มีอยู่จริงไหม?

**คำสั่ง (ตัดสิน):**
4. ทิ้งคอลัมน์อนาคตออกจาก `features_v1` → เก็บใน `features_clean`
5. เขียนเหตุผล 2-3 บรรทัดใน cell ท้ายข้อ (สำคัญ! การอธิบายให้คนอื่นฟังได้
   คือหลักฐานว่าเราเข้าใจจริง)

**ผลลัพธ์ที่คาด:** `features_clean` กลับมาเหลือ 20 คอลัมน์ ไม่มีคอลัมน์จากอนาคต"""),

("code", """# TODO: สืบตามขั้น 1-3 แล้วประหารคอลัมน์อนาคตทิ้ง
____ = None   # ← ลบบรรทัดนี้ แล้วเขียนโค้ดของคุณแทน

features_clean = ____

checks.check("ex_04_06", features_clean)"""),

("md", """*เขียนเหตุผลของคุณตรงนี้ (double-click เพื่อแก้):*

> `att_next_month_pct` ใช้ไม่ได้เพราะ ...

<details><summary>คำใบ้ 1 (แนวทาง)</summary>

ไล่จากนิยาม: แถวของเดือน t มีไว้ทำนาย "จะหายเดือน t+1 ไหม" —
feature ตัวนี้คือ % เข้าเรียนของเดือน t+1 เอง เด็กที่หายเดือน t+1
attendance เดือน t+1 ย่อมเป็น 0 แน่นอน (หายไปแล้วจะมาเรียนได้ยังไง)
มันเลยไม่ใช่ "ตัวทำนาย" แต่คือ **คำตอบที่ปลอมตัวมา** — ณ สิ้นเดือน t
ค่านี้ยังไม่เกิดขึ้นในโลกจริง โมเดลที่ train ด้วยมันจะพังทันทีตอนใช้งานจริง

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- หลักฐาน: `.groupby("churned_next_month", dropna=False)[...].agg(["mean", "count"])`
- ประหาร: `.drop(columns=["att_next_month_pct"])`

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_04_06.py"""),

("md", """> ⚠️ **กับดัก! บทเรียนที่แพงที่สุดของ ML — ขอสรุปให้จำขึ้นใจ**
>
> **Temporal leakage** = feature ของเดือน t แอบใช้ข้อมูลหลังสิ้นเดือน t
> วิธีดมกลิ่น: (1) ผลดีเกินจริงกะทันหัน (AP/accuracy กระโดดผิดธรรมชาติ) = สงสัยไว้ก่อนเสมอ
> (2) ชื่อ/นิยาม feature มีกลิ่นอนาคต — *next, final, last, total ทั้งซีซัน*
> (3) feature มาจากตารางที่ถูก update ย้อนหลัง เช่น "สถานะล่าสุด" แทน "สถานะ ณ เวลานั้น"
>
> **Treatment leakage** (โดนกันเยอะไม่แพ้กัน): สมมติเราเอา **log การโทรของ mentor**
> มาเป็น feature — ฟังดูดีใช่ไหม? แต่ mentor โทร*เพราะ*เด็กดูเสี่ยง การถูกโทรจึงเป็น
> **ผลของความเสี่ยง ไม่ใช่เหตุ** โมเดลจะเรียนว่า "ถูกโทร = จะหาย" ทั้งที่การโทร
> อาจช่วยรั้งเด็กไว้ด้วยซ้ำ → intervention ของเราเอง (โทร, ส่วนลด, ข้อความตาม)
> ห้ามใช้เป็น feature ทำนาย label ที่ intervention นั้นพยายามเปลี่ยน
>
> ระบบเรามียามเฝ้าประตูให้ชั้นหนึ่ง: `contracts.FORBIDDEN_FEATURE_COLS` +
> `checks.expect_no_leakage_cols` — แต่ยามจับได้เฉพาะ*ชื่อ*ที่รู้จัก
> จมูกของคุณต้องทำงานเสมอครับ"""),

("code", """# checkpoint: ตารางสะอาดเวอร์ชัน canonical (canonical ไม่เคยมีคอลัมน์อนาคตอยู่แล้ว)
features_clean_ok = feats_canonical.copy()
print("features_clean_ok:", features_clean_ok.shape)"""),

# ============================================================ Unit 7 — ex_04_07
("md", """## [แนวคิด] 4.7 เซฟงาน + ให้ contract ตรวจรอบสุดท้าย

ตาราง features คือ **สินค้าส่งมอบ** ของบทนี้ — บท 05 (โมเดลแรก + baseline)
จะเปิดไฟล์นี้มาใช้ทันที ก่อนส่งของเราให้ยามตรวจ 2 ชั้น:

1. `contracts.validate_df("features_monthly", df)` — เช็คคอลัมน์บังคับ, ชนิดข้อมูล,
   รูปแบบเดือน และ**คอลัมน์ต้องห้าม (leakage)** — ผ่านคืน True, ไม่ผ่าน raise พร้อม
   ข้อความไทยบอกจุดผิด
2. บันทึกด้วย `to_csv(..., index=False)` — ลืม `index=False` เมื่อไร จะได้คอลัมน์ผี
   `Unnamed: 0` ติดไฟล์ไปหลอกบทถัดไป

นิสัยดีของทีม data ที่อยากปลูกไว้: **validate ก่อน save เสมอ** — พังให้พังตอนนี้
ดีกว่าไปพังเงียบๆ ในบท 05"""),

("code", """# ตัวอย่าง (เคสคู่ขนาน): contract ทำงานยังไง — ลองกับตาราง labels
print("ตารางดี:", contracts.validate_df("labels_monthly", labels))

broken = labels.drop(columns=["active"])          # แกล้งทำพัง: ลบคอลัมน์บังคับ
try:
    contracts.validate_df("labels_monthly", broken)
except AssertionError as e:
    print("contract จับได้:", e)"""),

("md", """### [แบบฝึกหัด 4.7] บันทึก features_monthly.csv + validate

ใช้ `features_clean` ของคุณ (ถ้าข้อ 4.6 ผ่านแล้ว) หรือ `features_clean_ok` ก็ได้

**คำสั่ง:**
1. เก็บตารางสุดท้ายใน `features_final`
2. `contracts.validate_df("features_monthly", features_final)` — ต้องผ่าน (ไม่ raise)
3. บันทึกลง `DATA_DIR / "features_monthly.csv"` ด้วย `index=False`
4. ส่ง `features_final` เข้า check

**ผลลัพธ์ที่คาด:** ไฟล์ `features_monthly.csv` โผล่ใน DATA_DIR ขนาด `(1610, 20)`"""),

("code", """# TODO: validate → save → check ตามคำสั่ง 1-4
____ = None   # ← ลบบรรทัดนี้ แล้วเขียนโค้ดของคุณแทน

features_final = ____

checks.check("ex_04_07", features_final)"""),

("md", """<details><summary>คำใบ้ 1 (แนวทาง)</summary>

สามขั้นจบ: ตั้งตัวแปรตารางสุดท้าย → ให้ contract ตรวจ (ถ้าโดนด่า
อ่านข้อความแล้วแก้ตามนั้น) → เซฟเป็น csv ตามพาธที่โจทย์บอก โดย DATA_DIR
เป็น Path ใช้เครื่องหมายทับต่อชื่อไฟล์ได้เลย และอย่าลืมปิด index ตอนเซฟ

</details>

<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>

- `.copy()` จากตารางสะอาดที่เลือกใช้
- `contracts.validate_df("features_monthly", ...)`
- `.to_csv(DATA_DIR / "features_monthly.csv", index=False)`

</details>"""),

("code", """# ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
# %load ../solutions/sol_04_07.py"""),

# ============================================================ สรุป
("md", """## สรุปสิ่งที่ได้จากบทนี้

- **Panel (นักเรียน, เดือน)** 1,610 แถว พร้อม features 4 กลุ่ม:
  attendance (`att_month_pct/cum/delta`) · exam (`exam_avg_score`, `new_attempts_month`)
  · engagement (`max_silent_weeks`, `streak_weeks`, `practice_pct`, `checkpoint_pct`)
  · static (`grade`, `live_or_replay`, `old_new`, `signup_lateness`, `n_subjects`)
  + `month_index`, `months_enrolled`
- **กฎเหล็ก**: ยืนที่สิ้นเดือน t มองย้อนหลังเท่านั้น — เครื่องมือคือ `churn_utils.cutoff`
- **แผลเป็นที่ตั้งใจให้เกิด**: feature AP 0.99 ที่แท้จริงคือเฉลยจากอนาคต —
  ต่อไปนี้เจออะไร "ดีเกินจริง" คุณจะขนลุกก่อนดีใจ และจำ treatment leakage
  (log การโทรของ mentor) ไว้ด้วยเสมอ
- ไฟล์ `features_monthly.csv` ผ่าน contract เรียบร้อย

**บทต่อไป (05):** เอาตารางนี้ไปเทรนโมเดลแรก — Logistic Regression ตัวจ้อยที่ต้อง
พิสูจน์ตัวเองว่าชนะ baseline ง่ายๆ อย่าง tier ของ mentor ให้ได้ก่อน
แล้วเราจะแบ่ง train/test แบบ**ตามเวลา** (ปี 68 สอน, ปี 69 สอบ) — เพราะอะไร?
คำตอบอยู่ในกฎเหล็กข้อเดิมที่คุณเพิ่งได้แผลมานั่นแหละครับ 😉"""),

("code", """# กันพลาดให้บท 05: บันทึกเวอร์ชัน canonical ทับไว้เสมอ (เนื้อหาเดียวกับเฉลย 4.7)
feats_canonical.to_csv(DATA_DIR / "features_monthly.csv", index=False)
print("features_monthly.csv พร้อมสำหรับบท 05 ✓", feats_canonical.shape)"""),

]
