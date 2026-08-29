# ML Churn Prediction — หลักสูตรสไตล์ DataCamp สำหรับ CEO 🎓

ทำนายนักเรียนคอร์สติวเข้ม (มี.ค.–ก.ย.) ที่มีแนวโน้มจะออกในแต่ละเดือน ด้วย scikit-learn
โดย **คุณเขียนโค้ด ML เองทุกบท** — PO เตรียมข้อมูล ระบบตรวจคำตอบ และเฉลยไว้ให้แล้ว

## เริ่มยังไง

```bash
cd ml-churn
source venv/bin/activate          # venv ติดตั้งไว้ให้แล้ว
jupyter notebook notebooks/       # เปิดสมุดบทเรียน
```

เริ่มที่ `notebooks/ch00_setup_tour.ipynb` แล้วไล่ตามลำดับ บทละ ~60–90 นาที

> ⚠️ **เครื่องนี้มี Anaconda** — 2 กับดักที่เจอมาแล้วจริง:
> 1. เปิด notebook ให้ใช้ `./venv/bin/jupyter notebook` เสมอ (ห้ามพิมพ์ `jupyter` เฉยๆ — PATH จะพาไป Anaconda)
> 2. ห้ามใช้ `python -m jupyter nbconvert` (มันค้นหา `jupyter-nbconvert` ผ่าน PATH → ได้ของ Anaconda
>    → kernel เป็น sklearn คนละ version → pickle โมเดลพัง) — `scripts/run_all.sh` เรียก
>    `./venv/bin/jupyter-nbconvert` ตรงๆ ให้แล้ว

## หลักสูตร 9 บท

| บท | เรื่อง | ได้อะไร |
|----|-------|---------|
| 00 | ติดตั้ง + ทัวร์ข้อมูล | เห็นธุรกิจตัวเองใน DataFrame ครั้งแรก |
| 01 | นิยามปัญหา + สร้าง label | labels_monthly + **การบ้าน: รวมบันทึกยกเลิกลง label sheet** |
| 02 | Data wrangling 2 ปี | ข้อมูล 68+69 เข้า schema เดียว |
| 03 | EDA | เห็น pattern churn ด้วยตาก่อนใช้โมเดล |
| 04 | Feature engineering ⭐ | ตาราง (นักเรียน, เดือน) + บทเรียน leakage |
| 05 | โมเดลแรก (LogReg) | PR curve, precision@30, เทียบ baseline |
| 06 | Validation + threshold | expanding window, CI, save โมเดลจริง |
| 07 | อ่านโมเดล | "ทำไมน้องคนนี้เสี่ยง" เป็นภาษาไทย |
| 08 | Deploy | risk report รายเดือน → แผนต่อ LINE/Eduwise |

## กติกาการเรียน (แบบ DataCamp)

- ทุกแบบฝึกหัดมี **skeleton เว้นช่อง `____`** → เติมเอง → รัน `checks.check(...)` เป็นปุ่ม submit
- ติดขัด: เปิด **คำใบ้ 1** (แนวทาง) → **คำใบ้ 2** (function ที่ใช้) → ค่อยดูเฉลย (`%load`)
- ห้ามดูเฉลยก่อนลองเอง อย่างน้อย 10 นาที 😄

## โหมดข้อมูล

- **default = SAMPLE**: ข้อมูลจำลอง 310 คน 2 ซีซัน (ฝัง pattern จริง) — ทุกสมุดรันได้ทันที
- **REAL**: `export ML_CHURN_DATA=real` หลังทำบท 02 เสร็จ (ข้อมูลอยู่ `data/processed/`)
- ข้อมูลจริงปี 69 ดึงจาก Supabase แล้วที่ `data/raw/2569_supabase/` (สคริปต์ `src/eduwise_extract.py` — read-only)
- ข้อมูลปี 68: วางไฟล์ Sheets/Excel export ที่ `data/raw/2568/`
- Label จริง: กรอกตาม `data/raw/labels/label_sheet_template.csv`

## โครงสร้าง

```
notebooks/       สมุดบทเรียน 9 บท (สร้างจาก notebooks_src/ ด้วย src/nb_build.py)
solutions/       เฉลยรายข้อ (%load ได้จากในสมุด)
src/             contracts, churn_utils (ฟังก์ชันกลาง), checks (ตัวตรวจ), extractor
data/sample/     ข้อมูลจำลอง (git-friendly, รันได้ทุกเครื่อง)
models/          โมเดลที่ save จากบท 06
reports/         risk_YYYY-MM.csv จากบท 08
scoring/         score_month.py = สคริปต์ production (บท 08 ให้เติม TODO)
scripts/         run_all.sh (รันทุกสมุด check สุขภาพ), smoke_test.py
docs/            data-dictionary.md, authoring-guide.md
```

## คำถามที่จะเจอแน่

**ทำไมต้องมี label sheet?** — Eduwise ทับประวัติการยกเลิก (ไม่มี drop date) เราจึงสร้าง label
ย้อนหลังจาก DB ไม่ได้ ต้องใช้บันทึกของคุณ นี่คืองานสำคัญที่สุดของโปรเจกต์ ไม่ใช่ตัวโมเดล

**โมเดลต้องดีแค่ไหนถึง deploy?** — เกณฑ์ที่ตกลงกัน: precision@30 บน holdout ปี 69
ต้องชนะ tier heuristic (red/orange) ที่ใช้อยู่ ถ้าไม่ชนะ = deploy ranked list แบบผสมไปก่อน
แล้วเก็บข้อมูลสะอาดๆ รอซีซัน 2570

*สร้างโดยคุณ PO · ก.ค. 2569 (Aug 2026) · แผน: ~/.claude/plans/po-python-rustling-hopcroft.md*
