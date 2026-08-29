# Authoring Guide — กติกาการเขียนบทเรียน (สำหรับผู้สร้าง notebook)

## รูปแบบไฟล์
- เขียน cell list ที่ `notebooks_src/chXX_<slug>_cells.py`:
  ```python
  TITLE = "01 — นิยามปัญหา + สร้าง label"
  CELLS = [
      ("md", """..."""),
      ("code", """..."""),
  ]
  ```
- build เป็น .ipynb: `./venv/bin/python -m src.nb_build notebooks_src/chXX_..._cells.py`
- execute ทดสอบ: `ML_CHURN_DATA=sample ./venv/bin/jupyter-nbconvert --to notebook --execute --inplace notebooks/chXX_*.ipynb`
  (**ห้าม** `python -m jupyter nbconvert` — มันหา jupyter-nbconvert ผ่าน PATH ไปเจอ Anaconda → kernel ผิด interpreter)
- เฉลย: `solutions/sol_XX_YY.py` (XX=บท, YY=ข้อ เช่น sol_01_02.py) — ไฟล์ต้องรันได้เดี่ยวๆ
  ต่อจาก cell ก่อนหน้าใน notebook (อ้างตัวแปรที่มีอยู่แล้วได้)
- ตัวตรวจ: `src/checks_chXX.py` ลงทะเบียนด้วย `@register("ex_XX_YY")` (ดู src/checks.py)

## โครงสร้าง 1 แบบฝึกหัด (unit) — ตามลำดับเป๊ะ
1. `md` **[แนวคิด]** ไทย ≤10 บรรทัด บอก "ทำไม" ก่อน "อย่างไร"
2. `code` **ตัวอย่าง** — โค้ดเต็มรันได้ บนเคสคู่ขนาน (ไม่ใช่เคสเดียวกับแบบฝึกหัด)
3. `md` **[แบบฝึกหัด X.Y ชื่อ]** — โจทย์ + คำสั่งเป็นข้อๆ (แบบ Instructions panel) + บอกผลลัพธ์ที่คาด (shape/head)
4. `code` **skeleton** — เว้น `____` + คอมเมนต์ไทย `# TODO:` · บรรทัดสุดท้ายเสมอ:
   `checks.check("ex_XX_YY", <ตัวแปรคำตอบ>)`
5. `md` **คำใบ้** — 2 ระดับใน `<details><summary>คำใบ้ 1 (แนวทาง)</summary>...</details>`
   และ `<details><summary>คำใบ้ 2 (function ที่ใช้)</summary>...</details>`
6. `code` **เฉลย** — cell มีแค่:
   ```
   # ถ้าอยากดูเฉลย: ลบ # หน้าบรรทัดล่าง แล้วรัน cell นี้ 2 ครั้ง (ครั้งแรก load, ครั้งสองรัน)
   # %load ../solutions/sol_XX_YY.py
   ```

## Convention กลาง (ตัดสินแล้ว — ทุกบทต้องเหมือนกัน)
- **หัวเรื่อง** (ตามแบบ ch04–06): ชื่อบท `# 0X — <ชื่อบท>` · แนวคิด `## [แนวคิด] X.Y <ชื่อ>` ·
  แบบฝึกหัด `### [แบบฝึกหัด X.Y] <ชื่อ>` · สรุป `## สรุปสิ่งที่ได้จากบทนี้` (ไม่มีอีโมจิในหัวเรื่อง)
- **สะพานเชื่อมบท**: ประโยคแรกของทุกบท (ยกเว้น 00) ต้องอ้างถึงบทก่อนหน้า 1 ประโยค
- **คำใบ้ 1** = ประโยคบรรยายแนวทางล้วน **ห้ามมีโค้ดแม้แต่บรรทัดเดียว**
- **คำใบ้ 2** = ชื่อ function/method + argument สำคัญเท่านั้น เช่น `` `.dt.strftime("%Y-%m")` · `merge(..., on=[...], how="left")` `` —
  **ห้าม** ใช้ชื่อตัวแปรของโจทย์ และห้ามประกอบเป็นบรรทัดที่ copy ไปรันได้เลย (โค้ดเต็มอยู่ใน solutions/ ที่เดียว)
- **ตัวตนผู้เล่า**: เรียกตัวเองว่า "ผม" (ไม่ใช่ "พี่ PO" บุรุษที่ 3) · เรียกผู้เรียนว่า "คุณ" หรือไม่เรียกเลย
- **สมมติฐาน mentor capacity** (ใช้ตรงกันทุกบท): โทรได้ ~30 สาย/เดือน สายละ 15–20 นาที
- **ข้อยกเว้น trap unit** (เช่น ex_04_06): อนุญาตให้ cell ตัวอย่างใช้เคสจริงของโจทย์ได้
  เพราะ "ความช็อค" คือบทเรียน — ห้ามผู้แก้ไขภายหลังเติมเคสคู่ขนานจนสปอยล์กับดัก
- **checks ที่ recompute จาก DATA_DIR สด**: ไม่บังคับต้องมี branch IS_SAMPLE แยก
  แต่ต้องมีคอมเมนต์หัวไฟล์อธิบายว่า expected คำนวณสดจึงใช้ได้ทุกโหมด

## กติกาสำคัญ
- **ภาษา**: อธิบายภาษาไทย เป็นกันเองแบบพี่สอนน้อง (ผู้เรียนคือ CEO เรียกว่า "คุณ" หรือไม่เรียกเลย)
  ศัพท์เทคนิคทับศัพท์อังกฤษได้ (feature, leakage, pipeline) · โค้ด/ตัวแปรอังกฤษ
- **หัว notebook ทุกบท**: md title + learning objectives → code cell setup:
  ```python
  import sys; sys.path.insert(0, "..")
  import pandas as pd
  from src import checks, churn_utils, contracts
  from src.config import DATA_DIR, IS_SAMPLE
  plt = churn_utils.plot_style()
  print("โหมดข้อมูล:", "SAMPLE (ข้อมูลจำลอง)" if IS_SAMPLE else f"REAL ({DATA_DIR})")
  ```
- **notebook ต้องรัน top-to-bottom ผ่านทั้งเล่มบน sample data** รวมทั้ง skeleton cell —
  ดังนั้น skeleton ที่มี `____` ต้องอยู่ในรูปที่รันแล้วไม่ตาย เช่น:
  ```python
  ____ = None  # TODO: แก้บรรทัดนี้
  labels = ____
  checks.check("ex_01_02", labels)   # ยังไม่ผ่านจนกว่าจะเติมถูก — ปกติ!
  ```
  ห้ามให้ skeleton โยน SyntaxError/NameError (ใช้ `____ = None` นำหน้าเสมอ) —
  `checks.check` คืน False ได้ ไม่ raise → nbconvert ผ่าน
- **cell เฉลย `%load` ต้อง comment ไว้** (ไม่รันจริงตอน nbconvert)
- **checks**: คำนวณค่า expected สดๆ จาก DATA_DIR + `src.churn_utils` (canonical) —
  ห้าม hardcode ตัวเลขที่ขึ้นกับ seed ถ้าเลี่ยงได้ · ถ้า `IS_SAMPLE` เป็น False เช็คแค่โครงสร้าง/ช่วงค่า
  ข้อความ fail = ไทย ชี้จุดผิด ไม่เฉลยตรงๆ ("จำนวนแถวไม่ตรง — ลืมกรอง active=1 หรือเปล่า?")
- **กล่องกับดัก**: จุด pitfall ประจำบท ใช้ md blockquote `> ⚠️ **กับดัก!** ...`
- **pandas 3.0 / sklearn 1.8**: ห้ามใช้ df.append, inplace chained assignment,
  sparse=... ใน OneHotEncoder (ใช้ default) · string dtype ระวัง .str ปกติใช้ได้
- **ขนาดบท**: 60–90 นาที ≈ 4–6 แบบฝึกหัด + อธิบาย · อย่ายัดทฤษฎีเกิน
- ท้ายบท: md **สรุปสิ่งที่ได้** + **โยงบทถัดไป** 2-3 บรรทัด
