# Data Dictionary — ml-churn

ทุกตารางใช้ schema เดียวกันทั้ง 2 ปี (นิยามเต็ม+กติกา validate อยู่ใน `src/contracts.py`)
key กลาง: `student_key` — ปี 2569 = Supabase profile uuid · ปี 2568 = `"2568-<รหัส>"`

## data/sample/ (synthetic — notebooks รันบนชุดนี้ default)

| ไฟล์ | 1 แถวคือ | คอลัมน์สำคัญ |
|---|---|---|
| students.csv | นักเรียน 1 คน/ปี | grade, school, target_school, live_or_replay (สด/เทป), old_new, signup_date, signup_lateness (เดือนหลัง มี.ค.), n_subjects, subject_ids ("1;4;5;6") |
| attendance_long.csv | (นักเรียน, วิชา, EP) — **EP-merge แล้ว** | ep_final_date, week_start, status (present/absent/leave) |
| exams.csv | ข้อสอบ 1 ชุด | exam_type (practice/checkpoint), chapter, subject_id |
| exam_attempts.csv | การทำข้อสอบ 1 ครั้ง (**distinct exam_id ต่อคน**) | submitted_at, percentage, passed |
| enrollment_events.csv | เหตุการณ์สมัคร/ยกเลิก | event_type (enroll/re_enroll/cancel/subject_drop), event_month, source, confidence · **cancel: event_month = เดือนสุดท้ายที่ยัง active (ออกตอนสิ้นเดือน)** |
| labels_monthly.csv | (นักเรียน, เดือนที่ active) | churned_next_month ∈ {0,1,NaN} · **NaN = censored** (ก.ย.=จบซีซัน / เดือนที่ t+1 ยังไม่จบ) |
| weekly_metrics.csv | (นักเรียน, สัปดาห์) | att_week_pct, att_cum_pct, practice/checkpoint done/total/pct/avg_score, silent_weeks, streak_weeks, tier |
| extra_features.csv | (นักเรียน, เดือน) | att_next_month_pct — **กับดัก leakage สำหรับ EX4.6 ห้ามเฉลยก่อนถึงบท** |

ข้อเท็จจริง sample (หลัง seed 68): 310 คน (160 ปี 68 + 150 ปี 69), churn rate ~8.8%,
peak ก.ค.–ส.ค., trap ให้ AP ~0.99, โมเดล LogReg train68→test69 ได้ AP ~0.22 (base ~0.10),
precision@30 model ~0.30 vs tier-heuristic ~0.23

## Subject IDs
1=คณิตศาสตร์ 2=ภาษาอังกฤษ 3=วิทย์กายภาพ 4=ฟิสิกส์ 5=เคมี 6=ชีววิทยา — มีข้อสอบเฉพาะ {1,4,5,6}

## นิยาม churn (fix แล้ว — บท 01 สอนที่มา)
- active เดือน t = enrolled-confirmed ตอนต้นเดือน
- churned_next_month=1 ⟺ ยกเลิก "ทุกวิชา" มีผลสิ้นเดือน t (หายไปเดือน t+1)
- ลดบางวิชา = subject_drop → เป็น feature ไม่ใช่ churn
- ออกเดือน ก.ย. = จบซีซันปกติ ไม่นับ churn (label NaN)
- ปี 2569 observed ถึง ~3 ส.ค. → label ได้ถึง t=2026-07 (known_through="2026-07")

## ข้อมูลจริง (data/raw/ → data/processed/ หลังทำบท 02)
- ปี 2569: `src/eduwise_extract.py` ดึงจาก Supabase (read-only) → data/raw/2569_supabase/
  (weekly_metrics_2569, attendance_long_2569, exam_attempts_2569, exams_2569, students_2569, enrollments_snapshot)
- ปี 2568: CEO วางไฟล์ Sheets/Excel export ใน data/raw/2568/ (schema ยังไม่รู้ — บท 02 audit)
- Label จริง: CEO กรอก data/raw/labels/ ตาม label_sheet_template.csv

## กับดักโดเมน 4 ข้อ (เคารพเสมอ — extractor จัดการให้แล้ว)
1. 1 EP สอน 2 รอบคนละวัน เข้ารอบไหนก็ = present (ห้ามนับรายแถว/รายวัน)
2. กรองช่วงเวลา attempts ด้วย submitted_at เท่านั้น (created_at = เวลา bulk sync — ขยะ)
3. ไม่มี retake → นับ distinct exam_id
4. ตัวหาร % ข้อสอบ = เฉพาะบทที่สอนแล้ว (+grace 2 สัปดาห์)

## Attendance ปี 69 — ไฟล์ไหนใช้ (เพิ่ม 11 ส.ค. 2026)
- `data/raw/2569_supabase/attendance_long_2569.csv` = ดิบจาก extractor (**อย่าใช้ตรงกับ features**
  — มี "ขาดปลอม" ของวิชาที่ลดแล้ว เพราะ Eduwise ทับประวัติวิชา + roster override)
- **ใช้ `data/raw/labels/attendance_long_2569_clean.csv`** — กรองด้วยวิชาที่เรียนจริงรายเดือน
  (`subject_enrollment_2569.csv` สร้างจากบิล SalesByProductReport ที่บอกจำนวนวิชา/บิล +
  attendance ระบุตัววิชา · กติกา: พฤติกรรมชนะเงิน — วิชาที่ยัง present ไม่ตัดจนกว่าหยุดมาจริง)
- `subjects_monthly_2569.csv` = จำนวนวิชาที่จ่ายรายเดือน → feature `n_subjects_month`
- สคริปต์: `scripts/build_subjects_monthly_2569.py` · เคสกำกวมอยู่ `subjects_review_2569.csv`
- ปี 68 แก้ที่ต้นทางแล้ว (convert_raw68 กรองด้วยวิชารายเดือนจาก subscription)

## Exclusion list (CEO ตัดสิน 11 ส.ค. 2026)
- `data/raw/labels/exclude_pairs_2569.csv` — 103 (คน,เดือน) ที่วิชาใน log เกินบิล (กำกวม)
  → **ch04 ต้อง filter คู่พวกนี้ออกจากตารางเทรน** (ตัดทั้งแถว ไม่ใช่แค่ attendance)
  เหตุผล: กันข้อมูลกำกวมปน · เสีย churn ตัวอย่างแค่ 10/178 (5.6%) · เหลือเทรน 2,803 แถว/231 churn
