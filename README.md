# Student Churn Early-Warning — Online Tutoring Business

> ⚠️ **DRAFT** — โครงและข้อเท็จจริงถูกเตรียมไว้แล้ว ส่วนที่เป็น `TODO(CEO)` เจ้าของโปรเจกต์ต้องเขียนเองด้วยภาษาตัวเองก่อนเปิด public

Predicting which students are at risk of cancelling their monthly subscription at a Thai online tutoring school (~660 students/season), so mentors can reach out **before** the cancellation happens.

I run this business. The model was built on our real operational data (attendance, exam checkpoints, engagement, payment timing) — and this public repo runs entirely on a **synthetic sample dataset** so that no real student data ever leaves the company.

<!-- TODO(CEO): 2-3 ประโยคของคุณเอง — ทำไมถึงลุกขึ้นมาสร้างโมเดลนี้ (เช่น churn เจ็บตรงไหนในธุรกิจจริง) -->

## Honest results (on real data)

| Metric (season 2569, out-of-fold) | Value |
|---|---|
| Average Precision | 0.213 |
| Precision@30 — model | 0.240 |
| Precision@30 — existing "tier" heuristic | **0.273** |

**The model lost to our existing human-designed tier heuristic** on the metric that matters operationally (precision in the top-30 list our mentors can actually call each month). The pre-registered deploy gate therefore says: don't replace the heuristic — **blend** the two ranked lists and keep monitoring.

I consider this the most valuable part of the project: the evaluation was honest enough to say "not yet."

<!-- TODO(CEO): 1-2 ประโยค — คุณได้เรียนรู้อะไรจากการที่โมเดล "แพ้" heuristic ของทีมตัวเอง -->

## What's inside (method highlights)

- **Right-censored labels** (ch01) — months where the outcome isn't knowable yet are labeled `NaN`, never assumed to be "stayed".
- **Leakage discipline** (ch04) — every feature is built through a `cutoff()` helper that only sees data available at prediction time; the course even plants a leakage trap (`att_next_month_pct`) that must be caught.
- **Time-aware validation** (ch06) — expanding-window cross-validation over months, because a random split would let the model peek at the future.
- **A metric tied to reality** (ch05/06) — precision@30, because 30 is roughly how many outreach calls our mentors can make per month; accuracy would be meaningless at a ~low monthly churn base rate.
- **Uncertainty** (ch06) — bootstrap confidence intervals before believing any comparison.
- **Engineering hygiene** — schema contracts (`src/contracts.py`), a sample/real data switch (`src/config.py`), one shared feature builder for both training and scoring (`src/churn_utils.py`) to prevent train/serve skew, and a smoke test gate (`scripts/smoke_test.py`).

## A few pictures (generated from the synthetic sample in this repo)

| | |
|---|---|
| ![Survival curve](figures/survival_curve.png) | ![Threshold vs mentor capacity](figures/threshold_vs_mentor_capacity.png) |
| *Survival curve — % of the March cohort still enrolled each month* | *Threshold sweep — the cutoff is chosen where mentor call capacity (~30/month) sits* |
| ![Model comparison](figures/model_comparison_bootstrap_ci.png) | ![Coefficients](figures/model_coefficients.png) |
| *LogReg vs HistGradientBoosting with bootstrap 95% CIs — overlapping, so keep the simpler model* | *What the model looks at (logistic regression coefficients)* |

> Numbers in these charts come from the **synthetic** dataset; the headline results table above is from the real (private) data.

## Reproduce (3 commands)

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
bash scripts/run_all.sh          # executes every notebook on the synthetic sample
./venv/bin/python scripts/smoke_test.py
```

## Repo structure

```
notebooks/       ch00–ch08: setup → labels → wrangling → EDA → features → model
                 → validation → interpretation → deploy (executed on sample data)
src/             config, schema contracts, shared feature/label builders, checkers
data/sample/     synthetic dataset (seeded generator: src/make_synthetic.py)
scoring/         monthly scoring script (production shape)
solutions/       per-exercise answer keys (course scaffolding)
docs/            data dictionary, course guide
```

## How this was built — and what is mine

This project is a structured, exercise-driven course **built with AI assistance (Claude) acting as the tutor**: the scaffolding, auto-checkers, and answer keys were AI-generated around my real business problem and my real data.

**The exercise solutions in notebooks ch01–ch07 are my own work** — label design, EDA, feature engineering, model training, validation, and interpretation, written by me and verified by the course's checkers.

The real-data ETL (Supabase extraction, payment reconciliation) exists privately and is excluded here for data-protection reasons (PDPA): this repo contains **zero real student records** — names in the sample data are synthetic (`น้อง...` pattern), generated with a fixed random seed.

<!-- TODO(CEO): ปรับย่อหน้าบนให้เป็นเสียงคุณเอง + ระบุสิ่งที่คุณภูมิใจว่าทำเองได้ -->

## What I'd do next

<!-- TODO(CEO): เลือก 2-3 ข้อที่คุณอยากทำจริงและอธิบายได้ เช่น:
- เก็บข้อมูลซีซันถัดไปแล้ว retrain + เทียบกับ blend list
- เพิ่ม features ฝั่ง engagement (Discord activity)
- ทดสอบ threshold ตาม capacity ของ mentor ที่เปลี่ยนไป
-->

---

## สรุปภาษาไทย

โปรเจกต์ทำนายความเสี่ยงที่นักเรียนจะยกเลิกคอร์สรายเดือนของโรงเรียนกวดวิชาออนไลน์ (~660 คน/ซีซัน) เพื่อให้ mentor โทรดูแลได้ก่อนเด็กหาย — เทรนบนข้อมูลจริงของธุรกิจ แต่ repo สาธารณะนี้รันบน**ข้อมูลสังเคราะห์ทั้งหมด**เพื่อคุ้มครองข้อมูลส่วนบุคคลของนักเรียน (PDPA)

ผลลัพธ์ตรงไปตรงมา: precision@30 ของโมเดล (0.240) **แพ้** ระบบ tier เดิมที่ทีมออกแบบเอง (0.273) → ข้อสรุปคือใช้แบบผสม (blend) ไม่ใช่แทนที่ — และผมถือว่าการประเมินที่ซื่อสัตย์พอจะบอกว่า "ยังไม่ชนะ" คือคุณค่าหลักของงานนี้

<!-- TODO(CEO): 2-3 ประโยคปิดท้ายภาษาไทยในเสียงของคุณ -->
