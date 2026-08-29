"""Generate data/sample/ — synthetic EWT-like data so every notebook runs
before real data is ready.

Planted structure (deterministic, seed=68):
  - ~300 students over 2 seasons (2568 complete Mar-Sep, 2569 censored at Jul)
  - churn risk truly depends on: low attendance, silent weeks, เทป (replay),
    late signup, fewer subjects, and peaks in Jul-Aug (seasonality)
  - leakage trap for EX4.6: data/sample/extra_features.csv has
    att_next_month_pct computed from month t+1 -> near-perfect "feature"

Run:  python -m src.make_synthetic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (EXAM_SUBJECTS, PROJECT_ROOT, SEASONS, month_seq)
from src import churn_utils, contracts

RNG = np.random.default_rng(68)
OUT = PROJECT_ROOT / "data" / "sample"

GRADES = ["ม.1", "ม.2", "ม.3"]
GRADE_P = [0.15, 0.35, 0.50]
TARGETS = ["MWIT", "KVIS", "TU", "อื่นๆ"]
TARGET_P = [0.35, 0.15, 0.40, 0.10]
SCHOOLS = ["สวนกุหลาบ", "สาธิตปทุมวัน", "บดินทรเดชา", "หาดใหญ่วิทยาลัย", "อื่นๆ"]
SCHOOL_P = [0.15, 0.12, 0.13, 0.10, 0.50]

THAI_NICKS = ["ต้นน้ำ", "ปราง", "ภูมิ", "ไอซ์", "เฟิร์น", "กันต์", "มายด์", "โอม",
              "พลอย", "เจได", "น้ำหวาน", "ภีม", "แพรวา", "ไทเกอร์", "อิง", "คุน"]


def season_weeks(year: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(SEASONS[year]["start_month"] + "-01")
    start = start - pd.Timedelta(days=start.weekday())  # monday
    return pd.date_range(start, periods=28, freq="7D")  # ~7 months


def gen_season(year: int, n_students: int, code_start: int):
    months = month_seq(SEASONS[year]["start_month"], SEASONS[year]["end_month"])
    weeks = season_weeks(year)

    students, events = [], []
    profiles = []
    for i in range(n_students):
        code = f"{code_start + i:04d}"
        student_key = f"{year}-{code}" if year == 2568 else (
            f"uuid-{year}-{code}")
        n_subj = int(RNG.choice([2, 3, 4], p=[0.15, 0.20, 0.65]))
        subj = sorted(RNG.choice(EXAM_SUBJECTS, size=n_subj, replace=False).tolist())
        lateness = int(RNG.choice([0, 0, 0, 1, 1, 2, 3], p=[.45, .15, .10, .12, .08, .06, .04]))
        live = RNG.choice(["สด", "เทป"], p=[0.7, 0.3])
        old_new = RNG.choice(["old", "new"], p=[0.25, 0.75])
        signup_month = months[lateness]
        signup_date = pd.Timestamp(signup_month + "-01") + pd.Timedelta(
            days=int(RNG.integers(0, 25)))

        # latent diligence drives both attendance and churn -> learnable signal
        diligence = float(np.clip(RNG.normal(0.78, 0.16), 0.05, 0.99))

        students.append({
            "student_key": student_key, "year": year, "student_code": code,
            "display_name": f"น้อง{RNG.choice(THAI_NICKS)} {code}",
            "grade": RNG.choice(GRADES, p=GRADE_P),
            "school": RNG.choice(SCHOOLS, p=SCHOOL_P),
            "target_school": RNG.choice(TARGETS, p=TARGET_P),
            "live_or_replay": live, "old_new": old_new,
            "signup_date": signup_date, "signup_lateness": lateness,
            "n_subjects": n_subj, "subject_ids": ";".join(map(str, subj)),
            "same_person_key": "",
        })
        events.append({
            "student_key": student_key, "year": year, "event_type": "enroll",
            "event_date": signup_date, "event_month": signup_month,
            "subject_ids": ";".join(map(str, subj)),
            "source": "label_sheet", "confidence": "high",
        })
        profiles.append({
            "student_key": student_key, "subjects": subj, "diligence": diligence,
            "live": live, "lateness": lateness, "n_subj": n_subj,
            "signup_month": signup_month,
        })

    students = pd.DataFrame(students)

    # exams: per subject, 9 chapters x practice + 4 checkpoints, staggered availability
    exams = []
    for s in EXAM_SUBJECTS:
        for ch in range(1, 10):
            exams.append({
                "exam_id": f"{year}-S{s}-P{ch}", "year": year, "subject_id": s,
                "exam_type": "practice", "chapter": str(ch),
                "total_questions": int(RNG.integers(10, 21)),
                "available_from_week": weeks[min(ch * 3 - 3, 27)],
            })
        for ck in range(1, 5):
            exams.append({
                "exam_id": f"{year}-S{s}-C{ck}", "year": year, "subject_id": s,
                "exam_type": "checkpoint", "chapter": f"C{ck}",
                "total_questions": 25,
                "available_from_week": weeks[min(ck * 6 - 1, 27)],
            })
    exams = pd.DataFrame(exams)

    # simulate week-by-week behaviour + churn hazard
    att_rows, atm_rows = [], []
    month_of_week = {w: w.strftime("%Y-%m") for w in weeks}
    for p in profiles:
        active = True
        churn_month = None
        wobble = RNG.normal(0, 0.05)
        fatigue = 0.0
        recent_absent = 0
        for wi, wk in enumerate(weeks):
            if month_of_week[wk] < p["signup_month"]:
                continue
            if not active:
                break
            month = month_of_week[wk]
            m_index = months.index(month) + 1 if month in months else 7
            # attendance probability decays with fatigue; replay students drift more
            p_att = p["diligence"] + wobble - fatigue
            if p["live"] == "เทป":
                p_att -= 0.08
            p_att = float(np.clip(p_att, 0.02, 0.98))
            week_present = 0
            for s in p["subjects"]:
                present = RNG.random() < p_att
                att_rows.append({
                    "student_key": p["student_key"], "year": year,
                    "subject_id": s, "episode_number": wi + 1,
                    "ep_final_date": wk + pd.Timedelta(days=int(RNG.integers(0, 6))),
                    "week_start": wk,
                    "status": "present" if present else (
                        "leave" if RNG.random() < 0.1 else "absent"),
                })
                if present:
                    week_present += 1
            if week_present == 0:
                recent_absent += 1
                fatigue += 0.04
            else:
                recent_absent = 0
                fatigue = max(0.0, fatigue - 0.02)

            # exam attempts: diligent students do newly available exams
            newly = exams[exams["available_from_week"] == wk]
            newly = newly[newly["subject_id"].isin(p["subjects"])]
            for _, ex in newly.iterrows():
                if RNG.random() < p_att * 0.9:
                    score = float(np.clip(RNG.normal(35 + 55 * p["diligence"], 12), 0, 100))
                    atm_rows.append({
                        "student_key": p["student_key"], "year": year,
                        "exam_id": ex["exam_id"], "subject_id": ex["subject_id"],
                        "exam_type": ex["exam_type"],
                        "submitted_at": wk + pd.Timedelta(
                            days=int(RNG.integers(0, 7)),
                            hours=int(RNG.integers(16, 23))),
                        "percentage": round(score, 1),
                        "passed": int(score >= 50),
                    })

            # month-end churn hazard (only evaluated on the last week of a month)
            is_month_end = (wi + 1 < len(weeks)
                            and month_of_week[weeks[wi + 1]] != month) or wi + 1 == len(weeks)
            if is_month_end and month < months[-1]:
                z = (-4.1
                     + 3.2 * (1 - p_att)                       # low attendance
                     + 0.55 * min(recent_absent, 3)            # silent streak
                     + (0.8 if p["live"] == "เทป" else 0.0)
                     + 0.35 * p["lateness"]
                     + 0.45 * (4 - p["n_subj"])
                     + (0.7 if m_index in (5, 6) else 0.0))    # Jul-Aug peak
                hazard = 1 / (1 + np.exp(-z))
                if RNG.random() < hazard:
                    active = False
                    churn_month = month
        if churn_month is not None:
            events.append({
                "student_key": p["student_key"], "year": year,
                "event_type": "cancel",
                "event_date": pd.Timestamp(churn_month + "-01") + pd.offsets.MonthEnd(0),
                "event_month": churn_month,
                "subject_ids": ";".join(map(str, p["subjects"])),
                "source": RNG.choice(["label_sheet", "line", "memory"], p=[.6, .3, .1]),
                "confidence": RNG.choice(["high", "medium"], p=[.8, .2]),
            })

    events = pd.DataFrame(events)
    events["event_date"] = pd.to_datetime(events["event_date"])
    attendance = pd.DataFrame(att_rows)
    attempts = pd.DataFrame(atm_rows)

    # censoring: 2569 season observed only through July (today ~Aug 2026)
    if year == 2569:
        obs_end = pd.Timestamp("2026-08-03")
        attendance = attendance[attendance["ep_final_date"] < obs_end]
        attempts = attempts[attempts["submitted_at"] < obs_end]
        known_through = "2026-07"
    else:
        known_through = SEASONS[year]["end_month"]

    labels = churn_utils.build_labels_monthly(events, year, known_through)
    weekly = churn_utils.compute_weekly_metrics(attendance, attempts, exams, year)
    return students, exams, attendance, attempts, events, labels, weekly


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [gen_season(2568, 160, 1001), gen_season(2569, 150, 2001)]
    names = ["students", "exams", "attendance_long", "exam_attempts",
             "enrollment_events", "labels_monthly", "weekly_metrics"]
    for i, name in enumerate(names):
        df = pd.concat([p[i] for p in parts], ignore_index=True)
        contracts.validate_df(name, df)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:20s} {len(df):7,d} rows")

    # leakage trap for EX4.6: "wonder feature" secretly computed from month t+1
    att = pd.concat([p[2] for p in parts], ignore_index=True)
    labels = pd.concat([p[5] for p in parts], ignore_index=True)
    # month by week_start (never by ep_final_date: a week's EPs can spill 1-2
    # days into the next month and would fake "next-month attendance")
    att["month"] = pd.to_datetime(att["week_start"]).dt.strftime("%Y-%m")
    nxt = (att.assign(present=att["status"].eq("present"))
              .groupby(["student_key", "month"], as_index=False)["present"].mean())
    nxt["att_next_month_pct"] = (100 * nxt["present"]).round(1)

    def prev_month(m: str) -> str:
        y, mm = map(int, m.split("-"))
        mm -= 1
        if mm == 0:
            y, mm = y - 1, 12
        return f"{y:04d}-{mm:02d}"

    # attach month t+1's attendance to row of month t  (this IS the leak)
    nxt["month"] = nxt["month"].map(prev_month)
    trap = labels[["student_key", "month"]].merge(
        nxt[["student_key", "month", "att_next_month_pct"]],
        on=["student_key", "month"], how="left")
    # fill 0 (= "gone") only where month t+1 is fully observed; keep NaN when
    # t+1 is censored (2569 observed through early Aug -> t=2026-07 stays NaN)
    fully_observed = trap["month"].lt("2026-07") | trap["month"].str.startswith("2025")
    trap.loc[fully_observed, "att_next_month_pct"] = (
        trap.loc[fully_observed, "att_next_month_pct"].fillna(0.0))
    trap.to_csv(OUT / "extra_features.csv", index=False)
    print(f"{'extra_features':20s} {len(trap):7,d} rows  (leakage trap)")

    churn = labels["churned_next_month"]
    print(f"\nchurn rate (labeled months): {churn.mean():.1%}  "
          f"({int(churn.sum())} churns / {churn.notna().sum()} labeled rows)")
    by_m = labels.groupby("month")["churned_next_month"].mean()
    print(by_m.round(3).to_string())


if __name__ == "__main__":
    main()
