"""Data contracts: one schema for both years (2568 sheets + 2569 Supabase).

validate_df(name, df) raises AssertionError with a Thai message on violation,
so notebooks can call it as a cheap "did I build this right?" gate.

Canonical key: student_key
  - 2569: Supabase profile uuid
  - 2568: "2568-<รหัสนักเรียน>"
students.csv carries same_person_key to link returning students across years.
"""
from __future__ import annotations

import pandas as pd

# name -> {column: kind}; kind in {"str","int","float","date","month","datetime"}
SCHEMAS: dict[str, dict[str, str]] = {
    "students": {
        "student_key": "str",
        "year": "int",              # 2568 | 2569 (Thai Buddhist year of the season)
        "student_code": "str",
        "display_name": "str",
        "grade": "str",             # ม.1 | ม.2 | ม.3 | ม.4 | อื่นๆ
        "school": "str",
        "target_school": "str",     # MWIT | KVIS | TU | อื่นๆ
        "live_or_replay": "str",    # สด | เทป | ผสม
        "old_new": "str",           # old | new
        "signup_date": "date",
        "signup_lateness": "int",   # months after season start (0 = tan Mar)
        "n_subjects": "int",
        "subject_ids": "str",       # ";"-joined, e.g. "1;4;5;6"
        "same_person_key": "str",   # blank if not matched across years
    },
    "attendance_long": {
        "student_key": "str",
        "year": "int",
        "subject_id": "int",
        "episode_number": "int",
        "ep_final_date": "date",    # last taught date of the EP (2-round merge applied)
        "week_start": "date",
        "status": "str",            # present | absent | leave
    },
    "exams": {
        "exam_id": "str",
        "year": "int",
        "subject_id": "int",
        "exam_type": "str",         # practice | checkpoint
        "chapter": "str",
        "total_questions": "int",
    },
    "exam_attempts": {
        "student_key": "str",
        "year": "int",
        "exam_id": "str",
        "subject_id": "int",
        "exam_type": "str",
        "submitted_at": "datetime",
        "percentage": "float",
        "passed": "int",            # 0|1
    },
    "enrollment_events": {
        "student_key": "str",
        "year": "int",
        "event_type": "str",        # enroll | re_enroll | cancel | subject_drop
        "event_date": "date",
        "event_month": "month",
        "subject_ids": "str",
        "source": "str",            # supabase | label_sheet | line | memory
        "confidence": "str",        # high | medium | low
    },
    "labels_monthly": {
        "student_key": "str",
        "year": "int",
        "month": "month",
        "active": "int",                 # 1 = enrolled-confirmed at month start
        "churned_next_month": "float",   # 1.0 | 0.0 | NaN (censored)
        "churn_date": "date",            # NaT unless churned
        "label_source": "str",
    },
    "weekly_metrics": {
        "student_key": "str",
        "year": "int",
        "week_start": "date",
        "week_end": "date",
        "att_week_pct": "float",
        "att_cum_pct": "float",
        "practice_done": "int",
        "practice_total": "int",
        "practice_pct": "float",
        "practice_avg_score": "float",
        "checkpoint_done": "int",
        "checkpoint_total": "int",
        "checkpoint_pct": "float",
        "checkpoint_avg_score": "float",
        "silent_weeks": "int",
        "streak_weeks": "int",
        "tier": "str",              # green | yellow | orange | red
    },
    "features_monthly": {
        # Built by the CEO in ch04. Feature columns beyond these are allowed;
        # validate_df checks the required identity/label columns + no-leakage names.
        "student_key": "str",
        "year": "int",
        "month": "month",
        "churned_next_month": "float",
    },
}

# Column names that must NEVER appear in a feature table (leakage guard).
FORBIDDEN_FEATURE_COLS = {
    "churn_date", "active_next_month", "att_next_month_pct", "label_source",
}

_MONTH_RE = r"^\d{4}-\d{2}$"


def _fail(msg: str):
    raise AssertionError(msg)


def validate_df(name: str, df: pd.DataFrame) -> bool:
    """Validate df against the contract. Returns True or raises AssertionError (Thai)."""
    if name not in SCHEMAS:
        _fail(f"ไม่รู้จักตาราง '{name}' — ตารางที่มี: {sorted(SCHEMAS)}")
    schema = SCHEMAS[name]
    missing = [c for c in schema if c not in df.columns]
    if missing:
        _fail(f"[{name}] ขาดคอลัมน์: {missing}")
    if len(df) == 0:
        _fail(f"[{name}] ตารางว่างเปล่า (0 แถว)")

    for col, kind in schema.items():
        s = df[col]
        if kind == "int":
            if not pd.api.types.is_integer_dtype(s):
                _fail(f"[{name}] คอลัมน์ '{col}' ควรเป็นจำนวนเต็ม แต่เป็น {s.dtype} — ลอง .astype(int)")
        elif kind == "float":
            if not pd.api.types.is_numeric_dtype(s):
                _fail(f"[{name}] คอลัมน์ '{col}' ควรเป็นตัวเลข แต่เป็น {s.dtype}")
        elif kind in ("date", "datetime"):
            if not pd.api.types.is_datetime64_any_dtype(s):
                _fail(f"[{name}] คอลัมน์ '{col}' ควรเป็น datetime — ลอง pd.to_datetime(...)")
        elif kind == "month":
            bad = s.dropna()[~s.dropna().astype(str).str.match(_MONTH_RE)]
            if len(bad):
                _fail(f"[{name}] คอลัมน์ '{col}' ต้องเป็นรูปแบบ 'YYYY-MM' — เจอค่าเช่น {bad.iloc[0]!r}")

    # Table-specific rules
    if name == "labels_monthly":
        if df.duplicated(["student_key", "month"]).any():
            _fail("[labels_monthly] มี (student_key, month) ซ้ำ — 1 คนต่อเดือนต้องมีแถวเดียว")
        vals = set(df["churned_next_month"].dropna().unique())
        if not vals <= {0.0, 1.0}:
            _fail(f"[labels_monthly] churned_next_month ต้องเป็น 0/1/NaN เท่านั้น เจอ {vals}")
        sep = df[df["month"].astype(str).str.endswith("-09")]
        if len(sep) and sep["churned_next_month"].notna().any():
            _fail("[labels_monthly] เดือน ก.ย. คือจบซีซัน ต้อง censored (NaN) เสมอ — ห้ามมี label")
    if name == "attendance_long":
        bad = set(df["status"].unique()) - {"present", "absent", "leave"}
        if bad:
            _fail(f"[attendance_long] status แปลกปลอม: {bad}")
        if df.duplicated(["student_key", "subject_id", "episode_number", "year"]).any():
            _fail("[attendance_long] มี (student, subject, EP) ซ้ำ — EP สอน 2 รอบต้อง merge เหลือแถวเดียว")
    if name == "exam_attempts":
        if df.duplicated(["student_key", "exam_id"]).any():
            _fail("[exam_attempts] มี (student, exam_id) ซ้ำ — ระบบไม่มี retake ต้อง distinct exam_id")
    if name == "features_monthly":
        hit = FORBIDDEN_FEATURE_COLS & set(df.columns)
        if hit:
            _fail(f"[features_monthly] พบคอลัมน์ต้องห้าม (leakage!): {hit}")
    return True
