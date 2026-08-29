"""Central config for the ml-churn project.

Every notebook starts with:

    from src.config import DATA_DIR, IS_SAMPLE

Data mode is controlled by the ML_CHURN_DATA env var:
  - unset or "sample"  -> data/sample     (synthetic, default: notebooks always run)
  - "real"             -> data/processed  (real EWT data, after ch02 is done)
  - any absolute path  -> that path
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_mode = os.environ.get("ML_CHURN_DATA", "sample")
if _mode == "sample":
    DATA_DIR = PROJECT_ROOT / "data" / "sample"
elif _mode == "real":
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
else:
    DATA_DIR = Path(_mode)

IS_SAMPLE = DATA_DIR == PROJECT_ROOT / "data" / "sample"

RAW_2568_DIR = PROJECT_ROOT / "data" / "raw" / "2568"
RAW_2569_DIR = PROJECT_ROOT / "data" / "raw" / "2569_supabase"
LABELS_DIR = PROJECT_ROOT / "data" / "raw" / "labels"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Season boundaries (คอร์สติวเข้ม runs Mar–Sep). Months are "YYYY-MM" strings.
SEASONS = {
    2568: {"start_month": "2025-03", "end_month": "2025-09"},
    2569: {"start_month": "2026-03", "end_month": "2026-09"},
}

# Subject ids (mirrors pm-webapp/report_engine.py). Only {1,4,5,6} have exams.
SUBJECT_NAMES = {
    1: "คณิตศาสตร์",
    2: "ภาษาอังกฤษ",
    3: "วิทย์กายภาพ",
    4: "ฟิสิกส์",
    5: "เคมี",
    6: "ชีววิทยา",
}
EXAM_SUBJECTS = [1, 4, 5, 6]


def month_seq(start: str, end: str) -> list[str]:
    """List of "YYYY-MM" from start to end inclusive. No external deps."""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def next_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    m += 1
    if m == 13:
        y, m = y + 1, 1
    return f"{y:04d}-{m:02d}"
