"""DataCamp-style answer checking.

In a notebook, the last line of every exercise cell is:

    from src import checks
    checks.check("ex_04_02", features)

Prints "ผ่าน ✓" on success, or a Thai diagnostic telling the learner what to
look at (never the full answer — hints live in the notebook's คำใบ้ blocks).

Chapter check modules (src/checks_ch00.py .. src/checks_ch08.py) register
validators with the @register decorator:

    from src.checks import register, expect_cols, expect_close

    @register("ex_01_02")
    def _(obj):
        expect_cols(obj, ["student_key", "month", "churned_next_month"])
        ...

Conventions for check authors:
  - Recompute expected values live from data/sample via src.churn_utils
    (canonical functions), don't hardcode magic numbers.
  - When src.config.IS_SAMPLE is False (real data), only check structure/ranges.
  - Raise CheckFail(thai_message) to fail with guidance.
"""
from __future__ import annotations

import importlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

_REGISTRY: dict[str, callable] = {}
_LOADED = False


class CheckFail(Exception):
    """Raise with a Thai message explaining what's wrong (not the answer)."""


def register(ex_id: str):
    def deco(fn):
        _REGISTRY[ex_id] = fn
        return fn
    return deco


def _load_all():
    global _LOADED
    if _LOADED:
        return
    here = Path(__file__).parent
    for mod in sorted(here.glob("checks_ch*.py")):
        importlib.import_module(f"src.{mod.stem}")
    _LOADED = True


def check(ex_id: str, obj=None) -> bool:
    """Validate an exercise answer. Returns True/False and prints feedback."""
    _load_all()
    if ex_id not in _REGISTRY:
        print(f"✗ ไม่รู้จักแบบฝึกหัด '{ex_id}' — เช็คว่าพิมพ์ id ถูกไหม")
        return False
    try:
        _REGISTRY[ex_id](obj)
    except CheckFail as e:
        print(f"✗ ยังไม่ผ่าน: {e}")
        return False
    except AssertionError as e:
        print(f"✗ ยังไม่ผ่าน: {e}")
        return False
    except Exception as e:  # learner passed something unexpected
        print(f"✗ เช็คไม่ได้ ({type(e).__name__}: {e}) — ดูว่าส่งตัวแปรถูกตัวไหม")
        traceback.print_exc(limit=1)
        return False
    print(f"ผ่าน ✓  ({ex_id}) เยี่ยมมากครับ 🎉")
    return True


# ----------------------------------------------------------- helper asserts

def expect_df(obj, name="ผลลัพธ์"):
    if not isinstance(obj, pd.DataFrame):
        raise CheckFail(f"{name} ต้องเป็น DataFrame แต่ได้ {type(obj).__name__}")
    return obj


def expect_cols(obj, cols, name="ตาราง"):
    expect_df(obj, name)
    missing = [c for c in cols if c not in obj.columns]
    if missing:
        raise CheckFail(f"{name} ขาดคอลัมน์ {missing} — เช็คชื่อคอลัมน์อีกที")


def expect_shape(obj, n_rows=None, name="ตาราง"):
    expect_df(obj, name)
    if n_rows is not None and len(obj) != n_rows:
        raise CheckFail(f"{name} มี {len(obj):,} แถว แต่ควรได้ {n_rows:,} แถว "
                        f"— ลองดูเงื่อนไขการกรอง/join อีกครั้ง")


def expect_close(value, expected, tol=1e-6, name="ค่า"):
    v, e = float(value), float(expected)
    if np.isnan(e):
        if not np.isnan(v):
            raise CheckFail(f"{name} ควรเป็น NaN แต่ได้ {v}")
        return
    if np.isnan(v) or abs(v - e) > tol:
        raise CheckFail(f"{name} = {v:.4f} แต่ค่าที่ถูกคือประมาณ {e:.4f}")


def expect_between(value, lo, hi, name="ค่า"):
    v = float(value)
    if not (lo <= v <= hi):
        raise CheckFail(f"{name} = {v:.4f} อยู่นอกช่วงที่คาดไว้ [{lo}, {hi}]")


def expect_no_leakage_cols(obj):
    from src.contracts import FORBIDDEN_FEATURE_COLS
    hit = FORBIDDEN_FEATURE_COLS & set(obj.columns)
    if hit:
        raise CheckFail(f"เจอคอลัมน์จากอนาคต (leakage!): {hit} — "
                        f"feature ต้องใช้ข้อมูล ≤ เดือน t เท่านั้น")


def sample_path(filename: str) -> Path:
    from src.config import DATA_DIR
    return DATA_DIR / filename
