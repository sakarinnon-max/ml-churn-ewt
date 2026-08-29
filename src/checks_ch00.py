"""Checks for ch00 — ติดตั้งเครื่องมือ + ทัวร์ข้อมูล

หมายเหตุ (convention กลาง): ค่า expected ทุกตัวคำนวณสดจาก DATA_DIR
(ผ่าน sample_path ซึ่งชี้ตาม ML_CHURN_DATA) ไม่ hardcode ตัวเลข —
จึงใช้ได้ทุกโหมด (SAMPLE/REAL) โดยไม่ต้องแยก branch IS_SAMPLE
ถ้าไฟล์ weekly_metrics.csv ยังไม่มีใน DATA_DIR (เช่น real mode ก่อนทำบท 02)
จะ fallback ไปเช็คเฉพาะโครงสร้าง/ช่วงค่าแทน
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks import CheckFail, expect_cols, expect_df, register, sample_path


def _load_wm() -> pd.DataFrame | None:
    p = sample_path("weekly_metrics.csv")
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["week_start", "week_end"])


def _expect_filled(name: str, value):
    if value is None:
        raise CheckFail(f"`{name}` ยังเป็น None อยู่ — เติมช่องว่าง ____ ก่อนนะครับ")


# ---------------------------------------------------------------- ex_00_01

@register("ex_00_01")
def _ex_00_01(obj):
    if not (isinstance(obj, tuple) and len(obj) == 3):
        raise CheckFail("ต้องส่งครบ 3 ตัวเป็น (wm, n_students, n_weeks) — "
                        "อย่าแก้บรรทัด checks.check ท้าย cell นะครับ")
    wm, n_students, n_weeks = obj

    _expect_filled("wm", wm)
    expect_df(wm, "wm")
    expect_cols(wm, ["student_key", "year", "week_start", "week_end", "att_week_pct", "tier"], "wm")
    if not pd.api.types.is_datetime64_any_dtype(wm["week_start"]):
        raise CheckFail("คอลัมน์ week_start ยังเป็นข้อความอยู่ — "
                        "ใส่ parse_dates=[...] ตอน read_csv หรือยัง?")

    _expect_filled("n_students", n_students)
    _expect_filled("n_weeks", n_weeks)
    for name, v in [("n_students", n_students), ("n_weeks", n_weeks)]:
        if isinstance(v, (pd.Series, pd.DataFrame, np.ndarray, list)):
            raise CheckFail(f"`{name}` ควรเป็นตัวเลขตัวเดียว — "
                            f"ใช้ .nunique() (ไม่ใช่ .unique() ที่คืนทั้งลิสต์)")

    exp = _load_wm()
    if exp is None:  # real mode, structure only
        return
    if len(wm) != len(exp):
        raise CheckFail(f"wm มี {len(wm):,} แถว แต่ไฟล์จริงมี {len(exp):,} แถว — "
                        f"โหลดครบทั้งไฟล์หรือเปล่า? (ไม่ต้องกรองอะไรในข้อนี้)")
    if int(n_students) != exp["student_key"].nunique():
        raise CheckFail("จำนวนนักเรียนยังไม่ตรง — ต้องนับแบบ 'ไม่ซ้ำ' "
                        "จากคอลัมน์ student_key ด้วย .nunique() (ไม่ใช่ len หรือ count)")
    if int(n_weeks) != exp["week_start"].nunique():
        raise CheckFail("จำนวนสัปดาห์ยังไม่ตรง — นับค่าไม่ซ้ำของคอลัมน์ week_start นะครับ")


# ---------------------------------------------------------------- ex_00_02

@register("ex_00_02")
def _ex_00_02(obj):
    _expect_filled("tier_counts", obj)
    if isinstance(obj, pd.DataFrame):
        raise CheckFail("ได้ DataFrame มา — ให้ .value_counts() บนคอลัมน์เดียว "
                        "เช่น last_week['tier'] จะได้ Series")
    if not isinstance(obj, pd.Series):
        raise CheckFail(f"tier_counts ควรเป็น Series (ผลจาก value_counts) "
                        f"แต่ได้ {type(obj).__name__}")

    exp_df = _load_wm()
    if exp_df is None:  # real mode, structure only
        if int(obj.sum()) <= 0:
            raise CheckFail("ผลรวมเป็น 0 — น่าจะยังนับไม่ถูก ลองไล่ทีละขั้นดูครับ")
        return

    w69 = exp_df[exp_df["year"] == 2569]
    exp = (w69.sort_values("week_start")
              .groupby("student_key").last()["tier"].value_counts())
    got = {str(k): int(v) for k, v in obj.items()}
    want = {str(k): int(v) for k, v in exp.items()}
    if got == want:
        return

    n69 = w69["student_key"].nunique()
    n_all = exp_df["student_key"].nunique()
    total = sum(got.values())
    if total == n_all and total != n69:
        raise CheckFail(f"รวมทุก tier ได้ {total} คน = นักเรียนทั้ง 2 ซีซัน — "
                        f"ลืมกรอง year == 2569 หรือเปล่า?")
    if total == len(w69):
        raise CheckFail("ตอนนี้นับทุกแถว (ทุกสัปดาห์) อยู่ — ต้องหยิบเฉพาะสัปดาห์ล่าสุด"
                        "ของแต่ละคนก่อน (sort_values → groupby → .last())")
    if total != n69:
        raise CheckFail(f"รวมทุก tier ได้ {total} แต่นักเรียนปี 2569 มี {n69} คน — "
                        f"เช็คลำดับ กรองปี → เรียงเวลา → groupby → last อีกรอบ")
    raise CheckFail("จำนวนรวมถูกแล้ว แต่สัดส่วนสียังไม่ตรง — "
                    "เรียงด้วย sort_values('week_start') ก่อนค่อย .last() หรือเปล่า?")


# ---------------------------------------------------------------- ex_00_03

@register("ex_00_03")
def _ex_00_03(obj):
    _expect_filled("weekly_avg", obj)
    if isinstance(obj, pd.DataFrame):
        raise CheckFail("ได้ DataFrame มา — เลือกคอลัมน์เดียวตอน groupby เช่น "
                        ".groupby('week_start')['att_week_pct'].mean() จะได้ Series")
    if not isinstance(obj, pd.Series):
        raise CheckFail(f"weekly_avg ควรเป็น Series แต่ได้ {type(obj).__name__}")

    exp_df = _load_wm()
    if exp_df is None:  # real mode, structure/range only
        if not pd.api.types.is_datetime64_any_dtype(obj.index):
            raise CheckFail("index ควรเป็นวันที่ (week_start ที่ parse แล้ว)")
        vals = obj.dropna()
        if len(vals) and not vals.between(0, 100).all():
            raise CheckFail("มีค่านอกช่วง 0–100 — ค่าเฉลี่ยของ % ต้องอยู่ในช่วงนี้")
        return

    w69 = exp_df[exp_df["year"] == 2569]
    exp = w69.groupby("week_start")["att_week_pct"].mean()

    if len(obj) == exp_df["week_start"].nunique() and len(obj) != len(exp):
        raise CheckFail(f"ได้ {len(obj)} สัปดาห์ = รวมทั้ง 2 ซีซัน — "
                        f"ลืมกรอง year == 2569 หรือเปล่า?")
    if len(obj) != len(exp):
        raise CheckFail(f"จำนวนสัปดาห์ไม่ตรง: ได้ {len(obj)} แต่ควรได้ {len(exp)} — "
                        f"groupby ด้วย week_start ใช่ไหม?")
    if not pd.api.types.is_datetime64_any_dtype(obj.index):
        raise CheckFail("index ควรเป็นวันที่ — เช็คว่าโหลดไฟล์ด้วย parse_dates "
                        "แล้ว groupby('week_start')")

    got = obj.sort_index()
    want = exp.sort_index()
    if list(got.index) != list(want.index):
        raise CheckFail("สัปดาห์ใน index ไม่ตรง — ใช้ week_start (ไม่ใช่ week_end) "
                        "เป็นตัว groupby นะครับ")
    diff = np.abs(got.to_numpy(dtype=float) - want.to_numpy(dtype=float))
    if np.nanmax(diff) > 0.01:
        raise CheckFail("จำนวนสัปดาห์ถูกแล้ว แต่ค่าเฉลี่ยยังไม่ตรง — "
                        "ใช้ .mean() ของคอลัมน์ att_week_pct ตรงๆ ใช่ไหม? "
                        "(อย่า fillna(0) ก่อนเฉลี่ยนะ สัปดาห์ไม่มีคลาสต้องเป็น NaN)")
