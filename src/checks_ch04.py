"""Checks for ch04 — Feature Engineering: ตาราง (นักเรียน, เดือน)

Expected values are recomputed live from DATA_DIR via the canonical
churn_utils.build_features_monthly (no hardcoded magic numbers).
If any input table is missing (e.g. real mode before ch02 is done),
fall back to structure/range checks only.

หมายเหตุ (recompute สด): ค่า expected ของทุกข้อคำนวณสดจาก DATA_DIR ผ่าน
churn_utils (canonical) ทุกครั้งที่รัน — ไม่มีเลข hardcode ที่ผูกกับ seed
จึงใช้ได้ทั้งโหมด sample และ real โดยไม่ต้องแยก branch IS_SAMPLE
(ตาม authoring-guide "checks ที่ recompute จาก DATA_DIR สด")
ถ้าไฟล์วัตถุดิบยังไม่ครบ จะถอยไปเช็คแค่โครงสร้าง/ช่วงค่าแทน
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks import (CheckFail, expect_cols, expect_df,
                        expect_no_leakage_cols, register, sample_path)

_ID = ["student_key", "year", "month"]
_CACHE: dict = {}

_INPUT_FILES = ["labels_monthly.csv", "attendance_long.csv", "exam_attempts.csv",
                "weekly_metrics.csv", "students.csv"]


def _expect_filled(name: str, value):
    if value is None:
        raise CheckFail(f"`{name}` ยังเป็น None อยู่ — เติมช่องว่าง ____ ก่อนนะครับ")


def _tables():
    """โหลดตารางวัตถุดิบทั้ง 5 จาก DATA_DIR (cache) — None ถ้าไฟล์ไม่ครบ"""
    if "tables" in _CACHE:
        return _CACHE["tables"]
    if not all(sample_path(f).exists() for f in _INPUT_FILES):
        _CACHE["tables"] = None
        return None
    labels = pd.read_csv(sample_path("labels_monthly.csv"), dtype={"month": str},
                         parse_dates=["churn_date"])
    att = pd.read_csv(sample_path("attendance_long.csv"),
                      parse_dates=["ep_final_date", "week_start"])
    atm = pd.read_csv(sample_path("exam_attempts.csv"), parse_dates=["submitted_at"])
    wm = pd.read_csv(sample_path("weekly_metrics.csv"),
                     parse_dates=["week_start", "week_end"])
    stu = pd.read_csv(sample_path("students.csv"), parse_dates=["signup_date"])
    _CACHE["tables"] = (labels, att, atm, wm, stu)
    return _CACHE["tables"]


def _canonical() -> pd.DataFrame | None:
    """features_monthly เวอร์ชัน canonical (คำนวณสด + cache) — กรรมการของบทนี้"""
    if "feats" in _CACHE:
        return _CACHE["feats"]
    t = _tables()
    if t is None:
        _CACHE["feats"] = None
        return None
    from src import churn_utils
    _CACHE["feats"] = churn_utils.build_features_monthly(*t)
    return _CACHE["feats"]


def _no_dup_keys(obj: pd.DataFrame, name: str):
    if obj.duplicated(_ID).any():
        raise CheckFail(f"{name} มี (student_key, year, month) ซ้ำ — "
                        f"1 คนต่อเดือนต้องมีแถวเดียว เช็คตอน concat/merge อีกที")


def _expect_rows(obj: pd.DataFrame, exp: pd.DataFrame, name: str):
    if len(obj) < len(exp):
        raise CheckFail(f"{name} มี {len(obj):,} แถว แต่ควรได้ {len(exp):,} — "
                        f"แถวหายไป (ทิ้งแถว NaN หรือ concat ไม่ครบทุกกลุ่มหรือเปล่า?)")
    if len(obj) > len(exp):
        raise CheckFail(f"{name} มี {len(obj):,} แถว แต่ควรได้ {len(exp):,} — "
                        f"แถวบวมขึ้น (merge ด้วย key ไม่ครบ หรือมีแถวซ้ำก่อน merge)")


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if np.isnan(f):
        return "NaN"
    return f"{f:.2f}"


def _expect_match_num(got: pd.DataFrame, exp: pd.DataFrame, col: str,
                      hint: str, tol: float = 0.01):
    """เทียบคอลัมน์ตัวเลขกับ canonical ราย (student, month) — NaN ต้องตรงกันด้วย"""
    if col not in got.columns:
        raise CheckFail(f"ขาดคอลัมน์ `{col}` — เช็คชื่อคอลัมน์อีกที")
    m = (exp[_ID + [col]].rename(columns={col: "_exp"})
         .merge(got[_ID + [col]].rename(columns={col: "_got"}), on=_ID, how="left"))
    e = pd.to_numeric(m["_exp"], errors="coerce")
    g = pd.to_numeric(m["_got"], errors="coerce")
    ok = (e.isna() & g.isna()) | ((e - g).abs() <= tol)
    bad = m[~ok]
    if len(bad):
        r = bad.iloc[0]
        raise CheckFail(
            f"`{col}` ยังไม่ตรง {len(bad):,} แถว (จาก {len(m):,}) — เช่น "
            f"{r['student_key']} เดือน {r['month']}: ได้ {_fmt(r['_got'])} "
            f"ควรได้ {_fmt(r['_exp'])} · {hint}")


def _expect_match_cat(got: pd.DataFrame, exp: pd.DataFrame, col: str):
    if col not in got.columns:
        raise CheckFail(f"ขาดคอลัมน์ `{col}` — merge static จาก students ครบหรือยัง?")
    m = (exp[_ID + [col]].rename(columns={col: "_exp"})
         .merge(got[_ID + [col]].rename(columns={col: "_got"}), on=_ID, how="left"))
    ok = (m["_exp"].isna() & m["_got"].isna()) | (m["_exp"].astype(str) == m["_got"].astype(str))
    bad = m[~ok]
    if len(bad):
        r = bad.iloc[0]
        raise CheckFail(f"`{col}` ไม่ตรง {len(bad):,} แถว — เช่น {r['student_key']}: "
                        f"ได้ {r['_got']!r} ควรได้ {r['_exp']!r} · "
                        f"merge จาก students ด้วย key student_key + how='left' ใช่ไหม?")


# ---------------------------------------------------------------- ex_04_01

@register("ex_04_01")
def _ex_04_01(obj):
    _expect_filled("panel", obj)
    expect_df(obj, "panel")
    need = ["student_key", "year", "month", "churned_next_month", "month_index"]
    expect_cols(obj, need, "panel")
    extra = set(obj.columns) - set(need)
    if extra:
        raise CheckFail(f"panel มีคอลัมน์เกินมา: {sorted(extra)} — "
                        f"เก็บแค่ 4 คอลัมน์หลัก + month_index พอครับ")
    _no_dup_keys(obj, "panel")

    if obj["month_index"].isna().any():
        raise CheckFail("month_index มีค่าว่าง — mapping ครอบคลุมทั้ง 2 ปีหรือยัง? "
                        "(key ต้องเป็นคู่ (year, month) ไม่ใช่ month เดี่ยวๆ)")
    mi = pd.to_numeric(obj["month_index"], errors="coerce")
    if mi.isna().any() or not mi.between(1, 12).all():
        raise CheckFail("month_index ควรเป็นเลขลำดับเดือนของซีซัน (มี.ค. = 1, ...) — "
                        "เจอค่าประหลาด ลองดูวิธีสร้าง mapping อีกที")

    t = _tables()
    if t is None:  # real mode ก่อน ch02 — เช็คโครงสร้างพอ
        return
    labels = t[0]
    exp_rows = int((labels["active"] == 1).sum())
    if len(obj) != exp_rows:
        n_nona = int(labels[labels["active"] == 1]["churned_next_month"].notna().sum())
        if len(obj) == n_nona:
            raise CheckFail(f"ได้ {len(obj):,} แถว = เท่าจำนวนแถวที่ label ไม่เป็น NaN — "
                            f"ห้ามทิ้งแถว censored นะครับ (เดือนล่าสุดคือแถวที่ต้องทำนายจริง!)")
        raise CheckFail(f"จำนวนแถวไม่ตรง: ได้ {len(obj):,} ควรได้ {exp_rows:,} — "
                        f"กรอง active == 1 แล้วหรือยัง?")

    got_nan = int(obj["churned_next_month"].isna().sum())
    exp_nan = int(labels[labels["active"] == 1]["churned_next_month"].isna().sum())
    if got_nan != exp_nan:
        raise CheckFail(f"แถว label NaN เหลือ {got_nan:,} แต่ควรมี {exp_nan:,} — "
                        f"อย่า fillna/dropna คอลัมน์ churned_next_month นะครับ")

    # month_index ต้องนับแยกตามปี — เทียบกับ mapping ที่คำนวณสด
    from src.config import SEASONS, month_seq
    idx_of = {}
    for y, season in SEASONS.items():
        for i, m in enumerate(month_seq(season["start_month"], season["end_month"]), start=1):
            idx_of[(y, m)] = i
    exp_idx = pd.Series([idx_of.get((y, m)) for y, m in zip(obj["year"], obj["month"])],
                        index=obj.index, dtype="float")
    if (pd.to_numeric(obj["month_index"]) != exp_idx).any():
        r = obj[pd.to_numeric(obj["month_index"]) != exp_idx].iloc[0]
        raise CheckFail(f"month_index ยังไม่ถูก — เช่น ปี {r['year']} เดือน {r['month']} "
                        f"ได้ {r['month_index']} · นับจากเดือนแรกของซีซัน 'ปีนั้นๆ' "
                        f"(2568 เริ่ม 2025-03, 2569 เริ่ม 2026-03) นะครับ")


# ---------------------------------------------------------------- ex_04_02

@register("ex_04_02")
def _ex_04_02(obj):
    _expect_filled("panel_att", obj)
    expect_df(obj, "panel_att")
    expect_cols(obj, _ID + ["att_month_pct", "att_cum_pct", "att_delta"], "panel_att")
    _no_dup_keys(obj, "panel_att")

    exp = _canonical()
    if exp is None:  # structure/range only
        v = obj["att_month_pct"].dropna()
        if len(v) and not v.between(0, 100).all():
            raise CheckFail("att_month_pct ควรอยู่ในช่วง 0–100")
        return
    _expect_rows(obj, exp, "panel_att")
    _expect_match_num(obj, exp, "att_month_pct",
                      "คิดจากแถวของเดือน t เท่านั้น (หลัง cutoff แล้วกรอง month == เดือนนั้น)")
    _expect_match_num(obj, exp, "att_cum_pct",
                      "สะสม = ทุกแถวใน att_cut (ผ่าน cutoff แล้ว ไม่ต้องกรองเดือนซ้ำ) — "
                      "ถ้าค่าเท่ากันทุกเดือนของคนเดียวกัน แปลว่าลืม cutoff!")
    _expect_match_num(obj, exp, "att_delta",
                      "delta = เดือนนี้ − เดือนก่อนหน้า (str(pd.Period(month) - 1)) "
                      "และถ้าฝั่งใดไม่มีคลาสต้องปล่อยเป็น NaN")


# ---------------------------------------------------------------- ex_04_03

@register("ex_04_03")
def _ex_04_03(obj):
    _expect_filled("panel_exam", obj)
    expect_df(obj, "panel_exam")
    expect_cols(obj, _ID + ["exam_avg_score", "new_attempts_month"], "panel_exam")
    _no_dup_keys(obj, "panel_exam")

    if obj["new_attempts_month"].isna().any():
        raise CheckFail("new_attempts_month มี NaN — คนไม่มี attempt ต้องเป็น 0 "
                        "(.fillna(0).astype(int) หลัง .map())")

    exp = _canonical()
    if exp is None:
        v = obj["exam_avg_score"].dropna()
        if len(v) and not v.between(0, 100).all():
            raise CheckFail("exam_avg_score ควรอยู่ในช่วง 0–100")
        return
    _expect_rows(obj, exp, "panel_exam")
    _expect_match_num(obj, exp, "exam_avg_score",
                      "เฉลี่ยสะสมจาก atm_cut ทั้งก้อน (cutoff ด้วย submitted_at) — "
                      "คนไม่มี attempt ปล่อย NaN ไว้ อย่าถม 0")
    _expect_match_num(obj, exp, "new_attempts_month",
                      "นับเฉพาะ attempt ที่ submitted_at >= วันแรกของเดือน t "
                      "(pd.Timestamp(month + '-01'))")


# ---------------------------------------------------------------- ex_04_04

@register("ex_04_04")
def _ex_04_04(obj):
    _expect_filled("panel_eng", obj)
    expect_df(obj, "panel_eng")
    expect_cols(obj, _ID + ["max_silent_weeks", "streak_weeks",
                            "practice_pct", "checkpoint_pct"], "panel_eng")
    _no_dup_keys(obj, "panel_eng")

    for col in ("max_silent_weeks", "streak_weeks"):
        if obj[col].isna().any():
            raise CheckFail(f"{col} มี NaN — เดือนที่ไม่มีข้อมูล weekly ต้องเป็น 0 "
                            f"(.fillna(0).astype(int))")

    exp = _canonical()
    if exp is None:
        return
    _expect_rows(obj, exp, "panel_eng")
    _expect_match_num(obj, exp, "max_silent_weeks",
                      "ค่า max ของ silent_weeks 'ในเดือน t' (groupby แล้ว .max())")
    _expect_match_num(obj, exp, "streak_weeks",
                      "snapshot สัปดาห์สุดท้ายของเดือน — sort ด้วย week_start ก่อนค่อย "
                      "drop_duplicates(keep='last') นะครับ")
    _expect_match_num(obj, exp, "practice_pct",
                      "snapshot สัปดาห์สุดท้ายของเดือน — ไม่ใช่ค่าเฉลี่ยทั้งเดือน "
                      "และไม่ต้อง fillna")
    _expect_match_num(obj, exp, "checkpoint_pct",
                      "snapshot สัปดาห์สุดท้ายของเดือน — ไม่ใช่ค่าเฉลี่ยทั้งเดือน "
                      "และไม่ต้อง fillna")


# ---------------------------------------------------------------- ex_04_05

_NUM_COLS_0405 = ["att_month_pct", "att_cum_pct", "att_delta", "practice_pct",
                  "checkpoint_pct", "exam_avg_score", "new_attempts_month",
                  "max_silent_weeks", "streak_weeks", "month_index",
                  "signup_lateness", "n_subjects", "months_enrolled",
                  "churned_next_month"]
_CAT_COLS_0405 = ["grade", "live_or_replay", "old_new"]


@register("ex_04_05")
def _ex_04_05(obj):
    _expect_filled("features", obj)
    expect_df(obj, "features")
    expect_no_leakage_cols(obj)
    _no_dup_keys(obj, "features")

    exp = _canonical()
    if exp is None:  # structure only
        expect_cols(obj, _ID + _CAT_COLS_0405 + ["months_enrolled"], "features")
        if (pd.to_numeric(obj["months_enrolled"], errors="coerce") < 1).any():
            raise CheckFail("months_enrolled ต้องขั้นต่ำ 1 — ใช้ .clip(lower=1)")
        return

    missing = set(exp.columns) - set(obj.columns)
    if missing:
        raise CheckFail(f"features ขาดคอลัมน์ {sorted(missing)} — "
                        f"merge ครบทั้ง 3 panel + static + months_enrolled หรือยัง?")
    extra = set(obj.columns) - set(exp.columns)
    if extra:
        raise CheckFail(f"features มีคอลัมน์เกิน: {sorted(extra)} — "
                        f"ถ้าเจอ _x/_y แปลว่า merge โดยไม่ drop คอลัมน์ซ้ำก่อน")
    _expect_rows(obj, exp, "features")

    if (pd.to_numeric(obj["months_enrolled"], errors="coerce") < 1).any():
        raise CheckFail("months_enrolled มีค่าต่ำกว่า 1 — เด็กที่สมัครเดือนนั้นพอดีนับเป็น 1 "
                        "(.clip(lower=1))")
    for col in _NUM_COLS_0405:
        _expect_match_num(obj, exp, col,
                          "เทียบกับ churn_utils.build_features_monthly (กรรมการ) แล้วยังเพี้ยน — "
                          "ไล่ดูข้อที่สร้างคอลัมน์นี้อีกครั้ง")
    for col in _CAT_COLS_0405:
        _expect_match_cat(obj, exp, col)


# ---------------------------------------------------------------- ex_04_06

@register("ex_04_06")
def _ex_04_06(obj):
    _expect_filled("features_clean", obj)
    expect_df(obj, "features_clean")
    if "att_next_month_pct" in obj.columns:
        raise CheckFail("ยังเจอ att_next_month_pct อยู่ในตาราง! — คอลัมน์นี้คือข้อมูล "
                        "เดือน t+1 (อนาคต) ต้อง drop ทิ้งสถานเดียว")
    expect_no_leakage_cols(obj)
    _no_dup_keys(obj, "features_clean")

    exp = _canonical()
    if exp is None:
        return
    missing = set(exp.columns) - set(obj.columns)
    if missing:
        raise CheckFail(f"เผลอ drop เกินไป — คอลัมน์ดีๆ หายไปด้วย: {sorted(missing)} "
                        f"(ประหารเฉพาะคอลัมน์จากอนาคตพอ)")
    if len(obj) != len(exp):
        raise CheckFail(f"จำนวนแถวเปลี่ยนจาก {len(exp):,} เป็น {len(obj):,} — "
                        f"โจทย์ให้ drop 'คอลัมน์' ไม่ใช่ drop แถวนะครับ")


# ---------------------------------------------------------------- ex_04_07

@register("ex_04_07")
def _ex_04_07(obj):
    _expect_filled("features_final", obj)
    expect_df(obj, "features_final")
    from src import contracts
    try:
        contracts.validate_df("features_monthly", obj)
    except AssertionError as e:
        raise CheckFail(f"contract ยังไม่ผ่าน: {e}")

    out_path = sample_path("features_monthly.csv")
    if not out_path.exists():
        raise CheckFail("ยังไม่เจอไฟล์ features_monthly.csv ใน DATA_DIR — "
                        "สั่ง to_csv(DATA_DIR / 'features_monthly.csv', index=False) หรือยัง?")
    saved = pd.read_csv(out_path, dtype={"month": str})
    if "Unnamed: 0" in saved.columns:
        raise CheckFail("ไฟล์ที่เซฟมีคอลัมน์ผี 'Unnamed: 0' — ลืม index=False ตอน to_csv ครับ")
    if len(saved) != len(obj):
        raise CheckFail(f"ไฟล์ที่เซฟมี {len(saved):,} แถว แต่ features_final มี "
                        f"{len(obj):,} — เซฟเวอร์ชันล่าสุดทับอีกรอบนะครับ")
    missing = set(obj.columns) - set(saved.columns)
    if missing:
        raise CheckFail(f"ไฟล์ที่เซฟขาดคอลัมน์ {sorted(missing)} — เซฟทับใหม่อีกรอบครับ")

    exp = _canonical()
    if exp is not None and len(obj) != len(exp):
        raise CheckFail(f"features_final มี {len(obj):,} แถว แต่ควรได้ {len(exp):,} — "
                        f"ใช้ features_clean (หรือ features_clean_ok) เต็มตารางนะครับ")
