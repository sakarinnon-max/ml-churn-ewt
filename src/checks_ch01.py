"""Checks for ch01 — นิยามปัญหา + สร้าง label

- ex_01_01..03 ใช้ toy 6 คนที่ fix ไว้ในสมุด (ไม่ขึ้นกับ seed/data mode)
  → คำนวณ expected จาก interval ที่นิยามไว้ตรงนี้ ผ่าน month_seq
- ex_01_04..05 คำนวณ expected สดๆ จาก DATA_DIR ด้วย canonical
  churn_utils.build_labels_monthly · ถ้าไม่ใช่ sample เช็คแค่โครงสร้าง/ช่วงค่า
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks import CheckFail, expect_cols, expect_df, register, sample_path
from src.config import IS_SAMPLE, month_seq

# ---------------------------------------------------------------- toy truth
# ตรงกับเรื่องราว 6 คนใน notebook (subject_drop ของ D ไม่ปิด interval ·
# cancel ก.ย. ของ E = จบซีซันปกติ)

_SEASON_END = "2026-09"
_KNOWN_THROUGH = "2026-07"
_TOY_INTERVALS = {
    "A": ("2026-03", "2026-09"),
    "B": ("2026-03", "2026-05"),
    "C": ("2026-06", "2026-08"),
    "D": ("2026-03", "2026-09"),
    "E": ("2026-03", "2026-09"),
    "F": ("2026-04", "2026-09"),
}
_TOY_NAIVE_ONES = {("B", "2026-05"), ("C", "2026-08"), ("E", "2026-09")}


def _toy_pairs() -> set[tuple[str, str]]:
    return {(sk, m) for sk, (s, e) in _TOY_INTERVALS.items() for m in month_seq(s, e)}


def _pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["student_key"].astype(str), df["month"].astype(str)))


def _expect_filled(name: str, value):
    if value is None:
        raise CheckFail(f"`{name}` ยังเป็น None อยู่ — เติมช่องว่าง ____ ก่อนนะครับ")


def _load_events() -> pd.DataFrame | None:
    p = sample_path("enrollment_events.csv")
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["event_date"])


def _canonical_labels(ev: pd.DataFrame) -> pd.DataFrame:
    from src import churn_utils
    return pd.concat([
        churn_utils.build_labels_monthly(ev, year=2568),
        churn_utils.build_labels_monthly(ev, year=2569, known_through="2026-07"),
    ], ignore_index=True)


# ---------------------------------------------------------------- ex_01_01

@register("ex_01_01")
def _ex_01_01(obj):
    _expect_filled("active_months", obj)
    expect_df(obj, "active_months")
    expect_cols(obj, ["student_key", "month"], "active_months")
    if len(obj) == 0:
        raise CheckFail("ตารางยังว่างอยู่ (0 แถว) — TODO 3 ต้องลบ # ออกแล้ว "
                        "append เข้า rows ในลูปด้วยนะครับ")
    if obj.duplicated(["student_key", "month"]).any():
        raise CheckFail("มี (student_key, month) ซ้ำ — append ซ้ำสองรอบ "
                        "หรือวนลูปซ้อนผิดชั้นหรือเปล่า?")

    got, want = _pairs(obj), _toy_pairs()
    if got == want:
        return

    d_months = {m for s, m in got if s == "D"}
    if 0 < len(d_months) < 7:
        raise CheckFail("น้อง D ควรได้ 7 เดือนเต็มซีซัน — subject_drop ถูกนับเป็น"
                        "การยกเลิกหรือเปล่า? (ต้องปิด interval ด้วยแถว 'cancel' เท่านั้น)")
    b_months = {m for s, m in got if s == "B"}
    if any(m > "2026-05" for m in b_months):
        raise CheckFail("น้อง B ยกเลิกมีผลสิ้น พ.ค. แต่มีเดือนหลังจากนั้นโผล่มา — "
                        "end ต้องมาจาก event_month ของแถว cancel (ไม่ใช่ SEASON_END เสมอไป)")
    f_months = {m for s, m in got if s == "F"}
    if "2026-03" in f_months:
        raise CheckFail("น้อง F สมัคร เม.ย. แต่มีแถว มี.ค. — start ต้องมาจาก "
                        "event_month ของแถว enroll ของแต่ละคน")
    missing, extra = want - got, got - want
    raise CheckFail(f"เดือนยังไม่ตรง: ขาด {len(missing)} คู่ / เกิน {len(extra)} คู่ "
                    f"(เป้าหมาย 33 แถว ตอนนี้ได้ {len(obj)}) — "
                    f"ไล่ดู start/end ของแต่ละคน แล้วอย่าลืมว่า month_seq รวมหัวท้าย")


# ---------------------------------------------------------------- ex_01_02

@register("ex_01_02")
def _ex_01_02(obj):
    _expect_filled("toy_labels_naive", obj)
    expect_df(obj, "toy_labels_naive")
    expect_cols(obj, ["student_key", "month", "churned_next_month"], "toy_labels_naive")
    col = obj["churned_next_month"]
    if col.isna().all():
        raise CheckFail("คอลัมน์ churned_next_month ยังว่างอยู่ — เติม TODO 2 "
                        "(เทียบ month กับผล .map(cancel_month)) ก่อนนะครับ")
    if len(obj) != 33:
        raise CheckFail(f"ได้ {len(obj)} แถว แต่ควรได้ 33 — ตาราง active_months "
                        f"จากข้อ 1.1 ต้องถูกก่อน ค่อยมาติด label ข้อนี้")
    if pd.api.types.is_bool_dtype(col):
        raise CheckFail("ได้ True/False มา — เกือบแล้ว! แปลงเป็นเลข 0/1 "
                        "ด้วย .astype(int) ต่อท้ายอีกนิดครับ")
    if not pd.api.types.is_integer_dtype(col):
        raise CheckFail(f"dtype ของ churned_next_month เป็น {col.dtype} — "
                        f"ข้อนี้ (ฉบับ naive) ต้องเป็นเลขจำนวนเต็ม 0/1")
    if not set(col.unique()) <= {0, 1}:
        raise CheckFail(f"ค่าต้องมีแค่ 0 กับ 1 แต่เจอ {sorted(set(col.unique()))[:5]}")

    n_ones = int(col.sum())
    if n_ones == 0:
        raise CheckFail("ยังไม่มีเลข 1 เลย — เทียบคอลัมน์ month กับ 'เดือน cancel "
                        "ของคนนั้น' (map จาก student_key) หรือยัง?")
    ones = _pairs(obj[col == 1])
    if ones == _TOY_NAIVE_ONES:
        return
    raise CheckFail(f"เลข 1 มี {n_ones} จุดที่ {sorted(ones)[:4]} — ควรมี 3 จุด "
                    f"ตรงเดือน cancel ของแต่ละคนเป๊ะๆ (event_month ของ cancel "
                    f"คือเดือนสุดท้ายที่ active ไม่ใช่เดือนที่หายไปแล้ว)")


# ---------------------------------------------------------------- ex_01_03

@register("ex_01_03")
def _ex_01_03(obj):
    _expect_filled("toy_labels", obj)
    expect_df(obj, "toy_labels")
    expect_cols(obj, ["student_key", "month", "churned_next_month"], "toy_labels")
    if len(obj) != 33:
        raise CheckFail(f"ได้ {len(obj)} แถว แต่ควรได้ 33 — ต้องทำต่อจาก "
                        f"toy_labels_naive ของข้อ 1.2 นะครับ")
    col = obj["churned_next_month"]
    if not pd.api.types.is_float_dtype(col):
        raise CheckFail(f"dtype เป็น {col.dtype} — int เก็บ NaN ไม่ได้ "
                        f"อย่าลบบรรทัด astype('float64') ใน skeleton ออกนะครับ")
    if col.isna().all():
        raise CheckFail("เป็น NaN หมดทั้งตาราง — แปลว่า label 0/1 จากข้อ 1.2 "
                        "ยังไม่มา ทำข้อ 1.2 ให้ผ่านก่อนแล้วรันไล่ลงมาใหม่ครับ")
    n_nan = int(col.isna().sum())
    if n_nan == 0:
        raise CheckFail("ยังไม่มี NaN สักแถว — TODO 2 ยังเป็นคอมเมนต์อยู่หรือเปล่า? "
                        "(ลบ # ออก แล้วใช้ .loc[mask, ...] ใส่ float('nan'))")

    nan_months = set(obj.loc[col.isna(), "month"].astype(str))
    if "2026-07" in nan_months:
        raise CheckFail("เดือน ก.ค. โดน censor ไปด้วย — ก.ค. ปิดยอดชัวร์แล้ว "
                        "(= KNOWN_THROUGH) เงื่อนไขต้องเป็น 'มากกว่า' ไม่ใช่ 'ตั้งแต่'")
    want_nan = {(s, m) for s, m in _toy_pairs() if m == "2026-09" or m > _KNOWN_THROUGH}
    got_nan = _pairs(obj[col.isna()])
    if got_nan != want_nan:
        if got_nan < want_nan and all(m == "2026-09" for _, m in got_nan):
            raise CheckFail("censor เฉพาะ ก.ย. อยู่ — อย่าลืมเงื่อนไขที่สอง: "
                            "เดือน > KNOWN_THROUGH (เชื่อมสองเงื่อนไขด้วย | และครอบวงเล็บ)")
        raise CheckFail(f"ตำแหน่ง NaN ยังไม่ตรง (ได้ {len(got_nan)} จุด ควรได้ "
                        f"{len(want_nan)}) — เดือนที่ต้องเป็น NaN คือ ก.ย. กับ "
                        f"เดือนที่มากกว่า KNOWN_THROUGH เท่านั้น")

    ones = _pairs(obj[col == 1.0])
    if ones != {("B", "2026-05")}:
        raise CheckFail("จุดที่เป็น 1 ควรเหลือแค่จุดเดียว (เดือนสุดท้ายของคนที่"
                        "ยกเลิกกลางซีซัน ก่อน KNOWN_THROUGH) — เช็คว่า mask "
                        "ไปทับเลข 1 ของข้อ 1.2 ผิดจุดหรือเปล่า")
    if int((col == 0.0).sum()) != 23:
        raise CheckFail("จำนวนแถวที่เป็น 0 ยังไม่ตรง — แถวที่ไม่โดน censor "
                        "ต้องคงค่าเดิมจากข้อ 1.2 ไว้ (อย่าไปแก้แถวอื่น)")


# ---------------------------------------------------------------- ex_01_04

_LABEL_COLS = ["student_key", "year", "month", "active",
               "churned_next_month", "churn_date", "label_source"]


@register("ex_01_04")
def _ex_01_04(obj):
    _expect_filled("labels", obj)
    expect_df(obj, "labels")
    expect_cols(obj, _LABEL_COLS, "labels")

    vals = set(obj["churned_next_month"].dropna().unique())
    if not vals <= {0.0, 1.0}:
        raise CheckFail(f"churned_next_month ต้องเป็น 0/1/NaN เท่านั้น แต่เจอ {vals}")
    if obj.duplicated(["student_key", "month"]).any():
        raise CheckFail("มี (student_key, month) ซ้ำ — 1 คนต่อเดือนต้องมีแถวเดียว "
                        "(concat ตารางเดิมซ้ำสองรอบหรือเปล่า?)")

    years = set(obj["year"].unique())
    if years == {2568} or years == {2569}:
        raise CheckFail(f"มีข้อมูลปีเดียว ({years}) — ต้องสร้างทั้ง 2568 และ 2569 "
                        f"แล้วต่อกันด้วย pd.concat นะครับ")

    # sanity 1: ก.ย. censored เสมอ
    sep = obj[obj["month"].astype(str).str.endswith("-09")]
    if len(sep) and sep["churned_next_month"].notna().any():
        raise CheckFail("เดือน ก.ย. คือจบซีซัน ต้องเป็น NaN ทุกแถว — "
                        "canonical function จัดการให้ ถ้าไม่ใช่แสดงว่าไปแก้ label เอง?")
    # sanity 2: เดือนที่ยังไม่ปิดยอดของปี 69 ต้อง censored
    aug69 = obj[obj["month"].astype(str) == "2026-08"]
    if len(aug69) and aug69["churned_next_month"].notna().any():
        raise CheckFail("ส.ค. 69 ยังมี label 0/1 อยู่ — วันนี้ 3 ส.ค. บันทึกยกเลิก"
                        "เดือนนี้ยังเก็บไม่ครบ ส่ง known_through='2026-07' "
                        "ให้ปี 2569 หรือยัง?")
    # sanity 3: churn rate รายเดือน (เฉพาะเดือนที่มี label) 3–30%
    rate = obj.groupby("month")["churned_next_month"].mean().dropna()
    bad = rate[(rate < 0.03) | (rate > 0.30)]
    if len(bad):
        m0 = bad.index[0]
        raise CheckFail(f"churn rate เดือน {m0} = {bad.iloc[0]:.1%} หลุดช่วง 3–30% "
                        f"ที่คาดไว้ — เช็คว่าเรียก build_labels_monthly ถูกปี/ถูก "
                        f"argument หรือเปล่า")

    ev = _load_events()
    if ev is None or not IS_SAMPLE:   # real mode → เช็คโครงสร้างพอ
        return
    exp = _canonical_labels(ev)
    if len(obj) != len(exp):
        raise CheckFail(f"ได้ {len(obj):,} แถว แต่ canonical ได้ {len(exp):,} แถว — "
                        f"โหลด enrollment_events.csv จาก DATA_DIR ครบทั้งไฟล์ไหม?")
    m = exp[["student_key", "month", "churned_next_month"]].merge(
        obj[["student_key", "month", "churned_next_month"]],
        on=["student_key", "month"], how="outer",
        suffixes=("_exp", "_got"), indicator=True)
    if (m["_merge"] != "both").any():
        n = int((m["_merge"] != "both").sum())
        raise CheckFail(f"คู่ (student, month) ไม่ตรงกับ canonical {n} แถว — "
                        f"เทียบ argument ของ build_labels_monthly กับคำใบ้อีกรอบ")
    a = m["churned_next_month_exp"].fillna(-1.0)
    b = m["churned_next_month_got"].fillna(-1.0)
    if not (a == b).all():
        n = int((a != b).sum())
        raise CheckFail(f"label ไม่ตรงกับ canonical {n} แถว — เช็ค known_through "
                        f"ของแต่ละปี (2568 จบแล้วปล่อย default · 2569 = '2026-07')")


# ---------------------------------------------------------------- ex_01_05

@register("ex_01_05")
def _ex_01_05(obj):
    _expect_filled("churn_by_month", obj)
    if isinstance(obj, pd.DataFrame):
        raise CheckFail("ได้ DataFrame มา — เลือกคอลัมน์เดียวตอน groupby เช่น "
                        ".groupby('month')['churned_next_month'].mean() จะได้ Series")
    if not isinstance(obj, pd.Series):
        raise CheckFail(f"churn_by_month ควรเป็น Series แต่ได้ {type(obj).__name__}")
    if isinstance(obj.index, pd.MultiIndex):
        raise CheckFail("index ซ้อนหลายชั้นอยู่ — groupby ด้วย 'month' "
                        "คอลัมน์เดียวพอครับ (ไม่ต้องใส่ year)")

    vals = obj.dropna()
    if len(vals) and float(vals.max()) > 1.0 + 1e-9:
        raise CheckFail("มีค่าเกิน 1 — อย่าเพิ่งคูณ 100 ครับ check ต้องการสัดส่วน 0–1 "
                        "(คูณ 100 เฉพาะตอน print ดูเฉยๆ)")

    ev = _load_events()
    if ev is None or not IS_SAMPLE:   # real mode → เช็คช่วงค่าพอ
        if len(vals) and not vals.between(0, 1).all():
            raise CheckFail("มีค่านอกช่วง 0–1 — churn rate เป็นสัดส่วน ต้องอยู่ช่วงนี้")
        return
    exp = _canonical_labels(ev).groupby("month")["churned_next_month"].mean()
    if len(obj) != len(exp):
        if len(obj) == int(exp.notna().sum()):
            raise CheckFail("เดือน censored หายไปจาก Series — อย่า dropna ครับ "
                            "เดือนที่เป็น NaN ต้องคงไว้ให้เห็นว่า 'ยังตอบไม่ได้'")
        raise CheckFail(f"ได้ {len(obj)} เดือน แต่ควรได้ {len(exp)} — "
                        f"groupby ด้วยคอลัมน์ month จาก labels ทั้ง 2 ปี (ข้อ 1.4)")
    got = obj.sort_index()
    want = exp.sort_index()
    if [str(i) for i in got.index] != [str(i) for i in want.index]:
        raise CheckFail("ชื่อเดือนใน index ไม่ตรง — ต้อง groupby ด้วยคอลัมน์ "
                        "'month' (รูปแบบ 'YYYY-MM') ของตาราง labels")
    if not np.allclose(got.to_numpy(dtype=float), want.to_numpy(dtype=float),
                       atol=1e-6, equal_nan=True):
        raise CheckFail("จำนวนเดือนถูกแล้ว แต่ค่าเฉลี่ยยังไม่ตรง — ใช้ .mean() "
                        "ของ churned_next_month ตรงๆ (อย่า fillna ก่อนเฉลี่ย "
                        "เดือน censored ต้องเป็น NaN)")
