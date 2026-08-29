"""Checks for ch02 — Data wrangling: รวม 2 ปีเข้า schema เดียว

หมายเหตุผู้ดูแล (convention "recompute สด" ตาม docs/authoring-guide.md):
- ex_02_01..03: ค่า expected ไม่ hardcode — คำนวณสดทุกครั้งจาก fixture คงที่
  data/sample/messy_2568_*.csv ด้วยสูตรเดียวกับเฉลย จึงใช้ได้ทุกโหมดข้อมูล
  โดยไม่ต้องมี branch IS_SAMPLE แยก
- MESSY_DIR ด้านล่างถูก "pin" ไว้ที่ data/sample เสมอ (จงใจไม่ตาม
  ML_CHURN_DATA/DATA_DIR) เพราะแบบฝึกหัด audit/clean/melt ของบทนี้ฝึกกับ
  ไฟล์ messy จำลองชุดเดียวกันเสมอ ไม่ว่าผู้เรียนจะรันโหมด sample หรือ real —
  อย่าแก้ให้ไปอ่านจาก DATA_DIR
- ex_02_04: ตรวจค่าใน report ตรงๆ (contract การันตีว่าคำตอบเหมือนกันทุกโหมด)
- ex_02_05: ทดสอบ guard function ของผู้เรียนด้วยเคสสังเคราะห์ ทั้งเคสที่ต้องผ่าน
  และเคสที่ต้องจับได้
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks import CheckFail, expect_cols, expect_df, register
from src.config import PROJECT_ROOT

MESSY_DIR = PROJECT_ROOT / "data" / "sample"
SUBJ = {"เลข": 1, "ฟิสิกส์": 4, "เคมี": 5, "ชีวะ": 6}

STUDENTS68_COLS = ["student_key", "year", "student_code", "display_name", "grade",
                   "live_or_replay", "old_new", "n_subjects", "subject_ids"]
ATT68_COLS = ["student_key", "subject_id", "week_no", "status"]
STATUS_MAP = {"มา": "present", "ขาด": "absent", "ลา": "leave"}


def _raw_students() -> pd.DataFrame:
    return pd.read_csv(MESSY_DIR / "messy_2568_students.csv", dtype=str)


def _raw_wide() -> pd.DataFrame:
    return pd.read_csv(MESSY_DIR / "messy_2568_attendance_wide.csv", dtype=str)


def _expected_students68() -> pd.DataFrame:
    df = _raw_students()
    df = df[df["รหัส"].notna()].copy()
    df = df.rename(columns={"รหัส": "student_code", "ชื่อเล่น": "display_name",
                            "ระดับชั้น": "grade", "สด/เทป": "live_or_replay",
                            "old / new": "old_new"})
    ids = df.apply(lambda r: sorted(SUBJ[c] for c in SUBJ if r[c] == "TRUE"), axis=1)
    df["subject_ids"] = ids.map(lambda x: ";".join(map(str, x)))
    df["n_subjects"] = ids.map(len)
    df["grade"] = df["grade"].str.strip()
    df["live_or_replay"] = df["live_or_replay"].str.strip()
    df["old_new"] = df["old_new"].str.strip().str.lower()
    df["year"] = 2568
    df["student_key"] = "2568-" + df["student_code"]
    return df[STUDENTS68_COLS].reset_index(drop=True)


def _expected_att68() -> pd.DataFrame:
    long = _raw_wide().melt(id_vars=["รหัส", "วิชา"], var_name="week",
                            value_name="status_th")
    long = long.dropna(subset=["status_th"])
    long["status"] = long["status_th"].str.strip().map(STATUS_MAP)
    long["subject_id"] = long["วิชา"].map(SUBJ)
    long["week_no"] = long["week"].str.removeprefix("W").astype(int)
    long["student_key"] = "2568-" + long["รหัส"]
    return long[ATT68_COLS].reset_index(drop=True)


def _as_int(value, name: str) -> int:
    if value is None:
        raise CheckFail(f"`{name}` ยังเป็น None อยู่ — เติมช่องว่าง ____ ก่อนนะครับ")
    if isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, list)):
        raise CheckFail(f"`{name}` ควรเป็นตัวเลขตัวเดียว ไม่ใช่ {type(value).__name__} — "
                        f"อย่าลืมปิดท้ายด้วย .sum() หรือ len(...)")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise CheckFail(f"`{name}` ควรเป็นตัวเลข แต่ได้ {value!r}")


# ---------------------------------------------------------------- ex_02_01

@register("ex_02_01")
def _ex_02_01(obj):
    if not isinstance(obj, dict):
        raise CheckFail("audit ต้องเป็น dict 3 ช่องตามโจทย์ — "
                        "อย่าแก้บรรทัด checks.check ท้าย cell นะครับ")
    for k in ("n_rows", "n_junk", "n_missing_school"):
        if k not in obj:
            raise CheckFail(f"dict ขาดช่อง '{k}' — ใช้ชื่อ key ตามโจทย์เป๊ะๆ นะครับ")
    n_rows = _as_int(obj["n_rows"], "n_rows")
    n_junk = _as_int(obj["n_junk"], "n_junk")
    n_school = _as_int(obj["n_missing_school"], "n_missing_school")

    raw = _raw_students()
    exp_rows = len(raw)
    exp_junk = int(raw["รหัส"].isna().sum())
    exp_school = int(raw["โรงเรียน"].isna().sum())

    if n_rows != exp_rows:
        if n_rows == exp_rows + 1:
            raise CheckFail("n_rows เกินมา 1 — บรรทัด header ไม่ใช่แถวข้อมูล "
                            "(len(df) ไม่นับ header อยู่แล้ว อย่าไปบวกเพิ่ม)")
        raise CheckFail(f"n_rows = {n_rows} ยังไม่ตรง — จำนวนแถวทั้งหมดคือ len(...) "
                        f"ของ DataFrame ตรงๆ ไม่ต้องกรองอะไรก่อน")
    if n_junk != exp_junk:
        raise CheckFail("n_junk ยังไม่ตรง — นิยามแถวขยะในโจทย์คือ 'รหัส' ว่าง: "
                        ".isna().sum() บนคอลัมน์เดียวนั้นพอ (อย่าใช้คอลัมน์ชื่อ)")
    if n_school != exp_school:
        junk_school = int(raw.loc[raw["รหัส"].isna(), "โรงเรียน"].isna().sum())
        if n_school == exp_school - junk_school:
            raise CheckFail("n_missing_school ขาดไปนิดเดียว — โจทย์ให้นับทั้งไฟล์ "
                            "'รวมแถวขยะด้วย' อย่าเพิ่งกรองแถวไหนออกในข้อนี้")
        raise CheckFail("n_missing_school ยังไม่ตรง — .isna().sum() บนคอลัมน์ 'โรงเรียน' "
                        "ทั้งไฟล์ตรงๆ เลยครับ")


# ---------------------------------------------------------------- ex_02_02

@register("ex_02_02")
def _ex_02_02(obj):
    if obj is None:
        raise CheckFail("students68 ยังเป็น None อยู่ — เขียนตามขั้น 1–6 แล้วเก็บผลไว้ใน "
                        "students68 ก่อนนะครับ")
    expect_df(obj, "students68")
    missing = [c for c in STUDENTS68_COLS if c not in obj.columns]
    if missing:
        raise CheckFail(f"students68 ขาดคอลัมน์ {missing} — เทียบชื่อกับโจทย์ขั้น 2 และ 6 อีกที "
                        f"(ระวัง 'old / new' ในชีตมี space รอบ / )")
    extra = [c for c in obj.columns if c not in STUDENTS68_COLS]
    if extra:
        raise CheckFail(f"มีคอลัมน์เกินมา: {extra} — ขั้นสุดท้ายเลือกเฉพาะ 9 คอลัมน์ตามโจทย์ข้อ 6")
    if list(obj.columns) != STUDENTS68_COLS:
        raise CheckFail("คอลัมน์ครบแล้ว แต่ลำดับยังไม่ตรงตามโจทย์ข้อ 6 — "
                        "เลือกด้วย df[[...]] เรียงตามลำดับใน contract")

    exp = _expected_students68()
    if len(obj) != len(exp):
        if len(obj) == len(_raw_students()):
            raise CheckFail(f"ได้ {len(obj)} แถวเท่าไฟล์ดิบเป๊ะ — แถวขยะ (รหัสว่าง) "
                            f"ยังไม่ถูกตัดออก (ขั้น 1: .notna())")
        raise CheckFail(f"ได้ {len(obj)} แถว แต่ควรได้ {len(exp)} — "
                        f"เช็คเงื่อนไขกรองแถวขยะอีกครั้ง (กรองจากคอลัมน์ 'รหัส' เท่านั้น)")

    if not pd.api.types.is_integer_dtype(obj["year"]):
        raise CheckFail("year ควรเป็นจำนวนเต็ม (ใส่ 2568 เป็นเลขตรงๆ ไม่ใช่ \"2568\") — "
                        "contract ทุกตารางใช้ year เป็น int")
    if set(obj["year"].unique()) != {2568}:
        raise CheckFail("year ต้องเป็น 2568 ทุกแถว")
    if not pd.api.types.is_integer_dtype(obj["n_subjects"]):
        raise CheckFail("n_subjects ควรเป็นจำนวนเต็ม — จำนวนสมาชิกของลิสต์วิชา (len)")

    got = obj.sort_values("student_key").reset_index(drop=True)
    want = exp.sort_values("student_key").reset_index(drop=True)
    if list(got["student_key"].astype(str)) != list(want["student_key"]):
        raise CheckFail("student_key ยังไม่ตรง — ต้องเป็น \"2568-\" + student_code "
                        "(string ต่อ string — อ่านไฟล์ด้วย dtype=str แล้วหรือยัง?)")

    bad_on = set(got["old_new"].dropna().unique()) - {"old", "new"}
    if bad_on:
        raise CheckFail(f"old_new มีค่าแปลกปลอม: {sorted(bad_on)} — "
                        f"ต้อง .str.strip().str.lower() ให้เหลือแค่ old/new")
    bad_lr = set(got["live_or_replay"].dropna().unique()) - {"สด", "เทป", "ผสม"}
    if bad_lr:
        raise CheckFail(f"live_or_replay มีค่าแปลกปลอม: {sorted(bad_lr)} — "
                        f"ค่าจากชีตมีช่องว่างท้าย ต้อง .str.strip()")
    if bool((got["grade"].astype(str).str.strip() != got["grade"].astype(str)).any()):
        raise CheckFail("grade ยังมีช่องว่างหัว/ท้ายหลงเหลือ — .str.strip() ครับ")

    if list(got["subject_ids"].astype(str)) != list(want["subject_ids"]):
        resorted = got["subject_ids"].astype(str).map(
            lambda s: ";".join(sorted(s.split(";"))))
        if list(resorted) == list(want["subject_ids"]):
            raise CheckFail("subject_ids เกือบแล้ว — รหัสวิชายังไม่ได้เรียง (sort) ก่อน join "
                            "contract ต้องการเรียงจากน้อยไปมากเสมอ")
        raise CheckFail("subject_ids ยังไม่ตรง — เก็บเฉพาะวิชาที่ค่า == \"TRUE\" "
                        "แล้วแปลงเป็นรหัสตัวเลขตาม dict SUBJ ก่อน join ด้วย \";\"")
    if list(map(int, got["n_subjects"])) != list(map(int, want["n_subjects"])):
        raise CheckFail("n_subjects ยังไม่ตรงกับจำนวนวิชาใน subject_ids — "
                        "นับจากลิสต์เดียวกันกับที่ใช้สร้าง subject_ids จะไม่มีวันเพี้ยน")
    for col, hint in [
        ("student_code", "มาจากคอลัมน์ 'รหัส' ตรงๆ ไม่ต้องแก้ค่า"),
        ("display_name", "rename จาก 'ชื่อเล่น' (ไม่ใช่ 'ชื่อ' ที่เป็นชื่อจริง)"),
        ("grade", "rename จาก 'ระดับชั้น' + .str.strip() (ค่าเพี้ยนแบบ ม3/3 ปล่อยไว้ก่อนได้)"),
        ("live_or_replay", "rename จาก 'สด/เทป' + .str.strip()"),
        ("old_new", "rename จาก 'old / new' + strip + lower"),
    ]:
        if list(got[col].fillna("?").astype(str)) != list(want[col].fillna("?").astype(str)):
            raise CheckFail(f"คอลัมน์ '{col}' ค่ายังไม่ตรง — {hint}")


# ---------------------------------------------------------------- ex_02_03

@register("ex_02_03")
def _ex_02_03(obj):
    if obj is None:
        raise CheckFail("att68 ยังเป็น None อยู่ — เขียนตามขั้น 1–5 แล้วเก็บผลไว้ใน att68 ก่อนครับ")
    expect_df(obj, "att68")
    expect_cols(obj, ATT68_COLS, "att68")
    extra = [c for c in obj.columns if c not in ATT68_COLS]
    if extra:
        raise CheckFail(f"มีคอลัมน์เกินมา: {extra} — ขั้นสุดท้ายเลือกแค่ 4 คอลัมน์ตามโจทย์")

    exp = _expected_att68()
    n_melted_all = len(_raw_wide()) * (len(_raw_wide().columns) - 2)
    if len(obj) == n_melted_all:
        raise CheckFail(f"ได้ {len(obj)} แถว = melt ครบทุกช่องรวมช่องว่าง — "
                        f"สัปดาห์ที่ยังไม่บันทึก (NaN) ไม่ใช่ 'ขาด' ต้อง dropna ทิ้ง (ขั้น 2)")
    if obj["status"].isna().any():
        raise CheckFail("status มี NaN ทั้งที่ dropna แล้ว — เจอค่ามีช่องว่างท้าย "
                        "(\"มา \" ไม่เท่ากับ \"มา\") ต้อง .str.strip() ก่อน map")
    bad = set(obj["status"].unique()) - {"present", "absent", "leave"}
    if bad:
        raise CheckFail(f"status มีค่าแปลกปลอม: {sorted(map(str, bad))} — "
                        f"ต้อง map เป็นภาษา contract: present/absent/leave")
    if len(obj) != len(exp):
        raise CheckFail(f"ได้ {len(obj):,} แถว แต่ควรได้ {len(exp):,} — "
                        f"เช็คลำดับ melt → dropna(subset=[\"status_th\"]) อีกครั้ง")
    if obj["subject_id"].isna().any():
        raise CheckFail("subject_id มี NaN — map ชื่อวิชาไทยด้วย dict SUBJ ครบทั้ง 4 วิชาหรือยัง?")
    if not pd.api.types.is_integer_dtype(obj["week_no"]):
        raise CheckFail(f"week_no ควรเป็น int แต่เป็น {obj['week_no'].dtype} — "
                        f"ตัด \"W\" ออกแล้ว .astype(int)")

    got_vc = {str(k): int(v) for k, v in obj["status"].value_counts().items()}
    want_vc = {str(k): int(v) for k, v in exp["status"].value_counts().items()}
    if got_vc != want_vc:
        raise CheckFail("จำนวนแถวถูกแล้ว แต่สัดส่วน present/absent/leave ยังไม่ตรง — "
                        "เช็ค dict ที่ใช้ map (มา→present, ขาด→absent, ลา→leave)")

    got_rows = sorted(map(tuple, zip(obj["student_key"].astype(str),
                                     obj["subject_id"].astype(int),
                                     obj["week_no"].astype(int),
                                     obj["status"].astype(str))))
    want_rows = sorted(map(tuple, zip(exp["student_key"], exp["subject_id"].astype(int),
                                      exp["week_no"].astype(int), exp["status"])))
    if got_rows != want_rows:
        raise CheckFail("ภาพรวมถูกเกือบหมดแล้ว แต่บางแถวยังไม่ตรง — ไล่เช็ค: "
                        "student_key = \"2568-\" + รหัส · subject_id จาก dict SUBJ · "
                        "week_no จากชื่อคอลัมน์ W ที่ melt มา")


# ---------------------------------------------------------------- ex_02_04

@register("ex_02_04")
def _ex_02_04(obj):
    if not isinstance(obj, dict):
        raise CheckFail("validation_report ต้องเป็น dict 4 ช่องตามโจทย์ — "
                        "อย่าแก้บรรทัด checks.check นะครับ")
    for k in ("attendance_ok", "attempts_ok", "dup_exam_pairs", "dup_ep_rows"):
        if k not in obj:
            raise CheckFail(f"dict ขาดช่อง '{k}' — ใช้ชื่อ key ตามโจทย์เป๊ะๆ")
        if obj[k] is None:
            raise CheckFail(f"ช่อง '{k}' ยังเป็น None — เติมช่องว่าง ____ ก่อนนะครับ")
    if obj["attendance_ok"] is not True:
        raise CheckFail("attendance_ok ต้องเป็น True ที่ได้จาก "
                        "contracts.validate_df(\"attendance_long\", att) จริงๆ — "
                        "ถ้ามัน raise ให้อ่านข้อความ error (มักเป็นเพราะลืม parse_dates)")
    if obj["attempts_ok"] is not True:
        raise CheckFail("attempts_ok ต้องเป็น True จาก "
                        "contracts.validate_df(\"exam_attempts\", atm) — "
                        "อ่าน submitted_at ด้วย parse_dates แล้วหรือยัง?")
    dup_exam = _as_int(obj["dup_exam_pairs"], "dup_exam_pairs")
    dup_ep = _as_int(obj["dup_ep_rows"], "dup_ep_rows")
    if dup_exam != 0:
        raise CheckFail(f"dup_exam_pairs = {dup_exam} — ควรเป็น 0 "
                        f"(นับด้วย .duplicated([\"student_key\", \"exam_id\"]).sum() บนตาราง attempts)")
    if dup_ep != 0:
        raise CheckFail(f"dup_ep_rows = {dup_ep} — ควรเป็น 0 (นับ .duplicated ด้วย 4 คอลัมน์ "
                        f"ตามโจทย์บนตาราง attendance)")


# ---------------------------------------------------------------- ex_02_05

@register("ex_02_05")
def _ex_02_05(fn):
    if fn is None or not callable(fn):
        raise CheckFail("ต้องส่ง 'ตัวฟังก์ชัน' assert_same_schema (ไม่ใส่วงเล็บเรียกมัน) — "
                        "อย่าแก้บรรทัด checks.check ท้าย cell นะครับ")

    base = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})

    def _must_pass(df_b, why):
        try:
            r = fn(base, df_b)
        except AssertionError as e:
            raise CheckFail(f"เคสที่ควรผ่าน ({why}) กลับโดน assert: {e}")
        except Exception as e:
            raise CheckFail(f"เคส ({why}) เจอ {type(e).__name__}: {e} — "
                            f"ฟังก์ชันไม่ควรพังกับตารางปกติ")
        if r is not True:
            raise CheckFail(f"เคส ({why}) ผ่าน assert หมดแล้ว แต่ฟังก์ชันคืน {r!r} — "
                            f"อย่าลืม return True (ขั้น 3)")

    def _must_catch(df_b, why, hint):
        try:
            fn(base, df_b)
        except AssertionError:
            return
        except Exception as e:
            raise CheckFail(f"เคส ({why}) ควรโดน assert (AssertionError) "
                            f"แต่กลับได้ {type(e).__name__} — {hint}")
        raise CheckFail(f"เคส ({why}) หลุดรอดไปได้ — {hint}")

    _must_pass(base.copy(), "ตารางเหมือนกันเป๊ะ")
    b = base.copy()
    b["x"] = b["x"].astype(float)
    _must_pass(b, "int64 กับ float64 — เป็นตัวเลขทั้งคู่ ถือว่าเข้ากัน")
    _must_pass(base[["y", "x"]], "คอลัมน์ชุดเดียวกันแต่คนละลำดับ — เทียบเป็น set ต้องผ่าน")

    _must_catch(base.drop(columns=["y"]), "คอลัมน์หายไปหนึ่งคอลัมน์",
                "เงื่อนไขข้อ 1 ต้องเทียบ set ของคอลัมน์ก่อนเข้า loop")
    extra = base.copy()
    extra["z"] = 0
    _must_catch(extra, "คอลัมน์เกินมาหนึ่งคอลัมน์",
                "set เท่ากันต้องจับได้ทั้งขาดและเกิน")
    s = base.copy()
    s["x"] = s["x"].astype(str)
    _must_catch(s, "คอลัมน์เดียวกันแต่ int vs string",
                "เงื่อนไขข้อ 2: ไม่ใช่ตัวเลขทั้งคู่ และ dtype ไม่เท่ากัน ต้อง assert ไม่ผ่าน")
