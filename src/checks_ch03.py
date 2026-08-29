"""Checks for ch03 — EDA: churn หน้าตาเป็นยังไง

หลักการ recompute สด (ตาม convention ใน docs/authoring-guide.md):
ทุกค่า expected คำนวณสดจากไฟล์ใน DATA_DIR ณ ตอนรัน (labels_monthly.csv,
weekly_metrics.csv, students.csv) ด้วยท่าเดียวกับเฉลย — ไม่ hardcode ตัวเลข
ที่ผูกกับ seed ของ sample จึง **ไม่ต้องแยก branch IS_SAMPLE**: โหมดไหน
DATA_DIR ชี้ไปไฟล์ชุดไหน expected ก็คำนวณจากชุดนั้นเอง ใช้ได้ทุกโหมด
ถ้าไฟล์ยังไม่มี (โหมด real ก่อนทำบท 02 เสร็จ) จะถอยไปเช็คแค่โครงสร้าง/ช่วงค่า

ex_03_05 เป็นโจทย์เปิด — ตรวจแค่ว่า "สรุปเป็นกลุ่มแล้วจริง และมีตัวเลขจริง"
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.checks import CheckFail, expect_cols, expect_df, register
from src.config import DATA_DIR

# ---------------------------------------------------------------- data loading

_CACHE: dict[str, object] = {}


def _labels() -> pd.DataFrame | None:
    if "labels" not in _CACHE:
        p = DATA_DIR / "labels_monthly.csv"
        _CACHE["labels"] = pd.read_csv(p, dtype={"month": str}) if p.exists() else None
    return _CACHE["labels"]  # type: ignore[return-value]


def _weekly() -> pd.DataFrame | None:
    if "wm" not in _CACHE:
        p = DATA_DIR / "weekly_metrics.csv"
        if p.exists():
            wm = pd.read_csv(p, parse_dates=["week_start", "week_end"])
            wm["month"] = wm["week_start"].dt.strftime("%Y-%m")
            _CACHE["wm"] = wm
        else:
            _CACHE["wm"] = None
    return _CACHE["wm"]  # type: ignore[return-value]


# ---------------------------------------------------------------- expected values

def _exp_churn_pivot() -> pd.DataFrame | None:
    lab = _labels()
    if lab is None:
        return None
    d = lab.copy()
    d["mon"] = d["month"].str[5:]
    return d.pivot_table(index="mon", columns="year",
                         values="churned_next_month", aggfunc="mean")


def _exp_active_pivot() -> pd.DataFrame | None:
    lab = _labels()
    if lab is None:
        return None
    d = lab.copy()
    d["mon"] = d["month"].str[5:]
    return d.pivot_table(index="mon", columns="year", values="active", aggfunc="sum")


def _exp_att_vs_label(how: str = "last") -> pd.DataFrame | None:
    lab, wm = _labels(), _weekly()
    if lab is None or wm is None:
        return None
    g = wm.sort_values("week_start").groupby(["student_key", "month"], as_index=False)
    snap = g.last() if how == "last" else g.first()
    return (lab.merge(snap[["student_key", "month", "att_cum_pct"]],
                      on=["student_key", "month"], how="left")
               .dropna(subset=["churned_next_month"]))


def _exp_silent_dist(how: str = "max") -> pd.DataFrame | None:
    lab, wm = _labels(), _weekly()
    if lab is None or wm is None:
        return None
    g = wm.sort_values("week_start").groupby(["student_key", "month"], as_index=False)
    sil = g["silent_weeks"].max() if how == "max" else g["silent_weeks"].last()
    svl = lab.merge(sil, on=["student_key", "month"], how="left").dropna(
        subset=["churned_next_month"])
    return pd.crosstab(svl["silent_weeks"], svl["churned_next_month"],
                       normalize="columns")


# ---------------------------------------------------------------- small helpers

def _not_none(obj, name: str, what: str) -> None:
    if obj is None:
        raise CheckFail(f"`{name}` ยังเป็น None อยู่ — เติมช่องว่าง ____ ให้เป็น{what} ก่อนนะครับ")


def _key(x) -> str:
    """Canonical label key: 3 / '03' / 3.0 -> '3' (เทียบ index/columns แบบไม่จู้จี้)."""
    try:
        return f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)


def _canon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(axis=0, how="all").copy()
    out.index = pd.Index([_key(i) for i in out.index])
    out.columns = pd.Index([_key(c) for c in out.columns])
    return out


def _values(df: pd.DataFrame, name: str) -> np.ndarray:
    try:
        return df.to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise CheckFail(f"{name} มีค่าที่ไม่ใช่ตัวเลขปนอยู่ — ตารางสรุปควรมีแต่ตัวเลข")


def _same_axes(got: pd.DataFrame, exp: pd.DataFrame, name: str) -> pd.DataFrame:
    g, e = _canon(got), _canon(exp)
    if set(g.columns) != set(e.columns):
        raise CheckFail(f"{name} คอลัมน์ยังไม่ตรง — ควรเป็น {list(exp.columns)} "
                        f"แต่ได้ {list(got.columns)[:6]}")
    if set(g.index) != set(e.index):
        raise CheckFail(f"{name} แถวยังไม่ตรง — ควรมี {len(e)} แถว: {list(exp.index)} "
                        f"แต่ได้ {list(got.index)[:8]}")
    return g.reindex(index=e.index, columns=e.columns)


def _cmp(got: pd.DataFrame, exp: pd.DataFrame, name: str, tol: float = 1e-6) -> None:
    g = _same_axes(got, exp, name)
    e = _canon(exp)
    gv, ev = _values(g, name), _values(e, name)
    bad = ~(np.isclose(gv, ev, atol=tol, rtol=0) | (np.isnan(gv) & np.isnan(ev)))
    if bad.any():
        i, j = np.argwhere(bad)[0]
        raise CheckFail(f"{name} ช่อง (แถว {exp.index[i]}, คอลัมน์ {exp.columns[j]}) "
                        f"= {gv[i, j]:.4f} แต่ควรได้ประมาณ {ev[i, j]:.4f} — "
                        f"ลองไล่ดูสูตรอีกรอบครับ")


def _like(got: pd.DataFrame, other: pd.DataFrame, scale: float = 1.0) -> bool:
    """True if got matches other (same axes, same values) — ใช้จับ 'ผิดแบบที่เดาได้'."""
    try:
        g, o = _canon(got), _canon(other)
        if set(g.index) != set(o.index) or set(g.columns) != set(o.columns):
            return False
        g = g.reindex(index=o.index, columns=o.columns)
        gv = g.to_numpy(dtype=float)
        ov = o.to_numpy(dtype=float) * scale
        return bool(np.all(np.isclose(gv, ov, atol=1e-4, rtol=0)
                           | (np.isnan(gv) & np.isnan(ov))))
    except Exception:
        return False


def _pct_scale_guard(got: pd.DataFrame, exp: pd.DataFrame, name: str) -> None:
    if _like(got, exp, scale=100.0):
        raise CheckFail(f"{name} เป็นหน่วย % (คูณ 100 ไปแล้ว) — "
                        f"เก็บเป็นสัดส่วน 0–1 ไว้ก่อน แล้วค่อยคูณ 100 ตอนพิมพ์/วาดกราฟ")


# ---------------------------------------------------------------- ex_03_01

@register("ex_03_01")
def _ex_03_01(obj):
    _not_none(obj, "churn_pivot", "ตาราง pivot")
    expect_df(obj, "churn_pivot")
    exp = _exp_churn_pivot()
    if exp is None:
        if obj.shape[1] < 1 or obj.empty:
            raise CheckFail("churn_pivot ยังว่างอยู่")
        return

    years = {_key(c) for c in exp.columns}
    idx = [_key(i) for i in obj.index]
    if set(idx) == years:
        raise CheckFail("สลับแกนกันอยู่ — index ควรเป็นเดือน (mon) ส่วน columns ควรเป็น year")
    if any(len(str(i)) > 2 for i in obj.index):
        raise CheckFail("แถวยังเป็นเดือนเต็มแบบ '2025-07' อยู่ — ต้องตัดเหลือเลขเดือน 2 หลักก่อน "
                        "(TODO 1) ไม่งั้นสองปีขึ้นคนละแถว เทียบกันไม่ได้")
    if {_key(c) for c in obj.columns} != years:
        raise CheckFail(f"คอลัมน์ควรเป็นปี {sorted(exp.columns)} — ใส่ columns=\"year\" "
                        f"ใน pivot_table แล้วหรือยัง (ตอนนี้ได้ {list(obj.columns)[:6]})")

    vals = _values(obj, "churn_pivot")
    if np.nanmin(vals) == 1.0 and np.nanmax(vals) == 1.0:
        raise CheckFail("ได้ 1.00 ทุกช่อง — values= ยังชี้ไปคอลัมน์อื่น (เช่น active ที่เป็น 1 หมด) "
                        "ต้องเป็น churned_next_month")
    act = _exp_active_pivot()
    if act is not None and _like(obj, act):
        raise CheckFail("นี่คือ 'จำนวนคน active' (ตัวอย่างข้างบน) ไม่ใช่ churn rate — "
                        "เปลี่ยน values= เป็น churned_next_month และ aggfunc= เป็นค่าเฉลี่ย")
    if np.nanmax(vals) > 1.5:
        _pct_scale_guard(obj, exp, "churn_pivot")
        raise CheckFail("ค่าสูงเกิน 1 — churn rate ต้องเป็นสัดส่วน 0–1 "
                        "(aggfunc ควรเป็นค่าเฉลี่ย ไม่ใช่ผลรวมของ 0/1)")
    _pct_scale_guard(obj, exp, "churn_pivot")
    _cmp(obj, exp, "churn_pivot")


# ---------------------------------------------------------------- ex_03_02

@register("ex_03_02")
def _ex_03_02(obj):
    _not_none(obj, "survival", "ตารางอัตรารอดสะสม")
    expect_df(obj, "survival")
    churn = _exp_churn_pivot()
    if churn is None:
        if obj.empty:
            raise CheckFail("survival ยังว่างอยู่")
        return
    retention = 1 - churn
    exp = retention.cumprod()

    _same_axes(obj, exp, "survival")   # ยิงข้อความเรื่องแกนก่อน ถ้าแกนเพี้ยน
    if _like(obj, churn):
        raise CheckFail("นี่ยังเป็น churn rate อยู่เลย — ขั้น 1 ต้องพลิกเป็น 'สัดส่วนที่อยู่ต่อ' ก่อน (1 - churn)")
    if _like(obj, retention):
        raise CheckFail("ได้ retention รายเดือนแล้ว แต่ยังไม่ได้คูณทบ — "
                        "survival ต้องสะสมด้วย .cumprod() (รอดเดือนนี้ได้ต้องรอดทุกเดือนก่อนหน้า)")
    if _like(obj, churn.cumprod()):
        raise CheckFail("คูณทบถูกแล้ว แต่คูณผิดตัว — ต้องคูณ 'สัดส่วนที่อยู่ต่อ' (1 - churn_pivot) "
                        "ไม่ใช่ churn rate เอง")
    if _like(obj, retention.cumsum()):
        raise CheckFail("ใช้ .cumsum() (บวกสะสม) อยู่ — ความน่าจะเป็นต่อเนื่องต้อง 'คูณ' ทบ ไม่ใช่บวก")
    _pct_scale_guard(obj, exp, "survival")
    _cmp(obj, exp, "survival")


# ---------------------------------------------------------------- ex_03_03

@register("ex_03_03")
def _ex_03_03(obj):
    _not_none(obj, "att_vs_label", "ตารางที่ merge แล้ว")
    expect_df(obj, "att_vs_label")
    expect_cols(obj, ["student_key", "month", "churned_next_month", "att_cum_pct"],
                "att_vs_label")
    exp = _exp_att_vs_label("last")
    lab = _labels()

    if obj["churned_next_month"].isna().any():
        raise CheckFail("ยังมีแถว label = NaN (censored) ปนอยู่ — "
                        "ปิดท้ายด้วย .dropna(subset=[\"churned_next_month\"]) ครับ")
    if exp is None:
        if obj.empty:
            raise CheckFail("att_vs_label ยังว่างอยู่")
        return

    if len(obj) != len(exp):
        if lab is not None and len(obj) == len(lab):
            raise CheckFail(f"ได้ {len(obj):,} แถว = เท่าตาราง labels ทั้งใบ — "
                            f"แถว censored ยังไม่ถูกตัด (ควรเหลือ {len(exp):,} แถว)")
        if len(obj) > len(exp):
            raise CheckFail(f"แถวบวมเป็น {len(obj):,} (ควรได้ {len(exp):,}) — "
                            f"แปลว่า merge เจอคู่ซ้ำ: สรุป wm ให้เหลือ (คน, เดือน) ละแถวเดียวก่อน "
                            f"และ on= ต้องเป็น list 2 คอลัมน์ [\"student_key\", \"month\"]")
        raise CheckFail(f"ได้แค่ {len(obj):,} แถว (ควรได้ {len(exp):,}) — "
                        f"merge ควรตั้งต้นจาก labels ด้วย how=\"left\" "
                        f"แล้วตัดเฉพาะแถวที่ churned_next_month เป็น NaN")

    got_keys = set(zip(obj["student_key"].astype(str), obj["month"].astype(str)))
    exp_keys = set(zip(exp["student_key"].astype(str), exp["month"].astype(str)))
    if got_keys != exp_keys:
        raise CheckFail("จำนวนแถวถูก แต่ชุด (student_key, month) ยังไม่ตรง — "
                        "เช็คว่า merge ด้วยกุญแจ 2 ดอกจริง และไม่ได้เผลอกรองอย่างอื่นทิ้ง")

    if obj["att_cum_pct"].isna().all():
        raise CheckFail("att_cum_pct เป็น NaN ทั้งคอลัมน์ — merge ไม่ติดสักแถว: "
                        "คอลัมน์ month ของ wm ต้องเป็นสตริงรูป \"YYYY-MM\" "
                        "แบบเดียวกับใน labels (.dt.strftime(\"%Y-%m\"))")

    got_map = dict(zip(zip(obj["student_key"].astype(str), obj["month"].astype(str)),
                       obj["att_cum_pct"].astype(float)))
    exp_map = dict(zip(zip(exp["student_key"].astype(str), exp["month"].astype(str)),
                       exp["att_cum_pct"].astype(float)))
    diff = [k for k, v in exp_map.items()
            if not (np.isclose(got_map[k], v, atol=1e-6, rtol=0)
                    or (np.isnan(got_map[k]) and np.isnan(v)))]
    if diff:
        first = _exp_att_vs_label("first")
        first_map = dict(zip(zip(first["student_key"].astype(str),
                                 first["month"].astype(str)),
                             first["att_cum_pct"].astype(float)))
        if all(np.isclose(got_map[k], first_map.get(k, np.nan), atol=1e-6, rtol=0)
               for k in diff):
            raise CheckFail("ค่า att_cum_pct เป็นของ 'สัปดาห์แรก' ของเดือน — "
                            "เราต้องการสถานะ ณ สิ้นเดือน t: sort_values(\"week_start\") "
                            "แล้วปิดท้ายด้วย .last()")
        raise CheckFail(f"ค่า att_cum_pct ยังไม่ตรง {len(diff):,} แถว — "
                        f"snapshot รายเดือนต้องมาจากสัปดาห์สุดท้ายของเดือนนั้น "
                        f"(sort_values ก่อน groupby แล้ว .last())")


# ---------------------------------------------------------------- ex_03_04

@register("ex_03_04")
def _ex_03_04(obj):
    _not_none(obj, "silent_dist", "ตารางสัดส่วน (crosstab)")
    expect_df(obj, "silent_dist")
    exp = _exp_silent_dist("max")
    if exp is None:
        if obj.empty:
            raise CheckFail("silent_dist ยังว่างอยู่")
        return

    label_keys = {_key(c) for c in exp.columns}          # {'0', '1'}
    if {_key(c) for c in obj.columns} != label_keys:
        if {_key(i) for i in obj.index} == label_keys:
            raise CheckFail("สลับแกนกันอยู่ — crosstab(แถว=silent_weeks, คอลัมน์=churned_next_month) "
                            "แล้ว normalize=\"columns\" ถึงจะได้สัดส่วนของแต่ละกลุ่ม")
        raise CheckFail(f"คอลัมน์ควรเป็น label 0/1 แต่ได้ {list(obj.columns)[:6]} — "
                        f"ตัวที่สองใน pd.crosstab ต้องเป็น churned_next_month")

    vals = _values(obj, "silent_dist")
    col_sum = np.nansum(vals, axis=0)
    if not np.allclose(col_sum, 1.0, atol=1e-6):
        row_sum = np.nansum(vals, axis=1)
        if np.allclose(row_sum, 1.0, atol=1e-6):
            raise CheckFail("normalize ผิดแกน — ตอนนี้แต่ละ 'แถว' รวมได้ 1 "
                            "แต่เราอยากให้แต่ละ 'กลุ่ม label' (คอลัมน์) รวมได้ 1 → normalize=\"columns\"")
        if np.nansum(vals) > 2:
            raise CheckFail("นี่คือจำนวนนับดิบ ยังไม่ได้ทำเป็นสัดส่วน — ใส่ normalize=\"columns\" "
                            "ใน pd.crosstab (สองกลุ่มขนาดต่างกันมาก เทียบจำนวนดิบไม่แฟร์)")
        raise CheckFail("แต่ละคอลัมน์ควรรวมได้ 1.0 — เช็ค normalize= ใน pd.crosstab อีกที")

    last_variant = _exp_silent_dist("last")
    if last_variant is not None and _like(obj, last_variant) and not _like(obj, exp):
        raise CheckFail("ใช้ค่า silent_weeks ของสัปดาห์สุดท้าย (.last()) อยู่ — "
                        "ข้อนี้ต้องการ 'เคยเงียบยาวสุดกี่สัปดาห์ในเดือนนั้น' → .max()")
    _cmp(obj, exp, "silent_dist", tol=1e-6)


# ---------------------------------------------------------------- ex_03_05

@register("ex_03_05")
def _ex_03_05(obj):
    _not_none(obj, "my_view", "ผลสรุปต่อกลุ่ม (Series หรือ DataFrame)")
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        raise CheckFail(f"my_view ควรเป็น Series หรือ DataFrame ที่ groupby แล้ว "
                        f"แต่ได้ {type(obj).__name__}")
    if len(obj) == 0:
        raise CheckFail("my_view ว่างเปล่า — กรองแรงไปหรือเปล่าครับ?")
    if len(obj) < 2:
        raise CheckFail("มีแค่กลุ่มเดียว — มุมที่น่าสนใจต้องเทียบอย่างน้อย 2 กลุ่ม "
                        "(เช่น สด vs เทป) ถึงจะบอกอะไรได้")

    cols = list(obj.columns) if isinstance(obj, pd.DataFrame) else [obj.name]
    if "student_key" in cols or obj.index.name == "student_key":
        raise CheckFail("ยังเป็นระดับ 'รายคน' อยู่ — ข้อนี้อยากได้ภาพรวมต่อกลุ่ม "
                        "(groupby ด้วยคุณสมบัติ เช่น live_or_replay / n_subjects / month)")
    lab = _labels()
    if lab is not None and len(obj) >= len(lab):
        raise CheckFail(f"my_view มี {len(obj):,} แถว = ยังเป็นข้อมูลรายแถวดิบ ไม่ใช่ค่าสรุป — "
                        f"ต้องผ่าน groupby(...) + ตัวสรุป (mean / size / agg) ก่อน")
    if len(obj) > 60:
        raise CheckFail(f"ได้ {len(obj):,} กลุ่ม — เยอะเกินกว่าจะอ่านด้วยตา "
                        f"ลองเลือกมิติที่มีไม่กี่ค่า (หรือ .head(20) หลังเรียงลำดับ)")

    if isinstance(obj, pd.Series):
        num = obj if pd.api.types.is_numeric_dtype(obj) else None
        if num is None:
            raise CheckFail("ค่าที่สรุปได้ไม่ใช่ตัวเลข — ปิดท้ายด้วยตัวสรุปที่เป็นตัวเลข "
                            "เช่น .mean() / .size() / .agg([\"mean\", \"size\"])")
        arr = num.to_numpy(dtype=float)
    else:
        numeric = obj.select_dtypes(include="number")
        if numeric.shape[1] == 0:
            raise CheckFail("ไม่มีคอลัมน์ตัวเลขเลย — ต้องมีค่าที่สรุปแล้วอย่างน้อย 1 คอลัมน์ "
                            "(เช่น churn rate จาก .mean() หรือจำนวนแถวจาก .size())")
        arr = numeric.to_numpy(dtype=float)
    if np.all(np.isnan(arr)):
        raise CheckFail("ค่าที่ได้เป็น NaN ทั้งหมด — merge ไม่ติด หรือกรองจนไม่เหลือแถว "
                        "(ลอง print ตารางหลัง merge ก่อน groupby ดูครับ)")
