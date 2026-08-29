# เฉลย 8.2 — เรียก "สคริปต์ production" จากข้างนอก แล้วอ่านไฟล์ผลลัพธ์
# (ท่านี้แหละที่ n8n/cron จะทำแทนคุณทุกสิ้นเดือน)
# ตัวจริงของข้อนี้คือเปิด scoring/score_month.py แล้วเติม TODO 3 จุดด้วยมือ —
# เติมเสร็จเมื่อไหร่ เปลี่ยน script เป็น "scoring/score_month.py" แล้วรันซ้ำ ต้องได้ CSV เหมือนกันเป๊ะ
# แล้วเปิดสองไฟล์เทียบบรรทัดต่อบรรทัด: จุดไหนของคุณต่างจากเฉลย จุดนั้นแหละบทเรียน
import subprocess

month = "2026-07"   # เดือน holdout เดียวกับข้อ 8.1 — CSV ต้องให้คะแนนตรงกับ risk_jul เป๊ะ
script = "scoring/score_month_solution.py"   # เปลี่ยนเป็น "scoring/score_month.py" เมื่อเติมไฟล์ของคุณเสร็จ

report_path = REPORTS_DIR / f"risk_{month}.csv"
report_path.unlink(missing_ok=True)
r = subprocess.run([sys.executable, script, "--month", month],
                   cwd=str(PROJECT_ROOT), capture_output=True, text=True)
print("\n".join(ln for ln in r.stdout.splitlines()
                if "⚠️" in ln or ln.startswith(("เขียน", "โมเดลเห็น"))))
if r.returncode != 0:
    print("\n".join(r.stderr.strip().splitlines()[-5:]))

risk_report = pd.read_csv(report_path)
show = [c for c in ["display_name", "risk_score", "att_month_pct", "reason_1"]
        if c in risk_report.columns]
print(f"อ่าน {report_path.name} ได้ {len(risk_report)} คน — 10 อันดับที่ mentor ควรโทรก่อน:")
print(risk_report.head(10)[show].to_string(index=False))
checks.check("ex_08_02", risk_report)
