# เฉลย 3.5 — ตัวอย่าง 1 มุม (ของคุณเป็นมุมอื่นก็ถูกได้เหมือนกัน)
# คำถาม: น้องดู "เทป" หายมากกว่าน้องเรียน "สด" จริงไหม?
my_view = (labels.merge(students[["student_key", "live_or_replay"]], on="student_key")
                 .groupby("live_or_replay")["churned_next_month"]
                 .agg(churn_rate="mean", n_rows="size"))       # ดูจำนวนแถวคู่กันเสมอ กลุ่มเล็กค่าเหวี่ยง
my_view["churn_rate"] = (my_view["churn_rate"] * 100).round(1)

print(my_view)
print("\nอ่านผล: เทป ~14% vs สด ~7% — ต่างกันราว 2 เท่า")
print("บอก mentor ว่า: เด็กสายเทปไม่มีจังหวะเจอหน้าใคร ควรมีเช็คอินเชิงรุกรายสัปดาห์")
print("(แต่ระวัง — เด็กที่เลือกเทปตั้งแต่แรกอาจว่างน้อย/ตั้งใจน้อยกว่าอยู่แล้ว")
print(" ไม่ได้แปลว่าบังคับให้มาเรียนสดแล้วจะไม่หาย — correlation ≠ causation)")

# มุมอื่นที่ลองต่อได้ (เปิดคอมเมนต์ดูได้เลย):
# my_view = (labels.merge(students[["student_key", "n_subjects"]], on="student_key")
#                  .groupby("n_subjects")["churned_next_month"].agg(["mean", "size"]))
# my_view = (labels.merge(students[["student_key", "signup_lateness"]], on="student_key")
#                  .groupby("signup_lateness")["churned_next_month"].agg(["mean", "size"]))

checks.check("ex_03_05", my_view)
