MSM Support Chatbot
ระบบแชทบอทสำหรับตอบคำถามเกี่ยวกับ MSM Support โดยใช้ FastAPI และ LINE Messaging API
ขั้นตอนการติดตั้ง 
1. Clone โปรเจค ใช้คำสั่งต่อไปนี้
   - git clone https://github.com/64160262/msm-support-chatbot.git
   - cd msm-support
2. สร้าง Virtual Environment ด้วยคำสั่ง
   - python -m venv venv
3. เปิดใช้งาน Virtual Environment
   - venv\Scripts\activate
การตั้งค่าฐานข้อมูล
1. ติดตั้ง Dependencies
   - pip install -r requirements.txt
2. สร้างฐานข้อมูล MySQL ด้วยคำสั่ง
   - CREATE DATABASE msm_support CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
3. แก้ไขการเชื่อมต่อฐานข้อมูลใน database.py
   - DATABASE_URL = "mysql+pymysql://username:password@localhost:3306/msm_support"
4. สร้างตาราง
  - python
  from database import engine
  from models import Base
  Base.metadata.create_all(bind=engine)
5. เพิ่มข้อมูลเริ่มต้น จากไฟล์ seeder.py จากนั้นเปิดการใช้งานด้วยคำสั่ง
   - python seeder.py

1. สร้าง LINE Bot ที่ [LINE Developers Console](https://developers.line.biz/)
2. แก้ไข Token ใน msm.py ที่ส่วน
   - line_bot_api = LineBotApi("YOUR_CHANNEL_ACCESS_TOKEN")
     handler = WebhookHandler("YOUR_CHANNEL_SECRET")
3. จากนั้นรันแอปพลิเคชัน ด้วย
   - uvicorn msm:app --reload
