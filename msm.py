from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
import re
import nltk
import pythainlp
from pythainlp.tokenize import word_tokenize
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot import LineBotApi, WebhookHandler

# Initialize FastAPI app
app = FastAPI()

# Initialize LineBot API and Webhook Handler
line_bot_api = LineBotApi("3Hh2AylEsGnyzZzqCXO8M8geW2w8LsjTLamhdPDPgDhR9BGIPhLlD/QrNaDqaFePyGbpWoAxYLi/yKGV7FT2PSigcnpeGhspHiVpNIDn6P5xosdPtPlHXevCSzl3G3U0Rhn6tFgKOhLlmkxjs9YkHgdB04t89/1O/w1cDnyilFU=")
handler = WebhookHandler("8425e5bab23fb36801b29f6c2c9e5778")

# Download NLTK resources
nltk.download('punkt')

# Load the MRC pipeline
mrcpipeline = pipeline("question-answering", model="MyMild/finetune_iapp_thaiqa")

# Define university-related FAQs
msm_rules = {
    "เปลี่ยนสถานะ สถานะลูกบ้าน สถานะ เปลี่ยนสถานะลูกบ้านยังไง เปลี่ยนสถานะลูกบ้านทำอย่างไร": """ขั้นตอนการเปลี่ยนสถานะลูกบ้าน
    1. เลือกเมนูการเปลี่ยนสถานะ
    2. เลือกปุ่ม "เพิ่มโครงการ +"
    3. กรอกข้อมูลโครงการที่ต้องการลงทะเบียน

    กรณีเจ้าของห้อง:
        - ให้นำเอกสารดังต่อไปนี้ จัดส่งตามที่อยู่ … 
        - โดยประดับอากรสแตมป์ 30 บาท

    กรณีผู้รับมอบอำนาจ:
        - ให้ทำการดาวน์โหลดหนังสือมอบอำนาจ ตามไฟล์ที่ระบุ 
        - กรอกข้อมูลให้ครบถ้วน 
        - จากนั้นแนบไฟล์ดังกล่าว 
        - พร้อมทั้งแนบสำเนาบัตรประชาชนเจ้าของห้อง

    กรณีผู้เช่า:
        - ให้ทำการดาวน์โหลดหนังสือยินยอมให้ใช้บริการ Smarty Application 
        - กรอกข้อมูลให้ครบถ้วน 
        - จากนั้นแนบไฟล์ดังกล่าว 
        - พร้อมทั้งแนบสำเนาบัตรประชาชนเจ้าของห้อง
        - จากนั้นรอการอนุมัติสิทธิ์การเข้าใช้งานแอปพลิเคชัน
    """,

    "Smartyทำอะไรได้บ้าง ฟีจเจอร์": """ฟีจเจอร์การใช้งานของ Smarty
    1. สามารถเช็คยอดค่าใช้จ่าย ด้วยเมนู "จ่ายค่าส่วนกลาง"
    2. สามารถชำระค่าส่วนกลาง และรับใบเสร็จได้ทันที
    3. ดูประวัติการชำระค่าส่วนกลาง
    4. ดูการจัดส่งพัสดุ (เฉพาะบริการจัดส่งพัสดุ ไปรษณีย์ไทย เท่านั้น)
    """,

    "จ่ายค่าส่วนกลาง ชำระค่าส่วนกลาง" :"""สามารถชำระค่าส่วนกลางได้ผ่าน Smarty Applicantion และรับใบเสร็จได้ทันที ผ่านเมนูค้างชำระ """,

    "การจัดส่งพัสดุ" : """Smarty Applicantion สามารถติดตามการจัดส่งพัสดุ (เฉพาะบริการจัดส่งพัสดุ ไปรษณีย์ไทย เท่านั้น)""",

    "ประวัติการชำระ ประวัติการชำระค่าส่วนกลาง ประวัติการชำระ ประวัติการจ่าย": """สามารถดูประวัติการชำระค่าส่วนกลางได้จากเมนู ประวัติการชำระ""",

    "สแกนบัตร บัตรประชาชน": """กรณีต้องการลงทะเบียนแต่ไม่สามารถสแกนบัตรประชาชนได้ (แจ้งว่า Error)
    เบื้องต้นให้ทำการอัพเดท version ของแอปพลิเคชัน ให้เป็นเวอร์ชันล่าสุด
    """,

    "ลงทะเบียนไม่ได้": """กรณีทำการลงทะเบียนใช้งาน Smarty ไม่ได้ 
    ให้ลูกทำบ้านทำการแจ้งรายละเอียดใน Form คนตกหล่นที่ Form Smart Living
    """,
    # "ขั้นตอนการลงทะเบียน ลงทะเบียนยังไง ลงทะเบียน": "https://cdn.discordapp.com/attachments/1189168145244618753/1307927834387415070/openPreregister.png?ex=673eb937&is=673d67b7&hm=81611e8b060e68d289a146a549171c0315296a8e9eab06724af5b3d6e7c716d7&",
    "ข้อมูลส่วนตัว เปลี่ยนแปลงรายละเอียด": """กรณีลูกบ้านต้องการเปลี่ยนแปลงรายละเอียดข้อมูลส่วนตัว 
    ให้ลูกทำบ้านทำการแจ้งรายละเอียดใน Form Smart Living
    """,

    # "สวัสดี Hi Hello หวัดดี ดีจ้า" : "สวัสดีครับ/ค่ะ🙏🏻 นี่คือ MSMBot💻 ที่จะช่วยตอบคำถามเกี่ยวกับ Smarty Support Hello there This is MSMBot💻 that'll help to answer question about Smarty Application",

    "ขอบคุณ THANKS THX": """ขอบคุณครับ/ค่ะ 🙇🏻 ที่ใช้งาน MSMBot, Thanks you for using our MSMBot💻
    """
}

# Preprocess university rules keys for better matching
preprocessed_msm_rules = {pythainlp.util.normalize(keyword): answer for keyword, answer in msm_rules.items()}

# Create context from all answers in msm_rules
university_context = ' '.join(preprocessed_msm_rules.values())

# Function to handle university questions
def handle_university_question(question: str) -> str:
    # Check if the input consists of special characters only
    if re.match(r'^[\W_]+$', question):
        return "ขออภัยไม่สามารถตอบคำถามนี้ได้🙇🏻🙏🏻"
    else:
        question = question.upper()
        matched_key = None
        max_matched_tokens = 0
        
        if question in msm_rules:
            return msm_rules[question]
        else:
            # Tokenize the question
            tokens = word_tokenize(question)
            # Check for substring matches
            for key in msm_rules:
                matched_tokens = sum(1 for token in tokens if token in key)
                if matched_tokens > max_matched_tokens:
                    max_matched_tokens = matched_tokens
                    matched_key = key
                
            # If a matching key is found, return its corresponding value
            if matched_key:
                return msm_rules[matched_key]
            
            # If no exact or substring match is found, use the model
            answer = mrcpipeline(question=question, context=university_context)
            return answer['answer']

# FastAPI Endpoints

@app.get("/")
async def verify_line_webhook(request: Request):
    """Handle GET requests for webhook verification"""
    challenge = request.query_params.get("hub.challenge")
    if challenge:
        return JSONResponse(content={"challenge": challenge}, status_code=200)
    return JSONResponse(content={"message": "Welcome to the chatbot API"}, status_code=200)

@app.post("/university")
async def university_chatbot(request: Request):
    """Handle POST requests for the university chatbot"""
    try:
        # Try to get data from multiple sources
        message = None
        
        # Check content type
        content_type = request.headers.get("content-type", "").lower()
        
        if "application/json" in content_type:
            # Handle JSON data
            body = await request.json()
            message = body.get("message")
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            # Handle form data
            form = await request.form()
            message = form.get("message")
        else:
            # Try query parameters
            message = request.query_params.get("message")

        if not message:
            raise HTTPException(
                status_code=400,
                detail="Message not found. Please provide a message in the request body, form data, or query parameters."
            )

        print(f"Received message: {message}")  # Debug print
        answer = handle_university_question(message)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/webhook")
async def line_webhook(request: Request):
    """Handle LINE Webhook"""
    try:
        body = await request.body()
        body_text = body.decode("utf-8")
        signature = request.headers.get("X-Line-Signature", "")

        try:
            handler.handle(body_text, signature)
        except InvalidSignatureError:
            raise HTTPException(status_code=400, detail="Invalid signature")
        except LineBotApiError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        return JSONResponse(content={"message": "OK"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# LINE bot message handler
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """Handle LINE text messages"""
    try:
        user_message = event.message.text
        if not user_message:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Please provide a valid question.")
            )
            return

        answer = handle_university_question(user_message)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=answer)
        )
    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"An error occurred: {str(e)}")
        )

# Add error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"}
    )

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
