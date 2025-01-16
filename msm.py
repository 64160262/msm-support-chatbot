from fastapi import FastAPI, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
import re
import nltk
import pythainlp
from pythainlp.tokenize import word_tokenize
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import LineBotApiError, InvalidSignatureError
from linebot import LineBotApi, WebhookHandler
from sqlalchemy.orm import Session
from database import get_db
from models import MsmRule
import schemas
from typing import Dict
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Initialize LineBot API and Webhook Handler
line_bot_api = LineBotApi("3Hh2AylEsGnyzZzqCXO8M8geW2w8LsjTLamhdPDPgDhR9BGIPhLlD/QrNaDqaFePyGbpWoAxYLi/yKGV7FT2PSigcnpeGhspHiVpNIDn6P5xosdPtPlHXevCSzl3G3U0Rhn6tFgKOhLlmkxjs9YkHgdB04t89/1O/w1cDnyilFU=")
handler = WebhookHandler("8425e5bab23fb36801b29f6c2c9e5778")

# Download NLTK resources
nltk.download('punkt')

# Load the MRC pipeline
mrcpipeline = pipeline("question-answering", model="MyMild/finetune_iapp_thaiqa")

def find_similar_keywords(question: str, msm_rules: dict, threshold: float = 0.2) -> str:
    question_tokens = set(word_tokenize(pythainlp.util.normalize(question.lower())))
    
    best_match = None
    highest_score = 0

    for key, value in msm_rules.items():
        key_tokens = set(word_tokenize(pythainlp.util.normalize(key.lower())))
        
        overlap = len(question_tokens & key_tokens)
        score = overlap / len(key_tokens) if key_tokens else 0
        
        if score > highest_score and score >= threshold:
            highest_score = score
            best_match = value

    return best_match if best_match else "ขออภัยไม่สามารถตอบคำถามนี้ได้🙇🏻🙏🏻"

def check_time_period(text):
    time_keywords = {
        'เดือนนี้': 'current_month',
        'เดือนที่แล้ว': 'last_month',
        'เดือนหน้า': 'next_month',
        'วันนี้': 'today'
    }
    
    for keyword in time_keywords:
        if keyword in text:
            return time_keywords[keyword]
    return None

def modify_response_with_time_period(response: str, time_period: str) -> str:
    time_prefix = {
        'current_month': 'สำหรับเดือนนี้ ',
        'last_month': 'สำหรับเดือนที่แล้ว ',
        'next_month': 'สำหรับเดือนหน้า ',
        'today': 'สำหรับวันนี้ '
    }
    return f"{time_prefix.get(time_period, '')}{response}"

def handle_msm_question(question: str, db: Session) -> str:
    print(f"Original question: {question}")
    question = question.strip()
    print(f"Stripped question: {question}")
    
    # ดึงข้อมูลจากฐานข้อมูล
    rules = db.query(MsmRule).all()
    msm_rules = {rule.keywords: rule.response for rule in rules}
    
    # เช็คคำถามเกี่ยวกับราคา
    price_keywords = ['เท่าไหร่', 'เท่าไร', 'กี่บาท', 'ราคา']
    if any(keyword in question for keyword in price_keywords):
        return "ขออภัยไม่สามารถตอบคำถามนี้ได้🙇🏻🙏🏻 โปรดติดต่อผู้จัดการอาคาร หรือตรวจสอบรายละเอียดผ่านแอปพลิเคชัน Smarty ค่ะ"
    
    if not question or re.match(r'^[\W_]+$', question):
        print("Input is empty or contains only special characters.")
        return "ขออภัยไม่สามารถตอบคำถามนี้ได้🙇🏻🙏🏻"

    # เช็คช่วงเวลา
    time_period = check_time_period(question)
    normalized_question = pythainlp.util.normalize(question.upper())
    print(f"Normalized question: {normalized_question}")
    
    # กรณีมีการระบุช่วงเวลา
    if time_period:
        # ตรวจสอบการตรงกันมั้ย
        for key, response in msm_rules.items():
            normalized_key = pythainlp.util.normalize(key.upper())
            if normalized_question == normalized_key:
                print(f"Exact match found: {key}")
                return modify_response_with_time_period(response, time_period)

        # ถ้าไม่ตรงกัน ตรวจสอบการตรงกันบางส่วน
        for key, response in msm_rules.items():
            normalized_key = pythainlp.util.normalize(key.upper())
            key_words = set(word_tokenize(normalized_key))
            question_words = set(word_tokenize(normalized_question))
            
            overlap = len(key_words.intersection(question_words))
            overlap_ratio = overlap / len(key_words) if key_words else 0
            
            if overlap_ratio >= 0.5:
                print(f"Partial match found: {key} (Overlap ratio: {overlap_ratio})")
                return modify_response_with_time_period(response, time_period)

        # ถ้ายังไม่พบ ใช้การค้นหาคำที่คล้ายกัน
        matched_context = find_similar_keywords(normalized_question, msm_rules, threshold=0.2)
        if matched_context != "ขออภัยไม่สามารถตอบคำถามนี้ได้🙇🏻🙏🏻":
            print(f"Similar keywords match found")
            return modify_response_with_time_period(matched_context, time_period)
    else:
        # กรณีไม่มีการระบุช่วงเวลา
        for key, response in msm_rules.items():
            normalized_key = pythainlp.util.normalize(key.upper())
            if normalized_question == normalized_key:
                print(f"Exact match found: {key}")
                return response

        for key, response in msm_rules.items():
            normalized_key = pythainlp.util.normalize(key.upper())
            key_words = set(word_tokenize(normalized_key))
            question_words = set(word_tokenize(normalized_question))
            
            overlap = len(key_words.intersection(question_words))
            overlap_ratio = overlap / len(key_words) if key_words else 0
            
            if overlap_ratio >= 0.4:
                print(f"Partial match found: {key} (Overlap ratio: {overlap_ratio})")
                return response

        matched_context = find_similar_keywords(normalized_question, msm_rules, threshold=0.2)
        if matched_context != "ขออภัยไม่สามารถตอบคำถามนี้ได้🙇🏻🙏🏻":
            print(f"Similar keywords match found")
            return matched_context

    print("No suitable response found. Returning fallback response.")
    return "ขออภัยไม่สามารถตอบคำถามนี้ได้🙇🏻🙏🏻"

# FastAPI Endpoints
@app.get("/")
async def verify_line_webhook(request: Request):
    challenge = request.query_params.get("hub.challenge")
    if challenge:
        return JSONResponse(content={"challenge": challenge}, status_code=200)
    return JSONResponse(content={"message": "Welcome to the chatbot API"}, status_code=200)

class ChatMessage(BaseModel):
    message: str

@app.post("/msm")
async def msm_chatbot(chat_message: ChatMessage, db: Session = Depends(get_db)):
    try:
        if not chat_message.message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        response = handle_msm_question(chat_message.message, db)
        return {"response": response}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/webhook")
async def line_webhook(request: Request):
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

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    try:
        user_message = event.message.text
        if not user_message:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="กรุณาระบุคำถามค่ะ")
            )
            return

        db = next(get_db())
        response = handle_msm_question(user_message, db)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response)
        )
    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"เกิดข้อผิดพลาด: {str(e)}")
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "เกิดข้อผิดพลาดภายในระบบ"}
    )

@app.post("/admin/rules/", response_model=schemas.MsmRule)
def create_rule(rule: schemas.MsmRuleCreate, db: Session = Depends(get_db)):
    db_rule = MsmRule(**rule.dict())
    db.add(db_rule)
    db.commit()
    db.refresh(db_rule)
    return db_rule

@app.get("/admin/rules/", response_model=list[schemas.MsmRule])
def read_rules(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    rules = db.query(MsmRule).offset(skip).limit(limit).all()
    return rules

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)