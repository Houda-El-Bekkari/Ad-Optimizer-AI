from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import requests

from sqlalchemy.orm import Session

from pydantic import BaseModel

# =========================================================
# DATABASE
# =========================================================

from app.core.config import settings
from app.core.security import create_access_token, get_current_user
from app.database.database import SessionLocal
from app.db_models.chatbot import ChatMemorySummaryResponse, ChatMessageCreate, ChatMessageDB, ChatMessageResponse
from app.db_models.user import UserDB
from app.services.chat_memory import build_question_with_memory, get_memory_summary, get_memory_text, update_memory_summary

# =========================================================
# AUTH
# =========================================================

from app.schemas.user_schema import UserCreate

from app.services.auth_service import create_user

# =========================================================
# CASE 1 SERVICES
# =========================================================
from fastapi.responses import FileResponse

from app.services.data_collect import run_data_collect
from app.services.preprocessing import run_preprocessing
from app.services.anomaly import run_anomaly
from app.services.correlation import run_correlation
from app.services.health_score import run_health_score
from app.services.causal import run_causal
from app.services.optimizer import run_optimizer
from app.services.xai import run_xai
from app.services.llm import run_agent

# =========================================================
# CASE 2 SERVICES
# =========================================================

from app.services.case2_segmentation import run_case2_segmentation
from app.services.case2_correlation import run_case2_correlation
from app.services.case2_strategy import run_case2_strategy
from app.services.case2_feature_engineering import (
    run_case2_feature_engineering
)
from app.services.case2_prediction import run_case2_prediction
from app.services.case2_comparison import run_case2_comparison
from app.services.case2_xai import run_case2_xai
from app.services.auth_service import authenticate_user
from app.services.llm import run_case2_final_response, run_dashboard_summary
from app.services.audit_report import run_audit_report
from app.services.audit_report_pdf import run_audit_report_pdf

from app.services.llm import (
    run_case2_final_response,
    run_dashboard_summary
)

CHATBOT_WEBHOOK_URL = settings.chatbot_webhook_url

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(

    title="AdOptimizer AI Backend",

    version="1.0.0"
)

# =========================================================
# CORS
# =========================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=settings.cors_origin_list,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# =========================================================
# DATABASE SESSION
# =========================================================

def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()

# =========================================================
# REQUEST MODELS
# =========================================================

class LLMRequest(BaseModel):

    question: str

    mode: str = "auto"

    user_email: str | None = None


class ChatbotAskRequest(BaseModel):

    question: str

    mode: str = "auto"


class LoginRequest(BaseModel):

    email: str

    password: str


class ChatMessageSaveRequest(BaseModel):

    question: str

    response: str

    mode: str = "auto"


class Case2StrategyRequest(BaseModel):

    objectif: str

    budget: float

    plateforme: str

    produit: str


def get_user_by_email(db: Session, user_email: str):

    email = user_email.strip()

    if not email:

        raise HTTPException(status_code=400, detail="User email is required")

    db_user = db.query(UserDB).filter(UserDB.email == email).first()

    if not db_user:

        raise HTTPException(status_code=404, detail="User not found")

    return db_user


def create_chatbot_message(db: Session, payload: ChatMessageCreate):

    db_user = get_user_by_email(db, payload.user_email)

    return create_chatbot_message_for_user(

        db,

        db_user,

        payload.question,

        payload.response,

        payload.mode
    )


def create_chatbot_message_for_user(

    db: Session,

    db_user: UserDB,

    question: str,

    response: str,

    mode: str = "auto"

):

    question = question.strip()

    response = response.strip()

    if not question:

        raise HTTPException(status_code=400, detail="Question is required")

    if not response:

        raise HTTPException(status_code=400, detail="Response is required")

    chat_message = ChatMessageDB(

        user_id=db_user.id,

        question=question,

        response=response,

        mode=(mode or "auto").strip() or "auto"
    )

    db.add(chat_message)

    db.commit()

    db.refresh(chat_message)

    update_memory_summary(db, db_user.id)

    return chat_message


def extract_chatbot_answer(result):

    if isinstance(result, dict):

        for key in ["answer", "response", "message", "text", "output"]:

            value = result.get(key)

            if isinstance(value, str) and value.strip():

                return value.strip()

        nested = result.get("data")

        if isinstance(nested, dict):

            return extract_chatbot_answer(nested)

    if isinstance(result, list) and result:

        return extract_chatbot_answer(result[0])

    return ""


def get_chatbot_message_for_user(db: Session, message_id: int, user_email: str):

    db_user = get_user_by_email(db, user_email)

    chat_message = db.query(ChatMessageDB).filter(

        ChatMessageDB.id == message_id,

        ChatMessageDB.user_id == db_user.id

    ).first()

    if not chat_message:

        raise HTTPException(status_code=404, detail="Chatbot message not found")

    return chat_message


def call_workflow(url: str, payload: dict):

    try:

        response = requests.post(url, json=payload, timeout=180)

        response.raise_for_status()

    except requests.RequestException as exc:

        raise HTTPException(

            status_code=502,

            detail=f"Unable to reach workflow: {exc}"

        ) from exc

    try:

        return response.json()

    except ValueError:

        return {"response": response.text}

# =========================================================
# HOME
# =========================================================

@app.get("/")

def home():

    return {

        "status": "running",

        "message": "AdOptimizer AI Backend is running"
    }

# =========================================================
# AUTH ROUTES
# =========================================================

@app.post("/signup")

def signup(

    user: UserCreate,

    db: Session = Depends(get_db)

):

    new_user = create_user(db, user)

    return {

        "message": "User registered successfully",

        "username": new_user.username,

        "email": new_user.email
    }

@app.post("/login")

def login(

    user: LoginRequest,

    db: Session = Depends(get_db)

):

    db_user = authenticate_user(

        db,

        user.email,

        user.password
    )

    if not db_user:

        return {

            "success": False,

            "message": "Invalid email or password"
        }

    access_token, expires_at = create_access_token(

        {

            "sub": str(db_user.id),

            "email": db_user.email,

            "role": db_user.role
        }
    )

    return {

        "success": True,

        "message": "Login successful",

        "access_token": access_token,

        "token_type": "bearer",

        "expires_at": expires_at.isoformat(),

        "user_id": db_user.id,

        "username": db_user.username,

        "email": db_user.email,

        "role": db_user.role
    }


@app.get("/me")

def me(current_user: UserDB = Depends(get_current_user)):

    return {

        "user_id": current_user.id,

        "username": current_user.username,

        "email": current_user.email,

        "role": current_user.role
    }


@app.post("/workflows/case1/audit")

def trigger_case1_audit(current_user: UserDB = Depends(get_current_user)):

    return call_workflow(settings.case1_audit_webhook_url, {"user_id": current_user.id})


@app.post("/workflows/case1/dashboard")

def get_case1_dashboard(current_user: UserDB = Depends(get_current_user)):

    return call_workflow(settings.case1_dashboard_webhook_url, {"user_id": current_user.id})


@app.post("/workflows/case2/campaign")

def generate_case2_campaign(

    payload: dict,

    current_user: UserDB = Depends(get_current_user)

):

    workflow_payload = dict(payload)

    workflow_payload["user_id"] = current_user.id

    return call_workflow(settings.case2_campaign_webhook_url, workflow_payload)


@app.get("/chatbot/history", response_model=list[ChatMessageResponse])

def chatbot_history(

    current_user: UserDB = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    return db.query(ChatMessageDB).filter(

        ChatMessageDB.user_id == current_user.id

    ).order_by(

        ChatMessageDB.created_at.asc(),

        ChatMessageDB.id.asc()

    ).all()


@app.get("/chatbot/memory", response_model=ChatMemorySummaryResponse | None)

def chatbot_memory(

    current_user: UserDB = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    return get_memory_summary(db, current_user.id)


@app.post("/chatbot/ask")

def ask_chatbot(

    payload: ChatbotAskRequest,

    current_user: UserDB = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    question = payload.question.strip()

    if not question:

        raise HTTPException(status_code=400, detail="Question is required")

    memory_summary = get_memory_text(db, current_user.id)

    enriched_question = build_question_with_memory(
        question,
        memory_summary
    )

    result = run_agent(
        enriched_question,
        payload.mode
    )

    answer = extract_chatbot_answer(result)

    if answer:

        create_chatbot_message(

            db,

            ChatMessageCreate(

                question=question,

                response=answer,

                mode=payload.mode,

                user_email=current_user.email
            )
        )

    latest_memory = get_memory_text(db, current_user.id)

    if isinstance(result, dict):

        result["memory_used"] = bool(memory_summary)

        result["memory_summary"] = latest_memory

        return result

    return {

        "data": result,

        "memory_used": bool(memory_summary),

        "memory_summary": latest_memory
    }


@app.post("/chatbot/messages", response_model=ChatMessageResponse)

def save_chatbot_message(

    payload: ChatMessageSaveRequest,

    current_user: UserDB = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    return create_chatbot_message_for_user(

        db,

        current_user,

        payload.question,

        payload.response,

        payload.mode
    )


@app.delete("/chatbot/messages/{message_id}")

def delete_chatbot_message(

    message_id: int,

    current_user: UserDB = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    chat_message = db.query(ChatMessageDB).filter(

        ChatMessageDB.id == message_id,

        ChatMessageDB.user_id == current_user.id

    ).first()

    if not chat_message:

        raise HTTPException(status_code=404, detail="Chatbot message not found")

    user_id = chat_message.user_id

    db.delete(chat_message)

    db.commit()

    update_memory_summary(db, user_id)

    return {

        "success": True,

        "message": "Chatbot message deleted successfully"
    }


@app.delete("/chatbot/history")

def delete_chatbot_history(

    current_user: UserDB = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    deleted_count = db.query(ChatMessageDB).filter(

        ChatMessageDB.user_id == current_user.id

    ).delete(synchronize_session=False)

    db.commit()

    update_memory_summary(db, current_user.id)

    return {

        "success": True,

        "deleted_count": deleted_count,

        "message": "Chatbot history deleted successfully"
    }
# =========================================================
# AGENT 3 — VIGILANT
# =========================================================

@app.post("/data-collect")

def data_collect():

    return run_data_collect()


@app.post("/preprocessing")

def preprocessing():

    return run_preprocessing()


@app.post("/anomaly")

def anomaly():

    return run_anomaly()

# =========================================================
# AGENT 2 — DECIDEUR
# =========================================================

@app.post("/correlation")

def correlation():

    return run_correlation()


@app.post("/health-score")

def health_score():

    return run_health_score()


@app.post("/causal")

def causal():

    return run_causal()


@app.post("/optimizer")

def optimizer():

    return run_optimizer()

# =========================================================
# AGENT 1 — CONSULTANT
# =========================================================

@app.post("/xai")

def xai():

    return run_xai()


@app.post("/llm")

def agent(

    req: LLMRequest,

    db: Session = Depends(get_db)

):

    question_for_llm = req.question

    if req.user_email:

        db_user = get_user_by_email(db, req.user_email)

        memory_summary = get_memory_text(db, db_user.id)

        question_for_llm = build_question_with_memory(req.question, memory_summary)

    result = run_agent(

        question_for_llm,

        req.mode
    )

    if req.user_email:

        answer = extract_chatbot_answer(result)

        if answer:

            create_chatbot_message(

                db,

                ChatMessageCreate(

                    question=req.question,

                    response=answer,

                    mode=req.mode,

                    user_email=req.user_email
                )
            )

    return result


@app.post("/dashboard-summary")

def dashboard_summary(payload: dict):

    return run_dashboard_summary(payload)

# =========================================================
# CASE 2 — CREATION NOUVELLE CAMPAGNE
# =========================================================

@app.post("/audit-report")
def audit_report():
    return run_audit_report()


@app.get("/audit-report/pdf")
def audit_report_pdf():
    result = run_audit_report_pdf()

    if result.get("status") != "success":
        raise HTTPException(status_code=500, detail=result.get("message", "PDF indisponible"))

    return FileResponse(
        path=result["output_file"],
        media_type="application/pdf",
        filename="adoptimizer_audit_report.pdf",
    )


# =========================
# CAS 2 — CREATION NOUVELLE CAMPAGNE
# =========================

@app.post("/case2/segmentation")

def case2_segmentation():

    return run_case2_segmentation()


@app.post("/case2/correlation")

def case2_correlation():

    return run_case2_correlation()


@app.post("/case2/strategy")

def case2_strategy(req: Case2StrategyRequest):

    return run_case2_strategy(

        req.model_dump()
    )


@app.post("/case2/feature-engineering")

def case2_feature_engineering():

    return run_case2_feature_engineering()


@app.post("/case2/prediction")

def case2_prediction():

    return run_case2_prediction()


@app.post("/case2/comparison")

def case2_comparison():

    return run_case2_comparison()


@app.post("/case2/xai")

def case2_xai():

    return run_case2_xai()


@app.post("/case2/final-response")

def case2_final_response():

    return run_case2_final_response()
