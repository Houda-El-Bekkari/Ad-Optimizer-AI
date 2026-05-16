from fastapi import FastAPI
from fastapi import Depends
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.orm import Session

from pydantic import BaseModel

# =========================================================
# DATABASE
# =========================================================

from app.database.database import SessionLocal

# =========================================================
# AUTH
# =========================================================

from app.schemas.user_schema import UserCreate

from app.services.auth_service import create_user

# =========================================================
# CASE 1 SERVICES
# =========================================================

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

from app.services.llm import (
    run_case2_final_response,
    run_dashboard_summary
)

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

    allow_origins=["*"],

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


class Case2StrategyRequest(BaseModel):

    objectif: str

    budget: float

    plateforme: str

    produit: str

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

    user: dict,

    db: Session = Depends(get_db)

):

    db_user = authenticate_user(

        db,

        user["email"],

        user["password"]
    )

    if not db_user:

        return {

            "success": False,

            "message": "Invalid email or password"
        }

    return {

        "success": True,

        "message": "Login successful",

        "username": db_user.username,

        "email": db_user.email,

        "role": db_user.role
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

def agent(req: LLMRequest):

    return run_agent(

        req.question,

        req.mode
    )


@app.post("/dashboard-summary")

def dashboard_summary(payload: dict):

    return run_dashboard_summary(payload)

# =========================================================
# CASE 2 — CREATION NOUVELLE CAMPAGNE
# =========================================================

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