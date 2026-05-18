from datetime import datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from app.database.database import Base


class ChatRequest(BaseModel):
    question: str
    response: str = ""
    mode: str = "auto"
    user_email: str | None = None


class ChatMessageCreate(BaseModel):
    question: str
    response: str
    mode: str = "auto"
    user_email: str


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    question: str
    response: str
    mode: str
    created_at: datetime


class ChatMemorySummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    summary: str
    message_count: int
    updated_at: datetime


class ChatMessageDB(Base):
    __tablename__ = "chatbot_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    mode = Column(String, default="auto", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ChatMemorySummaryDB(Base):
    __tablename__ = "chatbot_memory_summaries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    summary = Column(Text, default="", nullable=False)
    message_count = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
