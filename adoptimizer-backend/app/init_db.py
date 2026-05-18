from app.database.database import engine

from app.db_models.chatbot import ChatMemorySummaryDB, ChatMessageDB

from app.db_models.user import UserDB

UserDB.metadata.create_all(bind=engine)

ChatMessageDB.metadata.create_all(bind=engine)

ChatMemorySummaryDB.metadata.create_all(bind=engine)

print("Database tables created successfully.")
