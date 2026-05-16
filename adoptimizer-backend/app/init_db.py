from app.database.database import engine

from app.db_models.user import UserDB

UserDB.metadata.create_all(bind=engine)

print("Users table created successfully.")