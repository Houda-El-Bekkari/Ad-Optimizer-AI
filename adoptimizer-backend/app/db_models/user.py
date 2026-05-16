from sqlalchemy import Column, Integer, String

from app.database.database import Base


class UserDB(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)

    username = Column(String, unique=True)

    email = Column(String, unique=True)

    password = Column(String)

    role = Column(String)