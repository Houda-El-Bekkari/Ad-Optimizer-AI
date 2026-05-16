from sqlalchemy.orm import Session

from passlib.context import CryptContext

from app.db_models.user import UserDB

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def create_user(db: Session, user):

    hashed_password = pwd_context.hash(
        user.password
    )

    db_user = UserDB(

        username=user.username,

        email=user.email,

        password=hashed_password,

        role=user.role
    )

    db.add(db_user)

    db.commit()

    db.refresh(db_user)

    return db_user
def authenticate_user(

    db: Session,

    email: str,

    password: str
):

    user = db.query(UserDB).filter(

        UserDB.email == email

    ).first()

    if not user:

        return None

    if not pwd_context.verify(

        password,

        user.password
    ):

        return None

    return user