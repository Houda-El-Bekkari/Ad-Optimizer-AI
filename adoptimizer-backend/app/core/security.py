import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.database import SessionLocal
from app.db_models.user import UserDB

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def create_access_token(data: dict) -> tuple[str, datetime]:
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    expires_at_ts = int(expires_at.timestamp())
    payload = data.copy()
    payload.update({"exp": expires_at_ts})
    header = {"alg": settings.jwt_algorithm, "typ": "JWT"}

    token = _encode_jwt(header, payload)

    return token, expires_at


def get_auth_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_auth_db),
) -> UserDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = _decode_jwt(token)

    try:
        user_id = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        user_id = int(user_id)
    except ValueError:
        raise credentials_exception

    user = db.query(UserDB).filter(UserDB.id == user_id).first()

    if not user:
        raise credentials_exception

    return user


def _encode_jwt(header: dict, payload: dict) -> str:
    encoded_header = _base64url_encode(json.dumps(header, separators=(",", ":")).encode())
    encoded_payload = _base64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{encoded_header}.{encoded_payload}".encode()
    signature = _sign(signing_input)

    return f"{encoded_header}.{encoded_payload}.{signature}"


def _decode_jwt(token: str) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        encoded_header, encoded_payload, signature = token.split(".")
        signing_input = f"{encoded_header}.{encoded_payload}".encode()
        expected_signature = _sign(signing_input)

        if not hmac.compare_digest(signature, expected_signature):
            raise credentials_exception

        header = json.loads(_base64url_decode(encoded_header))

        if header.get("alg") != settings.jwt_algorithm:
            raise credentials_exception

        payload = json.loads(_base64url_decode(encoded_payload))
        exp = int(payload.get("exp", 0))

        if exp < int(datetime.now(timezone.utc).timestamp()):
            raise credentials_exception

        return payload
    except (ValueError, json.JSONDecodeError, TypeError):
        raise credentials_exception


def _sign(value: bytes) -> str:
    digest = hmac.new(
        settings.jwt_secret_key.encode(),
        value,
        hashlib.sha256,
    ).digest()

    return _base64url_encode(digest)


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
