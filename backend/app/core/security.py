from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import uuid4

from jose import jwt
from passlib.context import CryptContext

from app.core.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
settings = get_settings()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def hash_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def create_session_id() -> str:
    return uuid4().hex


def create_access_token(subject: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_expires_minutes)
    payload = {"sub": subject, "role": role, "type": "access", "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def create_refresh_token(subject: str, session_id: str) -> tuple[str, datetime]:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expires_days)
    payload = {
        "sub": subject,
        "sid": session_id,
        "type": "refresh",
        "jti": uuid4().hex,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256"), expire


def decode_token(token: str) -> dict[str, str]:
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
