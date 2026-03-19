from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import (
    create_access_token,
    create_refresh_token,
    create_session_id,
    hash_password,
    hash_token,
    verify_password,
)
from app.models import AuthSession, User


async def create_user_and_session(
    db: AsyncSession,
    *,
    name: str,
    email: str,
    password: str,
    role: str,
) -> tuple[User, str, str]:
    existing = await db.execute(select(User).where(User.email == email.lower()))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        id=create_session_id(),
        name=name,
        email=email.lower(),
        password_hash=hash_password(password),
        role=role,
        is_active=True,
    )
    session_id = create_session_id()
    refresh_token, refresh_expiry = create_refresh_token(user.id, session_id)
    session = AuthSession(
        id=session_id,
        user=user,
        refresh_token_hash=hash_token(refresh_token),
        expires_at=refresh_expiry,
    )
    db.add(user)
    db.add(session)
    await db.commit()
    await db.refresh(user)
    return user, create_access_token(user.id, user.role), refresh_token


async def login_user(db: AsyncSession, *, email: str, password: str) -> tuple[User, str, str]:
    result = await db.execute(select(User).where(User.email == email.lower()))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    session_id = create_session_id()
    refresh_token, refresh_expiry = create_refresh_token(user.id, session_id)
    session = AuthSession(
        id=session_id,
        user_id=user.id,
        refresh_token_hash=hash_token(refresh_token),
        expires_at=refresh_expiry,
    )
    db.add(session)
    await db.commit()
    return user, create_access_token(user.id, user.role), refresh_token


async def refresh_session(db: AsyncSession, *, refresh_token: str) -> tuple[User, str, str]:
    payload = _decode_refresh_token(refresh_token)
    result = await db.execute(select(AuthSession).where(AuthSession.id == payload["sid"]))
    session = result.scalar_one_or_none()
    if session is None or session.revoked_at is not None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is revoked")
    if session.expires_at <= datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    if session.refresh_token_hash != hash_token(refresh_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token mismatch")

    user_result = await db.execute(select(User).where(User.id == session.user_id))
    user = user_result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    new_session_id = create_session_id()
    new_refresh_token, refresh_expiry = create_refresh_token(user.id, new_session_id)
    session.revoked_at = datetime.now(timezone.utc)
    db.add(
        AuthSession(
            id=new_session_id,
            user_id=user.id,
            refresh_token_hash=hash_token(new_refresh_token),
            expires_at=refresh_expiry,
        )
    )
    await db.commit()
    return user, create_access_token(user.id, user.role), new_refresh_token


async def revoke_session(db: AsyncSession, *, refresh_token: str) -> None:
    payload = _decode_refresh_token(refresh_token)
    result = await db.execute(select(AuthSession).where(AuthSession.id == payload["sid"]))
    session = result.scalar_one_or_none()
    if session is None:
        return
    session.revoked_at = datetime.now(timezone.utc)
    await db.commit()


def _decode_refresh_token(refresh_token: str) -> dict[str, str]:
    from app.core.security import decode_token

    try:
        payload = decode_token(refresh_token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc
    if payload.get("type") != "refresh" or "sid" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    return payload
