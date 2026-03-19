from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.deps import get_current_user, get_db
from app.schemas import AuthResponse, LoginRequest, LogoutRequest, MeResponse, RefreshRequest, SignupRequest, UserRead
from app.services.auth import create_user_and_session, login_user, refresh_session, revoke_session

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, db: AsyncSession = Depends(get_db)) -> AuthResponse:
    user, access_token, refresh_token = await create_user_and_session(
        db,
        name=payload.name,
        email=payload.email,
        password=payload.password,
        role=payload.role,
    )
    return AuthResponse(
        user=UserRead.model_validate(user),
        tokens={"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"},
    )


@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)) -> AuthResponse:
    user, access_token, refresh_token = await login_user(db, email=payload.email, password=payload.password)
    return AuthResponse(
        user=UserRead.model_validate(user),
        tokens={"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"},
    )


@router.post("/refresh", response_model=AuthResponse)
async def refresh(payload: RefreshRequest, db: AsyncSession = Depends(get_db)) -> AuthResponse:
    user, access_token, refresh_token = await refresh_session(db, refresh_token=payload.refresh_token)
    return AuthResponse(
        user=UserRead.model_validate(user),
        tokens={"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"},
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: LogoutRequest, db: AsyncSession = Depends(get_db)) -> None:
    await revoke_session(db, refresh_token=payload.refresh_token)


@router.get("/me", response_model=MeResponse)
async def me(current_user=Depends(get_current_user)) -> MeResponse:
    return MeResponse(user=UserRead.model_validate(current_user))

