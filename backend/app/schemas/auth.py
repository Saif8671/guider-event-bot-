from typing import Literal

from pydantic import EmailStr, Field

from app.schemas.common import APIModel, AuthTokens, UserRead


class SignupRequest(APIModel):
    name: str = Field(min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    role: Literal["organizer", "attendee"] = "attendee"


class LoginRequest(APIModel):
    email: EmailStr
    password: str


class RefreshRequest(APIModel):
    refresh_token: str


class LogoutRequest(APIModel):
    refresh_token: str


class AuthResponse(APIModel):
    user: UserRead
    tokens: AuthTokens


class MeResponse(APIModel):
    user: UserRead
