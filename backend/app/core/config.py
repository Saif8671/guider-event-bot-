from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Event Platform API"
    environment: str = "development"
    database_url: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/event_platform")
    jwt_secret: str = Field(default="change-me")
    jwt_access_token_expires_minutes: int = 60
    jwt_refresh_token_expires_days: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()

