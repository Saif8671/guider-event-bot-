from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = "EventFlow API"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:5173")


def get_settings() -> Settings:
    """Load runtime settings from the environment."""
    return Settings(
        app_name=os.getenv("APP_NAME", "EventFlow API"),
        environment=os.getenv("ENVIRONMENT", "development"),
        api_prefix=os.getenv("API_PREFIX", "/api/v1"),
        cors_origins=tuple(
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
            if origin.strip()
        ),
    )

