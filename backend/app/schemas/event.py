from datetime import datetime

from pydantic import Field

from app.schemas.common import APIModel, EventRead


class EventCreate(APIModel):
    title: str = Field(min_length=2, max_length=255)
    description: str = Field(default="", max_length=50_000)
    category: str = Field(default="general", max_length=100)
    location_type: str = Field(default="offline", max_length=50)
    location_name: str | None = Field(default=None, max_length=255)
    location_address: str | None = Field(default=None, max_length=255)
    online_url: str | None = Field(default=None, max_length=500)
    start_at: datetime
    end_at: datetime
    timezone: str = Field(min_length=1, max_length=100)
    cover_image_url: str | None = Field(default=None, max_length=500)
    capacity: int = Field(default=0, ge=0)
    is_featured: bool = False


class EventUpdate(APIModel):
    title: str | None = Field(default=None, min_length=2, max_length=255)
    description: str | None = Field(default=None, max_length=50_000)
    category: str | None = Field(default=None, max_length=100)
    location_type: str | None = Field(default=None, max_length=50)
    location_name: str | None = Field(default=None, max_length=255)
    location_address: str | None = Field(default=None, max_length=255)
    online_url: str | None = Field(default=None, max_length=500)
    start_at: datetime | None = None
    end_at: datetime | None = None
    timezone: str | None = Field(default=None, max_length=100)
    cover_image_url: str | None = Field(default=None, max_length=500)
    capacity: int | None = Field(default=None, ge=0)
    is_featured: bool | None = None
    status: str | None = Field(default=None, max_length=50)


class EventListResponse(APIModel):
    items: list[EventRead]

