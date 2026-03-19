from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserRead(APIModel):
    id: str
    name: str
    email: EmailStr
    role: str


class AuthTokens(APIModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class EventRead(APIModel):
    id: str
    organizer_id: str
    title: str
    slug: str
    description: str
    category: str
    location_type: str
    location_name: str | None
    location_address: str | None
    online_url: str | None
    start_at: datetime
    end_at: datetime
    timezone: str
    cover_image_url: str | None
    status: str
    capacity: int
    is_featured: bool
    tickets: list["TicketRead"] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None
