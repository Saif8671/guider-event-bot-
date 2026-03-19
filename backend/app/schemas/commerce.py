from datetime import datetime

from pydantic import Field

from app.schemas.common import APIModel


class TicketCreate(APIModel):
    name: str = Field(min_length=2, max_length=255)
    ticket_type: str = Field(default="free", max_length=50)
    price_cents: int = Field(default=0, ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    quantity_total: int = Field(default=1, ge=1)
    sale_starts_at: datetime | None = None
    sale_ends_at: datetime | None = None
    requires_approval: bool = False
    is_active: bool = True


class TicketRead(APIModel):
    id: str
    event_id: str
    name: str
    ticket_type: str
    price_cents: int
    currency: str
    quantity_total: int
    quantity_sold: int
    remaining_quantity: int
    sale_starts_at: datetime | None = None
    sale_ends_at: datetime | None = None
    requires_approval: bool
    is_active: bool
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TicketListResponse(APIModel):
    items: list[TicketRead]


class RSVPCreate(APIModel):
    ticket_id: str | None = None


class RSVPRead(APIModel):
    id: str
    event_id: str
    user_id: str
    ticket_id: str | None = None
    status: str
    qr_code_token: str
    checked_in_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class OrderRead(APIModel):
    id: str
    user_id: str
    event_id: str
    ticket_id: str | None = None
    payment_provider: str
    provider_payment_id: str | None = None
    amount_cents: int
    currency: str
    status: str
    refunded_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PurchaseResponse(APIModel):
    order: OrderRead
    rsvp: RSVPRead | None = None
