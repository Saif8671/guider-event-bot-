from app.schemas.auth import AuthResponse, LoginRequest, LogoutRequest, MeResponse, RefreshRequest, SignupRequest
from app.schemas.commerce import (
    OrderRead,
    PurchaseResponse,
    RSVPCreate,
    RSVPRead,
    TicketCreate,
    TicketListResponse,
    TicketRead,
)
from app.schemas.common import APIModel, AuthTokens, EventRead, UserRead
from app.schemas.event import EventCreate, EventListResponse, EventUpdate

EventRead.model_rebuild(_types_namespace={"TicketRead": TicketRead})

__all__ = [
    "APIModel",
    "AuthResponse",
    "AuthTokens",
    "EventCreate",
    "EventListResponse",
    "EventRead",
    "EventUpdate",
    "OrderRead",
    "PurchaseResponse",
    "LoginRequest",
    "LogoutRequest",
    "MeResponse",
    "RSVPCreate",
    "RSVPRead",
    "RefreshRequest",
    "SignupRequest",
    "TicketCreate",
    "TicketListResponse",
    "TicketRead",
    "UserRead",
]
