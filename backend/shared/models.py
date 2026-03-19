from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from backend.shared.ids import new_id


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EventStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class LocationType(str, Enum):
    PHYSICAL = "physical"
    VIRTUAL = "virtual"
    HYBRID = "hybrid"


class ApprovalMode(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    WAITLIST = "waitlist"


class RegistrationStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    WAITLISTED = "waitlisted"
    CANCELLED = "cancelled"


class TicketStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"


class CheckInResult(str, Enum):
    SUCCESS = "success"
    DUPLICATE = "duplicate"
    INVALID = "invalid"
    REVOKED = "revoked"
    NOT_APPROVED = "not_approved"


class FeedbackChannel(str, Enum):
    IN_APP = "in_app"
    EMAIL = "email"
    QR = "qr"


@dataclass(slots=True)
class Organization:
    name: str
    slug: str
    id: str = field(default_factory=lambda: new_id("org"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class User:
    organization_id: str
    email: str
    name: str
    role: str
    id: str = field(default_factory=lambda: new_id("usr"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Event:
    organization_id: str
    title: str
    slug: str
    description: str = ""
    status: EventStatus = EventStatus.DRAFT
    start_at: datetime = field(default_factory=utcnow)
    end_at: datetime = field(default_factory=utcnow)
    timezone: str = "UTC"
    location_type: LocationType = LocationType.PHYSICAL
    location_name: str = ""
    location_address: str = ""
    virtual_url: str = ""
    capacity: int | None = None
    approval_mode: ApprovalMode = ApprovalMode.AUTO
    ticket_type: str = "general"
    cover_image_url: str = ""
    guest_list_visible: bool = True
    id: str = field(default_factory=lambda: new_id("evt"))
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def is_published(self) -> bool:
        return self.status == EventStatus.PUBLISHED


@dataclass(slots=True)
class RegistrationQuestion:
    event_id: str
    label: str
    field_key: str
    question_type: str
    required: bool = False
    sort_order: int = 0
    id: str = field(default_factory=lambda: new_id("rq"))


@dataclass(slots=True)
class RegistrationAnswer:
    registration_id: str
    question_id: str
    answer_text: str = ""
    answer_json: str = ""
    id: str = field(default_factory=lambda: new_id("ra"))


@dataclass(slots=True)
class Registration:
    event_id: str
    guest_name: str
    guest_email: str
    guest_phone: str = ""
    status: RegistrationStatus = RegistrationStatus.PENDING
    source: str = "web"
    approved_by: str | None = None
    id: str = field(default_factory=lambda: new_id("reg"))
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Ticket:
    registration_id: str
    event_id: str
    token_hash: str
    token_prefix: str
    qr_payload: str
    status: TicketStatus = TicketStatus.ACTIVE
    id: str = field(default_factory=lambda: new_id("tic"))
    issued_at: datetime = field(default_factory=utcnow)
    revoked_at: datetime | None = None


@dataclass(slots=True)
class Payment:
    registration_id: str
    event_id: str
    provider: str
    amount: float
    currency: str
    status: str
    provider_payment_id: str | None = None
    id: str = field(default_factory=lambda: new_id("pay"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Invoice:
    payment_id: str
    invoice_number: str
    pdf_url: str = ""
    id: str = field(default_factory=lambda: new_id("inv"))
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Notification:
    event_id: str
    channel: str
    template_key: str
    status: str
    registration_id: str | None = None
    sent_at: datetime | None = None
    id: str = field(default_factory=lambda: new_id("ntf"))


@dataclass(slots=True)
class Device:
    organization_id: str
    device_name: str
    device_type: str
    status: str = "active"
    last_seen_at: datetime | None = None
    id: str = field(default_factory=lambda: new_id("dev"))


@dataclass(slots=True)
class ScanSession:
    event_id: str
    device_id: str
    started_at: datetime = field(default_factory=utcnow)
    ended_at: datetime | None = None
    id: str = field(default_factory=lambda: new_id("scan"))


@dataclass(slots=True)
class CheckIn:
    event_id: str
    registration_id: str
    ticket_id: str
    device_id: str
    method: str
    result: CheckInResult
    checked_in_at: datetime = field(default_factory=utcnow)
    meta_json: str = ""
    id: str = field(default_factory=lambda: new_id("chk"))


@dataclass(slots=True)
class BotMessage:
    event_id: str
    channel: FeedbackChannel | str
    message_text: str
    user_id: str | None = None
    intent: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: new_id("msg"))


@dataclass(slots=True)
class FeedbackSurvey:
    event_id: str
    title: str
    is_active: bool = True
    created_at: datetime = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: new_id("fs"))


@dataclass(slots=True)
class FeedbackResponse:
    event_id: str
    registration_id: str
    rating: int
    feedback_text: str = ""
    would_return: bool | None = None
    survey_id: str | None = None
    submitted_at: datetime = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: new_id("fr"))


@dataclass(slots=True)
class EventIncident:
    event_id: str
    severity: str
    title: str
    description: str = ""
    reported_by: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: new_id("inc"))


@dataclass(slots=True)
class AuditLog:
    event_id: str
    actor_type: str
    action: str
    payload_json: str = ""
    actor_id: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    id: str = field(default_factory=lambda: new_id("log"))


@dataclass(slots=True)
class RegistrationResult:
    registration: Registration
    ticket: Ticket | None
    status: RegistrationStatus


@dataclass(slots=True)
class TicketValidationResult:
    valid: bool
    registration_id: str | None
    ticket_id: str | None
    guest_name: str | None
    status: RegistrationStatus | None
    already_checked_in: bool
    reason: str = ""


@dataclass(slots=True)
class ScanResult:
    status: CheckInResult
    message: str
    registration_id: str | None = None
    ticket_id: str | None = None
    guest_name: str | None = None
    already_checked_in: bool = False
    checkin_id: str | None = None


@dataclass(slots=True)
class EventSummary:
    event_id: str
    registrations: int
    approved: int
    checked_in: int
    feedback_count: int
    average_rating: float | None

