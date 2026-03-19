from __future__ import annotations

from dataclasses import dataclass, field

from backend.shared.models import (
    ApprovalMode,
    AuditLog,
    CheckIn,
    CheckInResult,
    Device,
    Event,
    EventIncident,
    EventStatus,
    EventSummary,
    FeedbackResponse,
    FeedbackSurvey,
    Invoice,
    Notification,
    Organization,
    Payment,
    Registration,
    RegistrationAnswer,
    RegistrationQuestion,
    RegistrationResult,
    RegistrationStatus,
    ScanSession,
    Ticket,
    TicketStatus,
    TicketValidationResult,
    User,
    utcnow,
)
from backend.shared.repository import InMemoryRepository
from backend.shared.security import generate_ticket_token, hash_token


@dataclass(slots=True)
class LumaStore:
    organizations: InMemoryRepository[Organization] = field(default_factory=InMemoryRepository)
    users: InMemoryRepository[User] = field(default_factory=InMemoryRepository)
    events: InMemoryRepository[Event] = field(default_factory=InMemoryRepository)
    questions: InMemoryRepository[RegistrationQuestion] = field(default_factory=InMemoryRepository)
    registrations: InMemoryRepository[Registration] = field(default_factory=InMemoryRepository)
    answers: InMemoryRepository[RegistrationAnswer] = field(default_factory=InMemoryRepository)
    tickets: InMemoryRepository[Ticket] = field(default_factory=InMemoryRepository)
    checkins: InMemoryRepository[CheckIn] = field(default_factory=InMemoryRepository)
    payments: InMemoryRepository[Payment] = field(default_factory=InMemoryRepository)
    invoices: InMemoryRepository[Invoice] = field(default_factory=InMemoryRepository)
    notifications: InMemoryRepository[Notification] = field(default_factory=InMemoryRepository)
    surveys: InMemoryRepository[FeedbackSurvey] = field(default_factory=InMemoryRepository)
    feedback: InMemoryRepository[FeedbackResponse] = field(default_factory=InMemoryRepository)
    incidents: InMemoryRepository[EventIncident] = field(default_factory=InMemoryRepository)
    audit_logs: InMemoryRepository[AuditLog] = field(default_factory=InMemoryRepository)
    devices: InMemoryRepository[Device] = field(default_factory=InMemoryRepository)
    scan_sessions: InMemoryRepository[ScanSession] = field(default_factory=InMemoryRepository)


class LumaService:
    def __init__(self, store: LumaStore | None = None) -> None:
        self.store = store or LumaStore()

    def create_organization(self, name: str, slug: str) -> Organization:
        organization = Organization(name=name, slug=slug)
        return self.store.organizations.add(organization)

    def create_user(self, organization_id: str, email: str, name: str, role: str) -> User:
        user = User(organization_id=organization_id, email=email, name=name, role=role)
        return self.store.users.add(user)

    def create_event(
        self,
        organization_id: str,
        title: str,
        slug: str,
        description: str = "",
        approval_mode: ApprovalMode = ApprovalMode.AUTO,
        capacity: int | None = None,
    ) -> Event:
        event = Event(
            organization_id=organization_id,
            title=title,
            slug=slug,
            description=description,
            approval_mode=approval_mode,
            capacity=capacity,
        )
        return self.store.events.add(event)

    def publish_event(self, event_id: str) -> Event:
        event = self._require_event(event_id)
        event.status = EventStatus.PUBLISHED
        self.store.events.update(event)
        return event

    def update_event(self, event_id: str, **changes) -> Event:
        event = self._require_event(event_id)
        for key, value in changes.items():
            if hasattr(event, key):
                setattr(event, key, value)
        event.updated_at = utcnow()
        self.store.events.update(event)
        return event

    def cancel_event(self, event_id: str) -> Event:
        event = self._require_event(event_id)
        event.status = EventStatus.CANCELLED
        self.store.events.update(event)
        return event

    def complete_event(self, event_id: str) -> Event:
        event = self._require_event(event_id)
        event.status = EventStatus.COMPLETED
        self.store.events.update(event)
        return event

    def add_registration_question(
        self,
        event_id: str,
        label: str,
        field_key: str,
        question_type: str,
        required: bool = False,
        sort_order: int = 0,
    ) -> RegistrationQuestion:
        self._require_event(event_id)
        question = RegistrationQuestion(
            event_id=event_id,
            label=label,
            field_key=field_key,
            question_type=question_type,
            required=required,
            sort_order=sort_order,
        )
        return self.store.questions.add(question)

    def register_attendee(
        self,
        event_id: str,
        guest_name: str,
        guest_email: str,
        guest_phone: str = "",
        answers: list[dict[str, str]] | None = None,
        source: str = "web",
    ) -> RegistrationResult:
        event = self._require_event(event_id)
        registration = Registration(
            event_id=event_id,
            guest_name=guest_name,
            guest_email=guest_email,
            guest_phone=guest_phone,
            source=source,
        )
        self.store.registrations.add(registration)

        for answer in answers or []:
            self.store.answers.add(
                RegistrationAnswer(
                    registration_id=registration.id,
                    question_id=answer["questionId"],
                    answer_text=answer.get("value", ""),
                    answer_json=answer.get("json", ""),
                )
            )

        if event.approval_mode == ApprovalMode.AUTO:
            registration.status = RegistrationStatus.APPROVED
            self.store.registrations.update(registration)
            ticket = self.issue_ticket(registration.id)
            return RegistrationResult(registration=registration, ticket=ticket, status=registration.status)

        if event.approval_mode == ApprovalMode.WAITLIST:
            registration.status = RegistrationStatus.WAITLISTED
            self.store.registrations.update(registration)
            return RegistrationResult(registration=registration, ticket=None, status=registration.status)

        registration.status = RegistrationStatus.PENDING
        self.store.registrations.update(registration)
        return RegistrationResult(registration=registration, ticket=None, status=registration.status)

    def approve_registration(self, registration_id: str, approved_by: str | None = None) -> RegistrationResult:
        registration = self._require_registration(registration_id)
        registration.status = RegistrationStatus.APPROVED
        registration.approved_by = approved_by
        self.store.registrations.update(registration)
        ticket = self._get_ticket_by_registration(registration_id)
        if ticket is None:
            ticket = self.issue_ticket(registration_id)
        return RegistrationResult(registration=registration, ticket=ticket, status=registration.status)

    def decline_registration(self, registration_id: str) -> RegistrationResult:
        registration = self._require_registration(registration_id)
        registration.status = RegistrationStatus.DECLINED
        self.store.registrations.update(registration)
        return RegistrationResult(registration=registration, ticket=self._get_ticket_by_registration(registration_id), status=registration.status)

    def waitlist_registration(self, registration_id: str) -> RegistrationResult:
        registration = self._require_registration(registration_id)
        registration.status = RegistrationStatus.WAITLISTED
        self.store.registrations.update(registration)
        return RegistrationResult(registration=registration, ticket=self._get_ticket_by_registration(registration_id), status=registration.status)

    def issue_ticket(self, registration_id: str) -> Ticket:
        registration = self._require_registration(registration_id)
        event = self._require_event(registration.event_id)
        existing = self._get_ticket_by_registration(registration_id)
        if existing is not None and existing.status == TicketStatus.ACTIVE:
            return existing
        token = generate_ticket_token()
        ticket = Ticket(
            registration_id=registration_id,
            event_id=event.id,
            token_hash=hash_token(token),
            token_prefix=token[:8],
            qr_payload=token,
        )
        self.store.tickets.add(ticket)
        self.store.audit_logs.add(
            AuditLog(
                event_id=event.id,
                actor_type="system",
                action="ticket.issued",
                payload_json=f'{{"ticketId":"{ticket.id}","registrationId":"{registration_id}"}}',
            )
        )
        return ticket

    def resend_ticket(self, registration_id: str) -> Ticket:
        existing = self._get_ticket_by_registration(registration_id)
        if existing is not None and existing.status == TicketStatus.ACTIVE:
            return existing
        return self.issue_ticket(registration_id)

    def revoke_ticket(self, ticket_id: str) -> Ticket:
        ticket = self._require_ticket(ticket_id)
        ticket.status = TicketStatus.REVOKED
        ticket.revoked_at = ticket.revoked_at or ticket.issued_at
        self.store.tickets.update(ticket)
        return ticket

    def validate_ticket(self, event_id: str, token: str, device_id: str) -> TicketValidationResult:
        event = self._require_event(event_id)
        ticket = self._find_ticket_by_token(token)
        if ticket is None or ticket.event_id != event.id:
            return TicketValidationResult(
                valid=False,
                registration_id=None,
                ticket_id=None,
                guest_name=None,
                status=None,
                already_checked_in=False,
                reason="invalid_token",
            )

        registration = self._require_registration(ticket.registration_id)
        already_checked_in = self._has_checkin(ticket.id)
        if ticket.status == TicketStatus.REVOKED:
            return TicketValidationResult(
                valid=False,
                registration_id=registration.id,
                ticket_id=ticket.id,
                guest_name=registration.guest_name,
                status=registration.status,
                already_checked_in=already_checked_in,
                reason="revoked",
            )

        return TicketValidationResult(
            valid=registration.status == RegistrationStatus.APPROVED,
            registration_id=registration.id,
            ticket_id=ticket.id,
            guest_name=registration.guest_name,
            status=registration.status,
            already_checked_in=already_checked_in,
            reason="" if registration.status == RegistrationStatus.APPROVED else "not_approved",
        )

    def record_checkin(
        self,
        event_id: str,
        registration_id: str,
        ticket_id: str,
        device_id: str,
        method: str,
        result: CheckInResult,
        meta_json: str = "",
    ) -> CheckIn:
        checkin = CheckIn(
            event_id=event_id,
            registration_id=registration_id,
            ticket_id=ticket_id,
            device_id=device_id,
            method=method,
            result=result,
            meta_json=meta_json,
        )
        self.store.checkins.add(checkin)
        return checkin

    def get_guest_list(self, event_id: str) -> list[dict[str, object]]:
        self._require_event(event_id)
        registrations = self.store.registrations.find(lambda item: item.event_id == event_id)
        tickets = {ticket.registration_id: ticket for ticket in self.store.tickets.find(lambda item: item.event_id == event_id)}
        return [
            {
                "registrationId": registration.id,
                "guestName": registration.guest_name,
                "guestEmail": registration.guest_email,
                "status": registration.status.value,
                "ticketStatus": tickets.get(registration.id).status.value if tickets.get(registration.id) else None,
            }
            for registration in registrations
        ]

    def submit_feedback(
        self,
        event_id: str,
        registration_id: str,
        rating: int,
        feedback_text: str = "",
        would_return: bool | None = None,
        survey_id: str | None = None,
    ) -> FeedbackResponse:
        self._require_event(event_id)
        feedback = FeedbackResponse(
            event_id=event_id,
            registration_id=registration_id,
            rating=rating,
            feedback_text=feedback_text,
            would_return=would_return,
            survey_id=survey_id,
        )
        self.store.feedback.add(feedback)
        return feedback

    def create_survey(self, event_id: str, title: str, is_active: bool = True) -> FeedbackSurvey:
        self._require_event(event_id)
        survey = FeedbackSurvey(event_id=event_id, title=title, is_active=is_active)
        return self.store.surveys.add(survey)

    def open_device(self, organization_id: str, device_name: str, device_type: str) -> Device:
        device = Device(organization_id=organization_id, device_name=device_name, device_type=device_type)
        return self.store.devices.add(device)

    def open_scan_session(self, event_id: str, device_id: str) -> ScanSession:
        self._require_event(event_id)
        self._require_device(device_id)
        session = ScanSession(event_id=event_id, device_id=device_id)
        return self.store.scan_sessions.add(session)

    def close_scan_session(self, session_id: str) -> ScanSession:
        session = self._require_scan_session(session_id)
        session.ended_at = session.ended_at or session.started_at
        self.store.scan_sessions.update(session)
        return session

    def create_incident(
        self,
        event_id: str,
        severity: str,
        title: str,
        description: str = "",
        reported_by: str | None = None,
    ) -> EventIncident:
        self._require_event(event_id)
        incident = EventIncident(
            event_id=event_id,
            severity=severity,
            title=title,
            description=description,
            reported_by=reported_by,
        )
        return self.store.incidents.add(incident)

    def get_event_summary(self, event_id: str) -> EventSummary:
        self._require_event(event_id)
        registrations = self.store.registrations.find(lambda item: item.event_id == event_id)
        approved = [item for item in registrations if item.status == RegistrationStatus.APPROVED]
        checkins = self.store.checkins.find(lambda item: item.event_id == event_id and item.result == CheckInResult.SUCCESS)
        feedback = self.store.feedback.find(lambda item: item.event_id == event_id)
        ratings = [item.rating for item in feedback if item.rating is not None]
        average_rating = sum(ratings) / len(ratings) if ratings else None
        return EventSummary(
            event_id=event_id,
            registrations=len(registrations),
            approved=len(approved),
            checked_in=len(checkins),
            feedback_count=len(feedback),
            average_rating=average_rating,
        )

    def _find_ticket_by_token(self, token: str) -> Ticket | None:
        token_hash = hash_token(token)
        matches = self.store.tickets.find(lambda item: item.token_hash == token_hash)
        return matches[0] if matches else None

    def _get_ticket_by_registration(self, registration_id: str) -> Ticket | None:
        matches = self.store.tickets.find(lambda item: item.registration_id == registration_id)
        return matches[0] if matches else None

    def _has_checkin(self, ticket_id: str) -> bool:
        return bool(self.store.checkins.find(lambda item: item.ticket_id == ticket_id and item.result == CheckInResult.SUCCESS))

    def _require_event(self, event_id: str) -> Event:
        event = self.store.events.get(event_id)
        if event is None:
            raise KeyError(f"event not found: {event_id}")
        return event

    def _require_registration(self, registration_id: str) -> Registration:
        registration = self.store.registrations.get(registration_id)
        if registration is None:
            raise KeyError(f"registration not found: {registration_id}")
        return registration

    def _require_ticket(self, ticket_id: str) -> Ticket:
        ticket = self.store.tickets.get(ticket_id)
        if ticket is None:
            raise KeyError(f"ticket not found: {ticket_id}")
        return ticket

    def _require_device(self, device_id: str) -> Device:
        device = self.store.devices.get(device_id)
        if device is None:
            raise KeyError(f"device not found: {device_id}")
        return device

    def _require_scan_session(self, session_id: str) -> ScanSession:
        session = self.store.scan_sessions.get(session_id)
        if session is None:
            raise KeyError(f"scan session not found: {session_id}")
        return session
