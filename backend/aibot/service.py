from __future__ import annotations

from dataclasses import dataclass, field

from backend.luma.service import LumaService
from backend.shared.models import BotMessage, CheckInResult, Device, FeedbackResponse, FeedbackChannel, ScanResult
from backend.shared.repository import InMemoryRepository


@dataclass(slots=True)
class AIBotStore:
    devices: InMemoryRepository[Device] = field(default_factory=InMemoryRepository)
    messages: InMemoryRepository[BotMessage] = field(default_factory=InMemoryRepository)
    feedback: InMemoryRepository[FeedbackResponse] = field(default_factory=InMemoryRepository)


class AIBotService:
    def __init__(self, luma_service: LumaService, store: AIBotStore | None = None) -> None:
        self.luma = luma_service
        self.store = store or AIBotStore()

    def register_device(self, organization_id: str, device_name: str, device_type: str) -> Device:
        device = self.luma.open_device(organization_id, device_name, device_type)
        self.store.devices.add(device)
        return device

    def scan_qr(self, event_id: str, device_id: str, token: str) -> ScanResult:
        validation = self.luma.validate_ticket(event_id=event_id, token=token, device_id=device_id)
        if not validation.valid:
            result = CheckInResult.INVALID if validation.reason == "invalid_token" else CheckInResult.NOT_APPROVED
            if validation.reason == "revoked":
                result = CheckInResult.REVOKED
            return ScanResult(
                status=result,
                message=validation.reason or "scan rejected",
                registration_id=validation.registration_id,
                ticket_id=validation.ticket_id,
                guest_name=validation.guest_name,
                already_checked_in=validation.already_checked_in,
            )

        if validation.already_checked_in:
            return ScanResult(
                status=CheckInResult.DUPLICATE,
                message="Attendee already checked in",
                registration_id=validation.registration_id,
                ticket_id=validation.ticket_id,
                guest_name=validation.guest_name,
                already_checked_in=True,
            )

        checkin = self.luma.record_checkin(
            event_id=event_id,
            registration_id=validation.registration_id or "",
            ticket_id=validation.ticket_id or "",
            device_id=device_id,
            method="qr",
            result=CheckInResult.SUCCESS,
        )
        return ScanResult(
            status=CheckInResult.SUCCESS,
            message="Checked in successfully",
            registration_id=validation.registration_id,
            ticket_id=validation.ticket_id,
            guest_name=validation.guest_name,
            already_checked_in=False,
            checkin_id=checkin.id,
        )

    def log_message(
        self,
        event_id: str,
        message_text: str,
        channel: FeedbackChannel | str = FeedbackChannel.IN_APP,
        user_id: str | None = None,
        intent: str | None = None,
    ) -> BotMessage:
        message = BotMessage(
            event_id=event_id,
            channel=channel,
            message_text=message_text,
            user_id=user_id,
            intent=intent,
        )
        self.store.messages.add(message)
        return message

    def submit_feedback(
        self,
        event_id: str,
        registration_id: str,
        rating: int,
        feedback_text: str = "",
        would_return: bool | None = None,
    ) -> FeedbackResponse:
        feedback = self.luma.submit_feedback(
            event_id=event_id,
            registration_id=registration_id,
            rating=rating,
            feedback_text=feedback_text,
            would_return=would_return,
        )
        self.store.feedback.add(feedback)
        return feedback

    def report_incident(self, event_id: str, severity: str, title: str, description: str = "", reported_by: str | None = None):
        return self.luma.create_incident(
            event_id=event_id,
            severity=severity,
            title=title,
            description=description,
            reported_by=reported_by,
        )

    def live_status(self, event_id: str) -> dict[str, object]:
        summary = self.luma.get_event_summary(event_id)
        incidents = len(self.luma.store.incidents.find(lambda item: item.event_id == event_id))
        return {
            "eventId": summary.event_id,
            "checkedInCount": summary.checked_in,
            "expectedCount": summary.approved,
            "activeIncidents": incidents,
            "feedbackCount": summary.feedback_count,
            "averageRating": summary.average_rating,
        }
