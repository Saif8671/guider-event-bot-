from __future__ import annotations

from backend.aibot.service import AIBotService
from backend.luma.service import LumaService
from backend.shared.models import ApprovalMode, CheckInResult


def test_registration_scan_and_feedback_flow() -> None:
    luma = LumaService()
    bot = AIBotService(luma)

    org = luma.create_organization("EventFlow", "eventflow")
    event = luma.create_event(org.id, "Demo Event", "demo-event", approval_mode=ApprovalMode.AUTO)
    luma.publish_event(event.id)

    device = bot.register_device(org.id, "Gate Scanner", "tablet")
    result = luma.register_attendee(event.id, "Guest One", "guest@example.com")

    assert result.ticket is not None
    scan = bot.scan_qr(event.id, device.id, result.ticket.qr_payload)
    assert scan.status == CheckInResult.SUCCESS

    feedback = bot.submit_feedback(event.id, result.registration.id, 5, "Great event", True)
    assert feedback.rating == 5

    live_status = bot.live_status(event.id)
    assert live_status["checkedInCount"] == 1
    assert live_status["feedbackCount"] == 1

