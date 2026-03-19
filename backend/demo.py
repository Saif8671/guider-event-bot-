from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.aibot.service import AIBotService
from backend.luma.service import LumaService
from backend.shared.models import ApprovalMode


def build_demo_system() -> tuple[LumaService, AIBotService]:
    luma = LumaService()
    bot = AIBotService(luma)
    return luma, bot


def main() -> None:
    luma, bot = build_demo_system()
    org = luma.create_organization("EventFlow", "eventflow")
    event = luma.create_event(
        org.id,
        "Founder Summit",
        "founder-summit",
        approval_mode=ApprovalMode.AUTO,
    )
    luma.publish_event(event.id)
    device = bot.register_device(org.id, "Gate Scanner", "tablet")
    registration = luma.register_attendee(event.id, "Saif Rahman", "saif@example.com")
    scan_result = bot.scan_qr(event.id, device.id, registration.ticket.qr_payload if registration.ticket else "")
    print(
        {
            "organization": org.name,
            "event": event.title,
            "device": device.device_name,
            "registrationStatus": registration.status.value,
            "scanStatus": scan_result.status.value,
            "message": scan_result.message,
            "liveStatus": bot.live_status(event.id),
        }
    )


if __name__ == "__main__":
    main()
