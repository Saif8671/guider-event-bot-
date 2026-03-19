from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import select

from app.core.security import hash_password
from app.db.session import async_session_maker
from app.models import Event, Ticket, User


async def seed_demo_data() -> None:
    async with async_session_maker() as db:
        organizer = await _get_or_create_user(
            db,
            email="organizer@example.com",
            name="Amina Organizer",
            role="organizer",
            password="password123",
        )
        await _get_or_create_user(
            db,
            email="attendee@example.com",
            name="Sam Attendee",
            role="attendee",
            password="password123",
        )

        event_specs = [
            {
                "slug": "future-builders-summit-2026",
                "title": "Future Builders Summit 2026",
                "description": "A hands-on meetup for founders, product builders, and operators shipping ambitious ideas.",
                "category": "conference",
                "location_type": "offline",
                "location_name": "Riyadh Front",
                "location_address": "Riyadh, Saudi Arabia",
                "online_url": None,
                "start_at": datetime.now(timezone.utc) + timedelta(days=14, hours=4),
                "end_at": datetime.now(timezone.utc) + timedelta(days=14, hours=8),
                "timezone": "Asia/Riyadh",
                "capacity": 250,
                "is_featured": True,
                "tickets": [
                    {"name": "General Admission", "ticket_type": "free", "price_cents": 0, "quantity_total": 100},
                    {"name": "Workshop Pass", "ticket_type": "paid", "price_cents": 2500, "quantity_total": 50},
                ],
            },
            {
                "slug": "product-design-night-online",
                "title": "Product Design Night",
                "description": "An online design critique and portfolio review session for product teams.",
                "category": "workshop",
                "location_type": "online",
                "location_name": "Virtual",
                "location_address": None,
                "online_url": "https://meet.example.com/product-design-night",
                "start_at": datetime.now(timezone.utc) + timedelta(days=7, hours=2),
                "end_at": datetime.now(timezone.utc) + timedelta(days=7, hours=4),
                "timezone": "Asia/Riyadh",
                "capacity": 150,
                "is_featured": False,
                "tickets": [
                    {"name": "RSVP", "ticket_type": "free", "price_cents": 0, "quantity_total": 150},
                ],
            },
            {
                "slug": "founder-breakfast-roundtable",
                "title": "Founder Breakfast Roundtable",
                "description": "An invite-only breakfast to trade notes on fundraising, hiring, and distribution.",
                "category": "networking",
                "location_type": "offline",
                "location_name": "Diriyah House",
                "location_address": "Diriyah, Riyadh",
                "online_url": None,
                "start_at": datetime.now(timezone.utc) + timedelta(days=21, hours=1),
                "end_at": datetime.now(timezone.utc) + timedelta(days=21, hours=3),
                "timezone": "Asia/Riyadh",
                "capacity": 40,
                "is_featured": True,
                "tickets": [
                    {"name": "Founder Seat", "ticket_type": "paid", "price_cents": 4500, "quantity_total": 24},
                ],
            },
        ]

        for spec in event_specs:
            event = await _get_or_create_event(db, organizer=organizer, **spec)
            await _ensure_tickets(db, event=event, tickets=spec["tickets"])


async def _get_or_create_user(
    db,
    *,
    email: str,
    name: str,
    role: str,
    password: str,
) -> User:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is not None:
        return user

    user = User(
        id=uuid4().hex,
        name=name,
        email=email,
        password_hash=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def _get_or_create_event(
    db,
    *,
    organizer: User,
    slug: str,
    title: str,
    description: str,
    category: str,
    location_type: str,
    location_name: str | None,
    location_address: str | None,
    online_url: str | None,
    start_at: datetime,
    end_at: datetime,
    timezone: str,
    capacity: int,
    is_featured: bool,
    tickets: list[dict[str, object]],
) -> Event:
    result = await db.execute(select(Event).where(Event.slug == slug))
    event = result.scalar_one_or_none()
    if event is not None:
        return event

    event = Event(
        id=uuid4().hex,
        organizer_id=organizer.id,
        title=title,
        slug=slug,
        description=description,
        category=category,
        location_type=location_type,
        location_name=location_name,
        location_address=location_address,
        online_url=online_url,
        start_at=start_at,
        end_at=end_at,
        timezone=timezone,
        capacity=capacity,
        is_featured=is_featured,
        status="published",
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return event


async def _ensure_tickets(db, *, event: Event, tickets: list[dict[str, object]]) -> None:
    for ticket_payload in tickets:
        name = str(ticket_payload["name"])
        result = await db.execute(select(Ticket).where(Ticket.event_id == event.id, Ticket.name == name))
        if result.scalar_one_or_none() is not None:
            continue

        ticket = Ticket(
            id=uuid4().hex,
            event_id=event.id,
            name=name,
            ticket_type=str(ticket_payload["ticket_type"]),
            price_cents=int(ticket_payload["price_cents"]),
            currency="USD",
            quantity_total=int(ticket_payload["quantity_total"]),
            quantity_sold=0,
            requires_approval=False,
            is_active=True,
        )
        db.add(ticket)
    await db.commit()


def main() -> None:
    asyncio.run(seed_demo_data())


if __name__ == "__main__":
    main()
