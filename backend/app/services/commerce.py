from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Event, Order, RSVP, Ticket, User
from app.schemas.commerce import RSVPCreate, TicketCreate
from app.services.events import ensure_owner, get_event_or_404


async def list_tickets(db: AsyncSession, *, event_id: str) -> list[Ticket]:
    result = await db.execute(select(Ticket).where(Ticket.event_id == event_id).order_by(Ticket.price_cents.asc()))
    return list(result.scalars().all())


async def create_ticket(
    db: AsyncSession,
    *,
    event: Event,
    current_user: User,
    payload: TicketCreate,
) -> Ticket:
    ensure_owner(event, current_user)
    ticket = Ticket(
        id=uuid4().hex,
        event_id=event.id,
        name=payload.name,
        ticket_type=payload.ticket_type,
        price_cents=payload.price_cents,
        currency=payload.currency.upper(),
        quantity_total=payload.quantity_total,
        sale_starts_at=payload.sale_starts_at,
        sale_ends_at=payload.sale_ends_at,
        requires_approval=payload.requires_approval,
        is_active=payload.is_active,
    )
    db.add(ticket)
    await db.commit()
    await db.refresh(ticket)
    return ticket


async def create_rsvp(
    db: AsyncSession,
    *,
    event_id: str,
    current_user: User,
    payload: RSVPCreate,
) -> RSVP:
    event = await get_event_or_404(db, event_id)
    if event.status != "published":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Event is not published")

    ticket = None
    if payload.ticket_id:
        result = await db.execute(select(Ticket).where(Ticket.id == payload.ticket_id, Ticket.event_id == event.id))
        ticket = result.scalar_one_or_none()
        if ticket is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    else:
        result = await db.execute(
            select(Ticket).where(Ticket.event_id == event.id, Ticket.is_active.is_(True), Ticket.price_cents == 0)
        )
        ticket = result.scalars().first()

    if ticket is not None and ticket.price_cents > 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Use ticket purchase for paid tickets")
    if ticket is not None and not ticket.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticket is not active")
    if ticket is not None and ticket.quantity_total > 0 and ticket.quantity_sold >= ticket.quantity_total:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ticket is sold out")

    if event.capacity > 0:
        count_result = await db.execute(
            select(func.count(RSVP.id)).where(RSVP.event_id == event.id, RSVP.status == "confirmed")
        )
        confirmed_count = count_result.scalar_one()
        if confirmed_count >= event.capacity:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Event is sold out")

    existing = await db.execute(select(RSVP).where(RSVP.event_id == event.id, RSVP.user_id == current_user.id))
    existing_rsvp = existing.scalar_one_or_none()
    if existing_rsvp is not None:
        return existing_rsvp

    rsvp = RSVP(
        id=uuid4().hex,
        event_id=event.id,
        user_id=current_user.id,
        ticket_id=ticket.id if ticket is not None else None,
        status="confirmed",
        qr_code_token=uuid4().hex,
    )
    db.add(rsvp)
    if ticket is not None:
        ticket.quantity_sold += 1
    await db.commit()
    await db.refresh(rsvp)
    return rsvp


async def purchase_ticket(
    db: AsyncSession,
    *,
    event_id: str,
    ticket_id: str,
    current_user: User,
) -> tuple[Order, RSVP | None]:
    event = await get_event_or_404(db, event_id)
    if event.status != "published":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Event is not published")

    ticket_result = await db.execute(
        select(Ticket).where(Ticket.id == ticket_id, Ticket.event_id == event.id).with_for_update()
    )
    ticket = ticket_result.scalar_one_or_none()
    if ticket is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    if not ticket.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticket is not active")
    if ticket.sale_starts_at and ticket.sale_starts_at > datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticket sale has not started")
    if ticket.sale_ends_at and ticket.sale_ends_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Ticket sale has ended")
    if ticket.quantity_total > 0 and ticket.quantity_sold >= ticket.quantity_total:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ticket is sold out")

    existing_order = await db.execute(
        select(Order).where(Order.event_id == event.id, Order.user_id == current_user.id, Order.ticket_id == ticket.id)
    )
    if existing_order.scalar_one_or_none() is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ticket already purchased")

    if event.capacity > 0:
        confirmed_result = await db.execute(
            select(func.count(RSVP.id)).where(RSVP.event_id == event.id, RSVP.status == "confirmed")
        )
        confirmed_count = confirmed_result.scalar_one()
        if confirmed_count >= event.capacity:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Event is sold out")

    ticket.quantity_sold += 1
    order = Order(
        id=uuid4().hex,
        user_id=current_user.id,
        event_id=event.id,
        ticket_id=ticket.id,
        payment_provider="manual",
        provider_payment_id=f"demo_{uuid4().hex[:12]}",
        amount_cents=ticket.price_cents,
        currency=ticket.currency,
        status="paid",
    )
    rsvp_result = await db.execute(select(RSVP).where(RSVP.event_id == event.id, RSVP.user_id == current_user.id))
    rsvp = rsvp_result.scalar_one_or_none()
    if rsvp is None:
        rsvp = RSVP(
            id=uuid4().hex,
            event_id=event.id,
            user_id=current_user.id,
            ticket_id=ticket.id,
            status="confirmed",
            qr_code_token=uuid4().hex,
        )
        db.add(rsvp)
    elif rsvp.ticket_id is None:
        rsvp.ticket_id = ticket.id

    db.add(order)
    await db.commit()
    await db.refresh(order)
    if rsvp is not None:
        await db.refresh(rsvp)
    return order, rsvp


async def get_order_or_404(db: AsyncSession, order_id: str) -> Order:
    result = await db.execute(select(Order).where(Order.id == order_id))
    order = result.scalar_one_or_none()
    if order is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    return order
