from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Event, User
from app.schemas.event import EventCreate, EventUpdate


def slugify(value: str) -> str:
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug or "event"


async def create_event(db: AsyncSession, *, organizer: User, payload: EventCreate) -> Event:
    slug = f"{slugify(payload.title)}-{uuid4().hex[:8]}"
    event = Event(
        id=uuid4().hex,
        organizer_id=organizer.id,
        title=payload.title,
        slug=slug,
        description=payload.description,
        category=payload.category,
        location_type=payload.location_type,
        location_name=payload.location_name,
        location_address=payload.location_address,
        online_url=payload.online_url,
        start_at=payload.start_at,
        end_at=payload.end_at,
        timezone=payload.timezone,
        cover_image_url=payload.cover_image_url,
        capacity=payload.capacity,
        is_featured=payload.is_featured,
        status="draft",
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return event


async def list_events(
    db: AsyncSession,
    *,
    published_only: bool = True,
    q: str | None = None,
    category: str | None = None,
    location_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[Event]:
    stmt = select(Event)
    conditions = []
    if published_only:
        conditions.append(Event.status == "published")
    if q:
        like = f"%{q}%"
        conditions.append(or_(Event.title.ilike(like), Event.description.ilike(like), Event.category.ilike(like)))
    if category:
        conditions.append(Event.category == category)
    if location_type:
        conditions.append(Event.location_type == location_type)
    if conditions:
        stmt = stmt.where(and_(*conditions))
    stmt = stmt.order_by(Event.is_featured.desc(), Event.start_at.asc()).limit(limit).offset(offset)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_event_or_404(db: AsyncSession, event_id: str) -> Event:
    result = await db.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return event


def ensure_owner(event: Event, user: User) -> None:
    if event.organizer_id != user.id and user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed")


async def update_event(db: AsyncSession, *, event: Event, payload: EventUpdate) -> Event:
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(event, key, value)
    await db.commit()
    await db.refresh(event)
    return event


async def delete_event(db: AsyncSession, *, event: Event) -> None:
    await db.delete(event)
    await db.commit()

