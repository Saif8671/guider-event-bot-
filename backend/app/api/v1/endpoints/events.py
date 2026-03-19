from fastapi import APIRouter, Depends, Query, status
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.deps import get_current_user, get_db
from app.schemas import EventCreate, EventListResponse, EventRead, EventUpdate
from app.services.events import create_event, delete_event, ensure_owner, get_event_or_404, list_events, update_event

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=EventListResponse)
async def get_events(
    q: str | None = Query(default=None),
    category: str | None = Query(default=None),
    location_type: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> EventListResponse:
    events = await list_events(
        db,
        published_only=True,
        q=q,
        category=category,
        location_type=location_type,
        limit=limit,
        offset=offset,
    )
    return EventListResponse(items=[EventRead.model_validate(event) for event in events])


@router.get("/{event_id}", response_model=EventRead)
async def get_event(event_id: str, db: AsyncSession = Depends(get_db)) -> EventRead:
    event = await get_event_or_404(db, event_id)
    if event.status != "published":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return EventRead.model_validate(event)


@router.post("", response_model=EventRead, status_code=status.HTTP_201_CREATED)
async def create_event_endpoint(
    payload: EventCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> EventRead:
    event = await create_event(db, organizer=current_user, payload=payload)
    return EventRead.model_validate(event)


@router.patch("/{event_id}", response_model=EventRead)
async def update_event_endpoint(
    event_id: str,
    payload: EventUpdate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> EventRead:
    event = await get_event_or_404(db, event_id)
    ensure_owner(event, current_user)
    updated = await update_event(db, event=event, payload=payload)
    return EventRead.model_validate(updated)


@router.delete("/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event_endpoint(
    event_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> None:
    event = await get_event_or_404(db, event_id)
    ensure_owner(event, current_user)
    await delete_event(db, event=event)


@router.post("/{event_id}/publish", response_model=EventRead)
async def publish_event(
    event_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> EventRead:
    event = await get_event_or_404(db, event_id)
    ensure_owner(event, current_user)
    event.status = "published"
    await db.commit()
    await db.refresh(event)
    return EventRead.model_validate(event)


@router.post("/{event_id}/unpublish", response_model=EventRead)
async def unpublish_event(
    event_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> EventRead:
    event = await get_event_or_404(db, event_id)
    ensure_owner(event, current_user)
    event.status = "draft"
    await db.commit()
    await db.refresh(event)
    return EventRead.model_validate(event)
