from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.deps import get_current_user, get_db
from app.schemas import RSVPCreate, RSVPRead
from app.services.commerce import create_rsvp

router = APIRouter(tags=["rsvps"])


@router.post("/events/{event_id}/rsvps", response_model=RSVPRead, status_code=status.HTTP_201_CREATED)
async def create_event_rsvp(
    event_id: str,
    payload: RSVPCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> RSVPRead:
    rsvp = await create_rsvp(db, event_id=event_id, current_user=current_user, payload=payload)
    return RSVPRead.model_validate(rsvp)
