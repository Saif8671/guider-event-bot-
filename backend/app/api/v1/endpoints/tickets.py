from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.deps import get_current_user, get_db
from app.schemas import TicketCreate, TicketListResponse, TicketRead
from app.services.commerce import create_ticket, list_tickets
from app.services.events import get_event_or_404

router = APIRouter(tags=["tickets"])


@router.get("/events/{event_id}/tickets", response_model=TicketListResponse)
async def get_event_tickets(event_id: str, db: AsyncSession = Depends(get_db)) -> TicketListResponse:
    tickets = await list_tickets(db, event_id=event_id)
    return TicketListResponse(items=[TicketRead.model_validate(ticket) for ticket in tickets])


@router.post("/events/{event_id}/tickets", response_model=TicketRead, status_code=status.HTTP_201_CREATED)
async def create_event_ticket(
    event_id: str,
    payload: TicketCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> TicketRead:
    event = await get_event_or_404(db, event_id)
    ticket = await create_ticket(db, event=event, current_user=current_user, payload=payload)
    return TicketRead.model_validate(ticket)
