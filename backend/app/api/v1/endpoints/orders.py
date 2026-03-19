from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.deps import get_current_user, get_db
from app.schemas import OrderRead, PurchaseResponse, RSVPRead
from app.services.commerce import get_order_or_404, purchase_ticket

router = APIRouter(tags=["orders"])


@router.get("/orders/{order_id}", response_model=OrderRead)
async def get_order(order_id: str, db: AsyncSession = Depends(get_db)) -> OrderRead:
    order = await get_order_or_404(db, order_id)
    return OrderRead.model_validate(order)


@router.post("/events/{event_id}/tickets/{ticket_id}/purchase", response_model=PurchaseResponse)
async def purchase_event_ticket(
    event_id: str,
    ticket_id: str,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> PurchaseResponse:
    order, rsvp = await purchase_ticket(db, event_id=event_id, ticket_id=ticket_id, current_user=current_user)
    return PurchaseResponse(order=OrderRead.model_validate(order), rsvp=None if rsvp is None else RSVPRead.model_validate(rsvp))
