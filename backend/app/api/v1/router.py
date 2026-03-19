from fastapi import APIRouter

from app.api.v1.endpoints import admin, auth, checkin, events, health, invitations, orders, rsvps, tickets

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(events.router)
api_router.include_router(tickets.router)
api_router.include_router(rsvps.router)
api_router.include_router(orders.router)
api_router.include_router(invitations.router)
api_router.include_router(checkin.router)
api_router.include_router(admin.router)

