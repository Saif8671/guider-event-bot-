from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    ticket_type: Mapped[str] = mapped_column(String(50), nullable=False, default="free")
    price_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    quantity_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quantity_sold: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sale_starts_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sale_ends_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False, server_default=func.false())
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=func.true())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    event = relationship("Event", back_populates="tickets")
    rsvps = relationship("RSVP", back_populates="ticket")
    orders = relationship("Order", back_populates="ticket")
