from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String

from app.db.session import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    action: Mapped[str] = mapped_column(String(255))
    actor: Mapped[str] = mapped_column(String(255))

