from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.sql import func

from app.models.base import Base


class Backtest(Base):
    __tablename__ = "backtests"

    id = Column(Integer, primary_key=True, index=True)

    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)

    status = Column(String, nullable=False, default="pending")

    total_reward = Column(Float, nullable=True)
    final_balance = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())