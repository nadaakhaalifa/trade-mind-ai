from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime
from sqlalchemy.sql import func

from app.models.base import Base

class TrainingResult(Base):
    __tablename__ = "training_results"

    id = Column(Integer, primary_key=True, index=True)

    training_run_id = Column(Integer, ForeignKey("training_runs.id"))

    episode = Column(Integer)
    total_reward = Column(Float)
    final_balance = Column(Float)
    epsilon = Column(Float)
    memory_size = Column(Integer)

    created_at = Column(DateTime(timezone=True), server_default=func.now())