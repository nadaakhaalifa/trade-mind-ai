from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.models.base import Base


# SQLAlchemy model for training jobs
class TrainingRun(Base):
    __tablename__ = "training_runs"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Link this training run to an experiment
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)

    # Example: DQN, PPO
    algorithm = Column(String, nullable=False)

    # Number of training episodes
    episodes = Column(Integer, nullable=False)

    # Example: pending, running, completed, failed
    status = Column(String, nullable=False, default="pending")

    # Training progress percentage (0 → 100)
    progress = Column(Integer, nullable=False, default=0)

    # Time when training started
    started_at = Column(DateTime(timezone=True), nullable=True)

    # Time when training finished
    finished_at = Column(DateTime(timezone=True), nullable=True)

    # Time when row was created
    created_at = Column(DateTime(timezone=True), server_default=func.now())