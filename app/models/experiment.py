from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.models.base import Base

# SQLALchemy model
class Experiment(Base):
    __tablename__ = "experiments"

    # Primary key: unique id for each experiment
    id = Column(Integer, primary_key=True, index=True)

    # Name of the experiment
    name = Column(String, nullable=False)

    # Short description
    description = Column(String, nullable=True)

    # Example: BTCUSD, ETHUSD, AAPL
    asset_symbol = Column(String, nullable=False)

    # Example: 1m, 5m, 1h, 1d
    timeframe = Column(String, nullable=False)

    # Starting money inside the simulation
    initial_balance = Column(Float, nullable=False)

    # Example: draft, ready, running, completed, failed
    status = Column(String, nullable=False, default="draft")

    # Auto-created timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())