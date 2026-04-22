from sqlalchemy import Column, Integer, String, DateTime, JSON
from datetime import datetime
from app.models.base import Base


class Environment(Base):
    __tablename__ = "environments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    asset_universe = Column(JSON, nullable=True)
    action_space = Column(String, nullable=False)
    fee_model = Column(JSON, nullable=True)
    slippage_model = Column(JSON, nullable=True)
    reward_function = Column(String, nullable=False)
    risk_constraints = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)