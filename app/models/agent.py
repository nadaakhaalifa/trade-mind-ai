from sqlalchemy import Column, Integer, String, DateTime, JSON
from datetime import datetime
from app.models.base import Base


class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    algorithm = Column(String, nullable=False)
    config = Column(JSON, nullable=True)
    status = Column(String, default="created")

    created_at = Column(DateTime, default=datetime.utcnow)