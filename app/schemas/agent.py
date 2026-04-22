from pydantic import BaseModel
from typing import Optional, Dict


class AgentCreate(BaseModel):
    name: str
    algorithm: str
    config: Optional[Dict] = None


class AgentResponse(BaseModel):
    id: int
    name: str
    algorithm: str
    config: Optional[Dict]
    status: str

    class Config:
        from_attributes = True