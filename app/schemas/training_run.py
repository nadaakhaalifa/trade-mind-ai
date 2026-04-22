from pydantic import BaseModel
from datetime import datetime
from typing import Optional


# Data the user sends when creating a training run
class TrainingRunCreate(BaseModel):
    experiment_id: int
    algorithm: str
    episodes: int


# Data we return back from the API
class TrainingRunResponse(BaseModel):
    id: int
    experiment_id: int
    algorithm: str
    episodes: int
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True