from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# Data the user sends when creating an experiment [input]
class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    asset_symbol: str
    timeframe: str
    initial_balance: float


# Data we return back from the API [output]
class ExperimentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    asset_symbol: str
    timeframe: str
    initial_balance: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True