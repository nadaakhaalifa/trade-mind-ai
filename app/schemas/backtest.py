from datetime import datetime
from pydantic import BaseModel


class BacktestCreate(BaseModel):
    training_run_id: int


class BacktestResponse(BaseModel):
    id: int
    training_run_id: int
    status: str
    total_reward: float | None
    final_balance: float | None
    created_at: datetime

    class Config:
        from_attributes = True