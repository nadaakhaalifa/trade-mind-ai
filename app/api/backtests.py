import os
import torch

from fastapi import APIRouter, HTTPException

from app.db.session import SessionLocal
from app.models.backtest import Backtest
from app.models.training_run import TrainingRun
from app.rl.agent import DQNAgent
from app.rl.environment import TradingEnvironment
from app.schemas.backtest import BacktestCreate


router = APIRouter(prefix="/backtests", tags=["Backtests"])


@router.post("/")
def create_backtest(backtest: BacktestCreate):
    db = SessionLocal()

    training_run = db.query(TrainingRun).filter(
        TrainingRun.id == backtest.training_run_id
    ).first()

    if not training_run:
        db.close()
        raise HTTPException(status_code=404, detail="Training run not found")

    if training_run.status != "completed":
        db.close()
        raise HTTPException(
            status_code=400,
            detail="Training run must be completed before backtesting",
        )

    if not training_run.model_path:
        db.close()
        raise HTTPException(
            status_code=400,
            detail="Training run has no saved model checkpoint",
        )

    if not os.path.exists(training_run.model_path):
        db.close()
        raise HTTPException(
            status_code=404,
            detail="Saved model file not found",
        )

    # Temporary fake market data.
    # Later we will replace this with unseen historical test data.
    prices = list(range(100, 150))

    env = TradingEnvironment(prices=prices)
    agent = DQNAgent()

    # Load trained model weights
    agent.network.load_state_dict(torch.load(training_run.model_path))
    agent.network.eval()

    # During backtesting, we do not want random actions.
    # The agent should only use the trained model.
    agent.epsilon = 0

    state = env.reset()
    done = False
    total_reward = 0
    
    trade_history = []
    step = 0   

    # Count what the model actually decided during backtesting
    action_counts = {
        "hold": 0,
        "buy": 0,
        "sell": 0,
    }

    while not done:
        action = agent.choose_action(state)

        price = state["prices"][-1]

        if action == 0:
           action_counts["hold"] += 1
           action_name = "hold"
        elif action == 1:
           action_counts["buy"] += 1
           action_name = "buy"
        elif action == 2:
           action_counts["sell"] += 1
           action_name = "sell"

        trade_history.append({
            "step": step,
            "action": action_name,
            "price": price,
        })

        state, reward, done = env.step(action)
        total_reward += reward
        step += 1

    new_backtest = Backtest(
        training_run_id=backtest.training_run_id,
        status="completed",
        total_reward=total_reward,
        final_balance=env.balance,
    )

    db.add(new_backtest)
    db.commit()
    db.refresh(new_backtest)

    response = {
        "id": new_backtest.id,
        "training_run_id": new_backtest.training_run_id,
        "status": new_backtest.status,
        "total_reward": new_backtest.total_reward,
        "final_balance": new_backtest.final_balance,
        "created_at": new_backtest.created_at,
        "actions": action_counts,
        "trades": trade_history,
    }

    db.close()
    return response