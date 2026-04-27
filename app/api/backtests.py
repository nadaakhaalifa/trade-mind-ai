import os
import torch

from fastapi import APIRouter, HTTPException

from app.db.session import SessionLocal
from app.models.backtest import Backtest
from app.models.training_run import TrainingRun
from app.rl.agent import DQNAgent
from app.rl.environment import TradingEnvironment
from app.schemas.backtest import BacktestCreate
from app.rl.market_data import load_prices_from_csv


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

    prices = load_prices_from_csv()

    env = TradingEnvironment(prices=prices)
    agent = DQNAgent()

    agent.network.load_state_dict(torch.load(training_run.model_path))
    agent.network.eval()

    agent.epsilon = 0

    state = env.reset()
    done = False
    total_reward = 0

    trade_history = []
    step = 0

    action_counts = {
        "hold": 0,
        "buy": 0,
        "sell": 0,
        "invalid_buy": 0,
        "invalid_sell": 0,
        "invalid_action": 0,
    }

    while not done:
        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)

        executed_action = info["executed_action"]
        price = info["price"]

        if executed_action in action_counts:
            action_counts[executed_action] += 1

        trade_history.append({
            "step": step,
            "chosen_action": info["chosen_action"],
            "executed_action": executed_action,
            "price": price,
            "reward": reward,
            "balance": info["balance"],
            "position": info["position"],
        })

        state = next_state
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