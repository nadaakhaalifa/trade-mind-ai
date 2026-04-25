from fastapi import APIRouter, HTTPException, Query
from app.db.session import SessionLocal
from app.models.training_run import TrainingRun
from app.models.experiment import Experiment
from app.schemas.training_run import TrainingRunCreate, TrainingRunResponse
from app.tasks.training_tasks import run_training_task
from app.models.training_result import TrainingResult

router = APIRouter(prefix="/training-runs", tags=["Training Runs"])


# Create a new training run
@router.post("/", response_model=TrainingRunResponse)
def create_training_run(training_run: TrainingRunCreate):
    db = SessionLocal()

    # Make sure the experiment exists first
    experiment = db.query(Experiment).filter(Experiment.id == training_run.experiment_id).first()

    if not experiment:
        db.close()
        raise HTTPException(status_code=404, detail="Experiment not found")

    new_training_run = TrainingRun(
        experiment_id=training_run.experiment_id,
        algorithm=training_run.algorithm,
        episodes=training_run.episodes,
    )

    db.add(new_training_run)
    db.commit()
    db.refresh(new_training_run)

    # Send the training job to Celery worker in the background
    run_training_task.delay(new_training_run.id)

    db.close()
    return new_training_run


# Get all training runs, optionally filtered by status
@router.get("/", response_model=list[TrainingRunResponse])
def get_training_runs(status: str | None = Query(default=None)):
    db = SessionLocal()

    query = db.query(TrainingRun)

    if status:
        query = query.filter(TrainingRun.status == status)

    training_runs = query.all()

    db.close()
    return training_runs

# Get one training run by id
@router.get("/{training_run_id}", response_model=TrainingRunResponse)
def get_training_run(training_run_id: int):
    db = SessionLocal()

    training_run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()

    if not training_run:
        db.close()
        raise HTTPException(status_code=404, detail="Training run not found")

    db.close()
    return training_run

# Get episode results for one training run
@router.get("/{training_run_id}/results")
def get_training_run_results(training_run_id: int):
    db = SessionLocal()

    training_run = db.query(TrainingRun).filter(
        TrainingRun.id == training_run_id
    ).first()

    if not training_run:
        db.close()
        raise HTTPException(status_code=404, detail="Training run not found")

    results = db.query(TrainingResult).filter(
        TrainingResult.training_run_id == training_run_id
    ).order_by(TrainingResult.episode).all()

    db.close()
    return results


# Get chart-ready metrics for one training run
@router.get("/{training_run_id}/metrics")
def get_training_run_metrics(training_run_id: int):
    db = SessionLocal()

    training_run = db.query(TrainingRun).filter(
        TrainingRun.id == training_run_id
    ).first()

    if not training_run:
        db.close()
        raise HTTPException(status_code=404, detail="Training run not found")

    results = db.query(TrainingResult).filter(
        TrainingResult.training_run_id == training_run_id
    ).order_by(TrainingResult.episode).all()

    db.close()

    return {
        "training_run_id": training_run_id,
        "episodes": [result.episode for result in results],
        "rewards": [result.total_reward for result in results],
        "balances": [result.final_balance for result in results],
        "epsilons": [result.epsilon for result in results],
        "memory_sizes": [result.memory_size for result in results],
    }