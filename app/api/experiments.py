from fastapi import APIRouter , HTTPException, Query
from app.db.session import SessionLocal
from app.models.experiment import Experiment
from app.schemas.experiment import ExperimentCreate, ExperimentResponse


router = APIRouter(prefix="/experiments", tags=["Experiments"])


# Create a new experiment
@router.post("/", response_model=ExperimentResponse)
def create_experiment(experiment: ExperimentCreate):
    db = SessionLocal()

    new_experiment = Experiment(
        name=experiment.name,
        description=experiment.description,
        asset_symbol=experiment.asset_symbol,
        timeframe=experiment.timeframe,
        initial_balance=experiment.initial_balance,
    )

    db.add(new_experiment) #Tells SQLAlchemy to prepare it for insert.
    db.commit() #Actually saves it in Postgres.
    db.refresh(new_experiment)#Reloads the saved row
    db.close()

    return new_experiment


# Get all experiments
@router.get("/", response_model=list[ExperimentResponse])
def get_experiments():
    db = SessionLocal()

    experiments = db.query(Experiment).all()

    db.close()
    return experiments

# Get one experiment by id
@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int):
    db = SessionLocal()

    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        db.close()
        raise HTTPException(status_code=404, detail="Experiment not found")

    db.close()
    return experiment