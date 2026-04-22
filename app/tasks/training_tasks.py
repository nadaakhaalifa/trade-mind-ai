import time
from datetime import datetime

from app.core.celery_app import celery
from app.db.session import SessionLocal
from app.models.training_run import TrainingRun
from app.models.experiment import Experiment

@celery.task
def run_training_task(training_run_id: int):
    db = SessionLocal()
    training_run = None

    try:
        training_run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()

        if not training_run:
            return {"error": "Training run not found"}

        training_run.status = "running"
        training_run.started_at = datetime.utcnow()
        db.commit()

        time.sleep(5)

        training_run.status = "completed"
        training_run.finished_at = datetime.utcnow()
        db.commit()

        return {"message": f"Training run {training_run_id} completed"}

    except Exception as e:
        if training_run:
            training_run.status = "failed"
            db.commit()
        return {"error": str(e)}

    finally:
        db.close()