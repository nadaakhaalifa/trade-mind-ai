from datetime import datetime

from app.core.celery_app import celery
from app.db.session import SessionLocal
from app.models.training_run import TrainingRun
from app.models.experiment import Experiment
from app.rl.trainer import DQNTrainer


@celery.task(name="app.tasks.training_tasks.run_training_task")
def run_training_task(training_run_id: int):
    db = SessionLocal()

    try:
        training_run = db.query(TrainingRun).filter(
            TrainingRun.id == training_run_id
        ).first()

        if not training_run:
            return {"error": "Training run not found"}

        experiment = db.query(Experiment).filter(
            Experiment.id == training_run.experiment_id
        ).first()

        if not experiment:
            training_run.status = "failed"
            db.commit()
            return {"error": "Experiment not found"}

        training_run.status = "running"
        training_run.progress = 0
        training_run.started_at = datetime.utcnow()
        db.commit()
        
        # fake data for noww
        prices = list(range(100, 150))

        trainer = DQNTrainer(
            prices=prices,
            episodes=training_run.episodes,
        )

        results = []

        for episode_result in trainer.train():
            results.append(episode_result)

            progress = int(
                episode_result["episode"] / training_run.episodes * 100
            )

            training_run = db.query(TrainingRun).filter(
                TrainingRun.id == training_run_id
            ).first()

            if not training_run:
                return {"error": "Training run not found during update"}

            training_run.progress = progress
            db.commit()

        training_run = db.query(TrainingRun).filter(
            TrainingRun.id == training_run_id
        ).first()

        if not training_run:
            return {"error": "Training run not found before completion"}

        training_run.status = "completed"
        training_run.progress = 100
        training_run.finished_at = datetime.utcnow()
        db.commit()

        return {
            "message": f"Training run {training_run_id} completed",
            "results": results,
        }

    except Exception as e:
        db.rollback()

        training_run = db.query(TrainingRun).filter(
            TrainingRun.id == training_run_id
        ).first()

        if training_run:
            training_run.status = "failed"
            db.commit()

        return {"error": str(e)}

    finally:
        db.close()