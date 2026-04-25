from datetime import datetime

from app.core.celery_app import celery
from app.db.session import SessionLocal
from app.models.experiment import Experiment
from app.models.training_result import TrainingResult
from app.models.training_run import TrainingRun
from app.rl.trainer import DQNTrainer


@celery.task(name="app.tasks.training_tasks.run_training_task")
def run_training_task(training_run_id: int):
    """
    Celery task that runs RL training in the background.
    Flow:
    1. Find training run
    2. Mark it as running
    3. Run DQN training
    4. Save each episode result
    5. Update progress
    6. Mark run as completed
    """
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

        # Mark the training run as started
        training_run.status = "running"
        training_run.progress = 0
        training_run.started_at = datetime.utcnow()
        db.commit()

        # Temporary fake market data.
        # Later we will replace this with real historical price data.
        prices = list(range(100, 150))

        trainer = DQNTrainer(
            prices=prices,
            episodes=training_run.episodes,
        )

        results = []

        for episode_result in trainer.train():
            # Save episode result in database
            result = TrainingResult(
                training_run_id=training_run_id,
                episode=episode_result["episode"],
                total_reward=episode_result["total_reward"],
                final_balance=episode_result["final_balance"],
                epsilon=episode_result["epsilon"],
                memory_size=episode_result["memory_size"],
            )

            db.add(result)

            # Update training progress based on completed episodes
            progress = int(
                episode_result["episode"] / training_run.episodes * 100
            )

            training_run = db.query(TrainingRun).filter(
                TrainingRun.id == training_run_id
            ).first()

            if not training_run:
                db.rollback()
                return {"error": "Training run not found during update"}

            training_run.progress = progress

            db.commit()

            results.append(episode_result)

        training_run = db.query(TrainingRun).filter(
            TrainingRun.id == training_run_id
        ).first()

        if not training_run:
            return {"error": "Training run not found before completion"}

        # Mark the training run as completed
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