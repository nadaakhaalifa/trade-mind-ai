from celery import Celery

celery = Celery(
    "trade_mind",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0",
    include=["app.tasks.training_tasks"] 
)

celery.conf.update(
    task_track_started=True
)