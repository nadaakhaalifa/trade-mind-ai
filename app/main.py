from fastapi import FastAPI
from sqlalchemy import text

from app.db.session import engine
from app.api import agents, experiments, training_runs


app = FastAPI()

app.include_router(agents.router)
app.include_router(experiments.router)
app.include_router(training_runs.router)


@app.get("/")
def root():
    return {"message": "TradeMind AI backend is running v2"}


@app.get("/test-db")
def test_db():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return {"status": "success", "result": [row[0] for row in result]}
    except Exception as e:
        return {"status": "error", "message": str(e)}