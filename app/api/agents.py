from fastapi import APIRouter
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.agent import Agent
from app.schemas.agent import AgentCreate, AgentResponse

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/agents", response_model=AgentResponse)
def create_agent(payload: AgentCreate):
    db: Session = SessionLocal()
    try:
        agent = Agent(
            name=payload.name,
            algorithm=payload.algorithm,
            config=payload.config,
            status="created",
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return agent
    finally:
        db.close()
        

@router.get("/agents", response_model=list[AgentResponse])
def list_agents():
    db: Session = SessionLocal()
    try:
        agents = db.query(Agent).all()
        return agents
    finally:
        db.close()