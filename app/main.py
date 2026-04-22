from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "TradeMind AI backend is running"}
