"""
main.py - FastAPI app. Run from the project root:
    uvicorn backend.main:app --host 0.0.0.0 --reload --port 8000
Docs (try the API in the browser): http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import ALLOWED_ORIGINS, FUEL_TYPES, TRANSMISSIONS
from backend.model_service import ModelService
from backend.schemas import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ModelService()      # models are loaded once at startup
    yield


app = FastAPI(title="AutoPulse API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    """Everything the frontend form needs for its dropdowns."""
    svc = app.state.service
    return {
        "fuel_types": FUEL_TYPES,
        "transmissions": TRANSMISSIONS,
        "makes": svc.catalog["makes"],
        "catalog": svc.catalog["models"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return app.state.service.predict(req)
    except Exception as exc:                 # never leak a stack trace to the client
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")


@app.get("/model-info")
def model_info():
    return app.state.service.model_info()