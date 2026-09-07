from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .model import model
from .schemas import PredictionResponse, SymptomInput

app = FastAPI(title="Smart Medical Diagnosis API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DISCLAIMER = "Educational screening only. This does not provide medical advice or replace a qualified clinician."
FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/symptoms")
def symptoms() -> dict[str, list[str]]:
    return {"symptoms": model.symptoms}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: SymptomInput) -> PredictionResponse:
    unknown = sorted(set(payload.symptoms) - set(model.symptoms))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown symptoms: {', '.join(unknown)}")
    return PredictionResponse(predictions=model.predict(payload.symptoms), disclaimer=DISCLAIMER)


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(FRONTEND)
