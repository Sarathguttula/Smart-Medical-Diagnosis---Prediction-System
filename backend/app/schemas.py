from pydantic import BaseModel, Field


class SymptomInput(BaseModel):
    symptoms: list[str] = Field(min_length=1, description="Symptoms selected by the user")


class Prediction(BaseModel):
    disease: str
    probability: float
    risk_level: str
    explanation: list[dict[str, float | str]]
    recommendations: list[str]


class PredictionResponse(BaseModel):
    predictions: list[Prediction]
    disclaimer: str
