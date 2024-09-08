from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from serving.predictor import Predictor


class Payload(BaseModel):
    text: List[str]


class Prediction(BaseModel):
    probs: List[List[float]]


app = FastAPI()
predictor = Predictor.default_from_model_registry()


@app.get("/health_check")
def health_check() -> str:
    return "ok"


@app.post("/predict", response_model=Prediction)
def predict(payload: Payload) -> Prediction:
    prediction = predictor.predict(text=payload.text)
    return Prediction(probs=prediction.tolist())
