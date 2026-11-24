import json
import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.features import build_features

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "model/metrics.json")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        state["metrics"] = json.load(f)
    yield
    state.clear()


app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure from sensor readings (AI4I 2020 dataset).",
    version="2.0.0",
    lifespan=lifespan,
)


class SensorReading(BaseModel):
    type: Literal["L", "M", "H"] = Field(..., description="Product quality variant")
    air_temperature_k: float = Field(..., gt=0)
    process_temperature_k: float = Field(..., gt=0)
    rotational_speed_rpm: float = Field(..., gt=0)
    torque_nm: float = Field(..., ge=0)
    tool_wear_min: float = Field(..., ge=0)


class Prediction(BaseModel):
    failure: bool
    failure_probability: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state, "version": "2.0.0"}


@app.get("/model-info")
def model_info():
    return {
        "version": "2.0.0", "model": "RandomForestClassifier",
        "dataset": "AI4I 2020 Predictive Maintenance",
        "features": "raw sensors + power, strain, temperature difference",
        "metrics": state.get("metrics", {}),
    }


@app.post("/predict", response_model=Prediction)
def predict(reading: SensorReading):
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = pd.DataFrame([{
        "Type": reading.type,
        "Air temperature [K]": reading.air_temperature_k,
        "Process temperature [K]": reading.process_temperature_k,
        "Rotational speed [rpm]": reading.rotational_speed_rpm,
        "Torque [Nm]": reading.torque_nm,
        "Tool wear [min]": reading.tool_wear_min,
    }])

    proba = float(model.predict_proba(build_features(raw))[0][1])
    return Prediction(failure=proba >= 0.5, failure_probability=round(proba, 4))
