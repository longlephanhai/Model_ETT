from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "linear_models.pkl"
WINDOW_SIZE = 24
FEATURE_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


class TimeSeriesPoint(BaseModel):
    date: str
    HUFL: float
    HULL: float
    MUFL: float
    MULL: float
    LUFL: float
    LULL: float
    OT: float


class PredictRequest(BaseModel):
    series: list[TimeSeriesPoint]
    steps: int = Field(default=1, ge=1)


class PredictResponse(BaseModel):
    forecast: list[float]
    future_dates: list[str]


app = FastAPI(title="ETDataset Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_models() -> list:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    models = joblib.load(MODEL_PATH)
    if not isinstance(models, list) or not models:
        raise RuntimeError("Unexpected model artifact format. Expected a non-empty list of pipelines.")

    return models


MODELS = load_models()
BASE_MODEL = MODELS[0]
FEATURE_NAMES = list(getattr(BASE_MODEL, "feature_names_in_", []))


def point_to_dict(point: TimeSeriesPoint) -> dict[str, float | str]:
    return point.model_dump()


def build_feature_row(history: Sequence[TimeSeriesPoint]) -> pd.DataFrame:
    if len(history) < WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"At least {WINDOW_SIZE} historical points are required to build model features.",
        )

    window = list(history[-WINDOW_SIZE:])
    if not FEATURE_NAMES:
        raise RuntimeError("Loaded model does not expose feature names.")

    latest = point_to_dict(window[-1])
    row: dict[str, float] = {}

    for feature_name in FEATURE_NAMES:
        if "_lag_" in feature_name:
            column_name, lag_text = feature_name.rsplit("_lag_", 1)
            lag = int(lag_text)
            source_index = -lag - 1
            row[feature_name] = float(point_to_dict(window[source_index])[column_name])
        else:
            row[feature_name] = float(latest[feature_name])

    return pd.DataFrame([row], columns=FEATURE_NAMES)


def predict_one_step(history: Sequence[TimeSeriesPoint]) -> float:
    feature_row = build_feature_row(history)
    prediction = BASE_MODEL.predict(feature_row)
    return float(prediction[0])


def parse_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise HTTPException(status_code=400, detail=f"Invalid timestamp: {value}")
    return timestamp


def infer_step_delta(series: Sequence[TimeSeriesPoint]) -> pd.Timedelta:
    if len(series) >= 2:
        last_timestamp = parse_timestamp(series[-1].date)
        previous_timestamp = parse_timestamp(series[-2].date)
        delta = last_timestamp - previous_timestamp
        if delta != pd.Timedelta(0):
            return delta

    return pd.Timedelta(hours=1)


def build_future_payload(series: Sequence[TimeSeriesPoint], steps: int) -> PredictResponse:
    if len(series) < WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {WINDOW_SIZE} rows to produce forecast points.",
        )

    forecast_values: list[float] = []
    future_dates: list[str] = []

    history = list(series)
    last_timestamp = parse_timestamp(history[-1].date)
    delta = infer_step_delta(history)

    for step_index in range(steps):
        prediction = predict_one_step(history)
        forecast_values.append(prediction)

        last_point = history[-1]
        future_timestamp = last_timestamp + delta * (step_index + 1)
        future_dates.append(future_timestamp.strftime("%Y-%m-%d %H:%M:%S"))

        history.append(
            TimeSeriesPoint(
                date=future_dates[-1],
                HUFL=last_point.HUFL,
                HULL=last_point.HULL,
                MUFL=last_point.MUFL,
                MULL=last_point.MULL,
                LUFL=last_point.LUFL,
                LULL=last_point.LULL,
                OT=prediction,
            )
        )

    return PredictResponse(forecast=forecast_values, future_dates=future_dates)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": "ETDataset Forecast API",
        "status": "ok",
        "model_path": str(MODEL_PATH),
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not request.series:
        raise HTTPException(status_code=400, detail="series cannot be empty.")

    return build_future_payload(request.series, request.steps)