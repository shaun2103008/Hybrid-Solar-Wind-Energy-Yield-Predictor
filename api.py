"""FastAPI Backend Server for Hybrid Solar-Wind Energy Yield Predictor.

Loads the trained Hybrid Voting Ensemble model (trained on 15-month SOLETE data)
and serves real-time energy yield predictions via REST API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "results" / "solete_best_predictor.joblib"
FEATURED_DATA_PATH = ROOT / "dataset" / "hybrid" / "solete" / "solete_featured_hourly.csv"

app = FastAPI(
    title="Hybrid Solar-Wind Energy Yield Prediction API",
    description="Real-time multi-model ensemble yield forecasting for co-located renewable plants.",
    version="1.0.0",
)

# Enable CORS so the Vercel frontend can call this backend from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model = None
feature_columns = []


@app.on_event("startup")
def load_model():
    global model, feature_columns
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Loaded trained model from {MODEL_PATH}")

    # Load feature column names
    if FEATURED_DATA_PATH.exists():
        df_sample = pd.read_csv(FEATURED_DATA_PATH, nrows=2)
        exclude = [
            "timestamp_utc", "P_Solar[kW]", "P_Gaia[kW]", "P_Hybrid[kW]",
            "GHI[kW1m2]", "POA Irr[kW1m2]", "WIND_SPEED[m1s]", "WIND_DIR[deg]",
            "TEMPERATURE[degC]", "HUMIDITY[%]", "Pressure[mbar]", "Azimuth[deg]", "Elevation[deg]",
            "target_hybrid_1h", "target_hybrid_2h", "target_hybrid_6h", "target_hybrid_24h",
            "target_solar_1h", "target_wind_1h"
        ]
        feature_columns = [c for c in df_sample.columns if c not in exclude]
        print(f"Registered {len(feature_columns)} feature columns for inference.")


class WeatherPredictionInput(BaseModel):
    irradiance_ghi: float = Field(0.45, description="Global Horizontal Irradiance in kW/m2 (0.0 to 1.0)")
    wind_speed: float = Field(7.5, description="Wind Speed in m/s (0.0 to 25.0)")
    temperature_C: float = Field(18.0, description="Ambient Temperature in degC (-10.0 to 45.0)")
    wind_direction: float = Field(220.0, description="Wind Direction in degrees (0 to 360)")
    humidity_pct: float = Field(0.70, description="Relative Humidity fraction (0.0 to 1.0)")
    pressure_mbar: float = Field(1013.0, description="Atmospheric Pressure in mbar")
    hour: int = Field(13, description="Hour of Day (0 to 23)")
    month: int = Field(6, description="Month of Year (1 to 12)")
    recent_hybrid_power_kW: Optional[float] = Field(None, description="Plant output 1 hour ago (kW)")


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Hybrid Solar-Wind Energy Yield Predictor",
        "facility": "SOLETE Co-Located Research Plant",
        "model": "Hybrid Voting Ensemble (R2: 0.9227)",
        "docs": "/docs",
    }


@app.post("/predict")
def predict_hybrid_yield(payload: WeatherPredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Approximate recent power if not supplied
    past_power = payload.recent_hybrid_power_kW
    if past_power is None:
        # Physical estimation baseline
        solar_est = max(0.0, payload.irradiance_ghi * 6.5) if 6 <= payload.hour <= 19 else 0.0
        wind_est = 11.0 * min(1.0, max(0.0, (payload.wind_speed - 3.0) / 9.0)) ** 3
        past_power = solar_est + wind_est

    # Construct input vector matching feature_columns
    row = {col: 0.0 for col in feature_columns}

    # Fill base features
    row["irradiance_ghi"] = payload.irradiance_ghi
    row["irradiance_poa"] = payload.irradiance_ghi * 1.05
    row["wind_speed"] = payload.wind_speed
    row["wind_direction"] = payload.wind_direction % 360.0
    row["temperature_C"] = payload.temperature_C
    row["humidity_pct"] = payload.humidity_pct
    row["pressure_mbar"] = payload.pressure_mbar
    row["solar_power_kW"] = max(0.0, payload.irradiance_ghi * 6.5) if 6 <= payload.hour <= 19 else 0.0
    row["wind_power_kW"] = min(11.0, max(0.0, 11.0 * ((payload.wind_speed - 3.0) / 9.0) ** 3)) if payload.wind_speed >= 3.0 else 0.0
    row["hybrid_power_kW"] = past_power

    # Aerodynamic vectors
    rad = np.radians(row["wind_direction"])
    row["wind_vector_u"] = -payload.wind_speed * np.sin(rad)
    row["wind_vector_v"] = -payload.wind_speed * np.cos(rad)

    # Cyclical
    row["hour"] = payload.hour
    row["month"] = payload.month
    row["sin_hour"] = np.sin(2 * np.pi * payload.hour / 24.0)
    row["cos_hour"] = np.cos(2 * np.pi * payload.hour / 24.0)
    row["sin_month"] = np.sin(2 * np.pi * (payload.month - 1) / 12.0)
    row["cos_month"] = np.cos(2 * np.pi * (payload.month - 1) / 12.0)
    row["is_daylight"] = 1 if payload.irradiance_ghi > 0.01 else 0

    # Lags & rolling
    for lag in [1, 2, 3, 6, 12, 24]:
        row[f"hybrid_lag_{lag}h"] = past_power
        row[f"solar_lag_{lag}h"] = row["solar_power_kW"]
        row[f"wind_lag_{lag}h"] = row["wind_power_kW"]

    for win in [3, 6]:
        row[f"hybrid_roll_mean_{win}h"] = past_power
        row[f"solar_roll_mean_{win}h"] = row["solar_power_kW"]
        row[f"wind_roll_mean_{win}h"] = row["wind_power_kW"]

    input_df = pd.DataFrame([row])[feature_columns]
    pred_hybrid = float(np.clip(model.predict(input_df)[0], 0.0, None))

    # Calculate solar vs wind component split
    solar_ratio = row["solar_power_kW"] / (row["solar_power_kW"] + row["wind_power_kW"] + 1e-6)
    pred_solar = float(pred_hybrid * solar_ratio)
    pred_wind = float(pred_hybrid * (1.0 - solar_ratio))

    capacity_kw = 18.0  # 7 kW solar + 11 kW wind
    capacity_factor = round((pred_hybrid / capacity_kw) * 100.0, 1)

    return {
        "predicted_hybrid_power_kW": round(pred_hybrid, 2),
        "predicted_solar_power_kW": round(pred_solar, 2),
        "predicted_wind_power_kW": round(pred_wind, 2),
        "plant_capacity_kW": capacity_kw,
        "capacity_factor_pct": min(100.0, capacity_factor),
        "unit": "kW",
        "model_used": "Hybrid Voting Ensemble (Random Forest + XGBoost + LightGBM)",
    }
