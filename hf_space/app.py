"""Gradio App & API for Hybrid Solar-Wind Energy Yield Prediction on Hugging Face Spaces.

Loads the trained Hybrid Voting Ensemble model (trained on 15-month SOLETE data).
Provides both an interactive UI and a REST API for the Vercel frontend.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "solete_best_predictor.joblib"
SAMPLE_FEATS_PATH = ROOT / "sample_features.csv"

# Load model and feature column names
print("Loading model...")
model = joblib.load(MODEL_PATH)
df_sample = pd.read_csv(SAMPLE_FEATS_PATH, nrows=2)
exclude = [
    "timestamp_utc", "P_Solar[kW]", "P_Gaia[kW]", "P_Hybrid[kW]",
    "GHI[kW1m2]", "POA Irr[kW1m2]", "WIND_SPEED[m1s]", "WIND_DIR[deg]",
    "TEMPERATURE[degC]", "HUMIDITY[%]", "Pressure[mbar]", "Azimuth[deg]", "Elevation[deg]",
    "target_hybrid_1h", "target_hybrid_2h", "target_hybrid_6h", "target_hybrid_24h",
    "target_solar_1h", "target_wind_1h"
]
feature_columns = [c for c in df_sample.columns if c not in exclude]
print(f"Model loaded with {len(feature_columns)} features.")


def predict_yield(
    irradiance_ghi: float,
    wind_speed: float,
    temperature_c: float,
    wind_direction: float,
    hour: int,
    month: int,
    past_hybrid_kw: float,
):
    """Predicts hybrid, solar, and wind generation from weather inputs."""
    # Approximate past power if 0
    if past_hybrid_kw <= 0.0:
        solar_est = max(0.0, (irradiance_ghi / 1000.0) * 6.5) if 6 <= hour <= 19 else 0.0
        wind_est = 11.0 * min(1.0, max(0.0, (wind_speed - 3.0) / 9.0)) ** 3 if wind_speed >= 3.0 else 0.0
        past_power = solar_est + wind_est
    else:
        past_power = past_hybrid_kw

    # Construct input row
    row = {col: 0.0 for col in feature_columns}
    ghi_kw = irradiance_ghi / 1000.0  # convert W/m2 to kW/m2 matching SOLETE
    row["irradiance_ghi"] = ghi_kw
    row["irradiance_poa"] = ghi_kw * 1.05
    row["wind_speed"] = wind_speed
    row["wind_direction"] = wind_direction % 360.0
    row["temperature_C"] = temperature_c
    row["humidity_pct"] = 0.70
    row["pressure_mbar"] = 1013.0
    row["solar_power_kW"] = max(0.0, ghi_kw * 6.5) if 6 <= hour <= 19 else 0.0
    row["wind_power_kW"] = min(11.0, max(0.0, 11.0 * ((wind_speed - 3.0) / 9.0) ** 3)) if wind_speed >= 3.0 else 0.0
    row["hybrid_power_kW"] = past_power

    # Aerodynamic vectors
    rad = np.radians(row["wind_direction"])
    row["wind_vector_u"] = -wind_speed * np.sin(rad)
    row["wind_vector_v"] = -wind_speed * np.cos(rad)

    # Cyclical
    row["hour"] = hour
    row["month"] = month
    row["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    row["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    row["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    row["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    row["is_daylight"] = 1 if ghi_kw > 0.01 else 0

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

    solar_ratio = row["solar_power_kW"] / (row["solar_power_kW"] + row["wind_power_kW"] + 1e-6)
    pred_solar = float(pred_hybrid * solar_ratio)
    pred_wind = float(pred_hybrid * (1.0 - solar_ratio))

    capacity_kw = 18.0
    capacity_factor = min(100.0, round((pred_hybrid / capacity_kw) * 100.0, 1))

    return (
        f"{pred_hybrid:.2f} kW",
        f"{pred_solar:.2f} kW",
        f"{pred_wind:.2f} kW",
        f"{capacity_factor:.1f}%",
    )


# Build clean Gradio Interface
with gr.Blocks(title="Hybrid Solar-Wind Energy Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Hybrid Solar-Wind Energy Yield Predictor")
    gr.Markdown("Real-time multi-model ensemble yield forecasting for co-located renewable energy plants.")

    with gr.Row():
        with gr.Column():
            irradiance = gr.Slider(0, 1000, value=550, step=10, label="Solar Irradiance (GHI in W/m²)")
            wind_speed = gr.Slider(0, 25, value=7.5, step=0.1, label="Wind Speed (m/s)")
            temp = gr.Slider(-10, 45, value=18, step=0.5, label="Ambient Temperature (°C)")
            wind_dir = gr.Slider(0, 360, value=220, step=5, label="Wind Direction (degrees)")
            with gr.Row():
                hour = gr.Slider(0, 23, value=13, step=1, label="Hour of Day (0-23)")
                month = gr.Slider(1, 12, value=6, step=1, label="Month (1-12)")
            past_power = gr.Number(value=4.5, label="Past Hour Hybrid Power (kW)")
            predict_btn = gr.Button("⚡ Predict Energy Yield", variant="primary")

        with gr.Column():
            gr.Markdown("### 📊 Predicted Output (1-Hour Ahead)")
            hybrid_out = gr.Textbox(label="Predicted Total Hybrid Power", elem_id="hybrid-output")
            with gr.Row():
                solar_out = gr.Textbox(label="Solar PV Contribution")
                wind_out = gr.Textbox(label="Wind Turbine Contribution")
            cf_out = gr.Textbox(label="Plant Capacity Factor")

    predict_btn.click(
        fn=predict_yield,
        inputs=[irradiance, wind_speed, temp, wind_dir, hour, month, past_power],
        outputs=[hybrid_out, solar_out, wind_out, cf_out],
        api_name="predict",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
