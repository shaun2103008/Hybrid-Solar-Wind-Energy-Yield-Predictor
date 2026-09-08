"""Data Cleaning and Feature Engineering Pipeline for Hybrid Solar-Wind Energy Yield Prediction.

Processes the co-located Hybrid Plant SCADA dataset:
- Cleans missing values, sensor anomalies, and formats timestamps.
- Engineers physical weather drivers (irradiance, wind vector components, air density).
- Extracts autoregressive lags, rolling window momentum, volatility, and cyclical time encodings.
- Creates future prediction targets (5-min, 15-min, 30-min, 1-hour ahead).
- Exports ready-to-train datasets for machine learning models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "hybrid" / "enr002_sample" / "hybrid_plant_scada.csv"
CLEANED_DIR = ROOT / "cleaned"
FEATURED_DIR = ROOT / "featured"
REPORT_DIR = ROOT / "metadata"


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values, types, and raw columns."""
    df = df.copy()

    # Parse and sort by timestamp
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values(by=["site_id", "timestamp_utc"]).reset_index(drop=True)

    # Fill or calculate missing columns
    if "fault_code" in df.columns:
        df["fault_code"] = df["fault_code"].fillna("Normal").astype(str)
        df["has_fault"] = (df["fault_code"] != "Normal").astype(int)

    # Power clipping: physical generation cannot be negative
    for col in ["ac_power_kW", "dc_power_kW", "farm_power_kW", "actual_power_kW", "turbine_power_kW"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    # Irradiance clipping
    for col in ["ghi_w_per_m2", "dni_w_per_m2", "dhi_w_per_m2", "poa_irradiance_w_m2"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    return df


def engineer_site_features(site_df: pd.DataFrame) -> pd.DataFrame:
    """Extract domain features for a single co-located hybrid plant time series."""
    df = site_df.copy().sort_values("timestamp_utc").reset_index(drop=True)

    # 1. Base Core Variables (Standardized names in both kW and MW)
    df["solar_power_kW"] = df["ac_power_kW"]
    df["solar_power_MW"] = df["solar_power_kW"] / 1000.0

    df["wind_power_kW"] = df["farm_power_kW"]
    df["wind_power_MW"] = df["wind_power_kW"] / 1000.0

    df["hybrid_power_kW"] = df["actual_power_kW"]
    df["hybrid_power_MW"] = df["hybrid_power_kW"] / 1000.0

    # 2. Solar Exogenous Drivers
    df["irradiance_ghi"] = df["ghi_w_per_m2"]
    df["irradiance_dni"] = df["dni_w_per_m2"]
    df["irradiance_dhi"] = df["dhi_w_per_m2"]
    df["irradiance_poa"] = df["poa_irradiance_w_m2"]
    df["cell_temp_C"] = df["cell_temp_C"]
    df["solar_elevation_deg"] = df["solar_elevation_deg"]
    df["clearness_index"] = df["clearness_index"]
    df["tracker_angle_deg"] = df["tracker_angle_deg"]

    # 3. Wind Exogenous Drivers
    df["wind_speed_hub"] = df["wind_speed_hub_m_per_s"]
    df["wind_direction_deg"] = df["wind_direction_deg"]
    df["pitch_angle_deg"] = df["pitch_angle_deg"]
    df["rotor_rpm"] = df["rpm_rotor"]
    df["turbulence_intensity"] = df["turbulence_intensity"]
    df["yaw_error_deg"] = df["yaw_error_deg"]

    # Decompose wind direction into orthogonal vector components (u, v)
    rad = np.radians(df["wind_direction_deg"])
    df["wind_vector_u"] = -df["wind_speed_hub"] * np.sin(rad)  # East-West component
    df["wind_vector_v"] = -df["wind_speed_hub"] * np.cos(rad)  # North-South component

    # 4. Atmospheric / Weather State
    df["temperature_C"] = df["temperature_C"]
    df["humidity_pct"] = df["humidity_pct"]
    df["pressure_hPa"] = df["pressure_hPa"]
    df["air_density_kg_m3"] = df["air_density_kg_m3"]
    df["cloud_cover_pct"] = df["cloud_cover_pct"]

    # 5. Temporal & Cyclical Encodings
    dt = df["timestamp_utc"].dt
    df["hour"] = dt.hour
    df["minute"] = dt.minute
    df["day_of_week"] = dt.dayofweek
    time_fraction = (df["hour"] * 60 + df["minute"]) / 1440.0
    df["sin_time_of_day"] = np.sin(2 * np.pi * time_fraction)
    df["cos_time_of_day"] = np.cos(2 * np.pi * time_fraction)
    df["is_daylight"] = (df["irradiance_ghi"] > 5.0).astype(int)

    # 6. Autoregressive Lags (Past History)
    # Sampling is 5 minutes: 1 step = 5m, 3 steps = 15m, 6 steps = 30m, 12 steps = 1h
    lag_steps = [1, 2, 3, 6, 12]
    for lag in lag_steps:
        mins = lag * 5
        df[f"solar_lag_{mins}m"] = df["solar_power_kW"].shift(lag)
        df[f"wind_lag_{mins}m"] = df["wind_power_kW"].shift(lag)
        df[f"hybrid_lag_{mins}m"] = df["hybrid_power_kW"].shift(lag)
        df[f"ghi_lag_{mins}m"] = df["irradiance_ghi"].shift(lag)
        df[f"wind_speed_lag_{mins}m"] = df["wind_speed_hub"].shift(lag)

    # 7. Rolling Window Statistics (Momentum & Volatility)
    # 15-minute rolling (3 timesteps)
    df["solar_roll_mean_15m"] = df["solar_power_kW"].rolling(window=3, min_periods=1).mean()
    df["wind_roll_mean_15m"] = df["wind_power_kW"].rolling(window=3, min_periods=1).mean()
    df["wind_roll_std_15m"] = df["wind_power_kW"].rolling(window=3, min_periods=1).std().fillna(0.0)
    df["hybrid_roll_mean_15m"] = df["hybrid_power_kW"].rolling(window=3, min_periods=1).mean()

    # 1-hour rolling (12 timesteps)
    df["solar_roll_mean_1h"] = df["solar_power_kW"].rolling(window=12, min_periods=1).mean()
    df["wind_roll_mean_1h"] = df["wind_power_kW"].rolling(window=12, min_periods=1).mean()
    df["wind_roll_std_1h"] = df["wind_power_kW"].rolling(window=12, min_periods=1).std().fillna(0.0)
    df["hybrid_roll_mean_1h"] = df["hybrid_power_kW"].rolling(window=12, min_periods=1).mean()

    # Rate of change (Ramping dynamics)
    df["solar_ramp_5m"] = df["solar_power_kW"].diff().fillna(0.0)
    df["wind_ramp_5m"] = df["wind_power_kW"].diff().fillna(0.0)
    df["ghi_ramp_5m"] = df["irradiance_ghi"].diff().fillna(0.0)
    df["wind_speed_ramp_5m"] = df["wind_speed_hub"].diff().fillna(0.0)

    # 8. Hybrid Complementarity & Interaction Metrics
    df["solar_wind_ratio"] = df["solar_power_kW"] / (df["wind_power_kW"] + 1.0)
    df["hybrid_capacity_factor"] = (df["hybrid_power_MW"] / df["capacity_mw"].iloc[0]).clip(0.0, 1.0)
    df["bess_soc_pct"] = df["bess_soc_pct"]
    df["bess_net_flow_kW"] = df["bess_discharge_kW"] - df["bess_charge_kW"]

    # 9. Future Targets (For Prediction Horizons)
    # Predict next 5 min, 15 min, 30 min, and 1 hour ahead
    df["target_hybrid_5m"] = df["hybrid_power_kW"].shift(-1)
    df["target_hybrid_15m"] = df["hybrid_power_kW"].shift(-3)
    df["target_hybrid_30m"] = df["hybrid_power_kW"].shift(-6)
    df["target_hybrid_1h"] = df["hybrid_power_kW"].shift(-12)

    df["target_solar_1h"] = df["solar_power_kW"].shift(-12)
    df["target_wind_1h"] = df["wind_power_kW"].shift(-12)

    return df


def main() -> None:
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    FEATURED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw co-located hybrid SCADA dataset: {RAW_PATH}")
    raw_df = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {raw_df.shape}")

    # Clean raw data
    clean_df = clean_raw_data(raw_df)
    clean_path = CLEANED_DIR / "hybrid_plant_cleaned.csv"
    clean_df.to_csv(clean_path, index=False)
    print(f"Saved cleaned dataset: {clean_path} ({clean_df.shape})")

    # Process each site separately and combine
    sites = clean_df["site_id"].unique().tolist()
    featured_dfs = []

    for i, site in enumerate(sites, 1):
        site_sub = clean_df[clean_df["site_id"] == site].copy()
        cap = site_sub["capacity_mw"].iloc[0]
        print(f"\nProcessing Site {i}: ID={site[:8]}... (Capacity: {cap:.1f} MW)")
        site_feat = engineer_site_features(site_sub)

        # Drop initial rows where lags are undefined and trailing rows where targets are undefined
        # Max lag = 12 steps, max lead = 12 steps
        site_feat_valid = site_feat.iloc[12:-12].copy().reset_index(drop=True)
        print(f"  Valid time steps after lag/lead alignment: {len(site_feat_valid)} rows, {site_feat_valid.shape[1]} columns")

        site_file = FEATURED_DIR / f"plant_{i}_{int(cap)}MW_featured.csv"
        site_feat_valid.to_csv(site_file, index=False)
        print(f"  Saved site feature dataset: {site_file.name}")
        featured_dfs.append(site_feat_valid)

    # Combined master dataset
    master_df = pd.concat(featured_dfs, ignore_index=True)
    master_path = FEATURED_DIR / "hybrid_plant_featured.csv"
    master_df.to_csv(master_path, index=False)
    print(f"\nSaved combined master featured dataset: {master_path} ({master_df.shape})")

    # Generate Feature Summary Report
    sample_cols = [
        "timestamp_utc", "solar_power_kW", "wind_power_kW", "hybrid_power_kW",
        "irradiance_ghi", "temperature_C", "wind_speed_hub", "wind_direction_deg",
        "cloud_cover_pct", "solar_lag_15m", "wind_lag_15m", "hybrid_lag_15m",
        "target_hybrid_1h"
    ]
    report = {
        "dataset_name": "Co-Located Hybrid Solar-Wind Plant SCADA Dataset",
        "sampling_resolution": "5 minutes",
        "total_valid_rows": len(master_df),
        "total_columns": master_df.shape[1],
        "plants": [
            {"plant_index": 1, "capacity_mw": round(float(clean_df[clean_df['site_id'] == sites[0]]['capacity_mw'].iloc[0]), 2)},
            {"plant_index": 2, "capacity_mw": round(float(clean_df[clean_df['site_id'] == sites[1]]['capacity_mw'].iloc[0]), 2)},
        ],
        "key_feature_samples": sample_cols,
        "correlations_with_hybrid_power": {
            "solar_power_kW": round(float(master_df["solar_power_kW"].corr(master_df["hybrid_power_kW"])), 3),
            "wind_power_kW": round(float(master_df["wind_power_kW"].corr(master_df["hybrid_power_kW"])), 3),
            "wind_speed_hub": round(float(master_df["wind_speed_hub"].corr(master_df["hybrid_power_kW"])), 3),
            "irradiance_ghi": round(float(master_df["irradiance_ghi"].corr(master_df["hybrid_power_kW"])), 3),
            "hybrid_lag_15m": round(float(master_df["hybrid_lag_15m"].corr(master_df["hybrid_power_kW"])), 3),
        }
    }
    report_path = REPORT_DIR / "feature_engineering_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Feature summary report written: {report_path}")


if __name__ == "__main__":
    main()
