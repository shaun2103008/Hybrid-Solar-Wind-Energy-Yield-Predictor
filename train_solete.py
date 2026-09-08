"""Complete Processing, Feature Engineering, and Model Benchmarking Pipeline for SOLETE Dataset.

SOLETE: 15-Month Co-located Wind Turbine + Solar PV + Meteorological Station (DTU Denmark).
Covers June 1, 2018 to September 1, 2019 (10,969 continuous hourly rows).
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "dataset" / "hybrid" / "solete" / "solete_60min.csv"
FEATURED_PATH = ROOT / "dataset" / "hybrid" / "solete" / "solete_featured_hourly.csv"
RESULTS_DIR = ROOT / "results"


def engineer_solete_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw SOLETE columns into ML-ready features."""
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # 1. Standardize Base Power & Physical Drivers
    df["solar_power_kW"] = df["P_Solar[kW]"].clip(lower=0.0)
    df["wind_power_kW"] = df["P_Gaia[kW]"].clip(lower=0.0)
    df["hybrid_power_kW"] = (df["solar_power_kW"] + df["wind_power_kW"]).clip(lower=0.0)

    df["irradiance_ghi"] = df["GHI[kW1m2]"].clip(lower=0.0)
    df["irradiance_poa"] = df["POA Irr[kW1m2]"].clip(lower=0.0)
    df["wind_speed"] = df["WIND_SPEED[m1s]"].clip(lower=0.0)
    df["wind_direction"] = df["WIND_DIR[deg]"] % 360.0
    df["temperature_C"] = df["TEMPERATURE[degC]"]
    df["humidity_pct"] = df["HUMIDITY[%]"]
    df["pressure_mbar"] = df["Pressure[mbar]"].clip(900.0, 1100.0)  # clip sensor artifacts
    df["solar_azimuth"] = df["Azimuth[deg]"]
    df["solar_elevation"] = df["Elevation[deg]"]

    # Aerodynamic wind vector decomposition
    rad = np.radians(df["wind_direction"])
    df["wind_vector_u"] = -df["wind_speed"] * np.sin(rad)
    df["wind_vector_v"] = -df["wind_speed"] * np.cos(rad)

    # 2. Temporal and Seasonal Cyclical Encodings
    dt = df["timestamp_utc"].dt
    df["hour"] = dt.hour
    df["month"] = dt.month
    df["day_of_year"] = dt.dayofyear
    df["day_of_week"] = dt.dayofweek

    # Daily cycle
    time_fraction = df["hour"] / 24.0
    df["sin_hour"] = np.sin(2 * np.pi * time_fraction)
    df["cos_hour"] = np.cos(2 * np.pi * time_fraction)

    # Annual seasonal cycle
    month_fraction = (df["month"] - 1) / 12.0
    df["sin_month"] = np.sin(2 * np.pi * month_fraction)
    df["cos_month"] = np.cos(2 * np.pi * month_fraction)

    df["is_daylight"] = (df["irradiance_ghi"] > 0.005).astype(int)

    # 3. Autoregressive Lags (1h, 2h, 3h, 6h, 12h, 24h)
    lags = [1, 2, 3, 6, 12, 24]
    for lag in lags:
        df[f"solar_lag_{lag}h"] = df["solar_power_kW"].shift(lag)
        df[f"wind_lag_{lag}h"] = df["wind_power_kW"].shift(lag)
        df[f"hybrid_lag_{lag}h"] = df["hybrid_power_kW"].shift(lag)
        if lag in [1, 2, 24]:
            df[f"ghi_lag_{lag}h"] = df["irradiance_ghi"].shift(lag)
            df[f"wind_speed_lag_{lag}h"] = df["wind_speed"].shift(lag)

    # 4. Rolling Window Momentum & Volatility
    for win in [3, 6]:
        df[f"solar_roll_mean_{win}h"] = df["solar_power_kW"].rolling(win, min_periods=1).mean()
        df[f"wind_roll_mean_{win}h"] = df["wind_power_kW"].rolling(win, min_periods=1).mean()
        df[f"wind_roll_std_{win}h"] = df["wind_power_kW"].rolling(win, min_periods=1).std().fillna(0.0)
        df[f"hybrid_roll_mean_{win}h"] = df["hybrid_power_kW"].rolling(win, min_periods=1).mean()

    # Ramping dynamics
    df["solar_ramp_1h"] = df["solar_power_kW"].diff().fillna(0.0)
    df["wind_ramp_1h"] = df["wind_power_kW"].diff().fillna(0.0)
    df["hybrid_ramp_1h"] = df["hybrid_power_kW"].diff().fillna(0.0)
    df["wind_speed_ramp_1h"] = df["wind_speed"].diff().fillna(0.0)
    df["ghi_ramp_1h"] = df["irradiance_ghi"].diff().fillna(0.0)

    # 5. Future Forecasting Targets
    df["target_hybrid_1h"] = df["hybrid_power_kW"].shift(-1)
    df["target_hybrid_2h"] = df["hybrid_power_kW"].shift(-2)
    df["target_hybrid_6h"] = df["hybrid_power_kW"].shift(-6)
    df["target_hybrid_24h"] = df["hybrid_power_kW"].shift(-24)
    df["target_solar_1h"] = df["solar_power_kW"].shift(-1)
    df["target_wind_1h"] = df["wind_power_kW"].shift(-1)

    # Drop 24 leading rows (due to 24h lag) and 24 trailing rows (due to 24h forecast lead)
    df_valid = df.iloc[24:-24].copy().reset_index(drop=True)
    return df_valid


def run_training_and_benchmarks():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" SOLETE 15-MONTH CO-LOCATED HYBRID ENERGY YIELD BENCHMARK")
    print("=" * 80)

    raw_df = pd.read_csv(RAW_PATH)
    print(f"Loaded Raw SOLETE: {len(raw_df)} hourly timestamps (457 continuous days)")

    featured_df = engineer_solete_features(raw_df)
    featured_df.to_csv(FEATURED_PATH, index=False)
    print(f"Engineered Features: {len(featured_df)} valid hours | {featured_df.shape[1]} columns")

    target_col = "target_hybrid_1h"
    exclude = [
        "timestamp_utc", "P_Solar[kW]", "P_Gaia[kW]", "P_Hybrid[kW]",
        "GHI[kW1m2]", "POA Irr[kW1m2]", "WIND_SPEED[m1s]", "WIND_DIR[deg]",
        "TEMPERATURE[degC]", "HUMIDITY[%]", "Pressure[mbar]", "Azimuth[deg]", "Elevation[deg]",
        "target_hybrid_1h", "target_hybrid_2h", "target_hybrid_6h", "target_hybrid_24h",
        "target_solar_1h", "target_wind_1h"
    ]
    feature_cols = [c for c in featured_df.columns if c not in exclude]

    X = featured_df[feature_cols].copy().fillna(0.0)
    y = featured_df[target_col].copy().fillna(0.0)
    timestamps = featured_df["timestamp_utc"]

    # Chronological Split: 80% Train (~12 months) / 20% Unseen Future (~3 months)
    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_ts = timestamps.iloc[split:]

    print(f"\nTrain Set: {len(X_train)} hours (~12 months: {timestamps.iloc[0].strftime('%Y-%m-%d')} to {timestamps.iloc[split-1].strftime('%Y-%m-%d')})")
    print(f"Test Set : {len(X_test)} hours (~3 months: {timestamps.iloc[split].strftime('%Y-%m-%d')} to {timestamps.iloc[-1].strftime('%Y-%m-%d')})")
    print(f"Features : {len(feature_cols)} predictor variables\n")

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.07, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.06, num_leaves=31, random_state=42, verbose=-1),
        "XGBoost": XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.06, subsample=0.85, random_state=42),
    }

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = []
    preds_dict = {}

    print("-" * 80)
    print(f"{'MODEL':<22} | {'R2 SCORE':<9} | {'MAE (kW)':<10} | {'RMSE (kW)':<10} | {'PEARSON r':<10} | {'TIME (s)':<8}")
    print("-" * 80)

    for name, model in models.items():
        t0 = time.time()
        if name == "Ridge Regression":
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        dt = time.time() - t0
        preds = np.clip(preds, 0.0, None)
        preds_dict[name] = preds

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r_val, _ = pearsonr(y_test, preds)

        results.append({
            "Model": name,
            "R2": round(r2, 4),
            "MAE_kW": round(mae, 3),
            "RMSE_kW": round(rmse, 3),
            "Pearson_r": round(r_val, 4),
            "Train_Time_s": round(dt, 2),
            "model_obj": model,
        })
        print(f"{name:<22} | {r2:<9.4f} | {mae:<10.3f} | {rmse:<10.3f} | {r_val:<10.4f} | {dt:<8.2f}")

    # Build Voting Ensemble
    ens = VotingRegressor([
        ("rf", models["Random Forest"]),
        ("xgb", models["XGBoost"]),
        ("lgb", models["LightGBM"]),
    ], weights=[0.40, 0.35, 0.25])
    t0 = time.time()
    ens.fit(X_train, y_train)
    ens_preds = np.clip(ens.predict(X_test), 0.0, None)
    ens_dt = time.time() - t0
    preds_dict["Hybrid Voting Ensemble"] = ens_preds

    ens_r2 = r2_score(y_test, ens_preds)
    ens_mae = mean_absolute_error(y_test, ens_preds)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ens_preds))
    ens_r, _ = pearsonr(y_test, ens_preds)

    results.append({
        "Model": "Hybrid Voting Ensemble",
        "R2": round(ens_r2, 4),
        "MAE_kW": round(ens_mae, 3),
        "RMSE_kW": round(ens_rmse, 3),
        "Pearson_r": round(ens_r, 4),
        "Train_Time_s": round(ens_dt, 2),
        "model_obj": ens,
    })
    print(f"{'Hybrid Voting Ensemble':<22} | {ens_r2:<9.4f} | {ens_mae:<10.3f} | {ens_rmse:<10.3f} | {ens_r:<10.4f} | {ens_dt:<8.2f}")
    print("=" * 80)

    res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    best_row = res_df.iloc[0]
    best_name = best_row["Model"]
    best_model = best_row["model_obj"]

    print(f"\nWINNER: {best_name} (R2 = {best_row['R2']}, MAE = {best_row['MAE_kW']} kW, Pearson r = {best_row['Pearson_r']})")

    # Save benchmark table
    res_save = res_df.drop(columns=["model_obj"])
    bench_path = RESULTS_DIR / "solete_benchmark_results.csv"
    res_save.to_csv(bench_path, index=False)
    print(f"Saved Benchmark CSV: {bench_path}")

    # Plot Actual vs Predicted over a continuous 2-week test sample
    plot_path = RESULTS_DIR / "solete_actual_vs_predicted.png"
    sample_window = 24 * 14  # 14 days = 336 hours
    ts_sub = test_ts.iloc[:sample_window]
    y_sub = y_test.iloc[:sample_window]
    pred_sub = preds_dict[best_name][:sample_window]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax1.plot(ts_sub, y_sub, label="Actual Real Hybrid Power (kW)", color="#111827", linewidth=2.0)
    ax1.plot(ts_sub, pred_sub, label=f"Predicted ({best_name})", color="#2563EB", linewidth=1.8, linestyle="--")
    ax1.set_title(f"SOLETE Co-Located Hybrid Plant: 1-Hour Ahead Yield Prediction ({best_name})", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Hybrid Power (kW)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    err = pred_sub - y_sub.values
    ax2.plot(ts_sub, err, color="#EF4444", linewidth=1.1, label="Error (Predicted - Actual)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax2.fill_between(ts_sub, 0, err, color="#EF4444", alpha=0.2)
    ax2.set_ylabel("Error (kW)", fontsize=11)
    ax2.set_xlabel("Timestamp (UTC)", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"Saved Evaluation Plot: {plot_path}")

    # Extract Feature Importances
    fi_path = RESULTS_DIR / "solete_top_features.csv"
    rf_model = models["Random Forest"]
    fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": rf_model.feature_importances_})
    fi_df = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_df["Importance_Pct"] = round((fi_df["Importance"] / fi_df["Importance"].sum()) * 100.0, 2)
    fi_df.to_csv(fi_path, index=False)

    print("\nTop 10 Most Predictive Features:")
    for i, r in fi_df.head(10).iterrows():
        print(f"  {i+1:2d}. {r['Feature']:<28} : {r['Importance_Pct']:5.2f}%")

    # Save Best Model
    model_path = RESULTS_DIR / "solete_best_predictor.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nSaved Best Model Artifact: {model_path}")


if __name__ == "__main__":
    run_training_and_benchmarks()
