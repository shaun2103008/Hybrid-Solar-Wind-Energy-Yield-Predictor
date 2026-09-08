"""Multi-Model Training and Benchmarking Suite for Hybrid Solar-Wind Energy Yield Prediction.

Trains, evaluates, and compares multiple machine learning algorithms:
1. Ridge Regression (Linear Baseline)
2. Random Forest Regressor (Bagging Ensemble)
3. Gradient Boosting Regressor (GBR)
4. LightGBM Regressor (Fast Leaf-wise Boosting)
5. XGBoost Regressor (Regularized Histogram Boosting)
6. Voting Ensemble (Stacking top performers)

Evaluates on unseen future test data (chronological split) using R², MAE, RMSE,
Normalized Error (% of plant capacity), and computes feature importances.
Saves comparison plots and the best performing model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "featured" / "hybrid_plant_featured.csv"
RESULTS_DIR = ROOT / "results"


def load_and_prepare_data(
    data_path: Path,
    target_col: str = "target_hybrid_15m",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], pd.Series]:
    """Load featured data, encode categoricals, and perform chronological train/test split."""
    df = pd.read_csv(data_path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    timestamps = df["timestamp_utc"]

    # Exclude non-feature columns
    exclude_cols = [
        "site_id", "timestamp_utc", "technology", "cloud_type", "fault_code",
        "target_hybrid_5m", "target_hybrid_15m", "target_hybrid_30m", "target_hybrid_1h",
        "target_solar_1h", "target_wind_1h"
    ]
    raw_features = [c for c in df.columns if c not in exclude_cols]

    # Convert booleans to int
    for b_col in ["has_bess", "is_daylight", "has_fault"]:
        if b_col in df.columns:
            df[b_col] = df[b_col].astype(int)

    # One-hot encode string categoricals
    cat_cols = ["interconnect_status", "hybrid_dispatch_mode"]
    cat_present = [c for c in cat_cols if c in df.columns]
    X_df = pd.get_dummies(df[raw_features], columns=cat_present, drop_first=True, dtype=int)

    # Drop any all-null columns and fill remaining NaNs
    X_df = X_df.dropna(axis=1, how="all").fillna(0.0)
    y_series = df[target_col].fillna(0.0)

    feature_names = X_df.columns.tolist()

    # Chronological Split: 80% Train (past), 20% Test (future unseen)
    split_idx = int(len(X_df) * 0.80)
    X_train = X_df.iloc[:split_idx]
    y_train = y_series.iloc[:split_idx]
    X_test = X_df.iloc[split_idx:]
    y_test = y_series.iloc[split_idx:]
    test_timestamps = timestamps.iloc[split_idx:]

    print(f"Loaded dataset: {data_path.name}")
    print(f"Total samples: {len(X_df)} (80% Train: {len(X_train)} rows | 20% Future Test: {len(X_test)} rows)")
    print(f"Input features count: {len(feature_names)}")
    print(f"Target variable: {target_col} (Mean: {y_series.mean()/1000:.2f} MW, Max: {y_series.max()/1000:.2f} MW)")

    return X_train, y_train, X_test, y_test, feature_names, test_timestamps


def get_models() -> dict[str, object]:
    """Define candidate model suite with tuned default hyperparameters."""
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=120, max_depth=6, learning_rate=0.07, num_leaves=31, random_state=42, verbose=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=120, max_depth=5, learning_rate=0.07, subsample=0.85, colsample_bytree=0.85, random_state=42
        ),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    test_timestamps: pd.Series,
    capacity_kw: float = 255245.0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], object, str]:
    """Train and evaluate candidate models and voting ensemble."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    results = []
    predictions_dict = {}

    print("\n" + "=" * 80)
    print(f"{'MODEL':<22} | {'R² SCORE':<9} | {'MAE (MW)':<10} | {'RMSE (MW)':<10} | {'NMAE (%)':<9} | {'TIME (s)':<8}")
    print("=" * 80)

    for name, model in models.items():
        t0 = time.time()
        # Scale for linear models, raw for trees
        if name == "Ridge Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        train_time = time.time() - t0
        preds = np.clip(preds, 0.0, None)  # Physical power output >= 0
        predictions_dict[name] = preds

        r2 = r2_score(y_test, preds)
        mae_kw = mean_absolute_error(y_test, preds)
        rmse_kw = np.sqrt(mean_squared_error(y_test, preds))
        nmae_pct = (mae_kw / capacity_kw) * 100.0

        results.append({
            "Model": name,
            "R2": round(r2, 4),
            "MAE_kW": round(mae_kw, 2),
            "RMSE_kW": round(rmse_kw, 2),
            "MAE_MW": round(mae_kw / 1000.0, 3),
            "RMSE_MW": round(rmse_kw / 1000.0, 3),
            "NMAE_pct": round(nmae_pct, 2),
            "Train_Time_s": round(train_time, 3),
            "model_obj": model,
        })
        print(f"{name:<22} | {r2:<9.4f} | {mae_kw/1000:<10.3f} | {rmse_kw/1000:<10.3f} | {nmae_pct:<9.2f} | {train_time:<8.3f}")

    # Build Voting Ensemble from top 3 tree models (RF, LightGBM, XGBoost)
    t0 = time.time()
    ensemble = VotingRegressor(
        estimators=[
            ("rf", models["Random Forest"]),
            ("lgb", models["LightGBM"]),
            ("xgb", models["XGBoost"]),
        ],
        weights=[0.25, 0.40, 0.35],
    )
    ensemble.fit(X_train, y_train)
    ens_preds = np.clip(ensemble.predict(X_test), 0.0, None)
    ens_time = time.time() - t0
    predictions_dict["Hybrid Voting Ensemble"] = ens_preds

    ens_r2 = r2_score(y_test, ens_preds)
    ens_mae = mean_absolute_error(y_test, ens_preds)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ens_preds))
    ens_nmae = (ens_mae / capacity_kw) * 100.0

    results.append({
        "Model": "Hybrid Voting Ensemble",
        "R2": round(ens_r2, 4),
        "MAE_kW": round(ens_mae, 2),
        "RMSE_kW": round(ens_rmse, 2),
        "MAE_MW": round(ens_mae / 1000.0, 3),
        "RMSE_MW": round(ens_rmse / 1000.0, 3),
        "NMAE_pct": round(ens_nmae, 2),
        "Train_Time_s": round(ens_time, 3),
        "model_obj": ensemble,
    })
    print(f"{'Hybrid Voting Ensemble':<22} | {ens_r2:<9.4f} | {ens_mae/1000:<10.3f} | {ens_rmse/1000:<10.3f} | {ens_nmae:<9.2f} | {ens_time:<8.3f}")
    print("=" * 80)

    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_model_name = best_row["Model"]
    best_model_obj = best_row["model_obj"]

    print(f"\nBest Performing Model: {best_model_name} (R² = {best_row['R2']}, MAE = {best_row['MAE_MW']} MW, NMAE = {best_row['NMAE_pct']}%)")
    return results_df, predictions_dict, best_model_obj, best_model_name


def plot_predictions(
    test_timestamps: pd.Series,
    y_test: pd.Series,
    predictions_dict: dict[str, np.ndarray],
    best_model_name: str,
    output_path: Path,
) -> None:
    """Generate high-contrast publication-ready comparison plot."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    y_test_mw = y_test / 1000.0
    best_preds_mw = predictions_dict[best_model_name] / 1000.0

    # Top Plot: Actual vs Best Model Predictions
    ax1.plot(test_timestamps, y_test_mw, label="Actual Hybrid Yield (MW)", color="#111827", linewidth=2.4, alpha=0.9)
    ax1.plot(test_timestamps, best_preds_mw, label=f"Predicted ({best_model_name})", color="#2563EB", linewidth=1.9, linestyle="--")

    # Add other models with subtle transparency
    colors = {"Random Forest": "#10B981", "XGBoost": "#F59E0B", "LightGBM": "#8B5CF6"}
    for m_name, col in colors.items():
        if m_name in predictions_dict and m_name != best_model_name:
            ax1.plot(test_timestamps, predictions_dict[m_name] / 1000.0, label=m_name, color=col, linewidth=1.1, alpha=0.55)

    ax1.set_title("Hybrid Solar-Wind Energy Yield Prediction (1-Hour Ahead Horizon on Unseen Future Data)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Power Output (MW)", fontsize=11, fontweight="semibold")
    ax1.legend(loc="upper right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bottom Plot: Residual Error
    best_array = best_preds_mw.values if hasattr(best_preds_mw, "values") else np.asarray(best_preds_mw)
    residuals_mw = best_array - y_test_mw.values
    ax2.plot(test_timestamps, residuals_mw, color="#EF4444", linewidth=1.2, label="Prediction Residual (Predicted - Actual)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.fill_between(test_timestamps, 0, residuals_mw, color="#EF4444", alpha=0.18)
    ax2.set_xlabel("Timestamp (UTC)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Error (MW)", fontsize=11, fontweight="semibold")
    ax2.legend(loc="lower right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved visual evaluation plot: {output_path}")


def extract_feature_importance(model: object, feature_names: list[str], output_path: Path) -> pd.DataFrame:
    """Extract and save feature importances from tree-based or ensemble models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    elif hasattr(model, "named_estimators_") and "xgb" in model.named_estimators_:
        importances = model.named_estimators_["xgb"].feature_importances_
    elif hasattr(model, "estimators_"):
        # Average across estimators
        importances = np.mean([est.feature_importances_ for est in model.estimators_ if hasattr(est, "feature_importances_")], axis=0)
    else:
        return pd.DataFrame()

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    fi_df["Importance_Pct"] = round((fi_df["Importance"] / fi_df["Importance"].sum()) * 100.0, 2)
    fi_df.to_csv(output_path, index=False)

    print("\nTop 10 Most Predictive Features:")
    for idx, row in fi_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['Feature']:<28} : {row['Importance_Pct']:5.2f}%")
    return fi_df


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, feature_names, test_timestamps = load_and_prepare_data(DATA_PATH)

    # Train and evaluate all models
    results_df, predictions_dict, best_model_obj, best_model_name = evaluate_models(
        X_train, y_train, X_test, y_test, feature_names, test_timestamps
    )

    # Save benchmark table
    results_save = results_df.drop(columns=["model_obj"])
    results_csv = RESULTS_DIR / "model_benchmark_results.csv"
    results_save.to_csv(results_csv, index=False)
    print(f"\nSaved benchmark table: {results_csv}")

    # Plot predictions
    plot_path = RESULTS_DIR / "actual_vs_predicted_comparison.png"
    plot_predictions(test_timestamps, y_test, predictions_dict, best_model_name, plot_path)

    # Extract feature importances
    fi_csv = RESULTS_DIR / "top_feature_importances.csv"
    extract_feature_importance(best_model_obj, feature_names, fi_csv)

    # Save the best model artifact for inference
    model_save_path = RESULTS_DIR / "best_hybrid_predictor.joblib"
    joblib.dump(best_model_obj, model_save_path)
    print(f"\nExported best model artifact: {model_save_path}")


if __name__ == "__main__":
    main()
