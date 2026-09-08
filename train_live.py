"""Live Training Script with Step-by-Step Output.

Demonstrates the difference between:
1. Tree Models (Random Forest & XGBoost): Training tree-by-tree / round-by-round.
2. Deep Neural Network (PyTorch MLP): Training epoch-by-epoch (50 Epochs).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "featured" / "hybrid_plant_featured.csv"


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    target_col = "target_hybrid_15m"
    exclude = [
        "site_id", "timestamp_utc", "technology", "cloud_type", "fault_code",
        "target_hybrid_5m", "target_hybrid_15m", "target_hybrid_30m", "target_hybrid_1h",
        "target_solar_1h", "target_wind_1h"
    ]
    raw_feats = [c for c in df.columns if c not in exclude]

    for b in ["has_bess", "is_daylight", "has_fault"]:
        if b in df.columns:
            df[b] = df[b].astype(int)

    cat_cols = [c for c in ["interconnect_status", "hybrid_dispatch_mode"] if c in df.columns]
    X_df = pd.get_dummies(df[raw_feats], columns=cat_cols, drop_first=True, dtype=int)
    X_df = X_df.dropna(axis=1, how="all").fillna(0.0)
    y_series = df[target_col].fillna(0.0)

    split = int(len(X_df) * 0.80)
    X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
    y_train, y_test = y_series.iloc[:split], y_series.iloc[split:]

    return X_train, y_train, X_test, y_test, X_df.columns.tolist()


# Define PyTorch Deep Learning Model
class HybridYieldNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_live():
    print("=" * 75)
    print(" LIVE MODEL TRAINING: HYBRID SOLAR-WIND ENERGY YIELD PREDICTOR")
    print("=" * 75)
    X_train, y_train, X_test, y_test, feats = prepare_data()
    print(f"Data Loaded: 1,344 Training rows | 336 Test rows | {len(feats)} Features\n")

    # =========================================================================
    # 1. RANDOM FOREST (Tree-by-Tree Building)
    # =========================================================================
    print("-" * 75)
    print(" MODEL 1: RANDOM FOREST REGRESSOR")
    print(" Trees are grown iteratively (Random Forest builds a forest of trees, not epochs)")
    print("-" * 75)

    rf = RandomForestRegressor(
        n_estimators=0,
        warm_start=True,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )

    total_trees = 150
    step = 15
    for n_trees in range(step, total_trees + 1, step):
        rf.n_estimators = n_trees
        t0 = time.time()
        rf.fit(X_train, y_train)
        dt = time.time() - t0
        preds = np.clip(rf.predict(X_test), 0.0, None)
        r2 = r2_score(y_test, preds)
        mae_mw = mean_absolute_error(y_test, preds) / 1000.0
        bar = "#" * (n_trees // 10) + "-" * (15 - (n_trees // 10))
        print(f"  Trees Built: [{bar}] {n_trees:3d}/150 | Test R2: {r2:6.4f} | Test MAE: {mae_mw:5.2f} MW | Step Time: {dt:4.2f}s")

    print(f"  -> Random Forest Done: 150 Trees Built. Final Test R2 = {r2:.4f}, MAE = {mae_mw:.2f} MW\n")

    # =========================================================================
    # 2. XGBOOST (Boosting Iterations)
    # =========================================================================
    print("-" * 75)
    print(" MODEL 2: XGBOOST REGRESSOR (100 Boosting Rounds)")
    print("-" * 75)
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mae",
    )
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=20,  # Print every 20 boosting rounds
    )
    xgb_preds = np.clip(xgb.predict(X_test), 0.0, None)
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_mae = mean_absolute_error(y_test, xgb_preds) / 1000.0
    print(f"  -> XGBoost Done: 100 Rounds Completed. Final Test R2 = {xgb_r2:.4f}, MAE = {xgb_mae:.2f} MW\n")

    # =========================================================================
    # 3. PYTORCH DEEP NEURAL NETWORK (50 Real Epochs with Backpropagation)
    # =========================================================================
    print("-" * 75)
    print(" MODEL 3: PYTORCH DEEP NEURAL NETWORK (Training across 50 Epochs)")
    print(" Uses Forward Pass -> MSE Loss -> Backprop Gradients -> Weight Update")
    print("-" * 75)

    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

    net = HybridYieldNet(input_dim=X_train_s.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=1e-4)

    epochs = 50
    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        out = net(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            net.eval()
            with torch.no_grad():
                test_out = net(X_test_t).numpy().flatten()
                test_preds_kw = scaler_y.inverse_transform(test_out.reshape(-1, 1)).flatten()
                test_preds_kw = np.clip(test_preds_kw, 0.0, None)
                r2_nn = r2_score(y_test, test_preds_kw)
                mae_nn = mean_absolute_error(y_test, test_preds_kw) / 1000.0

            print(f"  Epoch [{epoch:2d}/50] | Train Loss (MSE): {loss.item():6.4f} | Test R2: {r2_nn:6.4f} | Test MAE: {mae_nn:5.2f} MW")

    print(f"\n  -> PyTorch Neural Network Done: 50 Epochs. Final Test R2 = {r2_nn:.4f}, MAE = {mae_nn:.2f} MW")
    print("=" * 75)
    print(" TRAINING SUMMARY")
    print(f"  Random Forest (150 Trees) : R2 = {r2:.4f} | MAE = {mae_mw:.2f} MW (WINNER)")
    print(f"  XGBoost (100 Rounds)      : R2 = {xgb_r2:.4f} | MAE = {xgb_mae:.2f} MW")
    print(f"  PyTorch Neural Net (50 Ep): R2 = {r2_nn:.4f} | MAE = {mae_nn:.2f} MW")
    print("=" * 75)


if __name__ == "__main__":
    train_live()
