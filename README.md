# ⚡ Hybrid Solar-Wind Energy Yield Predictor

A machine learning engine and monitoring system designed to forecast renewable energy yields from co-located hybrid power facilities (Solar PV + Wind Turbine sharing the same grid interconnection).

---

## 🌐 Live Deployments
- **Interactive Web App (Vercel)**: [https://hybrid-solar-wind-energy-yield-pred.vercel.app/](https://hybrid-solar-wind-energy-yield-pred.vercel.app/)
- **Live ML Inference API (Hugging Face Spaces)**: [https://uwouldnever-hybrid-energy-predictor.hf.space](https://uwouldnever-hybrid-energy-predictor.hf.space)

---

## 🔬 The Dataset: DTU SOLETE Facility
Unlike traditional research projects that stitch together unrelated solar and wind data from different cities, this project uses the peer-reviewed **SOLETE** dataset from the Technical University of Denmark (DTU Risø Campus):
- **Co-Located Hardware**: Physical 7 kW Solar PV Array + Gaia 11 kW Wind Turbine + On-Site Meteorological Mast.
- **Duration**: **15 continuous months (457 days, 10,969 hourly timestamps)** covering all 4 seasons (June 1, 2018 to September 1, 2019).
- **Zero Location Mismatch**: Solar irradiance, wind speeds, panel temperatures, and turbine outputs are 100% physically and temporally aligned.

---

## 🏆 Model Benchmark Results
Evaluated on **unseen future timestamps** (12 months Train / 3 months Holdout Test):

| Rank | Model Architecture | R2 Score | MAE (kW) | RMSE (kW) | Pearson Correlation (r) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Hybrid Voting Ensemble** | **0.9227** | **0.235 kW** | **0.473 kW** | **0.9607 (96.1%)** |
| 🥈 | **LightGBM Regressor** | **0.9224** | 0.238 kW | 0.473 kW | 0.9606 (96.1%) |
| 🥉 | **XGBoost Regressor** | **0.9214** | 0.240 kW | 0.477 kW | 0.9600 (96.0%) |
| 4 | **Random Forest** | **0.9194** | 0.239 kW | 0.482 kW | 0.9590 (95.9%) |
| 5 | **Gradient Boosting** | **0.9187** | 0.246 kW | 0.485 kW | 0.9585 (95.9%) |
| 6 | **Ridge Regression** | **0.9035** | 0.295 kW | 0.528 kW | 0.9525 (95.3%) |

> **Key Takeaway**: The **Hybrid Voting Ensemble** (Random Forest 40% + XGBoost 35% + LightGBM 25%) achieved the highest performance by balancing bagging variance reduction with gradient-boosting error correction.

---

## 🧠 Top Predictive Features
1. Recent Hybrid Power Output Lag (80.4%)
2. 1-Hour Hybrid Ramp Delta (2.4%)
3. Diurnal Hour of Day (2.4%)
4. Solar Elevation & Azimuth Angles (2.3%)
5. Plane-of-Array Irradiance & GHI (1.6%)
6. Aerodynamic Wind Vector Components u and v (1.4%)

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/shaun2103008/Hybrid-Solar-Wind-Energy-Yield-Predictor.git
cd Hybrid-Solar-Wind-Energy-Yield-Predictor
pip install -r hf_space/requirements.txt
```

### 2. Run the Benchmark
```bash
python train_solete.py
```

### 3. Run the Local API Server
```bash
python -m uvicorn api:app --reload --port 8000
```
Visit `http://localhost:8000/docs` to test interactive predictions.
