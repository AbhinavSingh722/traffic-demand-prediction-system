# Traffic Demand Prediction System

> **Industry-level ML Pipeline** for predicting urban traffic demand using ensemble learning.
> Built for **Innovexa Catalyst** | Achieves **96%+ R2 accuracy** with LightGBM + XGBoost ensemble.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

This project implements a complete **Traffic Demand Prediction System** that forecasts vehicle traffic demand (vehicles/hour) using historical traffic, weather, road infrastructure, and location data. The system uses an ensemble of gradient boosting models to achieve production-grade accuracy.

### Architecture

```
traffic_prediction/
├── .streamlit/
│   └── config.toml              # Dark theme configuration
├── data/
│   └── smart_city_traffic.csv   # Synthetic dataset (125K records)
├── notebooks/
│   └── EDA_and_modeling.py      # Complete ML pipeline
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py      # Synthetic data generation
│   ├── data_preprocessing.py    # Cleaning, encoding, scaling
│   ├── feature_engineering.py   # 5 engineered features
│   ├── model_training.py        # RF, XGBoost, LightGBM training
│   ├── ensemble.py              # Weighted ensemble (55/45)
│   └── evaluate.py              # Metrics & visualizations
├── models/
│   ├── random_forest.pkl        # Trained Random Forest
│   ├── xgboost.pkl              # Trained XGBoost
│   ├── lightgbm.pkl             # Trained LightGBM
│   ├── ensemble_config.json     # Ensemble weights
│   ├── label_encoders.pkl       # Fitted encoders
│   ├── scaler.pkl               # Fitted scaler
│   ├── feature_names.json       # Feature ordering
│   └── demand_stats.json        # Training data statistics
├── app/
│   └── streamlit_app.py         # Interactive web application
├── reports/
│   ├── project_report.md        # Full methodology report
│   ├── dataset_documentation.md # Dataset details
│   ├── model_comparison.csv     # Model metrics table
│   └── plots/                   # All 6+ EDA visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model Performance

| Model | R2 Score | RMSE | MAE | MAPE (%) |
|-------|----------|------|-----|----------|
| Random Forest | 97.87% | 16.66 | 9.80 | 7.96% |
| XGBoost | 98.37% | 14.59 | 8.74 | 7.85% |
| LightGBM | 98.37% | 14.58 | 8.76 | 8.14% |
| **Ensemble (55% LGBM + 45% XGB)** | **98.39%** | **14.49** | **8.67** | **7.81%** |

> All models significantly exceed their minimum R2 thresholds. The ensemble achieves the best overall performance.

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the Repository
```bash
git clone <repository-url>
cd traffic_prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML Pipeline
This generates the dataset, trains all models, and produces evaluation artifacts:
```bash
python notebooks/EDA_and_modeling.py
```

### 4. Run the Streamlit App Locally
```bash
streamlit run app/streamlit_app.py
```

---

## Dataset Description

| Feature | Type | Description |
|---------|------|-------------|
| Geohash_Location | Categorical | City zone identifier (20 zones) |
| Timestamp | DateTime | Record timestamp |
| Day_of_Week | Categorical | Day name (Monday-Sunday) |
| Hour | Integer | Hour of day (0-23) |
| Road_Type | Categorical | Highway / Arterial / Residential / Collector |
| Number_of_Lanes | Integer | Road lane count (1-8) |
| Traffic_Signals | Integer | Number of traffic signals nearby (0-5) |
| Large_Vehicles_Count | Integer | Count of large vehicles (0-40) |
| Temperature | Float | Temperature in Celsius (-5 to 45) |
| Humidity | Float | Humidity percentage (10-100) |
| Rainfall | Float | Rainfall in mm (0-50) |
| Weather_Conditions | Categorical | Clear / Cloudy / Rain / Snow / Fog |
| Nearby_Landmarks | Categorical | Nearest landmark type |
| Event_Indicator | Binary | Whether a special event is occurring |
| **Traffic_Demand** | **Integer** | **TARGET: vehicles per hour** |

### Engineered Features (5 mandatory)
| Feature | Formula |
|---------|---------|
| Peak_Hour_Flag | 1 if hour in {7,8,9,17,18,19}, else 0 |
| Weekend_Flag | 1 if day in {Saturday, Sunday}, else 0 |
| Traffic_Density_Score | 0.4*norm(lanes) + 0.3*norm(signals) + 0.3*norm(large_vehicles) |
| Weather_Impact_Score | 0.25*norm(temp_dev) + 0.25*norm(humidity) + 0.35*norm(rainfall) + 0.15*weather_severity |
| Rush_Hour_Indicator | Encoded: off_peak=0, morning_rush=1, evening_rush=2 |

---

## Streamlit Application

### User Inputs
1. **Location** — Dropdown of city zones
2. **Road Type** — Highway / Arterial / Residential / Collector
3. **Weather** — Clear / Cloudy / Rain / Snow / Fog
4. **Date** — Calendar picker (DD/MM/YYYY)
5. **Time** — Time picker (HH:MM)

### Prediction Outputs
1. **Predicted Traffic Demand** — Vehicles/hour metric
2. **Congestion Level** — Low/Medium/High with percentage
3. **Peak Traffic Alert** — Yes/No with time horizon
4. **24-Hour Forecast Chart** — Interactive Plotly line chart

### Live App
> **Streamlit Cloud URL:** *[To be deployed]*

---

## Demo Video
> **Video Link:** *[To be recorded - 5-10 minutes]*

---

## Key Technologies

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-Learn, XGBoost, LightGBM |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web Application | Streamlit |
| Deployment | Streamlit Cloud |
| Serialization | joblib |

---

## Deployment to Streamlit Cloud

1. Push repository to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set main file path: `app/streamlit_app.py`
6. Click **Deploy**

---

## License

This project is built for educational and demonstration purposes for Innovexa Catalyst.

---

*Built with precision by Innovexa Catalyst*
