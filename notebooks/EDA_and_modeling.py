"""
EDA_and_modeling.py
====================
Complete pipeline script for the Traffic Demand Prediction System.
Executes all phases: dataset generation, EDA, preprocessing, feature engineering,
model training, hyperparameter tuning, evaluation, ensemble learning, and serialization.

Usage:
    python notebooks/EDA_and_modeling.py

Author: Innovexa Catalyst
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from pathlib import Path

from src.generate_dataset import generate_dataset, save_dataset
from src.data_preprocessing import (
    load_data,
    handle_missing_values,
    remove_duplicates,
    encode_categoricals,
    scale_features,
    split_data,
    save_preprocessor_artifacts,
    CATEGORICAL_COLS,
    DROP_COLS,
    TARGET_COL,
)
from src.feature_engineering import add_all_features
from src.model_training import (
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    save_model,
)
from src.ensemble import ensemble_predict, save_ensemble_config
from src.evaluate import (
    evaluate_model,
    print_evaluation_report,
    compare_models,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_demand_distribution,
    plot_hourly_traffic,
    plot_weather_impact,
    plot_correlation_heatmap,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(PROJECT_ROOT) / "data"
MODEL_DIR = Path(PROJECT_ROOT) / "models"
REPORTS_DIR = Path(PROJECT_ROOT) / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

DATASET_PATH = DATA_DIR / "smart_city_traffic.csv"

# Create directories
for d in [DATA_DIR, MODEL_DIR, REPORTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def phase1_generate_data():
    """Phase 1: Generate synthetic dataset."""
    print("\n" + "#" * 70)
    print("#  PHASE 1: DATASET GENERATION")
    print("#" * 70)

    if DATASET_PATH.exists():
        print(f"[INFO] Dataset already exists at {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH, parse_dates=["Timestamp"])
    else:
        df = generate_dataset()
        save_dataset(df, output_dir=DATA_DIR)

    print(f"\n[SUMMARY] Dataset shape: {df.shape}")
    print(f"[SUMMARY] Columns: {list(df.columns)}")
    print(f"\n[SUMMARY] Data Types:")
    print(df.dtypes)
    print(f"\n[SUMMARY] Missing Values:")
    print(df.isnull().sum())
    return df


def phase2_eda(df):
    """Phase 2: Exploratory Data Analysis with all 6 required visualizations."""
    print("\n" + "#" * 70)
    print("#  PHASE 2: EXPLORATORY DATA ANALYSIS")
    print("#" * 70)

    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    print("\n--- Value Distributions (Categorical) ---")
    for col in ["Road_Type", "Weather_Conditions", "Day_of_Week", "Nearby_Landmarks"]:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts())

    print("\n--- Target Variable Statistics ---")
    print(f"  Mean:   {df[TARGET_COL].mean():.2f}")
    print(f"  Median: {df[TARGET_COL].median():.2f}")
    print(f"  Std:    {df[TARGET_COL].std():.2f}")
    print(f"  Min:    {df[TARGET_COL].min():.2f}")
    print(f"  Max:    {df[TARGET_COL].max():.2f}")

    # --- All 6 Required Visualizations ---
    print("\n--- Generating Visualizations ---")

    # 1. Traffic Demand Distribution
    plot_demand_distribution(df, save_path=PLOTS_DIR / "01_demand_distribution.png")
    print("  [OK] Traffic Demand Distribution")

    # 2. Hourly Traffic Analysis
    plot_hourly_traffic(df, save_path=PLOTS_DIR / "02_hourly_traffic.png")
    print("  [OK] Hourly Traffic Analysis")

    # 3. Weather Impact Analysis
    plot_weather_impact(df, save_path=PLOTS_DIR / "03_weather_impact.png")
    print("  [OK] Weather Impact Analysis")

    # 4. Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    plot_correlation_heatmap(numeric_df, save_path=PLOTS_DIR / "04_correlation_heatmap.png")
    print("  [OK] Correlation Heatmap")

    # Outlier detection summary
    print("\n--- Outlier Analysis (IQR Method) ---")
    for col in ["Traffic_Demand", "Temperature", "Humidity", "Rainfall",
                 "Large_Vehicles_Count"]:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            print(f"  {col}: {outliers} outliers ({outliers/len(df)*100:.2f}%)")


def phase3_preprocess(df):
    """Phase 3: Data Preprocessing."""
    print("\n" + "#" * 70)
    print("#  PHASE 3: DATA PREPROCESSING")
    print("#" * 70)

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    return df


def phase4_feature_engineering(df):
    """Phase 4: Feature Engineering - all 5 mandatory features."""
    print("\n" + "#" * 70)
    print("#  PHASE 4: FEATURE ENGINEERING")
    print("#" * 70)

    df = add_all_features(df, encode_rush_hour=True)

    # Validate all 5 features exist
    required_features = [
        "Peak_Hour_Flag", "Weekend_Flag", "Traffic_Density_Score",
        "Weather_Impact_Score", "Rush_Hour_Indicator",
    ]
    for feat in required_features:
        assert feat in df.columns, f"Missing engineered feature: {feat}"
        print(f"  [OK] {feat} - range: [{df[feat].min():.4f}, {df[feat].max():.4f}]")

    return df


def phase5_encode_and_split(df):
    """Phase 5: Encode, scale, and split the data."""
    print("\n" + "#" * 70)
    print("#  PHASE 5: ENCODE, SCALE, SPLIT")
    print("#" * 70)

    # Drop Timestamp before encoding
    df_model = df.drop(columns=DROP_COLS, errors="ignore")

    df_model, encoders = encode_categoricals(df_model, fit=True)
    df_model, scaler = scale_features(df_model, fit=True)

    save_preprocessor_artifacts(encoders, scaler, output_dir=MODEL_DIR)

    X_train, X_test, y_train, y_test = split_data(df_model)

    # Save feature names for later use
    feature_names = [c for c in X_train.columns]
    print(f"[INFO] Feature count: {len(feature_names)}")
    print(f"[INFO] Features: {feature_names}")

    return X_train, X_test, y_train, y_test, encoders, scaler, feature_names


def phase6_train_models(X_train, y_train):
    """Phase 6: Train all 3 models with hyperparameter tuning."""
    print("\n" + "#" * 70)
    print("#  PHASE 6: MODEL TRAINING & TUNING")
    print("#" * 70)

    # Train Random Forest
    rf_model, rf_params = train_random_forest(X_train, y_train, tune=True, n_iter=30)
    save_model(rf_model, MODEL_DIR / "random_forest.pkl")

    # Train XGBoost
    xgb_model, xgb_params = train_xgboost(X_train, y_train, tune=True, n_iter=40)
    save_model(xgb_model, MODEL_DIR / "xgboost.pkl")

    # Train LightGBM
    lgbm_model, lgbm_params = train_lightgbm(X_train, y_train, tune=True, n_iter=40)
    save_model(lgbm_model, MODEL_DIR / "lightgbm.pkl")

    return rf_model, xgb_model, lgbm_model


def phase7_evaluate(rf_model, xgb_model, lgbm_model, X_train, X_test,
                     y_train, y_test, feature_names):
    """Phase 7: Evaluate all models and ensemble."""
    print("\n" + "#" * 70)
    print("#  PHASE 7: EVALUATION")
    print("#" * 70)

    all_results = []

    # Random Forest evaluation
    rf_pred = rf_model.predict(X_test)
    rf_results = evaluate_model(y_test, rf_pred, "Random Forest")
    print_evaluation_report(rf_results)
    all_results.append(rf_results)

    # XGBoost evaluation
    xgb_pred = xgb_model.predict(X_test)
    xgb_results = evaluate_model(y_test, xgb_pred, "XGBoost")
    print_evaluation_report(xgb_results)
    all_results.append(xgb_results)

    # LightGBM evaluation
    lgbm_pred = lgbm_model.predict(X_test)
    lgbm_results = evaluate_model(y_test, lgbm_pred, "LightGBM")
    print_evaluation_report(lgbm_results)
    all_results.append(lgbm_results)

    # Ensemble evaluation
    print("\n" + "=" * 60)
    print("  ENSEMBLE: 55% LightGBM + 45% XGBoost")
    print("=" * 60)
    ens_pred = ensemble_predict(lgbm_model, xgb_model, X_test)
    ens_results = evaluate_model(y_test, ens_pred, "Ensemble (55% LGBM + 45% XGB)")
    print_evaluation_report(ens_results)
    all_results.append(ens_results)

    # Save ensemble config
    save_ensemble_config(output_dir=MODEL_DIR)

    # Comparison table
    comparison_df = compare_models(all_results)

    # --- Remaining Visualizations ---

    # 5. Feature Importance (LightGBM)
    plot_feature_importance(
        lgbm_model, feature_names,
        title="LightGBM Feature Importance",
        save_path=PLOTS_DIR / "05_feature_importance_lgbm.png",
    )
    print("  [OK] Feature Importance (LightGBM)")

    # 5b. Feature Importance (XGBoost)
    plot_feature_importance(
        xgb_model, feature_names,
        title="XGBoost Feature Importance",
        save_path=PLOTS_DIR / "05b_feature_importance_xgb.png",
    )
    print("  [OK] Feature Importance (XGBoost)")

    # 6. Actual vs Predicted (Ensemble)
    plot_actual_vs_predicted(
        y_test, ens_pred,
        title="Ensemble: Actual vs Predicted Traffic Demand",
        save_path=PLOTS_DIR / "06_actual_vs_predicted_ensemble.png",
    )
    print("  [OK] Actual vs Predicted (Ensemble)")

    # Additional: Actual vs Predicted for individual models
    plot_actual_vs_predicted(
        y_test, lgbm_pred,
        title="LightGBM: Actual vs Predicted",
        save_path=PLOTS_DIR / "06b_actual_vs_predicted_lgbm.png",
    )

    # --- Validate R² Thresholds ---
    print("\n" + "=" * 70)
    print("  R2 THRESHOLD VALIDATION")
    print("=" * 70)

    thresholds = {
        "Random Forest": 0.85,
        "XGBoost": 0.93,
        "LightGBM": 0.95,
        "Ensemble (55% LGBM + 45% XGB)": 0.96,
    }

    all_passed = True
    for res in all_results:
        name = res["Model"]
        r2 = res["R2"]
        threshold = thresholds.get(name, 0)
        passed = r2 >= threshold
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}: R2={r2:.6f} (threshold: {threshold:.2f})")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  ALL MODELS MEET THEIR R2 THRESHOLDS!")
    else:
        print("\n  WARNING: Some models did not meet thresholds. Review tuning.")

    print("=" * 70)
    return all_results, comparison_df


def phase8_end_to_end_test(feature_names):
    """Phase 8: End-to-end prediction test."""
    print("\n" + "#" * 70)
    print("#  PHASE 8: END-TO-END PREDICTION TEST")
    print("#" * 70)

    import joblib
    from src.model_training import load_model
    from src.ensemble import ensemble_predict as ens_pred_func

    # Load models from disk
    rf = load_model(MODEL_DIR / "random_forest.pkl")
    xgb_m = load_model(MODEL_DIR / "xgboost.pkl")
    lgbm = load_model(MODEL_DIR / "lightgbm.pkl")
    encoders, scaler = joblib.load(MODEL_DIR / "label_encoders.pkl"), \
                        joblib.load(MODEL_DIR / "scaler.pkl")

    # Create a sample input
    sample = {
        "Geohash_Location": "tdr1x0",
        "Day_of_Week": "Monday",
        "Hour": 8,
        "Road_Type": "Highway",
        "Number_of_Lanes": 6,
        "Traffic_Signals": 1,
        "Large_Vehicles_Count": 12,
        "Temperature": 25.0,
        "Humidity": 55.0,
        "Rainfall": 0.0,
        "Weather_Conditions": "Clear",
        "Nearby_Landmarks": "Office Complex",
        "Event_Indicator": 0,
    }

    # Add engineered features
    from src.feature_engineering import add_features_for_inference
    sample = add_features_for_inference(sample)

    # Create DataFrame in feature order
    sample_df = pd.DataFrame([sample])
    sample_df = sample_df[feature_names]

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in sample_df.columns and col in encoders:
            le = encoders[col]
            known = set(le.classes_)
            sample_df[col] = sample_df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            sample_df[col] = le.transform(sample_df[col])

    # Scale numerics
    from src.data_preprocessing import NUMERICAL_COLS
    cols_to_scale = [c for c in NUMERICAL_COLS if c in sample_df.columns]
    sample_df[cols_to_scale] = scaler.transform(sample_df[cols_to_scale])

    # Predict
    rf_p = rf.predict(sample_df)[0]
    xgb_p = xgb_m.predict(sample_df)[0]
    lgbm_p = lgbm.predict(sample_df)[0]
    ens_p = ens_pred_func(lgbm, xgb_m, sample_df)[0]

    print(f"\n  Sample Input: Monday, 8 AM, Highway, Clear Weather")
    print(f"  Random Forest:  {rf_p:.1f} vehicles/hour")
    print(f"  XGBoost:        {xgb_p:.1f} vehicles/hour")
    print(f"  LightGBM:       {lgbm_p:.1f} vehicles/hour")
    print(f"  Ensemble:       {ens_p:.1f} vehicles/hour")
    print(f"\n  [OK] End-to-end prediction test PASSED!")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    """Run the complete ML pipeline."""
    print("\n" + "*" * 70)
    print("*  TRAFFIC DEMAND PREDICTION SYSTEM - FULL PIPELINE")
    print("*  Innovexa Catalyst")
    print("*" * 70)

    # Phase 1: Generate / Load Data
    df = phase1_generate_data()

    # Phase 2: EDA
    phase2_eda(df)

    # Phase 3: Preprocess
    df = phase3_preprocess(df)

    # Phase 4: Feature Engineering
    df = phase4_feature_engineering(df)

    # Phase 5: Encode, Scale, Split
    X_train, X_test, y_train, y_test, encoders, scaler, feature_names = \
        phase5_encode_and_split(df)

    # Phase 6: Train Models
    rf_model, xgb_model, lgbm_model = phase6_train_models(X_train, y_train)

    # Phase 7: Evaluate
    all_results, comparison_df = phase7_evaluate(
        rf_model, xgb_model, lgbm_model,
        X_train, X_test, y_train, y_test, feature_names,
    )

    # Save comparison table
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv")
    print(f"[INFO] Model comparison saved to {REPORTS_DIR / 'model_comparison.csv'}")

    # Save feature names for Streamlit
    import json
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    print(f"[INFO] Feature names saved to {MODEL_DIR / 'feature_names.json'}")

    # Save training data stats for congestion level mapping
    stats = {
        "demand_mean": float(y_train.mean()),
        "demand_std": float(y_train.std()),
        "demand_q25": float(y_train.quantile(0.25)),
        "demand_q50": float(y_train.quantile(0.50)),
        "demand_q75": float(y_train.quantile(0.75)),
        "demand_min": float(y_train.min()),
        "demand_max": float(y_train.max()),
    }
    with open(MODEL_DIR / "demand_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Demand stats saved to {MODEL_DIR / 'demand_stats.json'}")

    # Phase 8: End-to-End Test
    phase8_end_to_end_test(feature_names)

    print("\n" + "*" * 70)
    print("*  PIPELINE COMPLETE - ALL PHASES EXECUTED SUCCESSFULLY")
    print("*" * 70)


if __name__ == "__main__":
    main()
