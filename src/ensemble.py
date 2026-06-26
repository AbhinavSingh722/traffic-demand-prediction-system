"""
ensemble.py
============
Ensemble learning module for the Traffic Demand Prediction System.
Implements a fixed-weight ensemble: 55% LightGBM + 45% XGBoost.

Author: Innovexa Catalyst
"""

import json
import numpy as np
from pathlib import Path


# Fixed ensemble weights as per specification
LGBM_WEIGHT = 0.55
XGB_WEIGHT = 0.45


def ensemble_predict(lgbm_model, xgb_model, X):
    """
    Generate ensemble predictions using weighted averaging.

    Formula:
        prediction = 0.55 × LightGBM_prediction + 0.45 × XGBoost_prediction

    Parameters
    ----------
    lgbm_model : fitted LGBMRegressor
        Trained LightGBM model.
    xgb_model : fitted XGBRegressor
        Trained XGBoost model.
    X : pd.DataFrame or np.ndarray
        Feature matrix for prediction.

    Returns
    -------
    np.ndarray
        Ensemble predictions.
    """
    lgbm_pred = lgbm_model.predict(X)
    xgb_pred = xgb_model.predict(X)

    ensemble_pred = LGBM_WEIGHT * lgbm_pred + XGB_WEIGHT * xgb_pred
    return ensemble_pred


def save_ensemble_config(output_dir="models"):
    """
    Save ensemble configuration as a JSON file.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save the config file.
    """
    config = {
        "ensemble_type": "weighted_average",
        "models": {
            "lgbm": {
                "weight": LGBM_WEIGHT,
                "file": "lightgbm.pkl",
            },
            "xgb": {
                "weight": XGB_WEIGHT,
                "file": "xgboost.pkl",
            },
        },
        "formula": f"prediction = {LGBM_WEIGHT} × LightGBM + {XGB_WEIGHT} × XGBoost",
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_path = output_path / "ensemble_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[INFO] Ensemble config saved to {config_path}")
    return config


def load_ensemble_config(model_dir="models"):
    """
    Load ensemble configuration from JSON.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the config file.

    Returns
    -------
    dict
        Ensemble configuration.
    """
    config_path = Path(model_dir) / "ensemble_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
