"""
model_training.py
==================
Model training module for the Traffic Demand Prediction System.
Trains, tunes, and serializes Random Forest, XGBoost, and LightGBM regressors.
Uses RandomizedSearchCV for hyperparameter optimization.

Author: Innovexa Catalyst
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb


RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Hyperparameter Search Spaces
# ---------------------------------------------------------------------------
RF_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [10, 15, 20, 25, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
}

XGB_PARAM_DIST = {
    "n_estimators": [300, 500, 800, 1000],
    "max_depth": [4, 6, 8, 10, 12],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "min_child_weight": [1, 3, 5, 7],
}

LGBM_PARAM_DIST = {
    "n_estimators": [300, 500, 800, 1000, 1200],
    "max_depth": [6, 8, 10, 12, 15, -1],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [31, 50, 80, 100, 127],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "min_child_samples": [5, 10, 20, 30],
}


def tune_model(model, param_dist, X_train, y_train, n_iter=50, cv=3,
               scoring="r2", random_state=RANDOM_STATE):
    """
    Tune a model using RandomizedSearchCV.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible estimator.
    param_dist : dict
        Hyperparameter search space.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training target.
    n_iter : int
        Number of random parameter combinations to try.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric for optimization.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (best estimator, best parameters, best CV score)
    """
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    print(f"[TUNE] Best CV R2: {search.best_score_:.6f}")
    print(f"[TUNE] Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def train_random_forest(X_train, y_train, tune=True, n_iter=50):
    """
    Train and optionally tune a Random Forest Regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    tune : bool
        Whether to perform hyperparameter tuning.
    n_iter : int
        Number of tuning iterations.

    Returns
    -------
    tuple
        (fitted model, best params or None)
    """
    print("\n" + "=" * 60)
    print("  TRAINING: Random Forest Regressor")
    print("=" * 60)

    base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    if tune:
        model, params, score = tune_model(
            base_model, RF_PARAM_DIST, X_train, y_train, n_iter=n_iter
        )
        return model, params
    else:
        base_model.fit(X_train, y_train)
        return base_model, None


def train_xgboost(X_train, y_train, tune=True, n_iter=50):
    """
    Train and optionally tune an XGBoost Regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    tune : bool
        Whether to perform hyperparameter tuning.
    n_iter : int
        Number of tuning iterations.

    Returns
    -------
    tuple
        (fitted model, best params or None)
    """
    print("\n" + "=" * 60)
    print("  TRAINING: XGBoost Regressor")
    print("=" * 60)

    base_model = xgb.XGBRegressor(
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )

    if tune:
        model, params, score = tune_model(
            base_model, XGB_PARAM_DIST, X_train, y_train, n_iter=n_iter
        )
        return model, params
    else:
        base_model.fit(X_train, y_train)
        return base_model, None


def train_lightgbm(X_train, y_train, tune=True, n_iter=50):
    """
    Train and optionally tune a LightGBM Regressor.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    tune : bool
        Whether to perform hyperparameter tuning.
    n_iter : int
        Number of tuning iterations.

    Returns
    -------
    tuple
        (fitted model, best params or None)
    """
    print("\n" + "=" * 60)
    print("  TRAINING: LightGBM Regressor")
    print("=" * 60)

    base_model = lgb.LGBMRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    if tune:
        model, params, score = tune_model(
            base_model, LGBM_PARAM_DIST, X_train, y_train, n_iter=n_iter
        )
        return model, params
    else:
        base_model.fit(X_train, y_train)
        return base_model, None


def save_model(model, filepath):
    """
    Serialize a trained model to disk using joblib.

    Parameters
    ----------
    model : fitted estimator
        Trained model to serialize.
    filepath : str or Path
        Output file path (.pkl).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[INFO] Model saved to {filepath}")


def load_model(filepath):
    """
    Load a serialized model from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the .pkl file.

    Returns
    -------
    fitted estimator
        The loaded model.
    """
    model = joblib.load(filepath)
    print(f"[INFO] Model loaded from {filepath}")
    return model
