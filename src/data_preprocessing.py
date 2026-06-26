"""
data_preprocessing.py
======================
Data preprocessing module for the Traffic Demand Prediction System.
Handles missing values, encoding, scaling, duplicate removal, and train/test split.

Author: Innovexa Catalyst
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# Columns to encode (categorical)
CATEGORICAL_COLS = [
    "Geohash_Location",
    "Day_of_Week",
    "Road_Type",
    "Weather_Conditions",
    "Nearby_Landmarks",
]

# Columns to scale (numerical)
NUMERICAL_COLS = [
    "Hour",
    "Number_of_Lanes",
    "Traffic_Signals",
    "Large_Vehicles_Count",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Event_Indicator",
]

TARGET_COL = "Traffic_Demand"

# Columns to drop before modeling
DROP_COLS = ["Timestamp"]


def load_data(filepath):
    """
    Load dataset from a CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    df = pd.read_csv(filepath, parse_dates=["Timestamp"])
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    """
    Handle missing values using median imputation for numeric columns
    and mode imputation for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()

    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  [IMPUTE] {col}: filled with median = {median_val:.2f}")

    # Categorical columns: fill with mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  [IMPUTE] {col}: filled with mode = {mode_val}")

    missing_after = df.isnull().sum().sum()
    print(f"[INFO] Missing values: {missing_before} -> {missing_after}")
    return df


def remove_duplicates(df):
    """
    Remove exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[INFO] Duplicates removed: {before - after} rows")
    return df.reset_index(drop=True)


def encode_categoricals(df, encoders=None, fit=True):
    """
    Encode categorical columns using LabelEncoder.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    encoders : dict or None
        Pre-fitted encoders (for inference mode).
    fit : bool
        If True, fit new encoders. If False, use provided encoders.

    Returns
    -------
    tuple
        (encoded DataFrame, dict of fitted encoders)
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"  [ENCODE] {col}: {len(le.classes_)} classes")
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


def scale_features(df, scaler=None, fit=True):
    """
    Scale numerical features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    scaler : StandardScaler or None
        Pre-fitted scaler (for inference mode).
    fit : bool
        If True, fit a new scaler. If False, use provided scaler.

    Returns
    -------
    tuple
        (scaled DataFrame, fitted StandardScaler)
    """
    df = df.copy()
    cols_to_scale = [c for c in NUMERICAL_COLS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        print(f"[INFO] Scaled {len(cols_to_scale)} numerical columns")
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df, scaler


def split_data(df, target_col=TARGET_COL, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    target_col : str
        Name of the target column.
    test_size : float
        Fraction of data for the test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    feature_cols = [c for c in df.columns if c != target_col and c not in DROP_COLS]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_preprocessor_artifacts(encoders, scaler, output_dir="models"):
    """
    Save fitted encoders and scaler for use in the Streamlit app.

    Parameters
    ----------
    encoders : dict
        Dictionary of fitted LabelEncoders.
    scaler : StandardScaler
        Fitted StandardScaler.
    output_dir : str or Path
        Directory to save artifacts.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(encoders, output_path / "label_encoders.pkl")
    joblib.dump(scaler, output_path / "scaler.pkl")
    print(f"[INFO] Preprocessor artifacts saved to {output_path}")


def load_preprocessor_artifacts(model_dir="models"):
    """
    Load fitted encoders and scaler from disk.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the artifacts.

    Returns
    -------
    tuple
        (encoders dict, StandardScaler)
    """
    model_path = Path(model_dir)
    encoders = joblib.load(model_path / "label_encoders.pkl")
    scaler = joblib.load(model_path / "scaler.pkl")
    return encoders, scaler


def preprocess_pipeline(filepath, save_dir="models"):
    """
    Full preprocessing pipeline: load → clean → encode → scale → split.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw CSV.
    save_dir : str or Path
        Directory to save preprocessor artifacts.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, encoders, scaler, df_processed)
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_data(filepath)
    df = handle_missing_values(df)
    df = remove_duplicates(df)

    # Drop timestamp for modeling (features extracted separately)
    df_model = df.drop(columns=DROP_COLS, errors="ignore")

    df_model, encoders = encode_categoricals(df_model, fit=True)
    df_model, scaler = scale_features(df_model, fit=True)

    save_preprocessor_artifacts(encoders, scaler, save_dir)

    X_train, X_test, y_train, y_test = split_data(df_model)

    print("=" * 60)
    return X_train, X_test, y_train, y_test, encoders, scaler, df_model
