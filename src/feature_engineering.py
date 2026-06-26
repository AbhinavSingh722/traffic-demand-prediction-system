"""
feature_engineering.py
=======================
Feature engineering module for the Traffic Demand Prediction System.
Creates all 5 mandatory engineered features:
  1. Peak Hour Flag
  2. Weekend Flag
  3. Traffic Density Score
  4. Weather Impact Score
  5. Rush Hour Indicator

Author: Innovexa Catalyst
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PEAK_HOURS = [7, 8, 9, 17, 18, 19]
MORNING_RUSH = [7, 8, 9]
EVENING_RUSH = [17, 18, 19]
WEEKEND_DAYS = ["Saturday", "Sunday"]

# Weather severity mapping (used for weather_impact_score)
WEATHER_SEVERITY = {
    "Clear": 0.0,
    "Cloudy": 0.3,
    "Fog": 0.5,
    "Rain": 0.7,
    "Snow": 1.0,
}

# Comfort baseline temperature (°C)
COMFORT_TEMP = 22.0


def create_peak_hour_flag(df, hour_col="Hour"):
    """
    Create binary indicator for peak traffic hours.
    Peak hours: 7-9 AM and 5-7 PM.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    hour_col : str
        Name of the hour column.

    Returns
    -------
    pd.Series
        Binary series (1 = peak hour, 0 = off-peak).
    """
    return df[hour_col].apply(lambda x: 1 if x in PEAK_HOURS else 0)


def create_weekend_flag(df, day_col="Day_of_Week"):
    """
    Create binary indicator for weekend days.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    day_col : str
        Name of the day-of-week column.

    Returns
    -------
    pd.Series
        Binary series (1 = weekend, 0 = weekday).
    """
    return df[day_col].apply(lambda x: 1 if x in WEEKEND_DAYS else 0)


def create_traffic_density_score(df, lanes_col="Number_of_Lanes",
                                  signals_col="Traffic_Signals",
                                  large_veh_col="Large_Vehicles_Count"):
    """
    Create composite traffic density score from traffic-related columns.

    Formula:
        score = 0.4 × norm(lanes) + 0.3 × norm(signals) + 0.3 × norm(large_vehicles)

    All sub-features are min-max normalized to [0, 1] before weighting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    lanes_col, signals_col, large_veh_col : str
        Column names for the traffic sub-features.

    Returns
    -------
    pd.Series
        Traffic density score in [0, 1].
    """
    def _min_max_norm(series):
        """Min-max normalize a series to [0, 1]."""
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(0.5, index=series.index)
        return (series - s_min) / (s_max - s_min)

    norm_lanes = _min_max_norm(df[lanes_col].astype(float))
    norm_signals = _min_max_norm(df[signals_col].astype(float))
    norm_large_veh = _min_max_norm(df[large_veh_col].astype(float))

    score = 0.4 * norm_lanes + 0.3 * norm_signals + 0.3 * norm_large_veh
    return score.round(4)


def create_weather_impact_score(df, temp_col="Temperature",
                                 humidity_col="Humidity",
                                 rainfall_col="Rainfall",
                                 weather_col="Weather_Conditions"):
    """
    Create composite weather impact score from weather-related columns.

    Formula:
        score = 0.25 × norm(|temp - 22|)
              + 0.25 × norm(humidity)
              + 0.35 × norm(rainfall)
              + 0.15 × weather_severity_encoded

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    temp_col, humidity_col, rainfall_col, weather_col : str
        Column names for the weather sub-features.

    Returns
    -------
    pd.Series
        Weather impact score in [0, 1].
    """
    def _min_max_norm(series):
        """Min-max normalize a series to [0, 1]."""
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(0.5, index=series.index)
        return (series - s_min) / (s_max - s_min)

    temp_deviation = np.abs(df[temp_col].astype(float) - COMFORT_TEMP)
    norm_temp_dev = _min_max_norm(temp_deviation)
    norm_humidity = _min_max_norm(df[humidity_col].astype(float))
    norm_rainfall = _min_max_norm(df[rainfall_col].astype(float))

    # Map weather conditions to severity
    weather_severity = df[weather_col].map(WEATHER_SEVERITY).fillna(0.3)

    score = (
        0.25 * norm_temp_dev
        + 0.25 * norm_humidity
        + 0.35 * norm_rainfall
        + 0.15 * weather_severity
    )
    return score.round(4)


def create_rush_hour_indicator(df, hour_col="Hour"):
    """
    Create categorical rush hour indicator.
    Categories: 'morning_rush', 'evening_rush', 'off_peak'.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    hour_col : str
        Name of the hour column.

    Returns
    -------
    pd.Series
        Categorical series with rush hour labels.
    """
    def _classify(hour):
        if hour in MORNING_RUSH:
            return "morning_rush"
        elif hour in EVENING_RUSH:
            return "evening_rush"
        else:
            return "off_peak"

    return df[hour_col].apply(_classify)


def add_all_features(df, encode_rush_hour=True):
    """
    Add all 5 mandatory engineered features to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Must contain raw (non-encoded) columns for
        Day_of_Week and Weather_Conditions if those columns exist in
        original (string) form. If already encoded, pass pre-encoded data
        and set flags accordingly.
    encode_rush_hour : bool
        If True, label-encode the rush_hour_indicator to numeric.

    Returns
    -------
    pd.DataFrame
        DataFrame with 5 new feature columns added.
    """
    df = df.copy()

    print("[FEATURE] Creating Peak Hour Flag...")
    df["Peak_Hour_Flag"] = create_peak_hour_flag(df)

    print("[FEATURE] Creating Weekend Flag...")
    df["Weekend_Flag"] = create_weekend_flag(df)

    print("[FEATURE] Creating Traffic Density Score...")
    df["Traffic_Density_Score"] = create_traffic_density_score(df)

    print("[FEATURE] Creating Weather Impact Score...")
    df["Weather_Impact_Score"] = create_weather_impact_score(df)

    print("[FEATURE] Creating Rush Hour Indicator...")
    df["Rush_Hour_Indicator"] = create_rush_hour_indicator(df)

    if encode_rush_hour:
        # Encode rush hour indicator: off_peak=0, morning_rush=1, evening_rush=2
        rush_map = {"off_peak": 0, "morning_rush": 1, "evening_rush": 2}
        df["Rush_Hour_Indicator"] = df["Rush_Hour_Indicator"].map(rush_map)

    print(f"[INFO] Added 5 engineered features. New shape: {df.shape}")
    return df


def add_features_for_inference(input_data):
    """
    Add engineered features for a single inference record (Streamlit app).
    Expects a dict or single-row DataFrame with raw values.

    Parameters
    ----------
    input_data : dict
        Dictionary with keys: Hour, Day_of_Week, Number_of_Lanes,
        Traffic_Signals, Large_Vehicles_Count, Temperature, Humidity,
        Rainfall, Weather_Conditions.

    Returns
    -------
    dict
        Updated dictionary with 5 engineered feature values.
    """
    hour = input_data.get("Hour", 12)
    day = input_data.get("Day_of_Week", "Monday")

    # Peak Hour Flag
    input_data["Peak_Hour_Flag"] = 1 if hour in PEAK_HOURS else 0

    # Weekend Flag
    input_data["Weekend_Flag"] = 1 if day in WEEKEND_DAYS else 0

    # Rush Hour Indicator (encoded)
    if hour in MORNING_RUSH:
        input_data["Rush_Hour_Indicator"] = 1
    elif hour in EVENING_RUSH:
        input_data["Rush_Hour_Indicator"] = 2
    else:
        input_data["Rush_Hour_Indicator"] = 0

    # Traffic Density Score (simple normalization for single record)
    lanes = input_data.get("Number_of_Lanes", 2)
    signals = input_data.get("Traffic_Signals", 1)
    large_veh = input_data.get("Large_Vehicles_Count", 5)

    # Use approximate dataset ranges for normalization
    norm_lanes = min(max((lanes - 1) / 7, 0), 1)
    norm_signals = min(max(signals / 5, 0), 1)
    norm_large_veh = min(max(large_veh / 40, 0), 1)
    input_data["Traffic_Density_Score"] = round(
        0.4 * norm_lanes + 0.3 * norm_signals + 0.3 * norm_large_veh, 4
    )

    # Weather Impact Score
    temp = input_data.get("Temperature", 22)
    humidity = input_data.get("Humidity", 50)
    rainfall = input_data.get("Rainfall", 0)
    weather = input_data.get("Weather_Conditions", "Clear")

    temp_dev = abs(temp - COMFORT_TEMP)
    norm_temp_dev = min(temp_dev / 27, 1)  # max deviation ~27 from [-5, 45]
    norm_humidity = min(max((humidity - 10) / 90, 0), 1)
    norm_rainfall = min(rainfall / 50, 1)
    weather_sev = WEATHER_SEVERITY.get(weather, 0.3)

    input_data["Weather_Impact_Score"] = round(
        0.25 * norm_temp_dev + 0.25 * norm_humidity
        + 0.35 * norm_rainfall + 0.15 * weather_sev, 4
    )

    return input_data
