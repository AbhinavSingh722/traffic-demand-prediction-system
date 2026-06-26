"""
generate_dataset.py
====================
Generates a realistic synthetic Smart City Traffic Dataset with 120,000+ records
and 15+ features. Uses statistical distributions to model real-world traffic
patterns including rush-hour peaks, weather effects, and weekday/weekend differences.

Author: Innovexa Catalyst
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
NUM_RECORDS = 125_000  # 120,000+ as required

# Geohash locations representing different city zones
GEOHASH_LOCATIONS = [
    "tdr1x0", "tdr1x1", "tdr1x2", "tdr1x3", "tdr1x4",
    "tdr1y0", "tdr1y1", "tdr1y2", "tdr1y3", "tdr1y4",
    "tdr1z0", "tdr1z1", "tdr1z2", "tdr1z3", "tdr1z4",
    "tdr2a0", "tdr2a1", "tdr2a2", "tdr2a3", "tdr2a4",
]

ROAD_TYPES = ["Highway", "Arterial", "Residential", "Collector"]
ROAD_TYPE_WEIGHTS = [0.25, 0.30, 0.25, 0.20]  # distribution weights

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Snow", "Fog"]
WEATHER_WEIGHTS = [0.35, 0.25, 0.20, 0.10, 0.10]

DAYS_OF_WEEK = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

NEARBY_LANDMARKS = [
    "Shopping Mall", "Hospital", "School", "Park",
    "Stadium", "Airport", "Train Station", "Office Complex",
    "University", "None",
]

# Base demand by road type (vehicles/hour)
ROAD_BASE_DEMAND = {
    "Highway": 350,
    "Arterial": 250,
    "Residential": 80,
    "Collector": 150,
}

# Weather impact multipliers on demand
WEATHER_DEMAND_MULTIPLIER = {
    "Clear": 1.0,
    "Cloudy": 0.95,
    "Rain": 0.82,
    "Snow": 0.70,
    "Fog": 0.78,
}

# Hourly demand multiplier (captures rush-hour peaks)
HOURLY_MULTIPLIER = {
    0: 0.15, 1: 0.10, 2: 0.08, 3: 0.07, 4: 0.09, 5: 0.20,
    6: 0.50, 7: 0.85, 8: 1.00, 9: 0.90, 10: 0.70, 11: 0.75,
    12: 0.80, 13: 0.75, 14: 0.70, 15: 0.72, 16: 0.80, 17: 0.95,
    18: 1.00, 19: 0.85, 20: 0.60, 21: 0.45, 22: 0.30, 23: 0.20,
}


def generate_dataset(num_records=NUM_RECORDS, random_state=RANDOM_STATE):
    """
    Generate a synthetic Smart City Traffic Dataset.

    Parameters
    ----------
    num_records : int
        Number of records to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 15+ features and the target variable Traffic_Demand.
    """
    rng = np.random.RandomState(random_state)

    # ----- Core categorical features -----
    geohash_location = rng.choice(GEOHASH_LOCATIONS, size=num_records)
    day_of_week = rng.choice(DAYS_OF_WEEK, size=num_records)
    hour = rng.randint(0, 24, size=num_records)

    road_type = rng.choice(
        ROAD_TYPES, size=num_records, p=ROAD_TYPE_WEIGHTS
    )

    weather_conditions = rng.choice(
        WEATHER_CONDITIONS, size=num_records, p=WEATHER_WEIGHTS
    )

    nearby_landmarks = rng.choice(NEARBY_LANDMARKS, size=num_records)

    # ----- Numeric features -----
    number_of_lanes = np.where(
        np.isin(road_type, ["Highway"]),
        rng.choice([4, 6, 8], size=num_records, p=[0.3, 0.5, 0.2]),
        np.where(
            np.isin(road_type, ["Arterial"]),
            rng.choice([2, 4, 6], size=num_records, p=[0.2, 0.5, 0.3]),
            np.where(
                np.isin(road_type, ["Collector"]),
                rng.choice([2, 4], size=num_records, p=[0.6, 0.4]),
                rng.choice([1, 2], size=num_records, p=[0.5, 0.5]),
            ),
        ),
    )

    traffic_signals = np.where(
        np.isin(road_type, ["Highway"]),
        rng.choice([0, 1], size=num_records, p=[0.8, 0.2]),
        np.where(
            np.isin(road_type, ["Arterial"]),
            rng.choice([2, 3, 4, 5], size=num_records),
            np.where(
                np.isin(road_type, ["Collector"]),
                rng.choice([1, 2, 3], size=num_records),
                rng.choice([0, 1, 2], size=num_records),
            ),
        ),
    )

    large_vehicles_count = np.clip(
        rng.poisson(
            lam=np.where(
                np.isin(road_type, ["Highway"]), 15,
                np.where(np.isin(road_type, ["Arterial"]), 8,
                         np.where(np.isin(road_type, ["Collector"]), 4, 1))
            ),
            size=num_records,
        ),
        0, 40,
    )

    # Weather-related numerics
    temperature = np.clip(
        rng.normal(loc=22, scale=10, size=num_records), -5, 45
    ).round(1)

    humidity = np.clip(
        np.where(
            np.isin(weather_conditions, ["Rain", "Snow", "Fog"]),
            rng.normal(loc=80, scale=10, size=num_records),
            rng.normal(loc=50, scale=15, size=num_records),
        ),
        10, 100,
    ).round(1)

    rainfall = np.where(
        np.isin(weather_conditions, ["Rain"]),
        np.clip(rng.exponential(scale=5, size=num_records), 0.1, 50),
        np.where(
            np.isin(weather_conditions, ["Snow"]),
            np.clip(rng.exponential(scale=2, size=num_records), 0.1, 20),
            0.0,
        ),
    ).round(2)

    event_indicator = rng.choice(
        [0, 1], size=num_records, p=[0.85, 0.15]
    )

    # Generate timestamps spanning 1 year
    start_date = pd.Timestamp("2024-01-01")
    timestamps = pd.date_range(
        start=start_date, periods=num_records, freq="4min"
    )
    # Shuffle to break sequential ordering
    timestamp_idx = rng.permutation(num_records)
    timestamps = timestamps[timestamp_idx]

    # ----- Target variable: Traffic_Demand -----
    base_demand = np.array([ROAD_BASE_DEMAND[rt] for rt in road_type], dtype=float)

    # Hourly multiplier
    hour_mult = np.array([HOURLY_MULTIPLIER[h] for h in hour], dtype=float)

    # Weather multiplier
    weather_mult = np.array(
        [WEATHER_DEMAND_MULTIPLIER[w] for w in weather_conditions], dtype=float
    )

    # Weekend reduction
    is_weekend = np.isin(day_of_week, ["Saturday", "Sunday"])
    weekend_mult = np.where(is_weekend, 0.65, 1.0)

    # Lane effect (more lanes → higher capacity → more demand)
    lane_mult = 1 + 0.05 * (number_of_lanes - 2)

    # Event boost
    event_mult = np.where(event_indicator == 1, 1.0 + rng.uniform(0.15, 0.30, size=num_records), 1.0)

    # Large vehicle effect
    large_veh_effect = 1 + 0.005 * large_vehicles_count

    # Temperature comfort effect: extreme temps reduce demand slightly
    temp_deviation = np.abs(temperature - 22)
    temp_mult = 1 - 0.003 * temp_deviation

    # Landmark effect
    landmark_mult = np.where(
        np.isin(nearby_landmarks, ["Stadium", "Shopping Mall", "Airport"]),
        1.15,
        np.where(
            np.isin(nearby_landmarks, ["Hospital", "Train Station", "Office Complex", "University"]),
            1.08,
            np.where(
                np.isin(nearby_landmarks, ["School", "Park"]),
                1.03,
                1.0,
            ),
        ),
    )

    # Combine all effects
    traffic_demand = (
        base_demand
        * hour_mult
        * weather_mult
        * weekend_mult
        * lane_mult
        * event_mult
        * large_veh_effect
        * temp_mult
        * landmark_mult
    )

    # Add realistic noise (±10%)
    noise = rng.normal(loc=1.0, scale=0.08, size=num_records)
    traffic_demand = traffic_demand * noise

    # Ensure non-negative and round
    traffic_demand = np.clip(traffic_demand, 5, None).round(0).astype(int)

    # ----- Assemble DataFrame -----
    df = pd.DataFrame({
        "Geohash_Location": geohash_location,
        "Timestamp": timestamps,
        "Day_of_Week": day_of_week,
        "Hour": hour,
        "Road_Type": road_type,
        "Number_of_Lanes": number_of_lanes,
        "Traffic_Signals": traffic_signals,
        "Large_Vehicles_Count": large_vehicles_count,
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Weather_Conditions": weather_conditions,
        "Nearby_Landmarks": nearby_landmarks,
        "Event_Indicator": event_indicator,
        "Traffic_Demand": traffic_demand,
    })

    # Inject ~1% missing values in non-critical columns for realism
    missing_cols = ["Temperature", "Humidity", "Large_Vehicles_Count", "Traffic_Signals"]
    for col in missing_cols:
        mask = rng.random(num_records) < 0.01
        df.loc[mask, col] = np.nan

    return df


def save_dataset(df, output_dir="data"):
    """
    Save the generated dataset to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The generated dataset.
    output_dir : str
        Directory to save the CSV file.

    Returns
    -------
    str
        Path to the saved CSV file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "smart_city_traffic.csv"
    df.to_csv(filepath, index=False)
    print(f"[INFO] Dataset saved to {filepath}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return str(filepath)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    df = generate_dataset()
    save_dataset(df, output_dir=project_root / "data")
    print("\n[INFO] Dataset generation complete!")
    print(df.describe())
