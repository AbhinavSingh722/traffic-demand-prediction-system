"""
test_feature_engineering.py
============================
Unit tests for the feature engineering module.
Validates all 5 mandatory engineered features produce correct outputs.

Author: Innovexa Catalyst
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    create_peak_hour_flag,
    create_weekend_flag,
    create_traffic_density_score,
    create_weather_impact_score,
    create_rush_hour_indicator,
    add_features_for_inference,
    PEAK_HOURS,
    MORNING_RUSH,
    EVENING_RUSH,
    WEEKEND_DAYS,
    WEATHER_SEVERITY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "Hour": [8, 12, 18, 3, 7],
        "Day_of_Week": ["Monday", "Saturday", "Friday", "Sunday", "Wednesday"],
        "Number_of_Lanes": [6, 2, 4, 2, 8],
        "Traffic_Signals": [0, 1, 3, 1, 0],
        "Large_Vehicles_Count": [15, 1, 8, 2, 20],
        "Temperature": [25.0, 10.0, 30.0, -2.0, 22.0],
        "Humidity": [45.0, 80.0, 60.0, 90.0, 50.0],
        "Rainfall": [0.0, 5.0, 10.0, 3.0, 0.0],
        "Weather_Conditions": ["Clear", "Rain", "Cloudy", "Snow", "Clear"],
    })


# ---------------------------------------------------------------------------
# Test Peak Hour Flag
# ---------------------------------------------------------------------------
class TestPeakHourFlag:
    def test_peak_hours_flagged(self, sample_df):
        result = create_peak_hour_flag(sample_df)
        # Hours 8, 18, 7 are peak; 12, 3 are not
        assert result.iloc[0] == 1  # 8 AM
        assert result.iloc[2] == 1  # 6 PM
        assert result.iloc[4] == 1  # 7 AM

    def test_off_peak_hours_not_flagged(self, sample_df):
        result = create_peak_hour_flag(sample_df)
        assert result.iloc[1] == 0  # 12 PM
        assert result.iloc[3] == 0  # 3 AM

    def test_output_is_binary(self, sample_df):
        result = create_peak_hour_flag(sample_df)
        assert set(result.unique()).issubset({0, 1})

    def test_all_24_hours(self):
        df = pd.DataFrame({"Hour": list(range(24))})
        result = create_peak_hour_flag(df)
        peak_count = result.sum()
        assert peak_count == len(PEAK_HOURS)


# ---------------------------------------------------------------------------
# Test Weekend Flag
# ---------------------------------------------------------------------------
class TestWeekendFlag:
    def test_weekend_flagged(self, sample_df):
        result = create_weekend_flag(sample_df)
        assert result.iloc[1] == 1  # Saturday
        assert result.iloc[3] == 1  # Sunday

    def test_weekday_not_flagged(self, sample_df):
        result = create_weekend_flag(sample_df)
        assert result.iloc[0] == 0  # Monday
        assert result.iloc[2] == 0  # Friday

    def test_output_is_binary(self, sample_df):
        result = create_weekend_flag(sample_df)
        assert set(result.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Test Traffic Density Score
# ---------------------------------------------------------------------------
class TestTrafficDensityScore:
    def test_score_in_range(self, sample_df):
        result = create_traffic_density_score(sample_df)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_higher_infra_gives_higher_score(self):
        df = pd.DataFrame({
            "Number_of_Lanes": [2, 8],
            "Traffic_Signals": [0, 5],
            "Large_Vehicles_Count": [0, 20],
        })
        result = create_traffic_density_score(df)
        assert result.iloc[1] > result.iloc[0]

    def test_identical_values_give_uniform_score(self):
        df = pd.DataFrame({
            "Number_of_Lanes": [4, 4, 4],
            "Traffic_Signals": [2, 2, 2],
            "Large_Vehicles_Count": [5, 5, 5],
        })
        result = create_traffic_density_score(df)
        # All the same inputs -> all same score (0.5 when min==max)
        assert len(result.unique()) == 1


# ---------------------------------------------------------------------------
# Test Weather Impact Score
# ---------------------------------------------------------------------------
class TestWeatherImpactScore:
    def test_score_in_range(self, sample_df):
        result = create_weather_impact_score(sample_df)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_snow_higher_than_clear(self):
        df = pd.DataFrame({
            "Temperature": [25.0, -2.0],
            "Humidity": [40.0, 90.0],
            "Rainfall": [0.0, 5.0],
            "Weather_Conditions": ["Clear", "Snow"],
        })
        result = create_weather_impact_score(df)
        assert result.iloc[1] > result.iloc[0]


# ---------------------------------------------------------------------------
# Test Rush Hour Indicator
# ---------------------------------------------------------------------------
class TestRushHourIndicator:
    def test_morning_rush(self, sample_df):
        result = create_rush_hour_indicator(sample_df)
        assert result.iloc[0] == "morning_rush"  # 8 AM
        assert result.iloc[4] == "morning_rush"  # 7 AM

    def test_evening_rush(self, sample_df):
        result = create_rush_hour_indicator(sample_df)
        assert result.iloc[2] == "evening_rush"  # 6 PM

    def test_off_peak(self, sample_df):
        result = create_rush_hour_indicator(sample_df)
        assert result.iloc[1] == "off_peak"   # 12 PM
        assert result.iloc[3] == "off_peak"   # 3 AM

    def test_only_three_categories(self, sample_df):
        result = create_rush_hour_indicator(sample_df)
        assert set(result.unique()).issubset({"morning_rush", "evening_rush", "off_peak"})


# ---------------------------------------------------------------------------
# Test Inference Feature Adder
# ---------------------------------------------------------------------------
class TestInferenceFeatures:
    def test_all_five_features_added(self):
        input_data = {
            "Hour": 8,
            "Day_of_Week": "Monday",
            "Number_of_Lanes": 6,
            "Traffic_Signals": 2,
            "Large_Vehicles_Count": 10,
            "Temperature": 25.0,
            "Humidity": 50.0,
            "Rainfall": 0.0,
            "Weather_Conditions": "Clear",
        }
        result = add_features_for_inference(input_data)

        assert "Peak_Hour_Flag" in result
        assert "Weekend_Flag" in result
        assert "Traffic_Density_Score" in result
        assert "Weather_Impact_Score" in result
        assert "Rush_Hour_Indicator" in result

    def test_peak_hour_correct(self):
        for hour in PEAK_HOURS:
            data = {"Hour": hour, "Day_of_Week": "Monday",
                    "Number_of_Lanes": 2, "Traffic_Signals": 1,
                    "Large_Vehicles_Count": 5, "Temperature": 22,
                    "Humidity": 50, "Rainfall": 0,
                    "Weather_Conditions": "Clear"}
            result = add_features_for_inference(data)
            assert result["Peak_Hour_Flag"] == 1

    def test_weekend_flag_correct(self):
        data = {"Hour": 12, "Day_of_Week": "Saturday",
                "Number_of_Lanes": 2, "Traffic_Signals": 1,
                "Large_Vehicles_Count": 5, "Temperature": 22,
                "Humidity": 50, "Rainfall": 0,
                "Weather_Conditions": "Clear"}
        result = add_features_for_inference(data)
        assert result["Weekend_Flag"] == 1

    def test_density_score_type(self):
        data = {"Hour": 12, "Day_of_Week": "Monday",
                "Number_of_Lanes": 4, "Traffic_Signals": 2,
                "Large_Vehicles_Count": 8, "Temperature": 22,
                "Humidity": 50, "Rainfall": 0,
                "Weather_Conditions": "Clear"}
        result = add_features_for_inference(data)
        assert isinstance(result["Traffic_Density_Score"], float)
        assert 0 <= result["Traffic_Density_Score"] <= 1
"""
test_ensemble.py
=================
Unit tests for the ensemble prediction module.

Author: Innovexa Catalyst
"""


class TestEnsemblePredict:
    def test_weight_sum_is_one(self):
        from src.ensemble import LGBM_WEIGHT, XGB_WEIGHT
        assert abs(LGBM_WEIGHT + XGB_WEIGHT - 1.0) < 1e-9

    def test_weights_are_correct(self):
        from src.ensemble import LGBM_WEIGHT, XGB_WEIGHT
        assert LGBM_WEIGHT == 0.55
        assert XGB_WEIGHT == 0.45

    def test_ensemble_predict_formula(self):
        from src.ensemble import ensemble_predict

        class MockModel:
            def __init__(self, constant):
                self.constant = constant
            def predict(self, X):
                return np.full(len(X), self.constant)

        lgbm = MockModel(100.0)
        xgb = MockModel(200.0)
        X = pd.DataFrame({"a": [1, 2, 3]})

        result = ensemble_predict(lgbm, xgb, X)
        expected = 0.55 * 100.0 + 0.45 * 200.0  # = 145.0
        np.testing.assert_allclose(result, expected)

    def test_ensemble_output_shape(self):
        from src.ensemble import ensemble_predict

        class MockModel:
            def predict(self, X):
                return np.ones(len(X))

        X = pd.DataFrame({"a": range(10)})
        result = ensemble_predict(MockModel(), MockModel(), X)
        assert len(result) == 10
