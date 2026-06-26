# Dataset Documentation: Smart City Traffic Dataset

## Overview

| Property | Value |
|----------|-------|
| Name | Smart City Traffic Dataset (Synthetic) |
| Records | 125,000 |
| Features | 15 (14 input + 1 target) |
| Engineered Features | 5 (added during pipeline) |
| Target Variable | Traffic_Demand (continuous, integer) |
| Generation Seed | random_state=42 |

---

## Feature Descriptions

### Original Features (15)

| # | Feature | Type | Range/Values | Description |
|---|---------|------|-------------|-------------|
| 1 | Geohash_Location | Categorical (20) | tdr1x0, tdr1x1, ..., tdr2a4 | City zone identifier using geohash encoding. 20 zones representing different urban areas. |
| 2 | Timestamp | DateTime | 2024-01-01 to 2024-12-13 | Record timestamp spanning ~1 year. Shuffled to break sequential ordering. |
| 3 | Day_of_Week | Categorical (7) | Monday - Sunday | Day name. Weekdays have ~1.5x more demand than weekends. |
| 4 | Hour | Integer | 0 - 23 | Hour of day. Key driver of demand via rush-hour patterns. |
| 5 | Road_Type | Categorical (4) | Highway, Arterial, Residential, Collector | Infrastructure type. Distribution: 25% Highway, 30% Arterial, 25% Residential, 20% Collector. |
| 6 | Number_of_Lanes | Integer | 1 - 8 | Lane count. Correlated with Road_Type (Highway: 4-8, Residential: 1-2). |
| 7 | Traffic_Signals | Integer | 0 - 5 | Nearby signal count. Highways have few (0-1), Arterials have many (2-5). |
| 8 | Large_Vehicles_Count | Integer | 0 - 40 | Count of trucks/buses. Poisson-distributed, mean varies by road type. |
| 9 | Temperature | Float | -5.0 to 45.0 | Temperature in Celsius. Normal distribution, mean=22, std=10. |
| 10 | Humidity | Float | 10.0 to 100.0 | Relative humidity. Higher in rain/snow/fog conditions. |
| 11 | Rainfall | Float | 0.0 to 50.0 | Rainfall in mm. Non-zero only for Rain/Snow conditions. |
| 12 | Weather_Conditions | Categorical (5) | Clear (35%), Cloudy (25%), Rain (20%), Snow (10%), Fog (10%) | Current weather. Directly impacts demand via multipliers. |
| 13 | Nearby_Landmarks | Categorical (10) | Shopping Mall, Hospital, School, Park, Stadium, Airport, Train Station, Office Complex, University, None | Nearest landmark. Influences demand by 3-15%. |
| 14 | Event_Indicator | Binary | 0 (85%), 1 (15%) | Whether a special event is occurring. Events boost demand by 15-30%. |
| 15 | **Traffic_Demand** | **Integer** | **5 - ~800+** | **TARGET: Vehicles per hour passing through the zone. Regression target.** |

### Engineered Features (5)

| # | Feature | Type | Range | Formula |
|---|---------|------|-------|---------|
| 16 | Peak_Hour_Flag | Binary | 0, 1 | 1 if Hour in {7,8,9,17,18,19} |
| 17 | Weekend_Flag | Binary | 0, 1 | 1 if Day_of_Week in {Saturday, Sunday} |
| 18 | Traffic_Density_Score | Float | [0, 1] | 0.4*norm(lanes) + 0.3*norm(signals) + 0.3*norm(large_veh) |
| 19 | Weather_Impact_Score | Float | [0, 1] | 0.25*norm(\|temp-22\|) + 0.25*norm(humidity) + 0.35*norm(rainfall) + 0.15*severity |
| 20 | Rush_Hour_Indicator | Integer | 0, 1, 2 | off_peak=0, morning_rush=1, evening_rush=2 |

---

## Synthetic Generation Methodology

### Base Demand Model

Traffic demand is computed as the product of multiple effects:

```
Traffic_Demand = Base_Demand
    * Hourly_Multiplier
    * Weather_Multiplier
    * Weekend_Multiplier
    * Lane_Effect
    * Event_Boost
    * Large_Vehicle_Effect
    * Temperature_Comfort_Effect
    * Landmark_Effect
    * Noise
```

### Component Details

#### Base Demand by Road Type
| Road Type | Base Demand (veh/hr) |
|-----------|---------------------|
| Highway | 350 |
| Arterial | 250 |
| Collector | 150 |
| Residential | 80 |

#### Hourly Multipliers
Peak at hours 8 and 18 (1.0), trough at hours 2-4 (0.07-0.09). Creates realistic bimodal rush-hour pattern.

#### Weather Demand Multipliers
| Condition | Multiplier |
|-----------|-----------|
| Clear | 1.00 |
| Cloudy | 0.95 |
| Rain | 0.82 |
| Fog | 0.78 |
| Snow | 0.70 |

#### Other Effects
- **Weekend**: 0.65x weekday demand
- **Lanes**: 1 + 0.05*(lanes - 2) multiplier
- **Events**: 1.15-1.30x random boost
- **Large vehicles**: 1 + 0.005*count
- **Temperature**: 1 - 0.003*|temp - 22|
- **Landmarks**: Stadium/Mall/Airport +15%, Hospital/Station/Office +8%, School/Park +3%
- **Noise**: Normal(1.0, 0.08) multiplicative noise

### Missing Values
~1% missing values injected in Temperature, Humidity, Large_Vehicles_Count, and Traffic_Signals for realism.

---

## Preprocessing Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| Missing value strategy | Median (numeric), Mode (categorical) | Preserves distribution with minimal bias at 1% missing rate |
| Encoding method | LabelEncoder for all categoricals | Tree-based models (RF, XGBoost, LightGBM) handle ordinal encoding effectively |
| Scaling method | StandardScaler | Normalizes features to zero mean, unit variance for consistent distance calculations |
| Train/test split | 80/20 | Standard split balancing training data volume and test set representativeness |
| Random seed | 42 | Fixed for full reproducibility |
| Duplicate handling | Drop exact duplicates | Prevents data leakage from repeated records |
| Outlier handling | Retained | Outliers are realistic (e.g., high demand during events on highways) and inform the model |
| Timestamp handling | Dropped for modeling | Temporal features (Hour, Day_of_Week) already extracted; raw timestamp adds no signal |

---

*Dataset documentation for Innovexa Catalyst | Traffic Demand Prediction System v1.0*
