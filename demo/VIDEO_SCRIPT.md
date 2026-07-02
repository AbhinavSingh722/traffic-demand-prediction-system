# Demo Video Script & Prompt

## Quick Info
- **Duration**: 5–10 minutes
- **Format**: Screen recording + voiceover (or text overlays)
- **Tools**: OBS Studio / Xbox Game Bar (`Win+G`) for recording, Clipchamp / CapCut for editing
- **Live App**: https://traffic-demand-prediction-system-8wsccguc27usbha2w5u7jg.streamlit.app/
- **GitHub**: https://github.com/AbhinavSingh722/traffic-demand-prediction-system

---

## Recording Setup

1. **Open these tabs in Chrome before recording:**
   - Tab 1: `demo/presentation.html` (open locally by double-clicking the file)
   - Tab 2: Your live Streamlit app (link above)
   - Tab 3: GitHub repo page (link above)
   - Tab 4: VS Code with the project open

2. **Screen resolution**: 1920x1080 (Full HD) recommended
3. **Browser**: Fullscreen mode (`F11`)

---

## Scene-by-Scene Script

### SCENE 1: Title & Introduction (30 seconds)
**Show**: `presentation.html` Slide 1

**Voiceover / Text Overlay**:
> "Hello! This is the Traffic Demand Prediction System, built for Innovexa Catalyst. This project predicts urban traffic demand using machine learning — specifically an ensemble of LightGBM and XGBoost models — achieving 98.39% R-squared accuracy. Let me walk you through the entire system."

---

### SCENE 2: Dataset Overview (45 seconds)
**Show**: Press right arrow to Slide 2

**Voiceover / Text Overlay**:
> "The system uses the Smart City Traffic Dataset containing over 125,000 records with 15+ features. The data includes temporal features like hour and day of week, road infrastructure details like road type and number of lanes, weather conditions including temperature, humidity, and rainfall, and location data using geohash encoding. The dataset was synthetically generated with realistic urban traffic patterns — including morning and evening rush hours, weather-dependent traffic variations, and weekend vs. weekday differences."

---

### SCENE 3: Feature Engineering (45 seconds)
**Show**: Press right arrow to Slide 3

**Voiceover / Text Overlay**:
> "We engineered 5 critical features to boost model performance. First, a Peak Hour Flag that identifies morning rush hours from 7 to 9 AM and evening rush from 5 to 7 PM. Second, a Weekend Flag for Saturday and Sunday patterns. Third, a Traffic Density Score — a weighted composite of lanes, signals, and large vehicle counts. Fourth, a Weather Impact Score combining temperature, humidity, rainfall, and severity. And fifth, a Rush Hour Category that classifies each hour as morning rush, evening rush, or off-peak."

---

### SCENE 4: ML Pipeline Architecture (45 seconds)
**Show**: Press right arrow to Slide 4

**Voiceover / Text Overlay**:
> "The end-to-end ML pipeline follows 5 stages. First, data generation creates 125,000 realistic records. Then preprocessing handles missing values, label encoding, and feature scaling. Feature engineering adds our 5 engineered features. Model training uses RandomizedSearchCV for hyperparameter tuning — 30 iterations for Random Forest, 40 each for XGBoost and LightGBM, all with 3-fold cross-validation. Finally, our weighted ensemble combines LightGBM at 55% weight with XGBoost at 45% to produce the final predictions."

---

### SCENE 5: Model Results (60 seconds)
**Show**: Press right arrow to Slide 5 (animated bars), then Slide 6 (metrics table)

**Voiceover (Slide 5)**:
> "Here are our model results. Watch the R-squared bars animate — Random Forest achieves 97.87%, far exceeding its 85% target. XGBoost hits 98.37%, beating its 93% target. LightGBM also reaches 98.37%, well above its 95% target. And our weighted ensemble achieves the best score of 98.39%, significantly surpassing the 96% minimum threshold."

**Voiceover (Slide 6)**:
> "Looking at the detailed metrics table — the ensemble model has the lowest RMSE at 14.49, lowest MAE at 8.67, and lowest MAPE at 7.81%. All four models pass their respective thresholds."

---

### SCENE 6: Live App Demo (3 minutes) — MOST IMPORTANT
**Show**: Switch to the live Streamlit app tab

**Voiceover**: "Now let me demonstrate the live web application, deployed on Streamlit Cloud."

**Action 1 — Highway, Clear, Monday 8 AM**:
> "Let's start with a highway on a clear Monday morning at 8 AM — a typical rush hour scenario."
- Set: Road Type = Highway, Weather = Clear, Date = a Monday, Time = 08:00
- Click "Predict Traffic Demand"
> "The system predicts approximately 489 vehicles per hour. Congestion level is HIGH at 61%, and it gives us a Peak Traffic Alert. Below, you can see the 24-hour forecast chart."

**Action 2 — Residential, Snow, Sunday 2 PM**:
> "Now let's try a residential road during snowfall on a Sunday afternoon."
- Change: Road Type = Residential, Weather = Snow, Date = a Sunday, Time = 14:00
- Click "Predict Traffic Demand"
> "The demand drops dramatically to about 57 vehicles per hour. Congestion is now MEDIUM at just 6%, and there's no peak traffic alert."

**Action 3 — Arterial, Rain, Friday 6 PM**:
> "One more scenario — an arterial road during rain on a Friday evening rush hour."
- Change: Road Type = Arterial, Weather = Rain, Date = a Friday, Time = 18:00
- Click "Predict Traffic Demand"
> "We see high demand again — the evening rush hour on an arterial road combined with rain creates significant congestion."
- Scroll down to show the 24-hour forecast chart
> "The interactive Plotly chart lets you hover over any point to see exact predicted values."

---

### SCENE 7: Code & GitHub Tour (1 minute)
**Show**: Switch to VS Code / GitHub repo tab

**Voiceover**:
> "The codebase is fully modular and production-ready."

- Show the `src/` folder structure briefly:
> "The source code is organized into 6 modules — generate_dataset, data_preprocessing, feature_engineering, model_training, ensemble, and evaluate."

- Show GitHub repo page:
> "The entire project is open-source on GitHub with comprehensive documentation including a project report, dataset documentation, and README with the live app link."

---

### SCENE 8: Conclusion (30 seconds)
**Show**: Switch back to `presentation.html` Slide 8

**Voiceover**:
> "To summarize — the Traffic Demand Prediction System achieves 98.39% R-squared accuracy using a weighted ensemble of LightGBM and XGBoost. It features 125,000 records, 5 engineered features, hyperparameter-tuned models, 6 EDA visualizations, a fully interactive Streamlit web application, and a clean, modular, documented codebase. Thank you for watching — built by Innovexa Catalyst."

---

## Editing Tips (Clipchamp or CapCut)

1. **Add text overlays** for key numbers: "98.39% R2", "125K Records", "5 Features"
2. **Add transitions** between scenes (cross-dissolve works best)
3. **Add background music** — use royalty-free lo-fi or corporate music
4. **Add an intro title card** (3 seconds): "Traffic Demand Prediction System"
5. **Add an outro card** (3 seconds): GitHub URL + Streamlit URL
6. **Speed up** any loading/waiting moments
7. **Export as MP4**, 1080p resolution

---

## AI Video Tool Prompt

If using Synthesia, Pictory, InVideo, or similar:

```
Create a professional 7-minute demo video for a machine learning project called
"Traffic Demand Prediction System" by Innovexa Catalyst.

The video should have a dark, modern tech aesthetic with purple/blue gradient
accents. Use clean sans-serif typography.

Structure:
1. INTRO (30s): Title card with project name, "98.39% R2 Accuracy", tech stack
   badges (Python, LightGBM, XGBoost, Streamlit)
2. DATASET (45s): Show "125,000 records", "15+ features" with icons for weather,
   road, time, location
3. FEATURES (45s): Display 5 feature cards with icons and formulas
4. PIPELINE (45s): Animated flow: Data > Preprocess > Features > Train > Ensemble
5. RESULTS (60s): Animated bar chart showing R2 scores for 4 models, all exceeding
   targets. Highlight Ensemble at 98.39%
6. LIVE DEMO (3min): Screen recording of the Streamlit app at
   https://traffic-demand-prediction-system-8wsccguc27usbha2w5u7jg.streamlit.app/
   showing 3 prediction scenarios
7. CONCLUSION (30s): Summary checkmarks, GitHub link, "Built by Innovexa Catalyst"

Tone: Professional, confident, technical but accessible.
Colors: Dark background (#0a0a1a), purple-blue gradients (#667eea to #764ba2),
green accents (#43e97b) for success metrics.
```
