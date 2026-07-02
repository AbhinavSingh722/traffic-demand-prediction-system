"""
Model Comparison Dashboard
===========================
Interactive comparison of all trained models with performance metrics,
actual vs. predicted scatter plots, and feature importance charts.

Author: Innovexa Catalyst
"""

import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Model Comparison | Traffic Prediction", page_icon="📊", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .metric-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box h3 { color: #667eea; margin: 0 0 0.3rem 0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-box .val { font-size: 2rem; font-weight: 800; margin: 0; }
    .metric-box .sub { color: #6a7a94; font-size: 0.78rem; margin: 0.3rem 0 0 0; }
    .pass-badge { background: linear-gradient(135deg, #00b09b, #96c93d); color: #fff; padding: 3px 12px; border-radius: 10px; font-size: 0.72rem; font-weight: 700; }
    .best-badge { background: linear-gradient(135deg, #667eea, #764ba2); color: #fff; padding: 3px 12px; border-radius: 10px; font-size: 0.72rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# -- Hardcoded model data (always available, matches CSV) --
MODEL_DATA = {
    "Random Forest":                  {"R2": 0.9787, "RMSE": 16.66, "MAE": 9.80, "MAPE": 7.96, "target": 0.85, "color": "#764ba2"},
    "XGBoost":                        {"R2": 0.9837, "RMSE": 14.59, "MAE": 8.74, "MAPE": 7.85, "target": 0.93, "color": "#f5576c"},
    "LightGBM":                       {"R2": 0.9837, "RMSE": 14.58, "MAE": 8.76, "MAPE": 8.14, "target": 0.95, "color": "#4facfe"},
    "Ensemble (55% LGBM + 45% XGB)":  {"R2": 0.9839, "RMSE": 14.49, "MAE": 8.67, "MAPE": 7.81, "target": 0.96, "color": "#43e97b"},
}

# ── Header ──
st.markdown("# 📊 Model Comparison Dashboard")
st.markdown("Compare Random Forest, XGBoost, LightGBM, and the Weighted Ensemble side by side.")
st.markdown("---")

# ── Top Metric Cards ──
cols = st.columns(4)
for i, (name, d) in enumerate(MODEL_DATA.items()):
    short = name.split("(")[0].strip()
    with cols[i]:
        badge = "best-badge" if "Ensemble" in name else "pass-badge"
        st.markdown(f"""
        <div class="metric-box">
            <h3>{short}</h3>
            <p class="val" style="color:{d['color']}">{d['R2']:.2%}</p>
            <p class="sub">RMSE: {d['RMSE']:.2f} | MAE: {d['MAE']:.2f}</p>
            <span class="{badge}">{'★ BEST' if 'Ensemble' in name else 'PASS'} ≥{d['target']:.0%}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ── R² Bar Chart ──
st.markdown("### R² Score Comparison")
fig = go.Figure()

models = list(MODEL_DATA.keys())
r2_vals = [MODEL_DATA[m]["R2"] for m in models]
colors = [MODEL_DATA[m]["color"] for m in models]
targets = [MODEL_DATA[m]["target"] for m in models]

fig.add_trace(go.Bar(
    x=[m.split("(")[0].strip() for m in models],
    y=r2_vals,
    marker_color=colors,
    text=[f"{v:.4f}" for v in r2_vals],
    textposition="outside",
    textfont=dict(size=14, color="white"),
))

# Target threshold markers
for i, t in enumerate(targets):
    fig.add_shape(type="line", x0=i-0.4, x1=i+0.4, y0=t, y1=t,
                  line=dict(color="rgba(235,51,73,0.5)", width=2, dash="dash"))
    fig.add_annotation(x=i, y=t-0.006, text=f"Target: {t:.0%}",
                       showarrow=False, font=dict(size=9, color="rgba(235,51,73,0.7)"))

fig.update_layout(
    yaxis_title="R² Score", yaxis_range=[0.82, 1.0],
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#a8b2d1"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    height=420, showlegend=False, margin=dict(t=30),
)
st.plotly_chart(fig, use_container_width=True)

# ── RMSE / MAE Comparison ──
st.markdown("### Error Metrics Comparison")
fig2 = go.Figure()
short_names = [m.split("(")[0].strip() for m in models]

fig2.add_trace(go.Bar(name="RMSE", x=short_names,
    y=[MODEL_DATA[m]["RMSE"] for m in models],
    marker_color=["rgba(118,75,162,0.7)","rgba(245,87,108,0.7)","rgba(79,172,254,0.7)","rgba(67,233,123,0.7)"]))
fig2.add_trace(go.Bar(name="MAE", x=short_names,
    y=[MODEL_DATA[m]["MAE"] for m in models],
    marker_color=["rgba(118,75,162,0.4)","rgba(245,87,108,0.4)","rgba(79,172,254,0.4)","rgba(67,233,123,0.4)"]))

fig2.update_layout(
    barmode="group", yaxis_title="Error Value",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#a8b2d1"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    height=380, legend=dict(orientation="h", y=1.05), margin=dict(t=30),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Full Metrics Table ──
st.markdown("### Detailed Metrics Table")
table_data = []
for name, d in MODEL_DATA.items():
    table_data.append({
        "Model": name, "R² Score": d["R2"], "RMSE": d["RMSE"],
        "MAE": d["MAE"], "MAPE (%)": d["MAPE"],
        "Target": f"≥{d['target']:.0%}", "Status": "★ BEST" if "Ensemble" in name else "✓ PASS"
    })
st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

# ── EDA Plots ──
st.markdown("---")
st.markdown("### 📈 EDA Visualisations")
plots_dir = PROJECT_ROOT / "reports" / "plots"
if plots_dir.exists():
    plot_files = sorted(plots_dir.glob("*.png"))
    if plot_files:
        plot_names = {
            "01_demand_distribution": "Traffic Demand Distribution",
            "02_hourly_traffic": "Hourly Traffic Patterns",
            "03_weather_impact": "Weather Impact Analysis",
            "04_correlation_heatmap": "Feature Correlation Heatmap",
            "05_feature_importance_lgbm": "Feature Importance (LightGBM)",
            "05b_feature_importance_xgb": "Feature Importance (XGBoost)",
            "06_actual_vs_predicted_ensemble": "Actual vs Predicted (Ensemble)",
        }
        for i in range(0, len(plot_files), 2):
            c1, c2 = st.columns(2)
            with c1:
                title = plot_names.get(plot_files[i].stem, plot_files[i].stem.replace("_"," ").title())
                st.markdown(f"**{title}**")
                st.image(str(plot_files[i]), use_column_width=True)
            if i+1 < len(plot_files):
                with c2:
                    title = plot_names.get(plot_files[i+1].stem, plot_files[i+1].stem.replace("_"," ").title())
                    st.markdown(f"**{title}**")
                    st.image(str(plot_files[i+1]), use_column_width=True)
    else:
        st.info("No EDA plots found. Run the ML pipeline to generate visualisations.")
else:
    st.info("Run the ML pipeline to generate EDA plots in `reports/plots/`.")

# ── Architecture ──
st.markdown("---")
st.markdown("### 🏗️ System Architecture")
st.markdown("""
| Component | Module | Description |
|-----------|--------|-------------|
| Data Generation | `generate_dataset.py` | 125K synthetic records with realistic patterns |
| Preprocessing | `data_preprocessing.py` | Missing values, label encoding, MinMax scaling |
| Feature Engineering | `feature_engineering.py` | 5 engineered features |
| Model Training | `model_training.py` | RF, XGBoost, LightGBM with RandomizedSearchCV |
| Ensemble | `ensemble.py` | 55% LightGBM + 45% XGBoost weighted average |
| Evaluation | `evaluate.py` | R², RMSE, MAE, MAPE + 6 visualisations |
| Web App | `streamlit_app.py` | Interactive prediction dashboard |
""")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#4a5568;font-size:0.85rem;'>Traffic Demand Prediction System v2.0 | Built by <strong>Innovexa Catalyst</strong></div>", unsafe_allow_html=True)
