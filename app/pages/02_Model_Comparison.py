"""
Model Comparison Dashboard
===========================
Interactive comparison of all trained models with performance metrics,
actual vs. predicted scatter plots, and feature importance charts.

Author: Innovexa Catalyst
"""

import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Project paths
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Model Comparison | Traffic Prediction",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_model_comparison():
    """Load model comparison CSV."""
    csv_path = PROJECT_ROOT / "reports" / "model_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_plots():
    """Collect all EDA plot paths."""
    plots_dir = PROJECT_ROOT / "reports" / "plots"
    if not plots_dir.exists():
        return {}
    plot_files = sorted(plots_dir.glob("*.png"))
    return {p.stem: str(p) for p in plot_files}


# ---------------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------------
st.markdown("# 📊 Model Comparison Dashboard")
st.markdown("Compare the performance of Random Forest, XGBoost, LightGBM, and the Weighted Ensemble side by side.")
st.markdown("---")

# -- Performance Metrics --
comparison = load_model_comparison()
if comparison is not None:
    st.markdown("### Performance Metrics")

    # R2 Score bar chart
    fig_r2 = go.Figure()

    colors = ["#764ba2", "#f5576c", "#4facfe", "#43e97b"]
    models = comparison["Model"].tolist() if "Model" in comparison.columns else comparison.iloc[:, 0].tolist()

    # Try to find R2 column
    r2_col = None
    for col in comparison.columns:
        if "r2" in col.lower() or "R2" in col:
            r2_col = col
            break

    if r2_col:
        r2_values = comparison[r2_col].tolist()

        fig_r2.add_trace(go.Bar(
            x=models,
            y=r2_values,
            marker_color=colors[:len(models)],
            text=[f"{v:.4f}" for v in r2_values],
            textposition="outside",
            textfont=dict(size=14, color="white"),
        ))

        # Add threshold lines
        thresholds = {"Random Forest": 0.85, "XGBoost": 0.93, "LightGBM": 0.95, "Ensemble": 0.96}
        for i, model in enumerate(models):
            for key, thresh in thresholds.items():
                if key.lower() in model.lower():
                    fig_r2.add_shape(
                        type="line",
                        x0=i - 0.4, x1=i + 0.4,
                        y0=thresh, y1=thresh,
                        line=dict(color="rgba(235,51,73,0.6)", width=2, dash="dash"),
                    )
                    fig_r2.add_annotation(
                        x=i, y=thresh - 0.008,
                        text=f"Target: {thresh:.0%}",
                        showarrow=False,
                        font=dict(size=10, color="rgba(235,51,73,0.8)"),
                    )

        fig_r2.update_layout(
            yaxis_title="R2 Score",
            yaxis_range=[0.80, 1.0],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#a8b2d1"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            height=450,
            showlegend=False,
        )

        st.plotly_chart(fig_r2, use_container_width=True)

    # Full metrics table
    st.markdown("### Detailed Metrics")
    st.dataframe(
        comparison.style.format(precision=4).set_properties(**{
            "background-color": "#1a1a2e",
            "color": "#e0e0e0",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("Model comparison data not found. Run the pipeline first.")

st.markdown("---")

# -- EDA Visualisations --
st.markdown("### 📈 EDA Visualisations")
st.markdown("Explore the 6 mandatory plots generated during exploratory data analysis.")

plots = load_plots()
if plots:
    plot_names = {
        "01_demand_distribution": "Traffic Demand Distribution",
        "02_hourly_traffic": "Hourly Traffic Analysis",
        "03_weather_impact": "Weather Impact Analysis",
        "04_correlation_heatmap": "Correlation Heatmap",
        "05_feature_importance_lgbm": "Feature Importance (LightGBM)",
        "05b_feature_importance_xgb": "Feature Importance (XGBoost)",
        "06_actual_vs_predicted_ensemble": "Actual vs Predicted (Ensemble)",
        "06b_actual_vs_predicted_lgbm": "Actual vs Predicted (LightGBM)",
    }

    # Create 2-column grid
    plot_keys = list(plots.keys())
    for i in range(0, len(plot_keys), 2):
        col1, col2 = st.columns(2)
        with col1:
            key = plot_keys[i]
            title = plot_names.get(key, key.replace("_", " ").title())
            st.markdown(f"**{title}**")
            st.image(plots[key], use_column_width=True)
        if i + 1 < len(plot_keys):
            with col2:
                key = plot_keys[i + 1]
                title = plot_names.get(key, key.replace("_", " ").title())
                st.markdown(f"**{title}**")
                st.image(plots[key], use_column_width=True)
else:
    st.warning("No EDA plots found in reports/plots/. Run the pipeline first.")

st.markdown("---")

# -- Architecture Info --
st.markdown("### 🏗️ System Architecture")
st.markdown("""
| Component | Module | Description |
|-----------|--------|-------------|
| Data Generation | `generate_dataset.py` | 125K synthetic records with realistic patterns |
| Preprocessing | `data_preprocessing.py` | Missing values, label encoding, MinMax scaling |
| Feature Engineering | `feature_engineering.py` | 5 engineered features (peak hour, weekend, density, weather, rush) |
| Model Training | `model_training.py` | RF, XGBoost, LightGBM with RandomizedSearchCV |
| Ensemble | `ensemble.py` | 55% LightGBM + 45% XGBoost weighted average |
| Evaluation | `evaluate.py` | R2, RMSE, MAE, MAPE metrics + 6 visualisations |
| Web App | `streamlit_app.py` | Interactive prediction dashboard |
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:0.85rem;'>"
    "Traffic Demand Prediction System v1.0 | Built by <strong>Innovexa Catalyst</strong>"
    "</div>",
    unsafe_allow_html=True,
)
