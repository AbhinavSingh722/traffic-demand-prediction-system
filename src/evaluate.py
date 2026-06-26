"""
evaluate.py
============
Evaluation module for the Traffic Demand Prediction System.
Provides metrics computation (R², RMSE, MAE, MAPE), comparison tables,
and visualization functions for model assessment.

Author: Innovexa Catalyst
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.
    model_name : str
        Name of the model for display.

    Returns
    -------
    dict
        Dictionary with R², RMSE, MAE, and MAPE.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # MAPE: avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    results = {
        "Model": model_name,
        "R2": round(r2, 6),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE (%)": round(mape, 4),
    }
    return results


def print_evaluation_report(results):
    """
    Print formatted evaluation report for a single model.

    Parameters
    ----------
    results : dict
        Dictionary from evaluate_model().
    """
    print(f"\n{'='*50}")
    print(f"  {results['Model']} - Evaluation Results")
    print(f"{'='*50}")
    print(f"  R2 Score:  {results['R2']:.6f}  ({results['R2']*100:.2f}%)")
    print(f"  RMSE:      {results['RMSE']:.4f}")
    print(f"  MAE:       {results['MAE']:.4f}")
    print(f"  MAPE:      {results['MAPE (%)']:.4f}%")
    print(f"{'='*50}\n")


def compare_models(all_results):
    """
    Create a comparison table of all model results.

    Parameters
    ----------
    all_results : list of dict
        List of result dictionaries from evaluate_model().

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame.
    """
    df = pd.DataFrame(all_results)
    df = df.set_index("Model")
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70 + "\n")
    return df


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted",
                              save_path=None):
    """
    Create scatter plot of actual vs predicted values.

    Parameters
    ----------
    y_true : array-like
        Actual target values.
    y_pred : array-like
        Predicted target values.
    title : str
        Plot title.
    save_path : str or Path, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample for performance if dataset is large
    n = len(y_true)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        y_t = np.array(y_true)[idx]
        y_p = np.array(y_pred)[idx]
    else:
        y_t = np.array(y_true)
        y_p = np.array(y_pred)

    ax.scatter(y_t, y_p, alpha=0.3, s=10, color="#4A90D9", edgecolors="none")

    # Perfect prediction line
    min_val = min(y_t.min(), y_p.min())
    max_val = max(y_t.max(), y_p.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2,
            label="Perfect Prediction")

    ax.set_xlabel("Actual Traffic Demand", fontsize=12)
    ax.set_ylabel("Predicted Traffic Demand", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Plot saved to {save_path}")
    plt.close(fig)
    return fig


def plot_feature_importance(model, feature_names, title="Feature Importance",
                             top_n=15, save_path=None):
    """
    Create bar chart of feature importances.

    Parameters
    ----------
    model : fitted model
        Model with feature_importances_ attribute.
    feature_names : list
        List of feature names.
    title : str
        Plot title.
    top_n : int
        Number of top features to display.
    save_path : str or Path, optional
        Path to save the figure.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    bars = ax.barh(
        range(len(indices)),
        importances[indices][::-1],
        color=colors,
        edgecolor="white",
    )

    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Plot saved to {save_path}")
    plt.close(fig)
    return fig


def plot_demand_distribution(df, target_col="Traffic_Demand", save_path=None):
    """
    Plot histogram + KDE of the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the target column.
    target_col : str
        Target column name.
    save_path : str or Path, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[target_col], kde=True, bins=60, color="#4A90D9",
                 edgecolor="white", alpha=0.7, ax=ax)
    ax.set_xlabel("Traffic Demand (vehicles/hour)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Traffic Demand Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_hourly_traffic(df, hour_col="Hour", target_col="Traffic_Demand",
                         save_path=None):
    """
    Plot average traffic demand by hour of day.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hour and target columns.
    save_path : str or Path, optional
        Path to save the figure.
    """
    hourly = df.groupby(hour_col)[target_col].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly.index, hourly.values, marker="o", linewidth=2.5,
            color="#E74C3C", markersize=8, markerfacecolor="#2C3E50")
    ax.fill_between(hourly.index, hourly.values, alpha=0.15, color="#E74C3C")

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Average Traffic Demand", fontsize=12)
    ax.set_title("Hourly Traffic Analysis", fontsize=14, fontweight="bold")
    ax.set_xticks(range(24))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_weather_impact(df, weather_col="Weather_Conditions",
                         target_col="Traffic_Demand", save_path=None):
    """
    Plot traffic demand by weather condition (box plot).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with weather and target columns.
    save_path : str or Path, optional
        Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {"Clear": "#2ECC71", "Cloudy": "#95A5A6", "Rain": "#3498DB",
               "Snow": "#9B59B6", "Fog": "#E67E22"}

    order = ["Clear", "Cloudy", "Fog", "Rain", "Snow"]
    existing = [w for w in order if w in df[weather_col].unique()]

    sns.boxplot(x=weather_col, y=target_col, data=df, order=existing,
                palette=palette, ax=ax, showfliers=False)

    ax.set_xlabel("Weather Condition", fontsize=12)
    ax.set_ylabel("Traffic Demand", fontsize=12)
    ax.set_title("Weather Impact on Traffic Demand", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_correlation_heatmap(df, save_path=None):
    """
    Plot full feature correlation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns.
    save_path : str or Path, optional
        Path to save the figure.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5,
                annot_kws={"size": 8}, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
