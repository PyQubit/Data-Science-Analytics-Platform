"""Shared Plotly chart helpers; all charts consume the current dataset/results."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def missingness_bar(df: pd.DataFrame):
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty: return None
    fig = px.bar(x=missing.index.astype(str), y=missing.values, labels={"x":"Column", "y":"Missing cells"}, title="Missing values by column")
    fig.update_layout(xaxis_tickangle=-35, margin=dict(b=100))
    return fig

def correlation_heatmap(df: pd.DataFrame, method: str = "pearson"):
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] > 200: raise ValueError("Correlation heatmap is limited to 200 numeric columns; select a smaller subset.")
    if numeric.shape[1] < 2: return None
    corr = numeric.corr(method=method, min_periods=2)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title=f"{method.title()} correlation matrix", aspect="auto")
    return fig

def distribution_chart(df: pd.DataFrame, column: str):
    s = df[column]
    if pd.api.types.is_numeric_dtype(s):
        return px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}", nbins=40)
    counts = s.astype("string").fillna("(missing)").value_counts().head(30).rename_axis(column).reset_index(name="count")
    return px.bar(counts, x=column, y="count", title=f"Top categories in {column}")

def scatter_chart(df: pd.DataFrame, x: str, y: str, color: str | None = None, max_points: int = 20000):
    plot_df = df[[c for c in [x,y,color] if c]].copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=42)
    return px.scatter(plot_df, x=x, y=y, color=color if color in plot_df.columns else None, title=f"{y} vs {x}", trendline=None)

def time_series_chart(df: pd.DataFrame, date_col: str, value_col: str, group_col: str | None = None):
    plot_df = df[[c for c in [date_col,value_col,group_col] if c]].copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors="coerce")
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[date_col,value_col]).sort_values(date_col)
    if plot_df.empty: return None
    return px.line(plot_df, x=date_col, y=value_col, color=group_col if group_col in plot_df else None, title=f"{value_col} over time")
