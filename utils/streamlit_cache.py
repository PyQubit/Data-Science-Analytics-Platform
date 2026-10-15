"""Streamlit cache wrappers keyed by a content fingerprint and computation options."""
from __future__ import annotations
import streamlit as st
import pandas as pd
from core.profiler import profile_dataset, descriptive_table
from core.insights_engine import generate_insights
from core.statistical_analysis import correlation_analysis
from utils.caching import dataframe_fingerprint

@st.cache_data(show_spinner=False, max_entries=8)
def cached_profile(_df: pd.DataFrame, fingerprint: str) -> dict:
    return profile_dataset(_df)

@st.cache_data(show_spinner=False, max_entries=8)
def cached_descriptive_table(_df: pd.DataFrame, fingerprint: str) -> pd.DataFrame:
    return descriptive_table(_df)

@st.cache_data(show_spinner=False, max_entries=8)
def cached_insights(_df: pd.DataFrame, fingerprint: str, max_insights: int = 20) -> list[dict]:
    return generate_insights(_df, max_insights=max_insights)

@st.cache_data(show_spinner=False, max_entries=16)
def cached_correlations(_df: pd.DataFrame, fingerprint: str, columns: tuple[str, ...], method: str) -> pd.DataFrame:
    return correlation_analysis(_df[list(columns)], method=method)

def fingerprint(df: pd.DataFrame) -> str:
    return dataframe_fingerprint(df)
