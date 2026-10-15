from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from core.statistical_analysis import hypothesis_test
from utils.streamlit_cache import cached_descriptive_table, cached_correlations, fingerprint
from core.visualization import correlation_heatmap

def render_statistics(df: pd.DataFrame | None, alpha: float = 0.05):
    st.title("Statistical Analysis")
    if df is None: st.info("Load a dataset first."); return
    st.subheader("Descriptive statistics")
    data_key = fingerprint(df)
    st.dataframe(cached_descriptive_table(df, data_key), use_container_width=True, hide_index=True)
    numeric = list(df.select_dtypes(include=np.number).columns)
    if len(numeric) >= 2:
        method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], key="corr_method")
        selected_corr_cols = st.multiselect("Columns included in correlation matrix (select up to 100)", numeric, default=numeric[:min(30, len(numeric))], key="corr_columns")
        if len(selected_corr_cols) > 100:
            st.warning("The correlation matrix is limited to the first 100 selected columns.")
            selected_corr_cols = selected_corr_cols[:100]
        corr = cached_correlations(df, data_key, tuple(selected_corr_cols), method) if len(selected_corr_cols) >= 2 else pd.DataFrame()
        if len(selected_corr_cols) >= 2:
            fig = correlation_heatmap(df[selected_corr_cols], method)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("A correlation heatmap is unavailable for the selected columns.")
        else:
            st.info("Select at least two numeric columns to build a correlation matrix.")
        with st.expander("Correlation coefficients"): st.dataframe(corr, use_container_width=True)
    st.subheader("Hypothesis testing")
    st.caption(f"Default significance level α = {alpha}. Choose a test that matches the variable types and study design; p-values do not measure effect size or practical importance.")
    options = {"Pearson correlation": "pearson", "Spearman rank correlation": "spearman", "Kendall rank correlation": "kendall",
               "Welch two-sample t-test": "welch_t", "Mann–Whitney U test": "mann_whitney", "Chi-square independence test": "chi_square",
               "Shapiro–Wilk normality test": "normality_shapiro"}
    with st.form("hypothesis_form"):
        test_label = st.selectbox("Test", list(options.keys()))
        x = st.selectbox("Variable X / outcome", list(df.columns), key="test_x")
        y = st.selectbox("Variable Y / grouping variable", list(df.columns), key="test_y")
        run = st.form_submit_button("Run selected test", type="primary")
    if run:
        try:
            result = hypothesis_test(df, options[test_label], x, y, alpha=alpha)
            clean = {k:v for k,v in result.items() if k not in {"contingency_table"}}
            st.json(clean)
            if "contingency_table" in result:
                st.write("Observed contingency table")
                st.dataframe(result["contingency_table"], use_container_width=True)
            if result["p_value"] < alpha: st.warning("The result meets the selected statistical significance threshold. Also review effect size and assumptions.")
            else: st.info("The result does not meet the selected statistical significance threshold; this is not proof of no effect.")
        except Exception as exc:
            st.error(f"Test could not be run: {exc}")
