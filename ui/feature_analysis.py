from __future__ import annotations
import streamlit as st
import pandas as pd
from core.feature_analysis import feature_summary, analyze_features, mutual_information_analysis
from utils.streamlit_cache import fingerprint

def render_feature_analysis(df: pd.DataFrame | None, random_state: int = 42):
    st.title("Feature Analysis")
    if df is None: st.info("Load a dataset first."); return
    target_options = ["(no target selected)"] + list(df.columns)
    target_choice = st.selectbox("Optional target variable", target_options)
    target = None if target_choice == "(no target selected)" else target_choice
    st.dataframe(feature_summary(df, target=target), use_container_width=True, hide_index=True)
    if target is not None:
        st.markdown("### Mutual information")
        st.caption("Optional univariate association scores. For very wide datasets, choose a manageable set of predictors; datasets over 20,000 rows are sampled deterministically for this calculation.")
        mi_options = [c for c in df.columns if c != target]
        with st.form("mutual_information_form"):
            mi_features = st.multiselect("Predictor features for mutual information (maximum 200)", mi_options,
                                         default=mi_options[:min(30, len(mi_options))], key="mi_features")
            run_mi = st.form_submit_button("Calculate mutual information", type="secondary")
        if run_mi:
            try:
                result_mi = mutual_information_analysis(df, target, mi_features, random_state=random_state)
                st.session_state.mi_result = {"fingerprint": fingerprint(df), "target": target, "features": tuple(mi_features), "result": result_mi}
            except Exception as exc:
                st.error(f"Mutual information could not be calculated: {exc}")
        saved_mi = st.session_state.get("mi_result")
        if saved_mi and saved_mi.get("fingerprint") == fingerprint(df) and saved_mi.get("target") == target and saved_mi.get("features") == tuple(mi_features):
            st.dataframe(saved_mi["result"]["scores"], use_container_width=True, hide_index=True)
            st.caption(saved_mi["result"]["limitation"] + (" A deterministic sample of rows was used." if saved_mi["result"]["sampled"] else ""))
    st.markdown("### Held-out model-based feature importance")
    st.caption("Trains a local Random Forest after preprocessing, holds out test rows, and uses permutation importance on that holdout. Results are predictive/model-dependent, not causal.")
    if target is None:
        st.info("Select a target variable above to run target-aware analysis."); return
    with st.form("feature_model_form"):
        task = st.selectbox("Prediction task", ["auto", "classification", "regression"])
        test_size = st.slider("Holdout fraction", min_value=.1, max_value=.5, value=.25, step=.05)
        run_model = st.form_submit_button("Train and evaluate model", type="primary")
    if run_model:
        with st.spinner("Fitting a model and estimating permutation importance..."):
            try:
                result = analyze_features(df, target=target, task=task, test_size=test_size, random_state=random_state)
                st.success(f"Task: {result['task']} · Train rows: {result['train_rows']:,} · Test rows: {result['test_rows']:,}")
                st.json(result["metrics"])
                if not result["importance"].empty:
                    st.dataframe(result["importance"], use_container_width=True, hide_index=True)
                    import plotly.express as px
                    fig = px.bar(result["importance"].sort_values("permutation_importance_mean"), x="permutation_importance_mean", y="feature", orientation="h", error_x="permutation_importance_std", title="Held-out permutation importance")
                    st.plotly_chart(fig, use_container_width=True)
                st.caption(result["limitation"])
            except Exception as exc:
                st.error(f"Feature model could not be evaluated: {exc}")
