from __future__ import annotations
import streamlit as st

def render_settings():
    st.title("Settings and Analysis Configuration")
    st.caption("These settings apply for this browser session unless you change environment variables before launching the app.")
    st.session_state.analysis_alpha = st.slider("Statistical significance level (alpha)", min_value=.001, max_value=.2, value=float(st.session_state.get("analysis_alpha", .05)), step=.001, format="%.3f")
    st.session_state.analysis_random_state = int(st.number_input("Random seed for reproducible sampling/models", min_value=0, max_value=2_147_483_647, value=int(st.session_state.get("analysis_random_state", 42)), step=1))
    st.session_state.analysis_chart_sample = int(st.number_input("Maximum rows for charting large datasets", min_value=1000, max_value=200000, value=int(st.session_state.get("analysis_chart_sample", 20000)), step=1000))
    st.write({"alpha":st.session_state.analysis_alpha, "random_state":st.session_state.analysis_random_state, "chart_sample_rows":st.session_state.analysis_chart_sample})
    st.info("Changing the random seed changes sampling and model results. Alpha affects displayed hypothesis-test conclusions, not underlying p-values.")
