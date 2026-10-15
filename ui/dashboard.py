from __future__ import annotations
import streamlit as st
import pandas as pd
from utils.streamlit_cache import cached_profile, cached_insights, fingerprint
from utils.formatting import format_bytes

def render_dashboard(df: pd.DataFrame | None):
    st.title("Data Science Analytics Platform")
    st.caption("Local-first exploratory analysis, statistical testing, pattern discovery, and reproducible reporting.")
    if df is None:
        st.info("Upload a dataset in the sidebar and select **Load dataset** to begin. Supported formats: CSV, XLSX/XLS, JSON, and Parquet.")
        st.markdown("### Analytical workflow")
        st.markdown("1. Load a real dataset and inspect its schema.\n2. Review data quality and choose cleaning transformations explicitly.\n3. Explore distributions, associations, and statistical tests.\n4. Investigate predictive feature importance, clusters, or anomalies when appropriate.\n5. Export a report with methodology and limitations.")
        return
    data_key = fingerprint(df)
    profile = cached_profile(df, data_key)
    cols = st.columns(4)
    cols[0].metric("Rows", f"{len(df):,}")
    cols[1].metric("Columns", f"{df.shape[1]:,}")
    cols[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    cols[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.caption(f"In-memory DataFrame size: {format_bytes(profile['memory_bytes'])}. These figures describe the currently selected (processed) dataset.")
    st.subheader("Current dataset preview")
    st.dataframe(df.head(12), use_container_width=True, hide_index=True)
    st.subheader("Quality warnings")
    if profile["quality_warnings"]:
        for warning in profile["quality_warnings"]: st.warning(warning)
    else:
        st.success("No issues were flagged by the configured screening checks. This is not a guarantee that the data is error-free.")
    st.subheader("Evidence-based insights")
    insights = cached_insights(df, data_key, max_insights=10)
    if not insights:
        st.info("No insight rules were triggered for this dataset.")
    for insight in insights:
        with st.expander(insight["title"]):
            st.write(insight["finding"])
            st.json(insight.get("evidence", {}))
            st.caption(f"{insight.get('category','')} · Sample size: {insight.get('sample_size','n/a')} · {insight.get('limitation','')}")
