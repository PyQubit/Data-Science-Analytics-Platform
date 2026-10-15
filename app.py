"""Entry point for the local Data Science Analytics Streamlit application."""
from __future__ import annotations
from pathlib import Path
import streamlit as st
from config.settings import SETTINGS
from core.data_loader import load_dataset, excel_sheet_names
from ui.dashboard import render_dashboard
from ui.exploration import render_upload_preview, render_quality_cleaning, render_exploration
from ui.statistics import render_statistics
from ui.feature_analysis import render_feature_analysis
from ui.patterns import render_patterns
from ui.reports import render_reports
from ui.settings import render_settings

st.set_page_config(page_title=SETTINGS.app_name, page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { color-scheme: light; }
.block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1500px; }
[data-testid="stMetric"] { border: 1px solid rgba(120,130,150,.22); padding: 14px; border-radius: 12px; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(120,130,150,.18); }
</style>
""", unsafe_allow_html=True)

for key, default in {"raw_df": None, "processed_df": None, "dataset_metadata": {}, "load_warnings": [],
                     "transformation_log": [], "analysis_alpha": SETTINGS.significance_level,
                     "analysis_random_state": SETTINGS.random_state, "analysis_chart_sample": SETTINGS.chart_sample_rows,
                     "generated_reports": None, "report_signature": None, "cluster_result": None, "anomaly_result": None, "mi_result": None}.items():
    if key not in st.session_state: st.session_state[key] = default

with st.sidebar:
    st.title("📊 Analytics")
    st.caption("Local data workbench")
    uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls", "json", "parquet"], help=f"Maximum file size: {SETTINGS.max_upload_mb} MB. Data is processed in memory.")
    sheet_name = None
    if uploaded is not None and Path(uploaded.name).suffix.lower() in {".xlsx", ".xls"}:
        try:
            sheets = excel_sheet_names(uploaded)
            sheet_name = st.selectbox("Excel sheet", sheets)
        except Exception as exc:
            st.error(f"Could not inspect workbook sheets: {exc}")
    if st.button("Load dataset", type="primary", use_container_width=True, disabled=uploaded is None):
        try:
            with st.spinner("Validating and loading data..."):
                result = load_dataset(uploaded, uploaded.name, sheet_name=sheet_name, max_size_mb=SETTINGS.max_upload_mb)
            st.session_state.raw_df = result.dataframe.copy(deep=True)
            st.session_state.processed_df = result.dataframe.copy(deep=True)
            st.session_state.dataset_metadata = result.metadata
            st.session_state.load_warnings = result.warnings
            st.session_state.transformation_log = []
            st.session_state.generated_reports = None
            st.session_state.report_signature = None
            st.session_state.cluster_result = None
            st.session_state.anomaly_result = None
            st.session_state.mi_result = None
            st.success(f"Loaded {result.metadata['rows']:,} rows × {result.metadata['columns']:,} columns.")
        except Exception as exc:
            st.error(f"Dataset could not be loaded: {exc}")
    if st.session_state.processed_df is not None:
        st.divider()
        st.markdown(f"**Loaded:** {st.session_state.dataset_metadata.get('filename', 'dataset')}")
        st.caption(f"{len(st.session_state.processed_df):,} rows · {len(st.session_state.processed_df.columns):,} columns")
        if st.button("Clear loaded data", use_container_width=True):
            for key, value in {"raw_df":None,"processed_df":None,"dataset_metadata":{},"load_warnings":[],"transformation_log":[],"generated_reports":None,"report_signature":None,"cluster_result":None,"anomaly_result":None,"mi_result":None}.items():
                st.session_state[key] = value
            st.rerun()
    page = st.radio("Workspace", ["Overview Dashboard", "Data Upload and Preview", "Data Quality and Cleaning", "Exploratory Data Analysis", "Statistical Analysis", "Feature Analysis", "Pattern Discovery", "Report Center", "Settings"], label_visibility="collapsed")
    st.divider()
    st.caption("Local-first · No API key required · No uploaded-data persistence")

df = st.session_state.processed_df
if page == "Overview Dashboard": render_dashboard(df)
elif page == "Data Upload and Preview": render_upload_preview(df, st.session_state.dataset_metadata, st.session_state.load_warnings)
elif page == "Data Quality and Cleaning": render_quality_cleaning(df)
elif page == "Exploratory Data Analysis": render_exploration(df)
elif page == "Statistical Analysis": render_statistics(df, float(st.session_state.analysis_alpha))
elif page == "Feature Analysis": render_feature_analysis(df, int(st.session_state.analysis_random_state))
elif page == "Pattern Discovery": render_patterns(df, int(st.session_state.analysis_random_state))
elif page == "Report Center": render_reports(df)
elif page == "Settings": render_settings()
