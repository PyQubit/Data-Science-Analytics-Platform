from __future__ import annotations
import streamlit as st
import pandas as pd
from utils.streamlit_cache import cached_profile, cached_descriptive_table, cached_insights, fingerprint
from core.report_generator import generate_html_report, generate_pdf_report, generate_excel_report

def render_reports(df: pd.DataFrame | None):
    st.title("Report Center")
    if df is None: st.info("Load a dataset first."); return
    st.write("Reports are built from the current processed dataset and calculated findings. Exports include methodology notes and limitations; generated reports may contain sensitive data from your upload, so store them carefully.")
    data_key = fingerprint(df)
    profile = cached_profile(df, data_key)
    default_charts = list(df.select_dtypes(include="number").columns[:2])
    if not default_charts and len(df.columns): default_charts = [df.columns[0]]
    chart_columns = st.multiselect("Columns to visualize in the exported reports (maximum 4)", list(df.columns), default=default_charts,
                                   help="Numeric columns produce histograms; other columns produce top-category charts. Two or more selected numeric columns also produce a correlation heatmap.")
    if len(chart_columns) > 4:
        st.warning("Only the first four selected columns will be used in the reports.")
        chart_columns = chart_columns[:4]
    report_data = {"profile": profile, "descriptive": cached_descriptive_table(df, data_key), "insights": cached_insights(df, data_key),
                   "chart_columns": chart_columns}
    if st.button("Generate HTML, PDF, and Excel reports", type="primary"):
        # Never leave a previous bundle available after a failed regeneration.
        st.session_state.generated_reports = None
        st.session_state.report_signature = None
        with st.spinner("Building reports from current analysis..."):
            try:
                logs = st.session_state.get("transformation_log", [])
                generated = {
                    "html": generate_html_report(df, report_data, logs),
                    "pdf": generate_pdf_report(df, report_data, logs),
                    "xlsx": generate_excel_report(df, report_data, logs),
                }
                st.session_state.generated_reports = generated
                st.session_state.report_signature = {"fingerprint": data_key, "chart_columns": tuple(chart_columns)}
                st.success("Reports generated.")
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")
    reports = st.session_state.get("generated_reports")
    signature = st.session_state.get("report_signature")
    reports_match_current_settings = bool(reports and signature == {"fingerprint": data_key, "chart_columns": tuple(chart_columns)})
    if reports_match_current_settings:
        c1,c2,c3 = st.columns(3)
        c1.download_button("Download HTML report", reports["html"], "analytics_report.html", "text/html", use_container_width=True)
        c2.download_button("Download PDF report", reports["pdf"], "analytics_report.pdf", "application/pdf", use_container_width=True)
        c3.download_button("Download Excel report", reports["xlsx"], "analytics_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    elif reports:
        st.info("The dataset or chart selection changed since these reports were generated. Generate reports again before downloading.")
    st.subheader("Report scope")
    st.write({"rows":len(df), "columns":len(df.columns), "missing_cells":int(df.isna().sum().sum()), "duplicate_rows":int(df.duplicated().sum()), "insights_included":len(report_data["insights"]), "transformations_logged":len(st.session_state.get("transformation_log", []))})
    st.caption("PDF reports are compact text reports. Interactive visualizations remain available in the app; the PDF export does not claim to embed every interactive chart.")
