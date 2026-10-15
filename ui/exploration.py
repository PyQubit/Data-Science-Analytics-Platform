from __future__ import annotations
import streamlit as st
import pandas as pd
import json
from utils.streamlit_cache import cached_profile, cached_descriptive_table, fingerprint
from core.visualization import missingness_bar, distribution_chart, scatter_chart, time_series_chart
from core.data_cleaner import apply_cleaning
from utils.formatting import format_bytes
from utils.exports import safe_csv_bytes

def render_upload_preview(df: pd.DataFrame | None, metadata: dict | None = None, load_warnings: list[str] | None = None):
    st.title("Data Upload and Preview")
    if df is None:
        st.info("Choose a supported file in the sidebar, then press Load dataset.")
        st.markdown("Supported formats: **CSV, Excel (.xlsx/.xls), JSON, Parquet**. Uploads are handled in memory and are not persisted by this app.")
        return
    meta = metadata or {}
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}"); c2.metric("Columns", f"{len(df.columns):,}"); c3.metric("Memory", format_bytes(int(df.memory_usage(index=True, deep=True).sum())))
    st.write(f"**Source:** {meta.get('filename', 'In-memory dataset')}  ")
    if load_warnings:
        with st.expander(f"Loader notes ({len(load_warnings)})", expanded=True):
            for note in load_warnings: st.warning(note)
    st.subheader("Preview")
    n = st.slider("Rows to preview", min_value=5, max_value=max(5, min(500, len(df) or 5)), value=min(25, max(5, len(df) or 5)), step=5)
    st.dataframe(df.head(n), use_container_width=True, hide_index=True)
    st.subheader("Schema")
    schema = pd.DataFrame({"column": df.columns.astype(str), "dtype": [str(df[c].dtype) for c in df.columns],
                           "non_missing": [int(df[c].notna().sum()) for c in df.columns],
                           "unique_non_missing": [int(df[c].nunique(dropna=True)) for c in df.columns]})
    st.dataframe(schema, use_container_width=True, hide_index=True)
    st.download_button("Download current dataset as CSV", data=safe_csv_bytes(df), file_name="analyzed_data.csv", mime="text/csv", help="Formula-like text cells are prefixed with an apostrophe to reduce spreadsheet-injection risk.")

def render_quality_cleaning(df: pd.DataFrame | None):
    st.title("Data Quality and Cleaning")
    if df is None:
        st.info("Load a dataset first."); return
    profile = cached_profile(df, fingerprint(df))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Missing cells", f"{profile['missing_cells']:,}")
    metric_cols[1].metric("Missing-cell rate", f"{profile['missing_pct']:.2f}%")
    metric_cols[2].metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
    metric_cols[3].metric("Memory", format_bytes(profile['memory_bytes']))
    st.subheader("Column-level quality profile")
    st.dataframe(profile["column_summary"], use_container_width=True, hide_index=True)
    fig = missingness_bar(df)
    if fig is not None: st.plotly_chart(fig, use_container_width=True)
    else: st.success("No missing values were observed in the current dataset.")
    st.subheader("Explicit cleaning operation")
    st.warning("The raw dataset is preserved. Review the operation and confirm it before applying; transformations affect only the current processed copy and are logged.")
    op_labels = {
        "fill_missing": "Fill missing values", "drop_missing_rows": "Remove rows with missing values",
        "drop_duplicates": "Remove duplicate rows", "convert_numeric": "Convert to numeric",
        "convert_datetime": "Parse as datetime", "strip_whitespace": "Trim text whitespace",
        "normalize_categories": "Normalize categories (trim/casefold)", "clip_outliers": "Clip numeric outliers to chosen quantiles",
        "standardize_numeric": "Standardize numeric columns", "drop_columns": "Drop columns"}
    with st.form("cleaning_form"):
        operation = st.selectbox("Operation", list(op_labels), format_func=lambda key: op_labels[key])
        selected = st.multiselect("Columns (leave empty to use all relevant columns)", list(df.columns))
        method = None; value = None; params = {}
        if operation == "fill_missing":
            method = st.selectbox("Imputation method", ["mean", "median", "mode", "constant", "forward_fill", "backward_fill"])
            if method == "constant": value = st.text_input("Replacement value", value="Unknown")
        if operation == "normalize_categories":
            params["casefold"] = st.checkbox("Convert text to lowercase", value=True)
        if operation == "clip_outliers":
            c1,c2 = st.columns(2)
            params["lower_quantile"] = c1.number_input("Lower quantile", min_value=0.0, max_value=.49, value=.01, step=.01)
            params["upper_quantile"] = c2.number_input("Upper quantile", min_value=.51, max_value=1.0, value=.99, step=.01)
        confirmed = st.checkbox("I have reviewed the operation and understand its effect.")
        submit = st.form_submit_button("Apply transformation", type="primary")
    if submit:
        if not confirmed:
            st.error("Please confirm that you reviewed the operation before applying it.")
        elif operation == "drop_columns" and not selected:
            st.error("Select one or more columns before dropping columns.")
        else:
            try:
                new_df, entry = apply_cleaning(df, operation, columns=selected or None, method=method, value=value, params=params)
                st.session_state.processed_df = new_df
                st.session_state.transformation_log = st.session_state.get("transformation_log", []) + [entry]
                # Analytic outputs are tied to their input data; invalidate them after each mutation.
                st.session_state.generated_reports = None
                st.session_state.report_signature = None
                st.session_state.cluster_result = None
                st.session_state.anomaly_result = None
                st.session_state.mi_result = None
                st.session_state.cleaning_message = f"Applied: {entry['description']}"
                st.rerun()
            except Exception as exc:
                st.error(f"Transformation was not applied: {exc}")
    if st.session_state.get("cleaning_message"):
        st.success(st.session_state.cleaning_message)
        st.session_state.cleaning_message = ""
    st.subheader("Transformation log")
    logs = st.session_state.get("transformation_log", [])
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
    else: st.caption("No cleaning transformations have been applied in this session.")
    c1,c2 = st.columns(2)
    if c1.button("Restore original dataset", disabled=st.session_state.get("raw_df") is None):
        st.session_state.processed_df = st.session_state.raw_df.copy(deep=True)
        st.session_state.transformation_log = []
        st.session_state.generated_reports = None
        st.session_state.report_signature = None
        st.session_state.cluster_result = None
        st.session_state.anomaly_result = None
        st.session_state.mi_result = None
        st.session_state.cleaning_message = "Restored original data and cleared the transformation log."
        st.rerun()
    c2.download_button("Download transformation log JSON", data=json.dumps(logs, ensure_ascii=False, indent=2, default=str), file_name="transformation_log.json", mime="application/json", disabled=not bool(logs))

def render_exploration(df: pd.DataFrame | None):
    st.title("Exploratory Data Analysis")
    if df is None: st.info("Load a dataset first."); return
    profile = cached_profile(df, fingerprint(df))
    left, right = st.columns(2)
    left.metric("Rows × columns", f"{len(df):,} × {len(df.columns):,}")
    right.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
    st.subheader("Descriptive statistics across column types")
    st.dataframe(cached_descriptive_table(df, fingerprint(df)), use_container_width=True, hide_index=True)
    st.subheader("Explore a feature distribution")
    col = st.selectbox("Feature", list(df.columns))
    st.plotly_chart(distribution_chart(df, col), use_container_width=True)
    numeric = list(df.select_dtypes(include="number").columns)
    if len(numeric) >= 2:
        st.subheader("Relationship exploration")
        xcol, ycol, color_col = st.columns(3)
        x = xcol.selectbox("X axis", numeric, index=0, key="scatter_x")
        y_default = 1 if len(numeric) > 1 else 0
        y = ycol.selectbox("Y axis", numeric, index=y_default, key="scatter_y")
        color_options = ["(no color grouping)"] + [c for c in df.columns if c not in {x, y}]
        color = color_col.selectbox("Color by", color_options, key="scatter_color")
        max_points = int(st.session_state.get("analysis_chart_sample", 20000))
        st.plotly_chart(scatter_chart(df, x, y, None if color == "(no color grouping)" else color, max_points=max_points), use_container_width=True)
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols and numeric:
        st.subheader("Time-series exploration")
        tc1, tc2, tc3 = st.columns(3)
        date_col = tc1.selectbox("Datetime column", datetime_cols, key="timeseries_date")
        value_col = tc2.selectbox("Measurement", numeric, key="timeseries_value")
        group_options = ["(no grouping)"] + [c for c in df.columns if c not in {date_col, value_col}]
        group_choice = tc3.selectbox("Optional group", group_options, key="timeseries_group")
        figure = time_series_chart(df, date_col, value_col, None if group_choice == "(no grouping)" else group_choice)
        if figure is not None: st.plotly_chart(figure, use_container_width=True)
        else: st.info("No valid datetime/numeric observations were available for this chart.")
    st.subheader("Data quality warnings")
    for warning in profile["quality_warnings"]: st.warning(warning)
