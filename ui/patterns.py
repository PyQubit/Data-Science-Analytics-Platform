from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
from core.pattern_discovery import discover_clusters
from core.anomaly_detection import detect_anomalies
from utils.streamlit_cache import fingerprint
from utils.exports import safe_csv_bytes

def render_patterns(df: pd.DataFrame | None, random_state: int = 42):
    st.title("Pattern Discovery and Anomaly Detection")
    if df is None: st.info("Load a dataset first."); return
    numeric = list(df.select_dtypes(include=np.number).columns)
    if not numeric:
        st.warning("Clustering and Isolation Forest currently require numeric features. Convert eligible columns in Data Quality and Cleaning first."); return
    st.subheader("Clustering")
    with st.form("cluster_form"):
        features = st.multiselect("Numeric features", numeric, default=numeric[:min(4,len(numeric))], key="cluster_features")
        algorithm = st.selectbox("Algorithm", ["kmeans", "dbscan"])
        n_clusters = st.slider("K-means cluster count", 2, max(2, min(12, max(2,len(df)-1))), min(3, max(2, len(df)-1)))
        eps = st.number_input("DBSCAN eps (standardized distance)", min_value=.05, max_value=10.0, value=.5, step=.05)
        min_samples = st.number_input("DBSCAN min_samples", min_value=2, max_value=100, value=5, step=1)
        run_cluster = st.form_submit_button("Discover clusters", type="primary")
    if run_cluster:
        try:
            with st.spinner("Clustering selected numeric features..."):
                result = discover_clusters(df, features, algorithm, n_clusters, eps, int(min_samples), random_state)
            st.session_state.cluster_result = {"signature": (fingerprint(df), tuple(features), algorithm, int(n_clusters), float(eps), int(min_samples), random_state), "result": result}
        except Exception as exc: st.error(f"Clustering could not be completed: {exc}")
    saved_cluster = st.session_state.get("cluster_result")
    cluster_signature = (fingerprint(df), tuple(features), algorithm, int(n_clusters), float(eps), int(min_samples), random_state)
    result = saved_cluster.get("result") if saved_cluster and saved_cluster.get("signature") == cluster_signature else None
    if saved_cluster and result is None:
        st.caption("Clustering parameters or dataset changed. Run clustering again to refresh the result.")
    if result:
        st.write({"algorithm":result["algorithm"], "features_used":result["features_used"], "cluster_counts":result["cluster_counts"], "silhouette_score":result["silhouette_score"], "explained_variance_ratio":result["explained_variance_ratio"]})
        import plotly.express as px
        fig = px.scatter(result["data"], x="pca_1", y="pca_2", color=result["data"]["cluster"].astype(str), hover_data=[c for c in result["features_used"][:5]], title="Cluster projection (PCA / first feature if only one input)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(result["limitation"])
        st.download_button("Download clustered data CSV", safe_csv_bytes(result["data"]), "clustered_data.csv", "text/csv", help="Formula-like text values are escaped for spreadsheet safety.")
    st.divider()
    st.subheader("Anomaly screening")
    with st.form("anomaly_form"):
        anomaly_features = st.multiselect("Features for anomaly model", numeric, default=numeric[:min(4,len(numeric))], key="anomaly_features")
        contamination = st.slider("Expected anomaly fraction / sensitivity setting", min_value=.01, max_value=.5, value=.05, step=.01)
        run_anomaly = st.form_submit_button("Run Isolation Forest", type="primary")
    if run_anomaly:
        try:
            with st.spinner("Fitting Isolation Forest..."):
                result = detect_anomalies(df, anomaly_features, contamination, random_state)
            st.session_state.anomaly_result = {"signature": (fingerprint(df), tuple(anomaly_features), float(contamination), random_state), "result": result}
        except Exception as exc: st.error(f"Anomaly detection could not be completed: {exc}")
    saved_anomaly = st.session_state.get("anomaly_result")
    anomaly_signature = (fingerprint(df), tuple(anomaly_features), float(contamination), random_state)
    anomaly = saved_anomaly.get("result") if saved_anomaly and saved_anomaly.get("signature") == anomaly_signature else None
    if saved_anomaly and anomaly is None:
        st.caption("Anomaly-detection parameters or dataset changed. Run the analysis again to refresh the result.")
    if anomaly:
        c1,c2 = st.columns(2); c1.metric("Flagged rows", f"{anomaly['anomaly_count']:,}"); c2.metric("Flagged share", f"{anomaly['anomaly_pct']:.2f}%")
        st.dataframe(anomaly["data"].sort_values("anomaly_score", ascending=False).head(100), use_container_width=True, hide_index=True)
        st.caption(anomaly["limitation"])
        st.download_button("Download anomaly results CSV", safe_csv_bytes(anomaly["data"]), "anomaly_results.csv", "text/csv", help="Formula-like text values are escaped for spreadsheet safety.")
