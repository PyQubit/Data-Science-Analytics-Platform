import numpy as np
import pandas as pd
import pytest
from core.feature_analysis import analyze_features, feature_summary, mutual_information_analysis
from core.pattern_discovery import discover_clusters
from core.anomaly_detection import detect_anomalies

def make_data(n=80):
    rng = np.random.default_rng(10)
    x = rng.normal(size=n)
    return pd.DataFrame({"signal":x, "noise":rng.normal(size=n), "category":np.where(x>0,"high","low"), "target":(x*3+rng.normal(scale=.2,size=n))})

def test_feature_model_excludes_target_and_returns_holdout_metrics():
    df = make_data()
    result = analyze_features(df, "target", task="regression", random_state=1)
    assert "target" not in result["predictors"]
    assert result["test_rows"] > 0
    assert "rmse" in result["metrics"]
    assert set(result["importance"].columns) >= {"feature","permutation_importance_mean"}

def test_feature_summary_includes_missingness_and_mutual_information():
    rng = np.random.default_rng(7)
    x = rng.normal(size=60)
    df = pd.DataFrame({"a":x, "category":np.where(x>0,"high","low"), "target":x*2+rng.normal(size=60)})
    df.loc[0, "a"] = np.nan
    row = feature_summary(df, "target").set_index("feature").loc["a"]
    assert row["missing_pct"] == pytest.approx(100/60)
    mi = mutual_information_analysis(df, "target", features=["a", "category"])
    assert mi["scores"].iloc[0]["mutual_information"] >= 0
    assert mi["rows_used"] == 60

def test_cluster_and_anomaly_edges():
    df = make_data(40)
    cluster = discover_clusters(df, ["signal","noise"], n_clusters=3)
    assert "cluster" in cluster["data"]
    anomaly = detect_anomalies(df, ["signal","noise"], contamination=.1)
    assert "anomaly_flag" in anomaly["data"]
    assert anomaly["anomaly_count"] > 0
