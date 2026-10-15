"""Isolation Forest anomaly screening; flags are prompts for review, not confirmed errors."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def detect_anomalies(df: pd.DataFrame, features: list[str], contamination: float = 0.05,
                     random_state: int = 42) -> dict:
    if not 0 < contamination <= .5: raise ValueError("contamination must be in (0, 0.5]")
    if not features: raise ValueError("Select at least one numeric feature.")
    missing = [c for c in features if c not in df.columns]
    if missing: raise ValueError(f"Unknown feature(s): {missing}")
    nonnumeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if nonnumeric: raise ValueError(f"Anomaly detection requires numeric features: {nonnumeric}")
    x = df[features].replace([np.inf,-np.inf],np.nan)
    valid_cols = [c for c in x if x[c].notna().any() and x[c].nunique(dropna=True)>1]
    if not valid_cols: raise ValueError("No non-constant numeric features remain.")
    x = x[valid_cols]
    if len(x) < 10: raise ValueError("At least 10 observations are recommended for Isolation Forest.")
    transformed = SimpleImputer(strategy="median").fit_transform(x)
    transformed = StandardScaler().fit_transform(transformed)
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    flags = model.fit_predict(transformed)
    output = df.copy()
    output["anomaly_flag"] = flags == -1
    output["anomaly_score"] = -model.score_samples(transformed)
    return {"data": output, "features_used": list(x.columns), "anomaly_count": int((flags == -1).sum()),
            "anomaly_pct": float((flags == -1).mean()*100), "contamination": contamination,
            "limitation": "Isolation Forest flags observations that are unusual relative to this dataset and feature set; flags are not proof of data errors or rare-event causes."}
