"""Numeric clustering and PCA projections with transparent input requirements."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def _prepare_numeric(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    if not features: raise ValueError("Select at least one numeric feature.")
    missing = [c for c in features if c not in df.columns]
    if missing: raise ValueError(f"Unknown feature(s): {missing}")
    nonnumeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if nonnumeric: raise ValueError(f"This module currently requires numeric features; non-numeric: {nonnumeric}")
    x = df[features].replace([np.inf, -np.inf], np.nan)
    valid_cols = [c for c in x if x[c].notna().any() and x[c].nunique(dropna=True)>1]
    if not valid_cols: raise ValueError("Selected features are empty or constant after validation.")
    x = x[valid_cols]
    array = SimpleImputer(strategy="median").fit_transform(x)
    scaled = StandardScaler().fit_transform(array)
    return x, scaled

def discover_clusters(df: pd.DataFrame, features: list[str], algorithm: str = "kmeans", n_clusters: int = 3,
                      eps: float = 0.5, min_samples: int = 5, random_state: int = 42) -> dict:
    x, scaled = _prepare_numeric(df, features)
    if len(x) < 4: raise ValueError("At least 4 rows are required for clustering.")
    if algorithm == "kmeans":
        if not 2 <= n_clusters < len(x): raise ValueError("K-means cluster count must be at least 2 and less than row count.")
        estimator = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = estimator.fit_predict(scaled)
    elif algorithm == "dbscan":
        if eps <= 0 or min_samples < 2: raise ValueError("DBSCAN eps must be positive and min_samples at least 2.")
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(scaled)
    else: raise ValueError("algorithm must be kmeans or dbscan")
    non_noise = labels != -1
    distinct = set(labels[non_noise])
    silhouette = None
    if len(distinct) >= 2 and int(non_noise.sum()) > len(distinct) and len(x) <= 20000:
        try: silhouette = float(silhouette_score(scaled[non_noise], labels[non_noise]))
        except ValueError: silhouette = None
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(scaled) if scaled.shape[1] >= 2 else np.column_stack([scaled[:,0], np.zeros(len(scaled))])
    output = df.copy()
    output["cluster"] = labels
    output["pca_1"] = coords[:,0]; output["pca_2"] = coords[:,1]
    return {"data": output, "algorithm": algorithm, "features_used": list(x.columns),
            "cluster_counts": {str(int(k)): int(v) for k,v in pd.Series(labels).value_counts().sort_index().items()},
            "silhouette_score": silhouette, "explained_variance_ratio": pca.explained_variance_ratio_.tolist() if scaled.shape[1] >= 2 else [1.0, 0.0],
            "limitation": "Clusters summarize geometric structure in the selected, standardized numeric features; they are not confirmed real-world categories."}
