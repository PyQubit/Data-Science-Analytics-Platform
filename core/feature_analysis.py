"""Feature association and held-out, model-dependent permutation importance."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

def _target_task(series: pd.Series) -> str:
    distinct = int(series.nunique(dropna=True))
    threshold = min(20, max(2, int(np.sqrt(max(len(series), 1)))))
    return "classification" if (not pd.api.types.is_numeric_dtype(series) or distinct <= threshold) else "regression"

def _mutual_information_for_feature(feature: pd.Series, target: pd.Series, task: str, random_state: int = 42) -> float:
    pair = pd.DataFrame({"feature": feature, "target": target}).dropna(subset=["target"]).copy()
    if len(pair) < 5 or pair["target"].nunique(dropna=True) < 2:
        return np.nan
    is_discrete = not (pd.api.types.is_numeric_dtype(pair["feature"]) and not pd.api.types.is_bool_dtype(pair["feature"]))
    if is_discrete:
        encoded, _ = pd.factorize(pair["feature"].astype("string").fillna("(missing)"), sort=True)
        x_values = encoded.astype(float)
    else:
        x_values = pd.to_numeric(pair["feature"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if not x_values.notna().any(): return np.nan
        x_values = x_values.fillna(x_values.median()).to_numpy(dtype=float)
    if len(np.unique(x_values)) < 2:
        return np.nan
    if task == "classification":
        y_encoded, _ = pd.factorize(pair["target"].astype("string"), sort=True)
        if len(np.unique(y_encoded)) < 2: return np.nan
        score = mutual_info_classif(np.asarray(x_values).reshape(-1, 1), y_encoded,
                                    discrete_features=[is_discrete], random_state=random_state)
    else:
        y_values = pd.to_numeric(pair["target"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y_values) & np.isfinite(np.asarray(x_values, dtype=float))
        if int(valid.sum()) < 5 or len(np.unique(y_values[valid])) < 2: return np.nan
        score = mutual_info_regression(np.asarray(x_values, dtype=float)[valid].reshape(-1, 1), y_values[valid],
                                       discrete_features=[is_discrete], random_state=random_state)
    return float(score[0])

def feature_summary(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    rows = []
    valid_target = target if target and target in df.columns else None
    target_kind = _target_task(df[valid_target].dropna()) if valid_target and df[valid_target].notna().any() else None
    for col in df.columns:
        s = df[col]
        row = {"feature": str(col), "dtype": str(s.dtype), "non_null": int(s.notna().sum()),
               "missing_pct": float(s.isna().mean()*100), "unique": int(s.nunique(dropna=True)),
               "variance": float(s.var()) if pd.api.types.is_numeric_dtype(s) and s.notna().sum()>1 else np.nan,
               "constant": bool(s.nunique(dropna=True) <= 1)}
        if valid_target and col != valid_target:
            if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(df[valid_target]):
                pair = pd.DataFrame({"x": s, "y": df[valid_target]}).apply(pd.to_numeric, errors="coerce").dropna()
                row["pearson_with_target"] = float(pair.corr().iloc[0,1]) if len(pair) >= 3 and pair.x.nunique()>1 and pair.y.nunique()>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def mutual_information_analysis(df: pd.DataFrame, target: str, features: list[str] | None = None,
                               random_state: int = 42, max_rows: int = 20000) -> dict:
    """Compute univariate feature-target mutual information on a deterministic sample if needed."""
    if target not in df.columns: raise ValueError("Target column does not exist")
    selected = list(dict.fromkeys(c for c in (features if features is not None else df.columns) if c in df.columns and c != target))
    if not selected: raise ValueError("Select at least one predictor column other than the target.")
    if len(selected) > 200: raise ValueError("Mutual information is limited to 200 selected predictor columns per run.")
    work = df[[target] + selected].dropna(subset=[target]).copy()
    if len(work) < 5: raise ValueError("At least five rows with a non-missing target are required.")
    rows_sampled = len(work)
    if max_rows < 5: raise ValueError("max_rows must be at least 5")
    if len(work) > max_rows:
        work = work.sample(n=max_rows, random_state=random_state)
    y = work[target]
    task = _target_task(y)
    results = []
    for feature in selected:
        value = _mutual_information_for_feature(work[feature], y, task, random_state=random_state)
        results.append({"feature": str(feature), "mutual_information": value,
                        "target": str(target), "target_task": task, "method": "Univariate mutual information (approximate nats)",
                        "rows_used": int(len(work))})
    scores = pd.DataFrame(results).sort_values("mutual_information", ascending=False, na_position="last").reset_index(drop=True)
    return {"target": target, "target_task": task, "features_scored": selected, "rows_available": rows_sampled,
            "rows_used": int(len(work)), "sampled": bool(rows_sampled > len(work)), "scores": scores,
            "limitation": "Mutual information is a univariate, sample-dependent association score in approximate nats. It does not establish causation and may miss interactions or be biased for high-cardinality features."}

def analyze_features(df: pd.DataFrame, target: str, task: str = "auto", test_size: float = .25,
                     random_state: int = 42, run_permutation: bool = True) -> dict:
    if target not in df.columns: raise ValueError("Target column does not exist")
    if not 0.1 <= test_size <= 0.5: raise ValueError("test_size must be between 0.1 and 0.5")
    work = df.dropna(subset=[target]).copy()
    y = work[target]
    if len(work) < 12: raise ValueError("At least 12 rows with a non-missing target are recommended for held-out feature analysis.")
    if task == "auto":
        task = _target_task(y)
    if task not in {"classification", "regression"}: raise ValueError("task must be auto, classification, or regression")
    if task == "classification":
        y = y.astype(str)
        if y.nunique() < 2: raise ValueError("Classification target must contain at least two classes.")
        if y.value_counts().min() < 2: raise ValueError("Every classification class needs at least two rows for a train/test split.")
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna(); work = work.loc[valid].copy(); y = y.loc[valid]
        if len(y) < 12 or y.nunique() < 2: raise ValueError("Regression target needs at least 12 valid, non-constant numeric values.")
    X = work.drop(columns=[target]).copy()  # Explicitly prevents target leakage.
    if X.shape[1] == 0: raise ValueError("At least one predictor column other than the target is required.")
    # Ignore fully missing and constant predictors to avoid ill-defined model preprocessing.
    keep = [c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
    X = X[keep]
    if X.shape[1] == 0: raise ValueError("No non-constant predictor columns remain after validation.")
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    # Normalize heterogeneous categorical scalars to text on the model copy only.
    # This avoids mixed int/string comparisons inside categorical imputers/encoders.
    for column in cat_cols:
        X[column] = X[column].map(lambda value: np.nan if pd.isna(value) else str(value))
    transformers = []
    if num_cols:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols))
    if cat_cols:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                     ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, max_categories=50))]), cat_cols))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1) if task == "classification" else RandomForestRegressor(n_estimators=150, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    stratify = y if task == "classification" and y.value_counts().min() >= 3 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    if task == "classification":
        metrics = {"accuracy": float(accuracy_score(y_test, predictions)), "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                   "classes": int(y.nunique())}
        permutation_scoring = "f1_macro"
    else:
        metrics = {"mae": float(mean_absolute_error(y_test, predictions)), "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                   "r2": float(r2_score(y_test, predictions)) if len(y_test)>1 else np.nan}
        permutation_scoring = "r2"
    importance_df = pd.DataFrame(columns=["feature", "permutation_importance_mean", "permutation_importance_std"])
    if run_permutation:
        perm = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=random_state, scoring=permutation_scoring, n_jobs=-1)
        importance_df = pd.DataFrame({"feature": X.columns.astype(str), "permutation_importance_mean": perm.importances_mean,
                                      "permutation_importance_std": perm.importances_std}).sort_values("permutation_importance_mean", ascending=False).reset_index(drop=True)
    return {"target": target, "task": task, "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
            "predictors": list(map(str, X.columns)), "metrics": metrics, "importance": importance_df,
            "limitation": "Permutation importance is model- and holdout-dependent; it indicates predictive contribution under this setup, not causation."}
