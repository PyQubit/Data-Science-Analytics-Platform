"""Exploratory profiling with explicit metrics and no opaque quality score."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

def profile_dataset(df: pd.DataFrame) -> dict:
    nrows, ncols = df.shape
    memory = int(df.memory_usage(index=True, deep=True).sum())
    missing_count = df.isna().sum()
    missing_pct = (missing_count / max(nrows, 1) * 100).round(2)
    column_rows = []
    for col in df.columns:
        s = df[col]
        nonnull = s.dropna()
        nunique = int(nonnull.nunique(dropna=True))
        dtype = str(s.dtype)
        row = {
            "column": str(col), "dtype": dtype, "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()), "missing_pct": float(missing_pct[col]),
            "unique": nunique, "unique_pct_non_null": round(nunique / max(int(s.notna().sum()), 1) * 100, 2),
            "memory_bytes": int(s.memory_usage(index=False, deep=True)),
            "constant": bool(nunique <= 1),
            "near_constant": bool(len(nonnull) > 0 and nunique > 1 and nonnull.value_counts(normalize=True, dropna=True).iloc[0] >= 0.98),
            "potential_identifier": bool(nrows >= 10 and nunique >= 0.95 * max(int(s.notna().sum()), 1) and nunique > 1),
        }
        if is_numeric_dtype(s) and not is_bool_dtype(s):
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().any():
                row.update({"mean": float(numeric.mean()), "median": float(numeric.median()),
                            "std": float(numeric.std()), "min": float(numeric.min()),
                            "max": float(numeric.max()), "q1": float(numeric.quantile(.25)),
                            "q3": float(numeric.quantile(.75)),
                            "skew": float(numeric.skew()) if numeric.notna().sum() > 2 else np.nan})
        elif is_datetime64_any_dtype(s):
            row.update({"min_date": str(s.min()) if s.notna().any() else None,
                        "max_date": str(s.max()) if s.notna().any() else None})
        elif is_bool_dtype(s):
            row["value_counts"] = {str(k): int(v) for k, v in s.value_counts(dropna=False).head(20).items()}
        else:
            row["top_values"] = {str(k)[:100]: int(v) for k, v in s.value_counts(dropna=True).head(10).items()}
        column_rows.append(row)
    numeric = df.select_dtypes(include=np.number)
    quality_warnings = []
    if nrows == 0: quality_warnings.append("No observations are available for analysis.")
    if df.isna().any().any(): quality_warnings.append(f"Missing values occur in {int(df.isna().any().sum())} column(s).")
    dup = int(df.duplicated().sum()) if nrows else 0
    if dup: quality_warnings.append(f"{dup:,} duplicate row(s) are present.")
    constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols: quality_warnings.append(f"Constant/one-value columns: {', '.join(constant_cols[:10])}.")
    high_missing = [str(c) for c in df.columns if nrows and df[c].isna().mean() >= .5]
    if high_missing: quality_warnings.append(f"At least 50% missing in: {', '.join(high_missing[:10])}.")
    return {
        "rows": int(nrows), "columns": int(ncols), "memory_bytes": memory,
        "duplicate_rows": dup, "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(float(df.isna().sum().sum() / max(nrows * ncols, 1) * 100), 2),
        "numeric_columns": [str(c) for c in numeric.columns],
        "categorical_columns": [str(c) for c in df.select_dtypes(include=["object", "category", "string", "bool"]).columns],
        "datetime_columns": [str(c) for c in df.columns if is_datetime64_any_dtype(df[c])],
        "column_summary": pd.DataFrame(column_rows), "quality_warnings": quality_warnings,
    }

def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return statistics for numeric and non-numeric columns in a single table."""
    rows = []
    for col in df.columns:
        s = df[col]
        row = {"column": str(col), "dtype": str(s.dtype), "count": int(s.notna().sum()),
               "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            num = pd.to_numeric(s, errors="coerce")
            row.update({"mean": num.mean(), "median": num.median(), "std": num.std(), "min": num.min(),
                        "q1": num.quantile(.25), "q3": num.quantile(.75), "max": num.max(),
                        "skew": num.skew(), "kurtosis": num.kurtosis()})
        else:
            mode = s.mode(dropna=True)
            row["mode"] = str(mode.iloc[0]) if not mode.empty else None
            row["top_frequency"] = int(s.value_counts(dropna=True).iloc[0]) if s.notna().any() else 0
        rows.append(row)
    return pd.DataFrame(rows, columns=["column", "dtype", "count", "missing", "unique", "mean", "median", "std", "min", "q1", "q3", "max", "skew", "kurtosis", "mode", "top_frequency"])
