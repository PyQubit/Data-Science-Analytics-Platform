"""Explicit, copy-preserving data cleaning operations with an audit log."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd

def apply_cleaning(df: pd.DataFrame, operation: str, columns: list[str] | None = None,
                   method: str | None = None, value: Any = None, params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply one reviewed transformation to a copy. Original ``df`` is never mutated."""
    params = params or {}
    out = df.copy(deep=True)
    selected = list(columns) if columns else list(out.columns)
    unknown = [c for c in selected if c not in out.columns]
    if unknown and operation not in {"drop_duplicates", "drop_missing_rows"}:
        raise ValueError(f"Unknown column(s): {unknown}")
    before_shape = out.shape
    before_missing = int(out.isna().sum().sum())
    description = operation
    affected_rows = 0
    affected_cols: set[str] = set()

    if operation == "drop_duplicates":
        subset = selected if columns else None
        mask = out.duplicated(subset=subset, keep="first")
        affected_rows = int(mask.sum())
        out = out.loc[~mask].copy()
        description = f"Removed {affected_rows} duplicate row(s)" + (f" using {selected}" if subset else " across all columns")
    elif operation == "drop_missing_rows":
        subset = selected if columns else None
        mask = out.isna().any(axis=1) if subset is None else out[subset].isna().any(axis=1)
        affected_rows = int(mask.sum())
        out = out.loc[~mask].copy()
        description = f"Removed {affected_rows} row(s) with missing values"
    elif operation == "fill_missing":
        if method not in {"mean", "median", "mode", "constant", "forward_fill", "backward_fill"}:
            raise ValueError("Choose mean, median, mode, constant, forward_fill, or backward_fill.")
        for col in selected:
            s = out[col]
            miss = int(s.isna().sum())
            if miss == 0: continue
            if method == "mean":
                if not pd.api.types.is_numeric_dtype(s): raise ValueError(f"Mean imputation requires a numeric column: {col}")
                fill_value = s.mean()
            elif method == "median":
                if not pd.api.types.is_numeric_dtype(s): raise ValueError(f"Median imputation requires a numeric column: {col}")
                fill_value = s.median()
            elif method == "mode":
                modes = s.mode(dropna=True)
                if modes.empty: continue
                fill_value = modes.iloc[0]
            elif method == "constant":
                fill_value = value if value is not None else "Unknown"
            elif method == "forward_fill":
                out[col] = s.ffill(); fill_value = None
            else:
                out[col] = s.bfill(); fill_value = None
            if method not in {"forward_fill", "backward_fill"}:
                out[col] = s.fillna(fill_value)
            affected_rows += miss - int(out[col].isna().sum())
            if miss: affected_cols.add(str(col))
        description = f"Imputed missing values using {method}"
    elif operation == "convert_numeric":
        for col in selected:
            original = out[col]
            converted = pd.to_numeric(original, errors="coerce")
            newly_invalid = original.notna() & converted.isna()
            affected_rows += int(newly_invalid.sum())
            if int((original.astype(str) != converted.astype(str)).sum()) > 0: affected_cols.add(str(col))
            out[col] = converted
        description = "Converted selected column(s) to numeric; unparseable non-null values became missing"
    elif operation == "convert_datetime":
        for col in selected:
            original = out[col]
            converted = pd.to_datetime(original, errors="coerce", utc=False)
            affected_rows += int((original.notna() & converted.isna()).sum())
            affected_cols.add(str(col))
            out[col] = converted
        description = "Parsed selected column(s) as datetime; unparseable non-null values became missing"
    elif operation == "strip_whitespace":
        for col in selected:
            if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
                original = out[col].copy()
                out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
                changed = int((original.astype(str) != out[col].astype(str)).sum())
                affected_rows += changed
                if changed: affected_cols.add(str(col))
        description = "Trimmed leading/trailing whitespace in text values"
    elif operation == "normalize_categories":
        for col in selected:
            if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]) or isinstance(out[col].dtype, pd.CategoricalDtype):
                original = out[col].copy()
                def normalize(x):
                    if not isinstance(x, str): return x
                    x = " ".join(x.strip().split())
                    return x.casefold() if params.get("casefold", True) else x
                out[col] = out[col].map(normalize)
                changed = int((original.astype(str) != out[col].astype(str)).sum())
                affected_rows += changed
                if changed: affected_cols.add(str(col))
        description = "Normalized category whitespace" + (" and case" if params.get("casefold", True) else "")
    elif operation == "drop_columns":
        out = out.drop(columns=selected)
        affected_cols.update(map(str, selected))
        description = f"Dropped {len(selected)} column(s)"
    elif operation == "clip_outliers":
        lower = float(params.get("lower_quantile", .01)); upper = float(params.get("upper_quantile", .99))
        if not (0 <= lower < upper <= 1): raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
        for col in selected:
            if not pd.api.types.is_numeric_dtype(out[col]): continue
            s = out[col]
            lo, hi = s.quantile(lower), s.quantile(upper)
            changed = int(((s < lo) | (s > hi)).sum())
            if changed:
                out[col] = s.clip(lo, hi); affected_rows += changed; affected_cols.add(str(col))
        description = f"Clipped numeric values to quantiles {lower:g}–{upper:g}; this changes values, not row count"
    elif operation == "standardize_numeric":
        for col in selected:
            if not pd.api.types.is_numeric_dtype(out[col]): raise ValueError(f"Scaling requires numeric columns: {col}")
            s = out[col].astype(float); sd = s.std(ddof=0)
            if sd and np.isfinite(sd): out[col] = (s - s.mean()) / sd; affected_cols.add(str(col))
        description = "Standardized selected numeric columns to mean 0 and population standard deviation 1"
    else:
        raise ValueError(f"Unsupported cleaning operation: {operation}")

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "operation": operation, "description": description, "columns": selected,
        "rows_before": int(before_shape[0]), "rows_after": int(out.shape[0]),
        "columns_before": int(before_shape[1]), "columns_after": int(out.shape[1]),
        "affected_rows": int(affected_rows), "affected_columns": sorted(affected_cols),
        "missing_cells_before": before_missing, "missing_cells_after": int(out.isna().sum().sum()),
        "parameters": {"method": method, "value": value, **params},
    }
    return out, entry
