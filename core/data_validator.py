"""Dataset validation and quality warnings."""
from __future__ import annotations
import pandas as pd

def validate_dataframe(df: pd.DataFrame) -> dict:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    warnings: list[str] = []
    if df.shape[0] == 0:
        warnings.append("The dataset contains no observations.")
    if df.shape[1] == 0:
        warnings.append("The dataset contains no columns.")
    if df.columns.duplicated().any():
        warnings.append("Duplicate column names exist; some downstream operations may be ambiguous.")
    if df.duplicated().any():
        warnings.append(f"{int(df.duplicated().sum()):,} duplicate rows were found; review before removing them.")
    missing_pct = df.isna().mean().mul(100) if len(df) else pd.Series(0.0, index=df.columns)
    for col, pct in missing_pct.items():
        if pct >= 50:
            warnings.append(f"{col!r} has {pct:.1f}% missing values.")
    for col in df.columns:
        non_null = df[col].dropna()
        if len(non_null) and non_null.nunique() <= 1:
            warnings.append(f"{col!r} is constant or contains only one distinct non-null value.")
    return {"valid": len(warnings) == 0, "warnings": warnings, "rows": int(len(df)), "columns": int(len(df.columns))}
