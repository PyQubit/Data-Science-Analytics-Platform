"""Safe downloadable text exports for spreadsheet-friendly clients."""
from __future__ import annotations
import pandas as pd

def safe_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export CSV while prefixing formula-like text cells to reduce spreadsheet injection risk.

    Numeric values remain numeric. The leading apostrophe is an intentional safety escape for
    text values that begin with =, +, -, @, tab, or carriage-return; those values differ from
    their in-memory representation in the downloaded CSV and are safer to open in spreadsheets.
    """
    safe = df.copy()
    def escape_text(value):
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value
    safe.columns = [escape_text(str(column)) for column in safe.columns]
    for column in safe.columns:
        if pd.api.types.is_object_dtype(safe[column]) or pd.api.types.is_string_dtype(safe[column]):
            safe[column] = safe[column].map(escape_text)
    return safe.to_csv(index=False).encode("utf-8-sig")
