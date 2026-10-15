"""Dataset fingerprint helpers for explicit cache-key construction."""
import hashlib
import pandas as pd

def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """A stable digest for invalidating cached results when values/schema change."""
    digest = hashlib.sha256()
    digest.update(repr(tuple(map(str, df.columns))).encode("utf-8"))
    digest.update(repr(tuple(map(str, df.dtypes))).encode("utf-8"))
    digest.update(str(df.shape).encode("ascii"))
    try:
        hashed = pd.util.hash_pandas_object(df, index=True, categorize=True).values
        digest.update(hashed.tobytes())
    except (TypeError, ValueError):
        digest.update(df.to_json(orient="split", date_format="iso", default_handler=str).encode("utf-8"))
    return digest.hexdigest()
