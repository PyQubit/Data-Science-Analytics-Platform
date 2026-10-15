"""Reusable validation functions."""
from pathlib import Path

def validate_extension(filename: str, allowed: set[str]) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise ValueError(f"Unsupported file extension {ext!r}.")

def validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not 0 < alpha < 1: raise ValueError("Significance level must be strictly between 0 and 1.")
    return alpha
