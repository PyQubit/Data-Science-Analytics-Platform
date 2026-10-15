"""Small display-formatting helpers."""
from __future__ import annotations

def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB": return f"{value:.1f} {unit}"
        value /= 1024

def format_number(value, digits: int = 3) -> str:
    try:
        if value is None: return "—"
        number = float(value)
        if number != number: return "—"
        return f"{number:,.{digits}g}"
    except (TypeError, ValueError):
        return str(value)
