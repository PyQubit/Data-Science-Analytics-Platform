"""Central application defaults."""
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    app_name: str = "Data Science Analytics"
    max_upload_mb: int = int(os.getenv("DSA_MAX_UPLOAD_MB", "200"))
    preview_rows: int = int(os.getenv("DSA_PREVIEW_ROWS", "100"))
    chart_sample_rows: int = int(os.getenv("DSA_CHART_SAMPLE_ROWS", "20000"))
    significance_level: float = float(os.getenv("DSA_ALPHA", "0.05"))
    random_state: int = int(os.getenv("DSA_RANDOM_STATE", "42"))

SETTINGS = Settings()
