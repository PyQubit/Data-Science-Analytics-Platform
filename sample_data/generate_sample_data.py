"""Generate the bundled synthetic retail data used for smoke tests and walkthroughs."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(seed: int = 42, rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=rows, freq="D")
    regions = rng.choice(["North", "South", "East", "West"], rows, p=[.28,.22,.25,.25]).astype(object)
    channels = rng.choice(["Online", "Store", "Partner"], rows, p=[.5,.35,.15]).astype(object)
    segments = rng.choice(["Consumer", "Small Business", "Enterprise"], rows, p=[.6,.28,.12]).astype(object)
    units = rng.integers(1, 30, rows)
    price = np.round(rng.uniform(8, 160, rows), 2)
    discount = rng.choice([0.0, .05, .10, .15, .20], rows, p=[.45,.15,.2,.12,.08])
    revenue = np.round(units * price * (1-discount), 2)
    returned = rng.choice([False, True], rows, p=[.93,.07])
    df = pd.DataFrame({"order_date": dates, "region": regions, "channel": channels, "customer_segment": segments,
                       "units": units, "unit_price": price, "discount": discount, "revenue": revenue, "returned": returned})
    # Deliberately included for cleaning-workflow demonstration; synthetic sample only.
    df.loc[rng.choice(rows, size=7, replace=False), "region"] = None
    df.loc[rng.choice(rows, size=5, replace=False), "unit_price"] = np.nan
    return df

if __name__ == "__main__":
    out = Path(__file__).with_name("retail_sample.csv")
    generate().to_csv(out, index=False)
    print(f"Wrote {out}")
