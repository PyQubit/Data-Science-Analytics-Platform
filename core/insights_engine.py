"""Deterministic, evidence-backed insight generation from observed dataset values."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def generate_insights(df: pd.DataFrame, max_insights: int = 20) -> list[dict]:
    if max_insights < 1: return []
    insights: list[dict] = []
    n, p = df.shape
    if p == 0 or n == 0:
        return [{"title": "Insufficient data", "category": "limitation", "finding": "The dataset has no observations or columns, so analytical insights cannot be computed.", "evidence": {"rows": n, "columns": p}, "sample_size": n, "limitation": "Load a non-empty tabular dataset."}]
    missing_pct = df.isna().mean() * 100
    for col, pct in missing_pct.sort_values(ascending=False).items():
        count = int(df[col].isna().sum())
        if count:
            insights.append({"title": f"Missingness in {col}", "category": "observed_fact",
                "finding": f"{count:,} of {n:,} rows ({pct:.2f}%) have a missing value in {col!r}.",
                "evidence": {"missing_count": count, "missing_percent": round(float(pct), 2)}, "sample_size": n,
                "limitation": "Missingness alone does not establish why values are absent or whether deletion/imputation is appropriate."})
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        insights.append({"title": "Duplicate records detected", "category": "observed_fact",
            "finding": f"{duplicate_count:,} row(s) ({duplicate_count / n * 100:.2f}%) exactly duplicate an earlier row across all columns.",
            "evidence": {"duplicate_rows": duplicate_count, "percent_rows": round(duplicate_count/n*100, 2)}, "sample_size": n,
            "limitation": "Duplicates may be valid repeated events; inspect domain meaning before removing them."})
    numeric = df.select_dtypes(include=np.number)
    for col in numeric.columns:
        s = pd.to_numeric(numeric[col], errors="coerce").dropna()
        if len(s) >= 8 and s.nunique() > 2:
            skew = float(s.skew())
            if np.isfinite(skew) and abs(skew) >= 1:
                direction = "right" if skew > 0 else "left"
                insights.append({"title": f"Skewed distribution: {col}", "category": "statistical_interpretation",
                    "finding": f"{col!r} shows a {direction}-skewed sample distribution (sample skewness {skew:.2f}).",
                    "evidence": {"sample_skewness": round(skew, 3), "median": float(s.median()), "mean": float(s.mean()), "n_non_missing": int(len(s))},
                    "sample_size": int(len(s)), "limitation": "Skewness is a descriptive statistic; it does not by itself identify a data problem or justify transformation."})
    if numeric.shape[1] >= 2:
        cols = list(numeric.columns[:100])
        corr = numeric[cols].corr(method="spearman", min_periods=max(5, min(20, n//5)))
        if len(cols) > 1:
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().dropna()
            upper = upper[upper.abs() >= .7].sort_values(key=lambda x: x.abs(), ascending=False)
            for (x, y), value in upper.head(5).items():
                pair = numeric[[x, y]].dropna()
                insights.append({"title": f"Strong rank association: {x} and {y}", "category": "observed_association",
                    "finding": f"{x!r} and {y!r} have a Spearman rank correlation of {value:.2f} across {len(pair):,} complete rows.",
                    "evidence": {"spearman_rho": round(float(value), 4), "complete_pair_rows": int(len(pair))}, "sample_size": int(len(pair)),
                    "limitation": "Association is not causation; outliers, confounding, time dependence, and data collection may affect the relationship."})
    for col in df.select_dtypes(include=["object", "category", "string", "bool"]).columns:
        s = df[col].dropna()
        if len(s) >= 10 and s.nunique() > 1:
            counts = s.value_counts(dropna=True)
            top_value = counts.index[0]; share = float(counts.iloc[0] / len(s) * 100)
            if share >= 50:
                insights.append({"title": f"Dominant category: {col}", "category": "observed_fact",
                    "finding": f"The most frequent value in {col!r} is {str(top_value)[:80]!r}, representing {share:.1f}% of non-missing values.",
                    "evidence": {"top_value": str(top_value), "count": int(counts.iloc[0]), "share_percent": round(share, 2), "non_missing_n": int(len(s))},
                    "sample_size": int(len(s)), "limitation": "A dominant category may be expected from the sampling design; consider the context before acting on it."})
    for col in numeric.columns:
        s = pd.to_numeric(numeric[col], errors="coerce").dropna()
        if len(s) >= 10 and s.nunique() > 2:
            q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3-q1
            if iqr > 0:
                count = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
                if count:
                    insights.append({"title": f"Potential IQR outliers: {col}", "category": "follow_up",
                        "finding": f"{count:,} value(s) in {col!r} fall outside the conventional 1.5×IQR screening fences.",
                        "evidence": {"potential_outliers": count, "n_non_missing": int(len(s)), "q1": float(q1), "q3": float(q3), "iqr": float(iqr)},
                        "sample_size": int(len(s)), "limitation": "These are statistical flags, not confirmed errors; skewed distributions may naturally produce many flags."})
    priority = {"observed_fact": 0, "observed_association": 1, "statistical_interpretation": 2, "follow_up": 3}
    insights.sort(key=lambda item: (priority.get(item["category"], 4), -(item.get("sample_size") or 0)))
    return insights[:max_insights]
