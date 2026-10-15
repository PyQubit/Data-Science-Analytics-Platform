"""Descriptive, association, and selected inferential statistics."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

def correlation_analysis(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be pearson, spearman, or kendall")
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] > 200:
        raise ValueError("Correlation matrix exceeds the 200-column safety limit; select a smaller subset of features first.")
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(method=method, min_periods=2)

def correlation_detail(df: pd.DataFrame, x: str, y: str, method: str = "pearson") -> dict:
    if x not in df or y not in df: raise ValueError("Selected column does not exist")
    if method not in {"pearson", "spearman", "kendall"}: raise ValueError("Unsupported correlation method")
    pair = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(pair)
    if n < 3: raise ValueError("At least 3 complete numeric observations are required")
    if pair[x].nunique() < 2 or pair[y].nunique() < 2:
        raise ValueError("Both selected variables must vary; correlation is undefined for a constant variable.")
    fn = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    result = fn(pair[x].to_numpy(), pair[y].to_numpy())
    corr = float(result.statistic if hasattr(result, "statistic") else result[0])
    pvalue = float(result.pvalue if hasattr(result, "pvalue") else result[1])
    confidence_interval = None
    if method == "pearson" and n > 3 and abs(corr) < 1:
        z = np.arctanh(corr)
        margin = stats.norm.ppf(.975) / np.sqrt(n - 3)
        confidence_interval = [float(np.tanh(z - margin)), float(np.tanh(z + margin))]
    return {"test": f"{method.title()} correlation", "x": x, "y": y, "n": n,
            "statistic": corr, "p_value": pvalue, "confidence_level": 0.95 if confidence_interval is not None else None,
            "confidence_interval_95": confidence_interval,
            "null_hypothesis": "Population association/correlation is zero.",
            "alternative_hypothesis": "Population association/correlation is non-zero.",
            "interpretation": "Association only; this result does not establish causation."}

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or len(b) < 2: return None
    pooled_var = ((len(a)-1)*np.var(a, ddof=1) + (len(b)-1)*np.var(b, ddof=1)) / (len(a)+len(b)-2)
    if not np.isfinite(pooled_var) or pooled_var <= 0: return 0.0 if np.mean(a) == np.mean(b) else None
    return float((np.mean(a)-np.mean(b))/np.sqrt(pooled_var))

def hypothesis_test(df: pd.DataFrame, test_type: str, x: str, y: str,
                    alpha: float = 0.05, group: str | None = None) -> dict:
    """Run a deliberately selected test and return assumptions/interpretation."""
    if not 0 < alpha < 1: raise ValueError("alpha must be strictly between 0 and 1")
    if x not in df.columns or y not in df.columns: raise ValueError("Selected column does not exist")
    if test_type in {"pearson", "spearman", "kendall"}:
        result = correlation_detail(df, x, y, test_type)
        result["alpha"] = alpha
        result["significant_at_alpha"] = bool(result["p_value"] < alpha)
        result["conclusion"] = "Reject the null hypothesis at the selected alpha." if result["p_value"] < alpha else "Insufficient evidence to reject the null hypothesis at the selected alpha."
        return result
    if test_type in {"welch_t", "mann_whitney"}:
        # x = numeric outcome, y = two-level grouping variable.
        numeric = pd.to_numeric(df[x], errors="coerce")
        groups = df[y]
        levels = list(groups.dropna().unique())
        if len(levels) != 2: raise ValueError(f"{y!r} must contain exactly two non-missing groups for this test (found {len(levels)}).")
        a = numeric.loc[groups == levels[0]].dropna().to_numpy(dtype=float)
        b = numeric.loc[groups == levels[1]].dropna().to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2: raise ValueError("Each group needs at least two numeric observations.")
        if test_type == "welch_t":
            result = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            statistic, pvalue = float(result.statistic), float(result.pvalue)
            effect = _cohens_d(a, b)
            assumptions = "Independent observations; numeric outcome; Welch test does not assume equal group variances. Check sampling/design and inspect distributions for severe issues."
            name = "Welch two-sample t-test"
        else:
            result = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
            statistic, pvalue = float(result.statistic), float(result.pvalue)
            # Rank-biserial correlation based on U, defined for the first group.
            effect = float(2 * statistic / (len(a)*len(b)) - 1)
            assumptions = "Independent observations and an ordinal/continuous outcome. A location-shift interpretation additionally depends on similarly shaped distributions."
            name = "Mann–Whitney U test"
        n = len(a) + len(b)
        difference_ci = None
        mean_difference = float(np.mean(a) - np.mean(b))
        if test_type == "welch_t":
            variance_a = np.var(a, ddof=1) / len(a)
            variance_b = np.var(b, ddof=1) / len(b)
            standard_error = float(np.sqrt(variance_a + variance_b))
            denominator = (variance_a**2 / (len(a)-1)) + (variance_b**2 / (len(b)-1))
            degrees = float((variance_a + variance_b)**2 / denominator) if denominator > 0 else float("inf")
            margin = float(stats.t.ppf(.975, degrees) * standard_error)
            difference_ci = [mean_difference - margin, mean_difference + margin]
        out = {"test": name, "x": x, "y": y, "groups": {str(levels[0]): len(a), str(levels[1]): len(b)},
               "n": n, "statistic": statistic, "p_value": pvalue, "effect_size": effect,
               "effect_size_name": "Cohen's d (group 1 minus group 2)" if test_type == "welch_t" else "Rank-biserial correlation (orientation follows group 1)",
               "mean_difference_group1_minus_group2": mean_difference,
               "confidence_level": 0.95 if difference_ci is not None else None,
               "mean_difference_confidence_interval_95": difference_ci,
               "group_means": {str(levels[0]): float(np.mean(a)), str(levels[1]): float(np.mean(b))},
               "null_hypothesis": "The two population distributions/means are equal under the test's assumptions.",
               "alternative_hypothesis": "The two populations differ (two-sided).", "assumptions": assumptions}
    elif test_type == "chi_square":
        contingency = pd.crosstab(df[x], df[y], dropna=True)
        if contingency.shape[0] < 2 or contingency.shape[1] < 2: raise ValueError("Chi-square requires at least two observed categories in each variable.")
        chi2, pvalue, dof, expected = stats.chi2_contingency(contingency, correction=False)
        n = int(contingency.to_numpy().sum())
        denom = n * min(contingency.shape[0]-1, contingency.shape[1]-1)
        cramer_v = float(math.sqrt(chi2 / denom)) if denom > 0 else None
        low_expected = int((expected < 5).sum())
        out = {"test": "Chi-square test of independence", "x": x, "y": y, "n": n,
               "statistic": float(chi2), "p_value": float(pvalue), "degrees_of_freedom": int(dof),
               "effect_size": cramer_v, "effect_size_name": "Cramér's V",
               "contingency_table": contingency, "expected_cells_below_5": low_expected,
               "null_hypothesis": "The categorical variables are independent.",
               "alternative_hypothesis": "The categorical variables are associated.",
               "assumptions": "Independent observations and mutually exclusive categories. A large number of expected cell counts below 5 weakens the usual chi-square approximation."}
    elif test_type == "normality_shapiro":
        vals = pd.to_numeric(df[x], errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 3: raise ValueError("Shapiro–Wilk requires at least 3 non-missing numeric observations.")
        if len(vals) > 5000:
            rng = np.random.default_rng(42); vals = rng.choice(vals, 5000, replace=False)
        stat, pvalue = stats.shapiro(vals)
        out = {"test": "Shapiro–Wilk normality test", "x": x, "n": len(vals), "statistic": float(stat),
               "p_value": float(pvalue), "null_hypothesis": "The sample is consistent with a normal distribution.",
               "alternative_hypothesis": "The sample departs from a normal distribution.",
               "assumptions": "Independent observations; for samples larger than 5,000, a deterministic random sample of 5,000 values is tested."}
    else:
        raise ValueError("Unsupported test type")
    out["alpha"] = alpha
    out["significant_at_alpha"] = bool(out["p_value"] < alpha)
    out["conclusion"] = "Reject the null hypothesis at the selected alpha." if out["p_value"] < alpha else "Insufficient evidence to reject the null hypothesis at the selected alpha."
    if out["p_value"] < alpha and test_type == "chi_square" and out.get("expected_cells_below_5", 0):
        out["caution"] = "Some expected cell counts are below 5; consider an exact/Monte Carlo method or combining categories only when substantively justified."
    return out

def mean_confidence_interval(values, confidence: float = 0.95) -> dict:
    """Student-t confidence interval for the population mean from independent numeric observations."""
    if not 0 < confidence < 1: raise ValueError("confidence must be strictly between 0 and 1")
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(array) < 2: raise ValueError("At least two non-missing observations are required")
    mean = float(np.mean(array)); se = float(stats.sem(array))
    margin = float(stats.t.ppf((1 + confidence) / 2, len(array)-1) * se)
    return {"n": int(len(array)), "mean": mean, "confidence_level": confidence,
            "lower": mean - margin, "upper": mean + margin, "standard_error": se,
            "method": "Two-sided Student-t interval; assumes independent observations and appropriate sampling."}

def adjust_pvalues(p_values: list[float], method: str = "fdr_bh", alpha: float = 0.05) -> dict:
    if not p_values: return {"adjusted_p_values": [], "reject": []}
    reject, corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return {"adjusted_p_values": corrected.tolist(), "reject": reject.tolist(), "method": method, "alpha": alpha}
