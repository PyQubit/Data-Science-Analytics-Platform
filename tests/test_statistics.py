import numpy as np
import pandas as pd
import pytest
from core.statistical_analysis import correlation_detail, hypothesis_test, adjust_pvalues

def test_pearson_correlation_known_result():
    df = pd.DataFrame({"x":[1,2,3,4,5], "y":[2,4,6,8,10]})
    result = correlation_detail(df, "x", "y", "pearson")
    assert result["n"] == 5
    assert result["statistic"] == pytest.approx(1.0)
    assert result["p_value"] < 1e-5

def test_welch_t_test_returns_effect_size():
    df = pd.DataFrame({"v":[1,2,3,10,11,12], "g":["a","a","a","b","b","b"]})
    result = hypothesis_test(df, "welch_t", "v", "g")
    assert result["n"] == 6
    assert result["effect_size"] is not None
    assert 0 <= result["p_value"] <= 1

def test_chi_square_and_bad_alpha():
    df = pd.DataFrame({"x":["a","a","b","b","a","b"], "y":["u","u","v","v","v","u"]})
    result = hypothesis_test(df, "chi_square", "x", "y")
    assert result["degrees_of_freedom"] == 1
    with pytest.raises(ValueError): hypothesis_test(df, "chi_square", "x", "y", alpha=1)

def test_multiple_test_correction():
    result = adjust_pvalues([.001,.03,.8])
    assert len(result["adjusted_p_values"]) == 3
    assert result["adjusted_p_values"][0] <= result["adjusted_p_values"][1]

def test_confidence_intervals_are_returned_for_supported_tests():
    from core.statistical_analysis import mean_confidence_interval
    ci = mean_confidence_interval([2, 3, 4, 5, 6])
    assert ci["lower"] < ci["mean"] < ci["upper"]
    df = pd.DataFrame({"x":[1,2,3,4,5,6], "y":[1,3,2,5,4,8]})
    corr = correlation_detail(df, "x", "y", "pearson")
    assert corr["confidence_interval_95"] is not None
    groups = pd.DataFrame({"value":[1,2,3,9,10,11], "group":["A"]*3+["B"]*3})
    result = hypothesis_test(groups, "welch_t", "value", "group")
    lo, hi = result["mean_difference_confidence_interval_95"]
    assert lo < result["mean_difference_group1_minus_group2"] < hi

def test_correlation_matrix_has_wide_dataset_guard():
    from core.statistical_analysis import correlation_analysis
    wide = pd.DataFrame(np.random.default_rng(0).normal(size=(5, 201)))
    with pytest.raises(ValueError, match="200-column"):
        correlation_analysis(wide)
