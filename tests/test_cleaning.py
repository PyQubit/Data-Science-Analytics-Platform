import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from core.data_cleaner import apply_cleaning

def test_fill_missing_does_not_mutate_raw_dataframe():
    raw = pd.DataFrame({"x":[1.0,None,3.0], "group":["a","b",None]})
    before = raw.copy(deep=True)
    cleaned, entry = apply_cleaning(raw, "fill_missing", columns=["x"], method="median")
    assert cleaned["x"].tolist() == [1.0,2.0,3.0]
    assert cleaned["group"].isna().sum() == 1
    assert entry["affected_rows"] == 1
    assert_frame_equal(raw, before)

def test_drop_duplicate_and_log():
    raw = pd.DataFrame({"x":[1,1,2], "y":["a","a","b"]})
    cleaned, entry = apply_cleaning(raw, "drop_duplicates")
    assert len(cleaned) == 2 and len(raw) == 3
    assert entry["rows_before"] == 3 and entry["rows_after"] == 2

def test_invalid_imputation_method_rejected():
    with pytest.raises(ValueError):
        apply_cleaning(pd.DataFrame({"x":[None,1]}), "fill_missing", method="guess")
