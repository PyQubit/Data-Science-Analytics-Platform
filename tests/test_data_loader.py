from io import BytesIO
import pandas as pd
import pytest
from core.data_loader import load_dataset

def test_csv_load_and_metadata():
    result = load_dataset(b"a,b\n1,2\n3,4\n", "sample.csv")
    assert result.dataframe.shape == (2,2)
    assert result.metadata["rows"] == 2
    assert result.metadata["columns"] == 2

def test_duplicate_column_names_are_normalized_with_warning():
    result = load_dataset(b"x,x,\n1,2,3\n", "dupes.csv")
    assert len(set(result.dataframe.columns)) == 3
    assert any("Duplicate column names" in warning for warning in result.warnings)

def test_invalid_extension_and_size_guard():
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(b"a\n1", "bad.exe")
    with pytest.raises(ValueError, match="exceeds"):
        load_dataset(b"a,b\n1,2\n", "big.csv", max_size_mb=0)

def test_json_records_load():
    result = load_dataset(b'[{"a":1,"b":"x"},{"a":2,"b":"y"}]', "data.json")
    assert result.dataframe.shape == (2,2)


def test_excel_sheet_selection_and_sheet_listing():
    from core.data_loader import excel_sheet_names
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"a":[1]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"b":[2,3]}).to_excel(writer, sheet_name="second", index=False)
    content = buffer.getvalue()
    assert excel_sheet_names(content) == ["first", "second"]
    result = load_dataset(content, "book.xlsx", sheet_name="second")
    assert result.dataframe.shape == (2,1)
    assert result.dataframe.columns.tolist() == ["b"]

def test_semicolon_csv_delimiter_is_detected():
    result = load_dataset("a;b\n1;2\n3;4\n".encode(), "semicolon.csv")
    assert result.dataframe.shape == (2,2)
    assert result.dataframe.iloc[1,1] == 4


def test_nested_json_values_are_preserved_as_json_text():
    content = b'[{"id":1,"payload":{"items":[1,2]}},{"id":2,"payload":{"items":[3]}}]'
    result = load_dataset(content, "nested.json")
    assert isinstance(result.dataframe.loc[0, "payload"], str)
    assert '"items"' in result.dataframe.loc[0, "payload"]
    assert any("Nested objects/arrays" in warning for warning in result.warnings)
