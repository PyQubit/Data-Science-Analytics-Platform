"""Safe local dataset ingestion for CSV, Excel, JSON, and Parquet."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from io import StringIO
import csv
import json
import warnings
from typing import Any
import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

@dataclass
class LoadResult:
    dataframe: pd.DataFrame
    warnings: list[str]
    metadata: dict[str, Any]

def _read_source(source: Any, filename: str, max_size_mb: int) -> bytes:
    max_bytes = max_size_mb * 1024 * 1024
    if isinstance(source, (bytes, bytearray)):
        content = bytes(source)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError("The selected file does not exist or is not a regular file.")
        if path.stat().st_size > max_bytes:
            raise ValueError(f"File exceeds the configured {max_size_mb} MB upload limit.")
        content = path.read_bytes()
    elif hasattr(source, "getvalue"):
        content = source.getvalue()
    elif hasattr(source, "read"):
        old_pos = source.tell() if hasattr(source, "tell") else None
        content = source.read(max_bytes + 1)
        if old_pos is not None and hasattr(source, "seek"):
            source.seek(old_pos)
    else:
        raise TypeError("source must be bytes, a filesystem path, or a file-like object")
    if len(content) > max_bytes:
        raise ValueError(f"File exceeds the configured {max_size_mb} MB upload limit.")
    if not content:
        raise ValueError("The uploaded file is empty.")
    return content

def _unique_column_names(columns: list[Any]) -> tuple[list[str], list[str]]:
    """Make duplicate/blank labels unique while preserving their original order."""
    counts: dict[str, int] = {}
    result: list[str] = []
    changed: list[str] = []
    for index, raw in enumerate(columns):
        base = str(raw).strip() or f"unnamed_column_{index + 1}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        name = base if count == 1 else f"{base}__{count}"
        while name in result:
            count += 1
            counts[base] = count
            name = f"{base}__{count}"
        result.append(name)
        if name != str(raw):
            changed.append(f"Column {raw!r} was normalized to {name!r}.")
    return result, changed

def _read_csv(content: bytes) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    decoded = None
    encoding_used = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            decoded = content.decode(enc)
            encoding_used = enc
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ValueError("Could not decode CSV using UTF-8 or common fallback encodings.")
    if encoding_used not in ("utf-8", "utf-8-sig"):
        notes.append(f"CSV was decoded using {encoding_used}; verify non-ASCII text if it looks unexpected.")
    try:
        sample = decoded[:65536]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
            notes.append("CSV delimiter could not be inferred confidently; comma was used.")
        try:
            header = next(csv.reader(StringIO(decoded), delimiter=delimiter))
            cleaned_header = [(name or "").strip() for name in header]
            nonempty_header = [name for name in cleaned_header if name]
            if len(nonempty_header) != len(set(nonempty_header)):
                notes.append("Duplicate column names were present in the original CSV header; the CSV parser assigned unique labels to keep columns addressable.")
            if any(not name for name in cleaned_header):
                notes.append("Blank column names were present in the original CSV header; the parser assigned placeholder labels.")
        except (StopIteration, csv.Error):
            pass
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame = pd.read_csv(BytesIO(content), encoding=encoding_used, sep=delimiter, low_memory=False)
        notes.extend(str(w.message) for w in caught)
        return frame, notes
    except pd.errors.EmptyDataError as exc:
        raise ValueError("CSV contains no tabular data.") from exc
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

def load_dataset(source: Any, filename: str, sheet_name: str | int | None = None,
                 max_size_mb: int = 200) -> LoadResult:
    """Load a supported file without silently dropping rows or columns."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format {ext or '(no extension)'}. Supported: CSV, XLSX, XLS, JSON, Parquet.")
    content = _read_source(source, filename, max_size_mb)
    notes: list[str] = []
    try:
        if ext == ".csv":
            frame, csv_notes = _read_csv(content)
            notes.extend(csv_notes)
        elif ext in {".xlsx", ".xls"}:
            try:
                frame = pd.read_excel(BytesIO(content), sheet_name=sheet_name if sheet_name is not None else 0)
            except ImportError as exc:
                raise ValueError("Excel support needs openpyxl for .xlsx and xlrd for legacy .xls files.") from exc
        elif ext == ".json":
            try:
                frame = pd.read_json(BytesIO(content))
            except ValueError:
                try:
                    obj = json.loads(content.decode("utf-8"))
                    if isinstance(obj, list):
                        frame = pd.DataFrame(obj)
                    elif isinstance(obj, dict):
                        frame = pd.json_normalize(obj)
                    else:
                        raise ValueError("JSON root must be an array of records or an object.")
                except Exception as exc:
                    raise ValueError(f"Could not parse JSON: {exc}") from exc
        else:
            try:
                frame = pd.read_parquet(BytesIO(content))
            except ImportError as exc:
                raise ValueError("Parquet support requires the optional 'pyarrow' dependency.") from exc
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read {filename}: {exc}") from exc

    if isinstance(frame, dict):
        raise ValueError("Multiple sheets were returned. Select one Excel sheet and retry.")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("The file did not produce a tabular dataset.")
    if frame.empty and len(frame.columns) == 0:
        raise ValueError("The file contains no rows or columns.")
    # Nested JSON values are not scalars and break many tabular operations (nunique,
    # duplicate detection, encoding). Preserve them as JSON text and disclose the conversion.
    if ext == ".json":
        for col in frame.columns:
            if frame[col].dtype == "object":
                has_nested = frame[col].dropna().map(lambda v: isinstance(v, (dict, list, tuple))).any()
                if has_nested:
                    frame[col] = frame[col].map(lambda v: json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list, tuple)) else v)
                    notes.append(f"Nested objects/arrays in JSON column {str(col)!r} were serialized as JSON text to keep the dataset tabular.")
    unique_names, name_notes = _unique_column_names(list(frame.columns))
    frame.columns = unique_names
    notes.extend(name_notes)
    if frame.empty:
        notes.append("Dataset has columns but zero data rows.")
    duplicated_rows = int(frame.duplicated().sum()) if len(frame) else 0
    mixed = []
    for col in frame.columns:
        # Object columns may legitimately mix values; make this visible rather than changing values.
        if frame[col].dtype == "object" and frame[col].dropna().map(lambda v: type(v).__name__).nunique() > 1:
            mixed.append(str(col))
    if mixed:
        notes.append("Potential mixed Python value types detected in: " + ", ".join(mixed[:20]) + (" …" if len(mixed) > 20 else "."))
    metadata = {
        "filename": Path(filename).name,
        "format": ext.lstrip("."),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "memory_bytes": int(frame.memory_usage(index=True, deep=True).sum()),
        "duplicate_rows": duplicated_rows,
        "column_names": list(map(str, frame.columns)),
        "selected_sheet": sheet_name if ext in {".xlsx", ".xls"} else None,
    }
    return LoadResult(frame, notes, metadata)

def excel_sheet_names(source: Any) -> list[str]:
    """Return workbook sheet names from an uploaded file or filesystem path."""
    if isinstance(source, (str, Path)):
        book = pd.ExcelFile(source)
    elif hasattr(source, "getvalue"):
        book = pd.ExcelFile(BytesIO(source.getvalue()))
    elif isinstance(source, (bytes, bytearray)):
        book = pd.ExcelFile(BytesIO(source))
    else:
        raise TypeError("Unsupported workbook source")
    return list(book.sheet_names)
