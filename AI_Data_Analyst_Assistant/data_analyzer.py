"""Safe file loading, profiling, and deterministic Pandas analysis helpers."""

from __future__ import annotations

import zipfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def load_uploaded_file(upload: Any) -> pd.DataFrame:
    """Validate and load a CSV or XLSX upload without writing it to disk."""
    filename = str(getattr(upload, "name", ""))
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in {"csv", "xlsx"}:
        raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")
    content = upload.getvalue() if hasattr(upload, "getvalue") else upload.read()
    if not content:
        raise ValueError("The uploaded file is empty.")

    try:
        if extension == "csv":
            frame = pd.read_csv(BytesIO(content))
        else:
            frame = pd.read_excel(BytesIO(content), engine="openpyxl")
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
        OSError,
        zipfile.BadZipFile,
    ) as exc:
        raise ValueError(f"Could not read the uploaded {extension.upper()} file: {exc}") from exc
    except ImportError as exc:
        raise ValueError(
            "Excel support requires openpyxl. Install the project requirements."
        ) from exc

    if frame.empty and len(frame.columns) == 0:
        raise ValueError("The uploaded file contains no rows or columns.")
    if len(frame.columns) == 0:
        raise ValueError("The uploaded file contains no columns.")
    if frame.empty:
        raise ValueError("The uploaded file contains column headers but no data rows.")
    frame.columns = [str(column).strip() for column in frame.columns]
    if any(not column for column in frame.columns):
        raise ValueError("Every column must have a non-empty header.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    return frame


def describe_dataset(frame: pd.DataFrame) -> Dict[str, Any]:
    """Return dataset quality counts, column metadata, and numeric statistics."""
    column_profile = pd.DataFrame(
        [
            {
                "Column": column,
                "Data type": str(frame[column].dtype),
                "Missing": int(frame[column].isna().sum()),
                "Unique values": int(frame[column].nunique(dropna=True)),
            }
            for column in frame.columns
        ]
    )
    numeric = frame.select_dtypes(include="number")
    numeric_summary = numeric.describe().replace([np.inf, -np.inf], np.nan)
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "columns": column_profile,
        "numeric_summary": numeric_summary,
    }


def find_date_columns(frame: pd.DataFrame, threshold: float = 0.8) -> List[str]:
    """Find native datetime columns and text columns that mostly parse as dates."""
    found: List[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append(column)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            present = series.dropna()
            if not present.empty:
                parsed = pd.to_datetime(present, errors="coerce", format="mixed")
                if float(parsed.notna().mean()) >= threshold:
                    found.append(column)
    return found


def detect_suspicious_columns(
    frame: pd.DataFrame, threshold: float = 0.8
) -> Dict[str, str]:
    """Identify text columns that can likely be converted to numeric or datetime."""
    suggestions: Dict[str, str] = {}
    date_columns = set(find_date_columns(frame, threshold))
    for column in frame.columns:
        series = frame[column]
        if column in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(series):
                suggestions[column] = "datetime"
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        present = series.dropna().astype(str).str.strip()
        if present.empty:
            continue
        numeric = pd.to_numeric(present.str.replace(",", "", regex=False), errors="coerce")
        if float(numeric.notna().mean()) >= threshold:
            suggestions[column] = "numeric"
    return suggestions


def build_cleaning_preview(
    frame: pd.DataFrame,
    remove_duplicates: bool = False,
    fill_missing: bool = False,
    convert_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Create an independent proposed cleaned frame and human-readable change log."""
    preview = frame.copy(deep=True)
    changes: List[Tuple[str, str]] = []
    if remove_duplicates:
        duplicates = int(preview.duplicated().sum())
        if duplicates:
            preview = preview.drop_duplicates().reset_index(drop=True)
            changes.append(("Duplicate rows", f"Remove {duplicates:,} duplicate rows"))
    if fill_missing:
        for column in preview.columns:
            missing = int(preview[column].isna().sum())
            if not missing:
                continue
            if pd.api.types.is_numeric_dtype(preview[column]):
                replacement = preview[column].median()
                if pd.isna(replacement):
                    continue
                preview[column] = preview[column].fillna(replacement)
                detail = f"Fill {missing:,} missing values with median {replacement:g}"
            else:
                modes = preview[column].mode(dropna=True)
                if modes.empty:
                    continue
                preview[column] = preview[column].fillna(modes.iloc[0])
                detail = f"Fill {missing:,} missing values with most common value"
            changes.append((column, detail))

    suggestions = detect_suspicious_columns(preview)
    for column in convert_columns or []:
        if column not in suggestions:
            raise ValueError(f"Column {column!r} is not a supported conversion candidate.")
        kind = suggestions[column]
        if kind == "numeric":
            converted = pd.to_numeric(
                preview[column].astype("string").str.replace(",", "", regex=False),
                errors="coerce",
            )
        else:
            converted = pd.to_datetime(preview[column], errors="coerce")
        newly_missing = int(converted.isna().sum() - preview[column].isna().sum())
        if newly_missing:
            changes.append(
                (column, f"Convert to {kind}; {newly_missing:,} unparseable values become missing")
            )
        else:
            changes.append((column, f"Convert column to {kind}"))
        preview[column] = converted
    return preview, changes


def execute_analysis(frame: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
    """Execute a validated analysis specification using Pandas only."""
    kind = operation.get("operation")
    aggregation = operation.get("aggregation")
    metric = operation.get("metric")
    allowed_aggregations = {"sum", "mean", "min", "max", "count"}

    if kind not in {"aggregate", "groupby", "top", "trend"}:
        raise ValueError("Unsupported analysis operation.")
    if aggregation not in allowed_aggregations:
        raise ValueError("Unsupported aggregation.")
    if metric not in frame.columns:
        raise ValueError("The selected metric column does not exist.")
    if aggregation != "count" and not pd.api.types.is_numeric_dtype(frame[metric]):
        raise ValueError("This aggregation requires a numeric metric column.")

    if kind == "aggregate":
        value = getattr(frame[metric], aggregation)()
        label = f"{aggregation}_{metric}"
        return pd.DataFrame([{label: value}])

    if kind in {"groupby", "top"}:
        group_by = operation.get("group_by")
        if group_by not in frame.columns:
            raise ValueError("The selected group-by column does not exist.")
        grouped = frame.groupby(group_by, dropna=False)[metric].agg(aggregation)
        result = grouped.rename(f"{aggregation}_{metric}").reset_index()
        result = result.sort_values(
            f"{aggregation}_{metric}", ascending=False, na_position="last"
        )
        if kind == "top":
            return result.head(1).reset_index(drop=True)
        return result.head(100).reset_index(drop=True)

    date_column = operation.get("date_column")
    if date_column not in frame.columns:
        raise ValueError("The selected date column does not exist.")
    if date_column not in find_date_columns(frame):
        raise ValueError("The selected trend column does not contain valid dates.")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    valid = dates.notna() & frame[metric].notna()
    if aggregation == "count":
        grouped_counts = frame.loc[valid].assign(
            _month=dates.loc[valid].dt.to_period("M").astype(str)
        ).groupby("_month", dropna=False)[metric].count()
        result = grouped_counts.rename("count").reset_index()
    else:
        grouped_values = frame.loc[valid].assign(
            _month=dates.loc[valid].dt.to_period("M").astype(str)
        ).groupby("_month", dropna=False)[metric].agg(aggregation)
        result = grouped_values.rename(f"{aggregation}_{metric}").reset_index()
    return result.rename(columns={"_month": "month"}).sort_values(
        "month"
    ).reset_index(drop=True).head(100)
