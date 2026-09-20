"""Strict ID-based loading and validation for M4 prediction artifacts."""

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path


class PredictionValidationError(ValueError):
    """Raised for fatal prediction contract violations."""


@dataclass(frozen=True)
class PredictionTable:
    ids: tuple
    columns: tuple
    values: tuple
    path: Path


def load_prediction_file(path, task, expected_ids=None):
    """Load one OOF or test CSV, validating schema and values."""
    prediction_path = Path(path)
    if not prediction_path.is_file():
        raise PredictionValidationError(f"Prediction file does not exist: {prediction_path}")
    task_type = task.get("type")
    id_column = str(task.get("id_column", "")).strip()
    expected_columns = _expected_columns(task)
    rows = []
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as prediction_file:
        reader = csv.DictReader(prediction_file)
        headers = [str(header).strip() for header in (reader.fieldnames or [])]
        if headers != expected_columns:
            raise PredictionValidationError(
                f"Wrong prediction columns in {prediction_path}: expected {expected_columns}, got {headers}"
            )
        for raw_row in reader:
            row = {str(key).strip(): value for key, value in raw_row.items()}
            identifier = _normalize_id(row[id_column])
            if any(existing[0] == identifier for existing in rows):
                raise PredictionValidationError(f"Duplicate prediction ID: {identifier!r}")
            numeric_values = tuple(_parse_prediction(row[column], column, task_type) for column in expected_columns[1:])
            rows.append((identifier, numeric_values))
    ids = tuple(row[0] for row in rows)
    if expected_ids is not None:
        expected = {_normalize_id(identifier) for identifier in expected_ids}
        actual = set(ids)
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise PredictionValidationError(f"Prediction ID mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        if len(ids) != len(expected):
            raise PredictionValidationError("Prediction row count does not match expected IDs")
    values = tuple(row[1] for row in rows)
    if values and all(row == values[0] for row in values):
        warnings.warn(f"Constant predictions detected in {prediction_path}", RuntimeWarning)
    if task_type == "multiclass_classification":
        for identifier, row in zip(ids, values):
            if not math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise PredictionValidationError(f"Multiclass probabilities for ID {identifier!r} must sum to 1")
    return PredictionTable(ids, tuple(expected_columns[1:]), values, prediction_path)


def load_configured_predictions(config, model, root=None):
    """Load configured OOF and test files for one model without auto-discovery."""
    base = Path(root) if root is not None else Path.cwd()
    prediction_config = config["prediction"]
    oof_path = base / prediction_config["oof_dir"] / f"{model}_oof.csv"
    test_path = base / prediction_config["test_dir"] / f"{model}_test.csv"
    oof = load_prediction_file(oof_path, config["task"])
    test = load_prediction_file(test_path, config["task"])
    if oof.columns != test.columns:
        raise PredictionValidationError(f"OOF/test schema mismatch for model {model!r}")
    return oof, test


def align_predictions(predictions, expected_ids=None):
    """Align prediction tables by ID and return ordered IDs and values."""
    if not predictions:
        raise PredictionValidationError("At least one prediction table is required")
    names = list(predictions)
    reference_ids = {_normalize_id(identifier) for identifier in (expected_ids or predictions[names[0]].ids)}
    ordered_ids = tuple(sorted(reference_ids))
    aligned = {}
    for name in names:
        table = predictions[name]
        table_ids = set(table.ids)
        if table_ids != reference_ids:
            raise PredictionValidationError(f"Prediction IDs do not align for model {name!r}")
        by_id = {identifier: values for identifier, values in zip(table.ids, table.values)}
        aligned[name] = tuple(by_id[identifier] for identifier in ordered_ids)
    return ordered_ids, aligned


def _expected_columns(task):
    task_type = task.get("type")
    id_column = str(task.get("id_column", "")).strip()
    if not id_column:
        raise PredictionValidationError("task.id_column must be configured")
    if task_type in {"regression", "binary_classification"}:
        return [id_column, "prediction"]
    if task_type == "multiclass_classification":
        classes = task.get("classes")
        if not isinstance(classes, list) or not classes:
            raise PredictionValidationError("task.classes is required for multiclass predictions")
        return [id_column] + [f"prob_{value}" for value in classes]
    raise PredictionValidationError(f"Unsupported task.type: {task_type!r}")


def _parse_prediction(value, column, task_type):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise PredictionValidationError(f"Non-numeric prediction in column {column!r}") from exc
    if not math.isfinite(number):
        raise PredictionValidationError(f"NaN or infinity in prediction column {column!r}")
    if task_type in {"binary_classification", "multiclass_classification"} and not 0 <= number <= 1:
        raise PredictionValidationError(f"Invalid probability in prediction column {column!r}: {number}")
    return number


def _normalize_id(value):
    identifier = str(value).strip()
    if not identifier:
        raise PredictionValidationError("IDs must not be empty")
    return identifier