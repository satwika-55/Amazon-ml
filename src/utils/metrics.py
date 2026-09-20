"""Configuration-driven metric implementations shared by M4."""

import math
import warnings


class MetricError(ValueError):
    """Raised when metric inputs or configuration are invalid."""


def evaluate_predictions(targets, predictions, config=None, metric_name=None, direction=None):
    """Evaluate predictions using the configured metric and direction."""
    if config is not None:
        metric_config = config.get("metric", {})
        metric_name = metric_name or metric_config.get("name")
        direction = direction or metric_config.get("direction")
    metric_name = str(metric_name or "").upper()
    if direction not in {"minimize", "maximize"}:
        raise MetricError("Metric direction must be 'minimize' or 'maximize'")
    targets = list(targets)
    predictions = list(predictions)
    if len(targets) != len(predictions):
        raise MetricError("Targets and predictions must have the same row count")
    if not targets:
        warnings.warn("Metric is undefined for an empty fold", RuntimeWarning)
        return float("nan")
    task_type = (config or {}).get("task", {}).get("type")
    if task_type == "multiclass_classification":
        classes = [str(value) for value in (config or {}).get("task", {}).get("classes", [])]
        if not classes:
            raise MetricError("Multiclass metrics require task.classes")
        target_indices = [_class_index(value, classes) for value in targets]
        probability_rows = [[_finite_number(item) for item in row] for row in predictions]
        if any(len(row) != len(classes) for row in probability_rows):
            raise MetricError("Multiclass prediction rows must match task.classes")
        if metric_name == "ACCURACY":
            return sum(max(range(len(row)), key=row.__getitem__) == target for row, target in zip(probability_rows, target_indices)) / len(targets)
        if metric_name == "LOGLOSS":
            return -sum(math.log(max(row[target], 1e-15)) for row, target in zip(probability_rows, target_indices)) / len(targets)
        raise MetricError(f"Metric {metric_name!r} is not supported for multiclass probabilities")
    values = [_finite_number(value) for value in targets + predictions]
    targets = values[:len(targets)]
    predictions = values[len(targets):]

    if metric_name == "RMSE":
        return math.sqrt(sum((prediction - target) ** 2 for target, prediction in zip(targets, predictions)) / len(targets))
    if metric_name == "MAE":
        return sum(abs(prediction - target) for target, prediction in zip(targets, predictions)) / len(targets)
    if metric_name == "MAPE":
        valid = [(target, prediction) for target, prediction in zip(targets, predictions) if target != 0]
        if not valid:
            warnings.warn("MAPE is undefined because all targets are zero", RuntimeWarning)
            return float("nan")
        return sum(abs((prediction - target) / target) for target, prediction in valid) / len(valid)
    if metric_name == "SMAPE":
        denominators = [abs(target) + abs(prediction) for target, prediction in zip(targets, predictions)]
        if any(denominator == 0 for denominator in denominators):
            warnings.warn("SMAPE is undefined for a zero target and prediction pair", RuntimeWarning)
            return float("nan")
        return sum(2 * abs(prediction - target) / denominator for target, prediction, denominator in zip(targets, predictions, denominators)) / len(targets)
    if metric_name == "RMSLE":
        if any(target < 0 or prediction < 0 for target, prediction in zip(targets, predictions)):
            raise MetricError("RMSLE does not accept negative values")
        return math.sqrt(sum((math.log1p(prediction) - math.log1p(target)) ** 2 for target, prediction in zip(targets, predictions)) / len(targets))
    if metric_name == "ACCURACY":
        return sum(target == prediction for target, prediction in zip(targets, predictions)) / len(targets)
    if metric_name == "LOGLOSS":
        return -sum(math.log(max(min(prediction, 1.0 - 1e-15), 1e-15)) if target == 1 else math.log(max(min(1.0 - prediction, 1.0 - 1e-15), 1e-15)) for target, prediction in zip(targets, predictions)) / len(targets)
    raise MetricError(f"Unsupported configured metric: {metric_name!r}")


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricError(f"Metric values must be numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise MetricError("Metric values must be finite")
    return number


def _class_index(value, classes):
    value_string = str(value)
    if value_string not in classes:
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise MetricError(f"Unknown class label: {value!r}") from exc
        if index < 0 or index >= len(classes):
            raise MetricError(f"Unknown class index: {value!r}")
        return index
    return classes.index(value_string)
