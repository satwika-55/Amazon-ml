"""Individual model evaluation and complementarity analysis for M4."""

import itertools
import math
import statistics
from collections import OrderedDict

from src.utils.metrics import evaluate_predictions
from src.utils.prediction_contract import align_predictions


def evaluate_individual_models(predictions, targets, folds, config):
    """Score validated OOF tables and return deterministic analysis data."""
    ids, aligned = align_predictions(predictions, expected_ids=targets.keys())
    target_values = [_target_value(targets[identifier], config["task"]) for identifier in ids]
    fold_values = [folds[identifier] for identifier in ids]
    model_reports = OrderedDict()
    for model in predictions:
        values = [list(row) if config["task"]["type"] == "multiclass_classification" else _prediction_value(row, config["task"]) for row in aligned[model]]
        fold_scores = OrderedDict()
        for fold in sorted(set(fold_values)):
            indices = [index for index, value in enumerate(fold_values) if value == fold]
            score = evaluate_predictions(
                [target_values[index] for index in indices],
                [values[index] for index in indices],
                config,
            )
            fold_scores[str(fold)] = _json_number(score)
        valid_scores = [score for score in fold_scores.values() if not math.isnan(score)]
        model_reports[model] = {
            "overall_score": _json_number(evaluate_predictions(target_values, values, config)),
            "fold_scores": fold_scores,
            "fold_mean": _json_number(statistics.fmean(valid_scores)) if valid_scores else None,
            "fold_std": _json_number(statistics.pstdev(valid_scores)) if len(valid_scores) > 1 else 0.0 if valid_scores else None,
            "valid_folds": len(valid_scores),
            "prediction_statistics": _statistics(values),
        }
    return {
        "models": model_reports,
        "prediction_correlation": prediction_correlation(predictions, config),
        "error_correlation": error_correlation(predictions, targets, config),
        "disagreement": disagreement_analysis(predictions, config),
        "target_quantiles": target_quantile_analysis(predictions, targets, config),
    }


def prediction_correlation(predictions, config):
    """Return pairwise correlation of aligned prediction vectors."""
    ids, aligned = align_predictions(predictions)
    vectors = {model: [_prediction_value(row, config["task"]) for row in aligned[model]] for model in predictions}
    result = OrderedDict()
    for first, second in itertools.combinations(predictions, 2):
        result[f"{first}__{second}"] = _json_number(_correlation(vectors[first], vectors[second]))
    return result


def error_correlation(predictions, targets, config):
    """Return pairwise residual correlation for regression models."""
    if config["task"]["type"] != "regression":
        return OrderedDict()
    ids, aligned = align_predictions(predictions, expected_ids=targets.keys())
    target_values = [float(targets[identifier]) for identifier in ids]
    errors = {
        model: [_prediction_value(row, config["task"]) - target for row, target in zip(aligned[model], target_values)]
        for model in predictions
    }
    result = OrderedDict()
    for first, second in itertools.combinations(predictions, 2):
        result[f"{first}__{second}"] = _json_number(_correlation(errors[first], errors[second]))
    return result


def disagreement_analysis(predictions, config):
    """Summarize pairwise disagreement without selecting thresholds."""
    ids, aligned = align_predictions(predictions)
    vectors = {model: [_prediction_value(row, config["task"]) for row in aligned[model]] for model in predictions}
    result = OrderedDict()
    for first, second in itertools.combinations(predictions, 2):
        differences = [left - right for left, right in zip(vectors[first], vectors[second])]
        absolute = [abs(value) for value in differences]
        relative = [abs(value) / max(abs(right), 1e-12) for value, right in zip(differences, vectors[second])]
        result[f"{first}__{second}"] = {
            "mean_difference": _json_number(statistics.fmean(differences)),
            "mean_absolute_difference": _json_number(statistics.fmean(absolute)),
            "mean_relative_difference": _json_number(statistics.fmean(relative)),
            "max_absolute_difference": _json_number(max(absolute)),
            "sample_count": len(ids),
        }
    return result


def target_quantile_analysis(predictions, targets, config):
    """Analyze configurable target quantile buckets for regression only."""
    if config["task"]["type"] != "regression" or not config.get("fusion", {}).get("target_quantile_analysis", {}).get("enabled", True):
        return OrderedDict()
    quantiles = config.get("fusion", {}).get("target_quantile_analysis", {}).get("quantiles", [0.25, 0.5, 0.75])
    target_items = sorted((float(value), identifier) for identifier, value in targets.items())
    if not target_items:
        return OrderedDict()
    thresholds = [_quantile([item[0] for item in target_items], float(quantile)) for quantile in quantiles]
    ids, aligned = align_predictions(predictions, expected_ids=targets.keys())
    buckets = OrderedDict((f"Q{index + 1}", []) for index in range(len(thresholds) + 1))
    for identifier in ids:
        bucket = 0
        while bucket < len(thresholds) and float(targets[identifier]) > thresholds[bucket]:
            bucket += 1
        buckets[f"Q{bucket + 1}"].append(identifier)
    result = OrderedDict()
    for model in predictions:
        values_by_id = {identifier: _prediction_value(row, config["task"]) for identifier, row in zip(ids, aligned[model])}
        model_buckets = OrderedDict()
        for bucket_name, bucket_ids in buckets.items():
            actual = [float(targets[identifier]) for identifier in bucket_ids]
            predicted = [values_by_id[identifier] for identifier in bucket_ids]
            errors = [prediction - target for target, prediction in zip(actual, predicted)]
            model_buckets[bucket_name] = {
                "count": len(bucket_ids),
                "target_min": _json_number(min(actual)) if actual else None,
                "target_max": _json_number(max(actual)) if actual else None,
                "metric": _json_number(evaluate_predictions(actual, predicted, config)) if actual else None,
                "mean_absolute_error": _json_number(statistics.fmean(abs(error) for error in errors)) if errors else None,
                "prediction_bias": _json_number(statistics.fmean(errors)) if errors else None,
            }
        result[model] = model_buckets
    return result


def _prediction_value(row, task):
    if task["type"] == "multiclass_classification":
        return float(max(range(len(row)), key=row.__getitem__))
    return float(row[0])


def _target_value(value, task):
    if task["type"] == "multiclass_classification":
        classes = [str(item) for item in task["classes"]]
        if str(value) not in classes:
            raise ValueError(f"Unknown target class: {value!r}")
        return float(classes.index(str(value)))
    return float(value)


def _statistics(values):
    return {
        "count": len(values),
        "min": _json_number(min(values)) if values else None,
        "max": _json_number(max(values)) if values else None,
        "mean": _json_number(statistics.fmean(values)) if values else None,
        "std": _json_number(statistics.pstdev(values)) if len(values) > 1 else 0.0 if values else None,
        "constant": bool(values) and len(set(values)) == 1,
    }


def _correlation(first, second):
    if len(first) != len(second) or len(first) < 2:
        return float("nan")
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    numerator = sum((left - first_mean) * (right - second_mean) for left, right in zip(first, second))
    first_scale = math.sqrt(sum((value - first_mean) ** 2 for value in first))
    second_scale = math.sqrt(sum((value - second_mean) ** 2 for value in second))
    return numerator / (first_scale * second_scale) if first_scale and second_scale else float("nan")


def _quantile(values, quantile):
    if not 0 <= quantile <= 1:
        raise ValueError("Configured quantiles must be between 0 and 1")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _json_number(value):
    return None if isinstance(value, float) and math.isnan(value) else float(value)
