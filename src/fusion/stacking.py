"""Dependency-free Ridge stacking and nested fusion evaluation."""

import math
import statistics
import warnings

from src.utils.metrics import evaluate_predictions


class RidgeMetaModel:
    """Small deterministic Ridge model supporting scalar and multiclass targets."""

    def __init__(self, coefficients, intercepts, alpha, task_type, classes=None):
        self.coefficients = coefficients
        self.intercepts = intercepts
        self.alpha = alpha
        self.task_type = task_type
        self.classes = classes or []

    def predict(self, features):
        outputs = []
        for row in features:
            raw = [intercept + sum(coefficient * value for coefficient, value in zip(coefficients, row)) for coefficients, intercept in zip(self.coefficients, self.intercepts)]
            if self.task_type == "multiclass_classification":
                positive = [max(value, 0.0) for value in raw]
                total = sum(positive)
                outputs.append([value / total for value in positive] if total else [1.0 / len(raw)] * len(raw))
            else:
                value = raw[0]
                if self.task_type == "binary_classification":
                    value = min(max(value, 0.0), 1.0)
                outputs.append([value])
        return outputs


def load_oof_predictions(paths, task):
    """Load explicitly named OOF files using the Phase 1 validator."""
    from src.utils.prediction_contract import load_prediction_file

    return {name: load_prediction_file(path, task) for name, path in paths.items()}


def train_meta_model(oof_predictions, targets, alpha=1.0, config=None):
    """Fit Ridge on OOF features only."""
    task = (config or {}).get("task", {"type": "regression"})
    features = _feature_matrix(oof_predictions)
    target_values, classes = _target_matrix(targets, task)
    return _fit_ridge(features, target_values, alpha, task["type"], classes)


def predict_with_meta_model(meta_model, base_predictions):
    """Generate predictions from aligned base-model feature vectors."""
    return meta_model.predict(_feature_matrix(base_predictions))


def nested_ridge_evaluation(predictions, targets, folds, config):
    """Evaluate Ridge by fitting each outer fold only on the other folds."""
    ids = sorted(set(targets) & set(folds))
    if set(ids) != set(targets) or set(ids) != set(folds):
        raise ValueError("Nested Ridge inputs must share identical target and fold IDs")
    matrix = _feature_matrix_by_id(predictions, ids)
    target_values, classes = _target_matrix([targets[identifier] for identifier in ids], config["task"])
    alpha = float(config.get("fusion", {}).get("ridge", {}).get("alpha", 1.0))
    if alpha < 0 or not math.isfinite(alpha):
        raise ValueError("Ridge alpha must be finite and nonnegative")
    outer_predictions = [None] * len(ids)
    fold_scores = {}
    for fold in sorted(set(folds[identifier] for identifier in ids)):
        train_indices = [index for index, identifier in enumerate(ids) if folds[identifier] != fold]
        valid_indices = [index for index, identifier in enumerate(ids) if folds[identifier] == fold]
        if not train_indices or not valid_indices:
            warnings.warn(f"Fold {fold} cannot be used for nested Ridge evaluation", RuntimeWarning)
            continue
        model = _fit_ridge([matrix[index] for index in train_indices], [target_values[index] for index in train_indices], alpha, config["task"]["type"], classes)
        predictions_for_fold = model.predict([matrix[index] for index in valid_indices])
        for index, prediction in zip(valid_indices, predictions_for_fold):
            outer_predictions[index] = prediction
        fold_scores[str(fold)] = evaluate_predictions(
            [targets[ids[index]] for index in valid_indices] if config["task"]["type"] == "multiclass_classification" else [_metric_target(targets[ids[index]], config["task"]) for index in valid_indices],
            predictions_for_fold if config["task"]["type"] == "multiclass_classification" else [_metric_prediction(prediction, config["task"]) for prediction in predictions_for_fold],
            config,
        )
    if any(prediction is None for prediction in outer_predictions):
        raise ValueError("Nested Ridge did not produce predictions for every row")
    valid_scores = list(fold_scores.values())
    return {
        "fold_scores": fold_scores,
        "fold_mean": statistics.fmean(valid_scores) if valid_scores else float("nan"),
        "fold_std": statistics.pstdev(valid_scores) if len(valid_scores) > 1 else 0.0 if valid_scores else float("nan"),
        "overall_score": evaluate_predictions(
            [targets[identifier] for identifier in ids] if config["task"]["type"] == "multiclass_classification" else [_metric_target(targets[identifier], config["task"]) for identifier in ids],
            outer_predictions if config["task"]["type"] == "multiclass_classification" else [_metric_prediction(prediction, config["task"]) for prediction in outer_predictions],
            config,
        ),
        "predictions": outer_predictions,
        "alpha": alpha,
    }


def _fit_ridge(features, targets, alpha, task_type, classes):
    if not features or not targets or len(features) != len(targets):
        raise ValueError("Ridge requires non-empty, equally sized features and targets")
    output_count = len(targets[0]) if isinstance(targets[0], list) else 1
    target_matrix = targets if isinstance(targets[0], list) else [[value] for value in targets]
    coefficients = []
    intercepts = []
    for output in range(output_count):
        response = [row[output] for row in target_matrix]
        coefficients_for_output, intercept = _solve_ridge(features, response, alpha)
        coefficients.append(coefficients_for_output)
        intercepts.append(intercept)
    return RidgeMetaModel(coefficients, intercepts, alpha, task_type, classes)


def _solve_ridge(features, response, alpha):
    dimension = len(features[0]) + 1
    gram = [[0.0] * dimension for _ in range(dimension)]
    rhs = [0.0] * dimension
    augmented = [[1.0] + list(row) for row in features]
    for row, target in zip(augmented, response):
        for left in range(dimension):
            rhs[left] += row[left] * target
            for right in range(dimension):
                gram[left][right] += row[left] * row[right]
    for index in range(1, dimension):
        gram[index][index] += alpha
    solution = _gaussian_solve(gram, rhs)
    return solution[1:], solution[0]


def _gaussian_solve(matrix, vector):
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    size = len(augmented)
    for pivot in range(size):
        pivot_row = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        if abs(augmented[pivot_row][pivot]) < 1e-12:
            raise ValueError("Ridge design matrix is singular")
        augmented[pivot], augmented[pivot_row] = augmented[pivot_row], augmented[pivot]
        divisor = augmented[pivot][pivot]
        augmented[pivot] = [value / divisor for value in augmented[pivot]]
        for row in range(size):
            if row == pivot:
                continue
            factor = augmented[row][pivot]
            augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[pivot])]
    return [row[-1] for row in augmented]


def _feature_matrix(predictions):
    models = list(predictions)
    if not models:
        raise ValueError("At least one base model is required for Ridge")
    row_count = len(predictions[models[0]])
    if any(len(predictions[model]) != row_count for model in models):
        raise ValueError("Ridge base-model vectors must have equal row counts")
    return [
        [float(value) for model in models for value in _as_row(predictions[model][row])]
        for row in range(row_count)
    ]


def _feature_matrix_by_id(predictions, ids):
    by_model = {}
    for model, table in predictions.items():
        by_model[model] = {identifier: row for identifier, row in zip(table.ids, table.values)} if hasattr(table, "ids") else None
    if any(values is None for values in by_model.values()):
        return _feature_matrix(predictions)
    return [[float(value) for model in predictions for value in _as_row(by_model[model][identifier])] for identifier in ids]


def _target_matrix(targets, task):
    if task["type"] != "multiclass_classification":
        return [float(value) for value in targets], []
    classes = [str(value) for value in task["classes"]]
    return [[1.0 if str(value) == class_name else 0.0 for class_name in classes] for value in targets], classes


def _metric_target(value, task):
    if task["type"] == "multiclass_classification":
        return float([str(item) for item in task["classes"]].index(str(value)))
    return float(value)


def _metric_prediction(value, task):
    if task["type"] == "multiclass_classification":
        return float(max(range(len(value)), key=value.__getitem__))
    return float(value[0])


def _as_row(value):
    return value if isinstance(value, (list, tuple)) else [value]
