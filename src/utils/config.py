"""Configuration loading and validation for the M4 pipeline."""

from pathlib import Path


SUPPORTED_TASKS = {
    "regression",
    "binary_classification",
    "multiclass_classification",
}


class ConfigurationError(ValueError):
    """Raised when the shared configuration is missing or invalid."""


def load_config(path=None):
    """Load and validate the shared YAML configuration."""
    config_path = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ConfigurationError(
            "Loading YAML configuration requires PyYAML; install declared project dependencies first."
        ) from exc

    with config_path.open("r", encoding="utf-8-sig") as config_file:
        config = yaml.safe_load(config_file)
    validate_config(config)
    return config


def validate_config(config):
    """Validate the configuration fields required by Phase 1."""
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must contain a mapping at its root.")

    required_sections = ("task", "data", "validation", "metric", "prediction", "fusion", "paths")
    missing_sections = [section for section in required_sections if not isinstance(config.get(section), dict)]
    if missing_sections:
        raise ConfigurationError(f"Missing required configuration sections: {', '.join(missing_sections)}")

    task = config["task"]
    task_type = task.get("type")
    if task_type not in SUPPORTED_TASKS:
        raise ConfigurationError(f"Unsupported task.type: {task_type!r}")
    for key in ("target", "id_column"):
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise ConfigurationError(f"task.{key} must be a non-empty string")
    if task_type == "multiclass_classification":
        classes = task.get("classes")
        if not isinstance(classes, list) or not classes or len(set(map(str, classes))) != len(classes):
            raise ConfigurationError("task.classes must contain unique classes for multiclass classification")

    validation = config["validation"]
    if not isinstance(validation.get("n_folds"), int) or validation["n_folds"] < 2:
        raise ConfigurationError("validation.n_folds must be an integer greater than one")
    if not isinstance(validation.get("folds_path"), str) or not validation["folds_path"].strip():
        raise ConfigurationError("validation.folds_path must be configured")

    metric = config["metric"]
    if not isinstance(metric.get("name"), str) or not metric["name"].strip():
        raise ConfigurationError("metric.name must be configured")
    if metric.get("direction") not in {"minimize", "maximize"}:
        raise ConfigurationError("metric.direction must be 'minimize' or 'maximize'")

    prediction = config["prediction"]
    for key in ("oof_dir", "test_dir", "merge_key"):
        if not isinstance(prediction.get(key), str) or not prediction[key].strip():
            raise ConfigurationError(f"prediction.{key} must be configured")

    fusion = config["fusion"]
    models = fusion.get("models")
    if not isinstance(models, list) or not models or any(not isinstance(model, str) or not model for model in models):
        raise ConfigurationError("fusion.models must be a non-empty list of model names")

    return config
