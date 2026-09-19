"""Placeholder interfaces for loading and validating source data."""


def load_raw_data(path):
    """TODO: Load the raw dataset from ``path``."""
    raise NotImplementedError


def validate_required_columns(data, required_columns):
    """TODO: Validate the required columns without assuming their names."""
    raise NotImplementedError
