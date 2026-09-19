"""Placeholder interfaces for out-of-fold prediction stacking."""


def load_oof_predictions(paths):
    """TODO: Load and validate OOF prediction files."""
    raise NotImplementedError


def train_meta_model(oof_predictions, targets):
    """TODO: Train a meta-model from aligned OOF predictions."""
    raise NotImplementedError


def predict_with_meta_model(meta_model, base_predictions):
    """TODO: Generate stacked predictions."""
    raise NotImplementedError
