"""Placeholder interfaces for a TF-IDF text baseline."""

# TEXT -> NUMBERS -> ML MODEL


def extract_tfidf_features(text_data):
    """TODO: Extract TF-IDF features."""
    raise NotImplementedError


def train_tfidf_model(features, targets):
    """TODO: Train the TF-IDF baseline."""
    raise NotImplementedError


def cross_validate_tfidf(features, targets):
    """TODO: Evaluate the TF-IDF baseline with cross-validation."""
    raise NotImplementedError


def predict_tfidf(model, features):
    """TODO: Generate TF-IDF predictions."""
    raise NotImplementedError
