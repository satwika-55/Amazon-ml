"""
Transformer-based text modeling for M2 pipeline.

Flow:

Clean Text
    |
    ↓
Sentence Transformer
    |
    ↓
Embeddings
    |
    ↓
ML Model
"""


import numpy as np

from sklearn.linear_model import Ridge

from src.utils.config import load_config



_transformer_model = None



def load_transformer():

    global _transformer_model


    if _transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer support is optional; install sentence-transformers "
                "only when the configured transformer candidate is enabled."
            ) from exc

        config = load_config()


        transformer_config = config.get(
            "text",
            {}
        ).get(
            "models",
            {}
        ).get("transformer", {})


        model_name = transformer_config.get(
            "model_name",
            "all-MiniLM-L6-v2"
        )


        _transformer_model = SentenceTransformer(
            model_name
        )


    return _transformer_model





def extract_transformer_embeddings(text_data):
    """
    Extract sentence embeddings.

    Input:
        cleaned text

    Output:
        numpy embedding matrix
    """


    model = load_transformer()


    config = load_config()


    transformer_config = config.get(
        "text",
        {}
    ).get(
        "models",
        {}
    ).get("transformer", {})


    batch_size = transformer_config.get(
        "batch_size",
        32
    )


    embeddings = model.encode(

        list(text_data),

        batch_size=batch_size,

        show_progress_bar=True,

        convert_to_numpy=True

    )


    return embeddings





def train_text_model(features, targets):
    """
    Train ML model on transformer embeddings.

    For regression baseline:
    Ridge Regression.
    """


    model = Ridge(
        alpha=1.0
    )


    model.fit(
        features,
        targets
    )


    return model