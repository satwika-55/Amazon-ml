"""
TF-IDF baseline model for M2 text pipeline.

Flow:

Clean Text
    |
    ↓
TF-IDF
    |
    ↓
ML Model
    |
    ↓
Prediction
"""


import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from src.utils.config import load_config



def build_vectorizer():
    """
    Create TF-IDF vectorizer from config.
    """


    config = load_config()

    tfidf_config = config.get(
        "text",
        {}
    ).get(
        "tfidf",
        {}
    )


    vectorizer = TfidfVectorizer(

        max_features=tfidf_config.get(
            "max_features",
            50000
        ),

        ngram_range=tuple(
            tfidf_config.get(
                "ngram_range",
                [1,2]
            )
        ),

        min_df=tfidf_config.get(
            "min_df",
            2
        ),

        max_df=tfidf_config.get(
            "max_df",
            0.95
        ),

        sublinear_tf=tfidf_config.get(
            "sublinear_tf",
            True
        )

    )


    return vectorizer





def extract_tfidf_features(
    train_text,
    val_text=None,
    test_text=None
):
    """
    Convert cleaned text into TF-IDF features.

    Fits only on train text.
    """


    vectorizer = build_vectorizer()


    X_train = vectorizer.fit_transform(
        train_text
    )


    X_val = None
    X_test = None


    if val_text is not None:

        X_val = vectorizer.transform(
            val_text
        )


    if test_text is not None:

        X_test = vectorizer.transform(
            test_text
        )


    return (
        vectorizer,
        X_train,
        X_val,
        X_test
    )





def train_tfidf_model(
    features,
    targets
):
    """
    Train Ridge regression model.
    """


    model = Ridge(
        alpha=1.0
    )


    model.fit(
        features,
        targets
    )


    return model





def cross_validate_tfidf(
    features,
    targets
):
    """
    Evaluate TF-IDF baseline.
    """


    config = load_config()


    cv_config = config.get(
        "training",
        {}
    )


    folds = cv_config.get(
        "folds",
        5
    )


    model = Ridge(
        alpha=1.0
    )


    scores = cross_val_score(

        model,

        features,

        targets,

        cv=folds,

        scoring=cv_config.get(
            "scoring",
            "neg_mean_squared_error"
        )

    )


    return scores





def predict_tfidf(
    model,
    features
):
    """
    Generate predictions.
    """


    return model.predict(
        features
    )





def save_tfidf_artifacts(
    vectorizer,
    model
):
    """
    Save trained vectorizer and model.
    """


    joblib.dump(
        vectorizer,
        "artifacts/tfidf_vectorizer.pkl"
    )


    joblib.dump(
        model,
        "artifacts/tfidf_model.pkl"
    )   