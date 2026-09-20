"""Placeholder interfaces for text cleaning and input preparation."""

# PURPOSE : RAW TEXT -> CLEANED TEXT
# then give to tfidfmodel

"""
Text preprocessing utilities for M2 text pipeline.
"""

import re
import html
import unicodedata

import pandas as pd

from src.utils.config import load_config


def clean_text(text):
    """
    Clean a single text value using shared config settings.
    """

    config = load_config()

    text_config = config.get("text", {})
    normalization = text_config.get("normalization", {})

    missing_value = text_config.get(
        "missing_text_value",
        ""
    )

    # Handle missing values
    if text is None or pd.isna(text):
        return missing_value

    text = str(text)

    # Remove HTML if enabled
    if normalization.get("remove_html", False):
        text = html.unescape(text)

        text = re.sub(
            r"<[^>]*>",
            " ",
            text
        )

    # Unicode normalization
    if normalization.get("unicode", False):
        text = unicodedata.normalize(
            "NFKC",
            text
        )

    # Lowercase only if enabled
    if normalization.get("lowercase", False):
        text = text.lower()

    # Preserve numbers and units by default
    # Do not remove digits/symbols aggressively

    # Normalize whitespace
    if normalization.get("whitespace", False):
        text = re.sub(
            r"\s+",
            " ",
            text
        )

    return text.strip()



def prepare_text_input(data):
    """
    Prepare text input for M2 models.

    Steps:
    - Read text columns from shared config
    - Handle missing values
    - Combine multiple text columns
    - Apply preprocessing
    """

    config = load_config()

    text_config = config.get("text", {})

    text_columns = text_config.get(
        "columns",
        []
    )

    missing_value = text_config.get(
        "missing_text_value",
        ""
    )

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame"
        )

    if not text_columns:
        raise ValueError(
            "No text columns configured in config.yaml"
        )

    missing_columns = [
        col for col in text_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing text columns in dataframe: {missing_columns}"
        )

    df = data.copy()

    # Fill missing text safely
    df[text_columns] = (
        df[text_columns]
        .fillna(missing_value)
    )

    # Convert all text columns to strings
    df[text_columns] = (
        df[text_columns]
        .astype(str)
    )

    # Combine all text sources
    df["combined_text"] = (
        df[text_columns]
        .agg(" ".join, axis=1)
    )

    # Clean combined text
    df["combined_text"] = (
        df["combined_text"]
        .apply(clean_text)
    )

    return df["combined_text"]