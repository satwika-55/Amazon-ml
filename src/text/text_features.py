"""
Manual text feature extraction for M2 pipeline.

Input:
    cleaned text

Output:
    numerical text features
"""


import re
import pandas as pd



def get_text_length(text):
    """
    Number of characters in text.
    """

    return len(text)



def get_word_count(text):
    """
    Number of words.
    """

    return len(
        text.split()
    )



def get_digit_count(text):
    """
    Count numerical characters.
    """

    return sum(
        c.isdigit()
        for c in text
    )



def get_uppercase_count(text):
    """
    Count uppercase letters.
    """

    return sum(
        c.isupper()
        for c in text
    )



def get_special_character_count(text):
    """
    Count special symbols.
    """

    return len(
        re.findall(
            r"[^a-zA-Z0-9\s]",
            text
        )
    )



def get_number_count(text):
    """
    Count complete numbers.

    Example:

    "iPhone 15 256GB"

    returns 2
    """

    return len(
        re.findall(
            r"\d+",
            text
        )
    )



def extract_text_features(text_series):
    """
    Extract all manual text features.

    Input:
        pandas Series containing text

    Output:
        dataframe of features
    """


    features = pd.DataFrame()


    features["text_length"] = (
        text_series
        .apply(get_text_length)
    )


    features["word_count"] = (
        text_series
        .apply(get_word_count)
    )


    features["digit_count"] = (
        text_series
        .apply(get_digit_count)
    )


    features["uppercase_count"] = (
        text_series
        .apply(get_uppercase_count)
    )


    features["special_character_count"] = (
        text_series
        .apply(get_special_character_count)
    )


    features["number_count"] = (
        text_series
        .apply(get_number_count)
    )


    return features