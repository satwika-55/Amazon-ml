"""
Validation utilities for M2 prediction outputs.

Checks:
- ID exists
- ID uniqueness
- row count
- missing predictions
"""


import pandas as pd



def validate_prediction_file(
    df,
    expected_rows=None
):
    """
    Validate prediction dataframe.

    Required format:

    id,prediction
    """



    # Check columns

    required = [
        "id",
        "prediction"
    ]


    for col in required:

        if col not in df.columns:

            raise ValueError(
                f"Missing column: {col}"
            )



    # Check duplicate IDs

    if not df["id"].is_unique:

        raise ValueError(
            "Duplicate IDs found"
        )



    # Check row count

    if expected_rows is not None:

        if len(df) != expected_rows:

            raise ValueError(
                f"Expected {expected_rows} rows but got {len(df)}"
            )



    # Check missing predictions

    if df["prediction"].isna().any():

        raise ValueError(
            "Missing prediction values found"
        )



    return True





def validate_id_alignment(
    original_ids,
    prediction_df
):
    """
    Check prediction IDs match original IDs.
    """


    original_ids = set(
        original_ids
    )


    prediction_ids = set(
        prediction_df["id"]
    )


    if original_ids != prediction_ids:

        raise ValueError(
            "Prediction IDs do not match original IDs"
        )


    return True