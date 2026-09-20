"""
Text prediction generation for M2 pipeline.

Creates:
    id,prediction
format outputs.
"""


import pandas as pd



def generate_text_predictions(
    data,
    model
):
    """
    Generate predictions in common format.

    Input:
        data:
            dataframe containing:
            id + processed features

        model:
            trained text model

    Output:
        dataframe:
            id,prediction
    """


    if not isinstance(data, pd.DataFrame):

        raise TypeError(
            "Input data must be pandas DataFrame"
        )


    if "id" not in data.columns:

        raise ValueError(
            "Missing id column"
        )


    ids = data["id"]


    # remove id column before prediction
    features = data.drop(
        columns=["id"]
    )


    predictions = model.predict(
        features
    )


    output = pd.DataFrame({

        "id": ids,

        "prediction": predictions

    })


    return output