"""
OOF and Test prediction generation
for M2 text pipeline.
"""


import pandas as pd

from sklearn.model_selection import KFold



def generate_oof_predictions(
    features,
    targets,
    ids,
    model_function,
    folds=5
):
    """
    Generate out-of-fold predictions.

    Parameters:

        features:
            TF-IDF / embedding features

        targets:
            training labels

        ids:
            original ids

        model_function:
            function that trains and returns model

        folds:
            number of folds

    Returns:

        dataframe:
            id,prediction
    """


    kf = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )


    oof_predictions = []



    for train_idx, val_idx in kf.split(features):


        X_train = features[train_idx]

        X_val = features[val_idx]


        y_train = targets.iloc[
            train_idx
        ]



        model = model_function(
            X_train,
            y_train
        )



        preds = model.predict(
            X_val
        )



        fold_df = pd.DataFrame({

            "id":
                ids.iloc[val_idx],

            "prediction":
                preds

        })


        oof_predictions.append(
            fold_df
        )



    oof_df = pd.concat(
        oof_predictions
    )


    return oof_df





def generate_test_predictions(
    train_features,
    train_targets,
    test_features,
    test_ids,
    model_function
):
    """
    Train on full training data
    and predict test data.
    """


    model = model_function(

        train_features,

        train_targets

    )


    predictions = model.predict(
        test_features
    )


    test_df = pd.DataFrame({

        "id":
            test_ids,

        "prediction":
            predictions

    })


    return test_df