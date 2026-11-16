import pandas as pd
from sklearn.metrics import mean_squared_error, fbeta_score
import numpy as np

def ImportTrainingData():
    train = pd.read_csv("../data/train_data.zip")
    return ImportData(train)

def ImportTestData():
    test = pd.read_csv("../data/test_data.csv")
    private = pd.read_csv("../data/test_data_private.csv")

    private = private.rename(columns={"true_risk": "risk"})

    test = pd.concat([test, private], ignore_index=True)

    return ImportData(test)


def ImportData(data):
    X_events = []
    y = []

    for event_id, group in data.groupby("event_id"):

        group = group.sort_values("time_to_tca", ascending=False)

        inputs = group[group["time_to_tca"] >= 2]
        targets = group[group["time_to_tca"] < 1]

        if inputs.empty or targets.empty:
            continue

        X_events.append(
            inputs.drop(columns=["event_id", "mission_id", "risk"])
        )

        target_risk = targets.iloc[-1]["risk"]
        y.append(int(target_risk >= -6))

    return X_events, y

def aggregate(X_events):
    X_agg = []

    all_num = pd.concat([
        event.select_dtypes(include="number")
        for event in X_events
    ])

    constant = []
    varying = []

    for col in all_num.columns: # check which are constant and not
        changes = any(
            event[col].nunique() > 1
            for event in X_events
        )
        if changes:
            varying.append(col)
        else:
            constant.append(col)

    for event in X_events: # aggregate events
        num = event.select_dtypes(include="number")
        cat = event.select_dtypes(exclude="number")

        summary = pd.concat([
            num[constant].iloc[-1],
            num[varying].mean().add_suffix("_mean"),
            num[varying].std().fillna(0).add_suffix("_std"),
            num[varying].iloc[0].add_suffix("_first"),
            num[varying].iloc[-1].add_suffix("_last"),
            cat.iloc[-1] # if c_object changed changed from TBA to other, keep the latter 
        ])

        summary["n_cdms"] = len(event)
        X_agg.append(summary)

    X_agg = pd.DataFrame(X_agg)

    X_agg = pd.get_dummies(X_agg, dtype=int)

    cov_cols = [
        col for col in X_agg.columns
        if "position_covariance_det" in col
    ]

    X_agg[cov_cols] = np.sign(X_agg[cov_cols]) * np.log1p(
        np.abs(X_agg[cov_cols])
    )

    return X_agg

def cost(y_prob, y_pred, y_test):
    mse = mean_squared_error(y_test, y_prob)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    return mse / f2 if f2 > 0 else float("inf") # common problem of only giving postive, resulting in 0-div

def evaluate(model, X_test, y_test, threshold = 0.01):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return cost(y_prob, y_pred, y_test)
