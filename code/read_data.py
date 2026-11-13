import pandas as pd

def ImportTrainingData():
    train = pd.read_csv("../data/train_data.zip")

    X_events = []
    y = []
    event_ids = []

    for event_id, group in train.groupby("event_id"): # Slight help by ChatGPT

        group = group.sort_values("time_to_tca", ascending=False)

        inputs = group[group["time_to_tca"] >= 2]
        targets = group[group["time_to_tca"] < 1]

        if inputs.empty or targets.empty:
            continue

        X_events.append(inputs.drop(columns=["event_id", "risk"])) # Add the event, event_id is stored elsewhere, risk is leakage

        target_risk = targets.iloc[-1]["risk"]
        y.append(int(target_risk >= -6)) # risk is 10-log

        event_ids.append(event_id)

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

    X_agg = pd.get_dummies(
        X_agg,
        columns=["mission_id", "c_object_type"],
        dtype=int
    )

    return X_agg