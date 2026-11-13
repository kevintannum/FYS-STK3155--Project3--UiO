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