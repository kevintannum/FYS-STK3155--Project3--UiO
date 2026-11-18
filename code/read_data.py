"""Data loading and event-level feature construction for the ESA Collision
Avoidance Challenge dataset (FYS-STK3155, Project 3).

Problem setup
-------------
The raw file is a long table of *conjunction data messages* (CDMs). Several CDMs
describe the same conjunction *event*, each issued at a different ``time_to_tca``
(days before Time of Closest Approach). The operational question is:

    using only information available at least ``INPUT_CUTOFF_DAYS`` days before
    TCA, what will the final risk be?

so every event is split into

* **inputs**  -- CDMs with ``time_to_tca >= INPUT_CUTOFF_DAYS`` (features), and
* **targets** -- CDMs with ``time_to_tca <  TARGET_CUTOFF_DAYS`` (labels).

Events without at least one CDM on each side are discarded.

Target-row selection
--------------------
Within an event the rows are sorted by ``time_to_tca`` in **descending** order,
so the row order runs from "far from TCA" to "close to TCA". Consequently

* ``targets.iloc[-1]``  -> the *last* target CDM  -> the one with the **smallest**
  ``time_to_tca``, i.e. the observation **closest to TCA**. This is the final,
  most authoritative risk assessment and is what we predict.
* ``inputs.iloc[-1]``   -> the *last* eligible input CDM -> the most recent
  information available at decision time ("latest" feature mode).
* ``inputs.iloc[0]``    -> the earliest eligible CDM.

Leakage policy
--------------
``risk`` and the two risk-derived columns ``max_risk_estimate`` /
``max_risk_scaling`` are *known at decision time* when they come from an input
CDM (``time_to_tca >= 2``), so using them is **not** target leakage -- an
operator two days before TCA can read them off the latest CDM. They are
nevertheless the single most informative predictors, so whether to include them
changes the research question. This is exposed as ``include_current_risk``.

Risk values taken from *target* CDMs (``time_to_tca < 1``) must never enter the
features; that is enforced by construction, since only ``inputs`` is ever
returned as ``X``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import fbeta_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Default location of the (zipped) training CSV, relative to ``code/``.
DEFAULT_TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "train_data.zip"

#: CDMs this far (days) before TCA or earlier may be used as features.
INPUT_CUTOFF_DAYS = 2.0

#: CDMs closer than this (days) to TCA are used only to read off the label.
TARGET_CUTOFF_DAYS = 1.0

#: ``risk`` is log10 of the collision probability, so -6 means Pc = 1e-6.
HIGH_RISK_THRESHOLD = -6.0

#: Columns derived from the *current* risk estimate. Dropped together when
#: ``include_current_risk=False``.
RISK_FEATURE_COLUMNS = ["risk", "max_risk_estimate", "max_risk_scaling"]

#: Columns that are identifiers / categories rather than measurements.
#: ``mission_id`` is an integer *label* for the protected satellite (19 distinct
#: values), not a quantity, so it is one-hot encoded rather than scaled.
CATEGORICAL_COLUMNS = ["mission_id", "c_object_type"]

#: Never a feature: it identifies the split unit.
ID_COLUMNS = ["event_id"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw(path: str | Path = DEFAULT_TRAIN_PATH) -> pd.DataFrame:
    """Read the raw CDM table. ``pandas`` decompresses the ``.zip`` directly."""
    return pd.read_csv(path)


def _iter_events(train: pd.DataFrame):
    """Yield ``(event_id, inputs, targets)`` for every *eligible* event.

    Rows within an event are sorted by ``time_to_tca`` descending, i.e. in
    chronological order (far from TCA first). Events lacking either an eligible
    input CDM or a target CDM are skipped.
    """
    for event_id, group in train.groupby("event_id", sort=True):
        group = group.sort_values("time_to_tca", ascending=False)

        inputs = group[group["time_to_tca"] >= INPUT_CUTOFF_DAYS]
        targets = group[group["time_to_tca"] < TARGET_CUTOFF_DAYS]

        if inputs.empty or targets.empty:
            continue

        yield event_id, inputs, targets


def _import_training_data(
    path: str | Path = DEFAULT_TRAIN_PATH,
    *,
    binary_target: bool,
    include_current_risk: bool = True,
):
    """Shared implementation behind the classification/regression importers."""
    train = load_raw(path)

    drop_columns = list(ID_COLUMNS)
    if not include_current_risk:
        drop_columns += RISK_FEATURE_COLUMNS
    else:
        # ``risk`` stays, but only ever from input CDMs (time_to_tca >= 2).
        pass

    X_events: list[pd.DataFrame] = []
    y: list[float | int] = []
    event_ids: list = []

    for event_id, inputs, targets in _iter_events(train):
        X_events.append(inputs.drop(columns=drop_columns))

        # Closest-to-TCA target CDM; see module docstring.
        target_risk = float(targets.iloc[-1]["risk"])

        y.append(int(target_risk >= HIGH_RISK_THRESHOLD) if binary_target else target_risk)
        event_ids.append(event_id)

    return X_events, np.asarray(y), np.asarray(event_ids)


def ImportTrainingDataClassification(
    path: str | Path = DEFAULT_TRAIN_PATH,
    *,
    include_current_risk: bool = True,
):
    """Load events with a **binary** target ``final_risk >= -6``.

    Returns
    -------
    X_events : list of DataFrame
        One DataFrame of eligible input CDMs per event.
    y : ndarray of int, shape (n_events,)
        1 if the final risk is at or above ``HIGH_RISK_THRESHOLD``.
    event_ids : ndarray, shape (n_events,)
        Event identifier, aligned with ``X_events`` and ``y``. Used to verify
        that the train/validation/test split is done at the event level.
    """
    return _import_training_data(
        path, binary_target=True, include_current_risk=include_current_risk
    )


def ImportTrainingDataRegression(
    path: str | Path = DEFAULT_TRAIN_PATH,
    *,
    include_current_risk: bool = True,
):
    """Load events with the **continuous** target ``final_risk`` (log10 Pc).

    Identical event filtering to :func:`ImportTrainingDataClassification`; only
    the target differs. ``y`` is ``float``, never thresholded.
    """
    return _import_training_data(
        path, binary_target=False, include_current_risk=include_current_risk
    )


def ImportTrainingData(path: str | Path = DEFAULT_TRAIN_PATH):
    """Backwards-compatible importer used by ``XGBoost.ipynb`` / ``RNN_model.ipynb``.

    Returns the original two-element tuple ``(X_events, y)`` with a binary
    target and ``risk`` dropped from the features. New code should call
    :func:`ImportTrainingDataClassification` or
    :func:`ImportTrainingDataRegression`, which also return ``event_ids``.
    """
    X_events, y, _ = ImportTrainingDataClassification(path, include_current_risk=False)
    return X_events, list(y)


DEFAULT_TEST_PATHS = (
    Path(__file__).resolve().parent.parent / "data" / "test_data.csv",
    Path(__file__).resolve().parent.parent / "data" / "test_data_private.csv",
)


def ImportTestData(paths=DEFAULT_TEST_PATHS):
    """Competition test set (public + private part) in the same format as
    :func:`ImportTrainingData`: ``(X_events, y)``, binary target, ``risk`` dropped.

    The private file stores the label column as ``true_risk``; it is renamed to
    ``risk`` before the two files are concatenated. Used by ``RNN_model.ipynb``.
    Not part of the shared split: never use it for model selection.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df.rename(columns={"true_risk": "risk"}))
    data = pd.concat(frames, ignore_index=True)

    drop_columns = list(ID_COLUMNS) + RISK_FEATURE_COLUMNS
    X_events, y = [], []
    for _, inputs, targets in _iter_events(data):
        X_events.append(inputs.drop(columns=[c for c in drop_columns if c in inputs.columns]))
        y.append(int(float(targets.iloc[-1]["risk"]) >= HIGH_RISK_THRESHOLD))
    return X_events, y


def cost(y_prob, y_pred, y_test):
    """Emil's score: MSE of the probabilities divided by F2 (inf if F2 == 0)."""
    mse = mean_squared_error(y_test, y_prob)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    return mse / f2 if f2 > 0 else float("inf")


def evaluate(model, X_test, y_test, threshold=0.01):
    """``cost`` of a fitted classifier with ``predict_proba`` at a given threshold."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return cost(y_prob, y_pred, y_test)


# ---------------------------------------------------------------------------
# Event-level feature construction
# ---------------------------------------------------------------------------
#
# Two representations are supported, selected with ``mode``:
#
# "latest"
#     One row per event: the *single* CDM closest to the 2-day cutoff. This is
#     the literal "what does an operator see two days before TCA" model, and is
#     the representation used for the main FFNN experiments.
#
# "aggregate"
#     One row per event summarising the whole eligible history: constant columns
#     taken as-is, varying columns summarised by mean / std / first / last, plus
#     ``n_cdms``. Strictly more information, but it is a *history* model and no
#     longer comparable to a single-CDM decision rule.
#
# Which columns count as "constant within an event" is a property of the data
# that must be decided from the **training events only** -- otherwise the
# feature definition itself has seen the validation/test events. That decision
# is therefore frozen into a "feature spec" by ``fit_feature_spec`` and replayed
# on every split by ``build_event_table``.
# ---------------------------------------------------------------------------


def fit_feature_spec(train_events: list[pd.DataFrame], mode: str = "latest") -> dict:
    """Decide the event-level column layout using **training events only**.

    Parameters
    ----------
    train_events : list of DataFrame
        The training split of ``X_events``.
    mode : {"latest", "aggregate"}

    Returns
    -------
    dict
        Serialisable description of the layout, consumed by
        :func:`build_event_table`.
    """
    if mode not in {"latest", "aggregate"}:
        raise ValueError(f"mode must be 'latest' or 'aggregate', got {mode!r}")

    columns = list(train_events[0].columns)
    numeric = [c for c in columns if c not in CATEGORICAL_COLUMNS]
    categorical = [c for c in columns if c in CATEGORICAL_COLUMNS]

    spec = {"mode": mode, "numeric": numeric, "categorical": categorical}

    if mode == "aggregate":
        # A column is "varying" if it changes within at least one training event.
        varying, constant = [], []
        for col in numeric:
            changes = any(event[col].nunique(dropna=False) > 1 for event in train_events)
            (varying if changes else constant).append(col)
        spec["constant"] = constant
        spec["varying"] = varying

    return spec


def build_event_table(events: list[pd.DataFrame], spec: dict) -> pd.DataFrame:
    """Collapse each event into exactly one row, following a fitted ``spec``.

    The returned frame is *raw*: still containing NaNs and the raw categorical
    columns. Imputation / encoding / scaling happen afterwards in
    :func:`make_preprocessor`, which is fitted on the training rows only.
    """
    mode = spec["mode"]
    rows = []

    for event in events:
        if mode == "latest":
            # Last row == smallest time_to_tca among eligible CDMs == the most
            # recent information available at the decision point.
            row = event.iloc[-1].copy()
            row["n_cdms"] = len(event)
        else:
            num = event[spec["numeric"]]
            summary = pd.concat(
                [
                    num[spec["constant"]].iloc[-1],
                    num[spec["varying"]].mean().add_suffix("_mean"),
                    num[spec["varying"]].std().fillna(0.0).add_suffix("_std"),
                    num[spec["varying"]].iloc[0].add_suffix("_first"),
                    num[spec["varying"]].iloc[-1].add_suffix("_last"),
                    # For categoricals keep the latest value: e.g. if
                    # c_object_type was refined from TBA to DEBRIS, the refined
                    # label is what the operator would act on.
                    event[spec["categorical"]].iloc[-1],
                ]
            )
            summary["n_cdms"] = len(event)
            row = summary
        rows.append(row)

    table = pd.DataFrame(rows).reset_index(drop=True)

    # ``pd.concat`` of Series can leave object dtype on numeric columns.
    for col in table.columns:
        if col not in CATEGORICAL_COLUMNS:
            table[col] = pd.to_numeric(table[col], errors="coerce")

    return table


def make_preprocessor(
    train_table: pd.DataFrame,
    categorical_columns: list[str] | None = None,
) -> ColumnTransformer:
    """Imputation + one-hot encoding + standardisation, as one fittable object.

    The column *lists* are read off ``train_table`` (the training split only),
    so the layout is decided without looking at held-out events. Fitting on the
    training rows and then calling ``transform`` on validation/test guarantees
    that medians, means, standard deviations and the one-hot vocabulary are all
    estimated without seeing held-out events. A single fitted ``OneHotEncoder``
    also guarantees an identical column layout across splits, which independent
    ``pd.get_dummies`` calls do not.

    Note that ``mission_id`` is integer-valued but *categorical*, so it is
    explicitly excluded from the numeric block; otherwise it would be both
    scaled and one-hot encoded.
    """
    categorical_columns = list(categorical_columns or CATEGORICAL_COLUMNS)
    categorical_columns = [c for c in categorical_columns if c in train_table.columns]

    numeric_columns = [
        c
        for c in train_table.columns
        if c not in categorical_columns
        and pd.api.types.is_numeric_dtype(train_table[c])
    ]

    numeric_pipe = Pipeline(
        [
            # Median is robust to the heavy tails in the covariance columns.
            # keep_empty_features keeps the column count stable even if some
            # column happens to be all-NaN inside a CV fold.
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Backwards-compatible helpers
# ---------------------------------------------------------------------------


def aggregate(X_events: list[pd.DataFrame]) -> pd.DataFrame:
    """Backwards-compatible aggregation used by ``XGBoost.ipynb``.

    Equivalent to ``build_event_table(X_events, fit_feature_spec(X_events,
    "aggregate"))`` followed by ``pd.get_dummies``.

    .. warning::
       The feature spec and the dummy columns are derived from *all* events
       passed in. Calling this on the full dataset before splitting lets the
       feature definition see held-out events. New code should use
       :func:`fit_feature_spec` on the training split only.
    """
    spec = fit_feature_spec(X_events, mode="aggregate")
    table = build_event_table(X_events, spec)
    X_agg = pd.get_dummies(table, dtype=int)

    # Added upstream (Emil): compress the enormous position-covariance determinants.
    cov_cols = [c for c in X_agg.columns if "position_covariance_det" in c]
    X_agg[cov_cols] = np.sign(X_agg[cov_cols]) * np.log1p(np.abs(X_agg[cov_cols]))
    return X_agg


def kelvins_loss(y_true, y_pred, *, beta: int = 2):
    """ESA-challenge-style score: MSE over high-risk events divided by F-beta.

    Parameters
    ----------
    y_true : array-like of float
        True final risk (log10 Pc), **not** thresholded.
    y_pred : array-like of float
        Predicted final risk (log10 Pc), **not** thresholded.
    beta : int
        Weight on recall in the F-score.

    Notes
    -----
    * Argument order is ``(y_true, y_pred)``, matching scikit-learn. The previous
      version of this file called ``fbeta_score(y_pred, y_test, ...)``, which
      silently swaps precision and recall and therefore reports the wrong
      F-score whenever the two classes are imbalanced -- which here they very
      much are.
    * ``fbeta_score`` needs *discrete* labels, so both arrays are thresholded at
      :data:`HIGH_RISK_THRESHOLD` before scoring. Passing continuous
      probabilities or risks straight in raises in scikit-learn.
    * The MSE term is computed over events that are **truly** high risk, which is
      what the ESA challenge scores; on an all-negative subset it is undefined
      and ``nan`` is returned.
    * This is a *reporting* metric, not a training loss: the thresholding makes
      it non-differentiable. Regression models are trained with plain MSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    true_high = y_true >= HIGH_RISK_THRESHOLD
    pred_high = y_pred >= HIGH_RISK_THRESHOLD

    f2 = fbeta_score(true_high, pred_high, beta=beta, zero_division=0.0)
    if not true_high.any() or f2 == 0.0:
        return float("nan")

    mse_high = mean_squared_error(y_true[true_high], y_pred[true_high])
    return float(mse_high / f2)


def loss(y_true, y_pred, *, beta: int = 2):
    """Deprecated alias for :func:`kelvins_loss` with corrected argument order."""
    warnings.warn(
        "read_data.loss() is deprecated; use kelvins_loss(y_true, y_pred). "
        "Note the argument order is (y_true, y_pred), as in scikit-learn.",
        DeprecationWarning,
        stacklevel=2,
    )
    return kelvins_loss(y_true, y_pred, beta=beta)


# ---------------------------------------------------------------------------
# Event-level splitting and design matrices
# ---------------------------------------------------------------------------


def event_level_split(
    event_ids,
    y,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    stratify_on=None,
    seed: int = 3155,
) -> dict[str, np.ndarray]:
    """Split **events** (never CDMs) into train / validation / test index arrays.

    Each element of ``event_ids`` is one conjunction event, and
    :func:`build_event_table` collapses each event to exactly one row, so
    splitting rows here *is* splitting events: it is impossible for two CDMs of
    the same event to land in different splits. The uniqueness assertion below
    makes that guarantee explicit rather than implicit.

    ``stratify=True`` keeps the (very small) positive class proportional across
    the three splits, which matters a great deal when well under 1 % of events
    are positive. For regression, pass the *binarised* target as ``stratify_on``
    so the handful of high-risk events are still spread proportionally while
    ``y`` itself stays continuous.
    """
    from sklearn.model_selection import train_test_split

    event_ids = np.asarray(event_ids)
    y = np.asarray(y)
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("event_ids are not unique -- the table is not one row per event")

    idx = np.arange(len(event_ids))
    strat_source = np.asarray(stratify_on) if stratify_on is not None else y
    strat = strat_source if stratify else None

    train_idx, hold_idx = train_test_split(
        idx, test_size=val_size + test_size, random_state=seed,
        stratify=strat,
    )
    # Split the held-out part so that `test_size` of the *original* data is test.
    rel_test = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        hold_idx, test_size=rel_test, random_state=seed,
        stratify=(strat_source[hold_idx] if stratify else None),
    )

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    return {"train": np.sort(train_idx), "val": np.sort(val_idx), "test": np.sort(test_idx)}


def prepare_design_matrices(
    X_events,
    y,
    event_ids,
    *,
    mode: str = "latest",
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify: bool = True,
    stratify_on=None,
    seed: int = 3155,
) -> dict:
    """End-to-end: split events, then fit all preprocessing on the training split.

    Order of operations matters and is the whole point of this function:

    1. split the **events** first;
    2. decide the feature layout (``fit_feature_spec``) from training events only;
    3. build the one-row-per-event tables for all three splits with that layout;
    4. fit imputer / one-hot encoder / scaler on the **training rows only** and
       ``transform`` validation and test.

    Doing step 4 before step 1 -- the usual mistake -- leaks the held-out events'
    medians and standard deviations into the training representation.

    Returns a dict with ``X_train/X_val/X_test`` (float32 ndarrays),
    ``y_train/y_val/y_test``, the matching ``ids_*``, ``feature_names``,
    ``preprocessor``, ``feature_spec`` and the raw ``tables``.
    """
    y = np.asarray(y)
    event_ids = np.asarray(event_ids)
    splits = event_level_split(
        event_ids, y, val_size=val_size, test_size=test_size,
        stratify=stratify, stratify_on=stratify_on, seed=seed,
    )

    train_events = [X_events[i] for i in splits["train"]]
    spec = fit_feature_spec(train_events, mode=mode)

    tables = {
        name: build_event_table([X_events[i] for i in idx], spec)
        for name, idx in splits.items()
    }

    pre = make_preprocessor(tables["train"])
    X_train = pre.fit_transform(tables["train"]).astype(np.float32)
    X_val = pre.transform(tables["val"]).astype(np.float32)
    X_test = pre.transform(tables["test"]).astype(np.float32)

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y[splits["train"]], "y_val": y[splits["val"]], "y_test": y[splits["test"]],
        "ids_train": event_ids[splits["train"]],
        "ids_val": event_ids[splits["val"]],
        "ids_test": event_ids[splits["test"]],
        "feature_names": list(pre.get_feature_names_out()),
        "preprocessor": pre,
        "feature_spec": spec,
        "splits": splits,
        "tables": tables,
    }


# ---------------------------------------------------------------------------
# Shared PCA pipeline for all three models (RNN / FFNN / XGBoost)
# ---------------------------------------------------------------------------
#
# One function builds every representation the group uses, from one event-level
# split, so the three models are trained and tested on the *same* events:
#
#   sequences  -- one time-ordered DataFrame per event, CDM-level, PCA-reduced
#                 (for the RNN; same shape as ImportTrainingData() returned)
#   aggregate  -- one row per event: mean / std / first / last of every
#                 sequence column + n_cdms (for the FFNN and XGBoost)
#
# Order of operations, each step fitted on TRAINING events only:
#
#   split events  ->  log10 on heavy-tailed columns  ->  median-impute
#   ->  standardise  ->  PCA (keep `variance_threshold` of the variance)
#   ->  [RNN: done]  ->  aggregate per event  ->  [FFNN / XGBoost: done]
#
# Why this order: PCA cannot take NaNs, and without scaling the covariance
# determinants (up to 1e46) would *be* the first component by themselves.
# Why no second PCA after aggregating: 4k+1 columns is already compact, and a
# second unsupervised reduction only compounds the information loss of the
# first.
#
# What bypasses the PCA: PCA keeps the highest-*variance* directions, not the
# most *predictive* ones. `risk` is the single strongest predictor, and
# `time_to_tca` is what makes the sequence a time series, so they (and the
# categoricals) pass through untouched rather than being rotated into 50
# components.
# ---------------------------------------------------------------------------

#: Numeric columns that bypass the PCA. `risk`-derived ones are removed from
#: this list when ``include_current_risk=False``.
PASSTHROUGH_COLUMNS = ["time_to_tca", "risk", "max_risk_estimate", "max_risk_scaling"]

#: Default share of variance the PCA keeps. On this data 0.95 keeps ~49 of 96
#: geometry/covariance columns; the variance is spread thinly (PC1 ~ 11 %).
DEFAULT_VARIANCE_THRESHOLD = 0.95


def _heavy_tailed_columns(cdm: pd.DataFrame, columns: list[str]) -> list[str]:
    """Non-negative columns whose 99th percentile exceeds 1000x their median.

    These are the covariance sigmas / determinants, which span 40 orders of
    magnitude and contain what look like sentinel values (several columns share
    the exact maximum 6.38e+07). A log10 makes them usable; the choice of
    columns is made on training CDMs only and replayed on the other splits.
    """
    out = []
    for c in columns:
        s = cdm[c].dropna()
        if len(s) == 0 or s.min() < 0:
            continue
        med, q99 = s.quantile(0.5), s.quantile(0.99)
        if (abs(q99) + 1e-12) / (abs(med) + 1e-12) > 1e3:
            out.append(c)
    return out


def _apply_log10(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in columns:
        df[c] = np.log10(df[c].clip(lower=0) + 1e-12)
    return df


def build_shared_datasets(
    path: str | Path = DEFAULT_TRAIN_PATH,
    *,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
    include_current_risk: bool = True,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 3155,
    cv_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """Build the PCA-reduced sequence and aggregate datasets for all models.

    Returns
    -------
    dict with keys
        ``splits``          : {"train"|"val"|"test": ndarray of event ids}
        ``y_class``         : {split: int array, final_risk >= -6}
        ``y_reg``           : {split: float array, final risk}
        ``sequences``       : {split: list of DataFrame}, one time-ordered
                              (far -> close to TCA) frame per event with the
                              columns in ``sequence_columns``; for the RNN
        ``sequence_columns``: passthrough + pc_1..pc_k + one-hot names
        ``aggregate``       : {split: DataFrame}, one row per event; for the
                              FFNN. Not re-scaled: fit a scaler on
                              ``aggregate["train"]`` if the model needs one.
        ``aggregate_trees`` : same events, columns and PCA as ``aggregate``, but
                              the non-PCA columns are in real units (not
                              centred or scaled, NaNs kept); for XGBoost.
        ``cv_folds``        : list of (fit_idx, val_idx) over the dev set =
                              train rows followed by val rows; shared by all.
        ``event_meta``      : {split: DataFrame} with event_id, n_cdms,
                              last_risk (raw, from the last pre-cutoff CDM) and
                              final_risk -- for baselines and plots
        ``n_components``, ``explained_variance_ratio``, ``log_columns``,
        ``pca_columns``, ``passthrough_columns``, ``transformer`` (fitted)
    """
    from sklearn.decomposition import PCA

    train = load_raw(path)
    ids, inputs, final_risk = [], [], []
    for event_id, inp, tgt in _iter_events(train):
        ids.append(event_id)
        inputs.append(inp.drop(columns=ID_COLUMNS))
        final_risk.append(float(tgt.iloc[-1]["risk"]))
    ids = np.asarray(ids)
    y_reg = np.asarray(final_risk)
    y_class = (y_reg >= HIGH_RISK_THRESHOLD).astype(int)

    # 1. one shared, stratified, event-level split
    splits = event_level_split(ids, y_class, val_size=val_size, test_size=test_size,
                               stratify=True, seed=seed)

    # 2. column roles
    passthrough = list(PASSTHROUGH_COLUMNS)
    if not include_current_risk:
        passthrough = [c for c in passthrough if c not in RISK_FEATURE_COLUMNS]
    all_cols = list(inputs[0].columns)
    categorical = [c for c in all_cols if c in CATEGORICAL_COLUMNS]
    pca_cols = [c for c in all_cols
                if c not in categorical and c not in PASSTHROUGH_COLUMNS
                and c not in RISK_FEATURE_COLUMNS]

    # 3. fit everything on the training CDM rows
    train_cdm = pd.concat([inputs[i] for i in splits["train"]], ignore_index=True)
    log_columns = _heavy_tailed_columns(train_cdm, pca_cols)

    pca_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=variance_threshold, svd_solver="full", random_state=seed)),
    ])
    pass_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer(
        [("pass", pass_pipe, passthrough), ("pca", pca_pipe, pca_cols), ("cat", cat_pipe, categorical)],
        remainder="drop", verbose_feature_names_out=False,
    )
    transformer.fit(_apply_log10(train_cdm, log_columns))

    pca = transformer.named_transformers_["pca"].named_steps["pca"]
    n_components = int(pca.n_components_)
    onehot_names = list(transformer.named_transformers_["cat"].get_feature_names_out(categorical))
    sequence_columns = passthrough + [f"pc_{i + 1}" for i in range(n_components)] + onehot_names
    n_numeric = len(passthrough) + n_components

    if verbose:
        print(f"shared datasets: {len(ids)} events | PCA keeps {n_components} of {len(pca_cols)} "
              f"columns for {variance_threshold:.0%} variance | log10 on {len(log_columns)} columns | "
              f"{len(sequence_columns)} sequence columns")

    # 4. transform every event, build sequences + aggregates. All CDM rows of a
    #    split go through the transformer in ONE call (per-event calls are ~100x
    #    slower), then are sliced back per event.
    numeric_cols = sequence_columns[:n_numeric]
    aggregate_columns = (
        [f"{c}_mean" for c in numeric_cols] + [f"{c}_std" for c in numeric_cols]
        + [f"{c}_first" for c in numeric_cols] + [f"{c}_last" for c in numeric_cols]
        + onehot_names + ["n_cdms"]
    )

    sequences, aggregate, aggregate_trees, event_meta = {}, {}, {}, {}
    for name, idx in splits.items():
        lengths = np.array([len(inputs[i]) for i in idx])
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        big = pd.concat([inputs[i] for i in idx], ignore_index=True)
        Z = transformer.transform(_apply_log10(big, log_columns)).astype(np.float32)

        # Tree-model copy: pass-through columns in their REAL units (no
        # centring / scaling, NaNs left for XGBoost). PC scores cannot be
        # "un-scaled": they are centred by construction, so they stay as is.
        Z_tree = Z.copy()
        Z_tree[:, :len(passthrough)] = big[passthrough].to_numpy(dtype=np.float32)

        def _agg(Zi):
            num = Zi[:, :n_numeric]
            std = num.std(axis=0, ddof=1) if len(num) > 1 else np.zeros(n_numeric, dtype=np.float32)
            return np.concatenate([num.mean(axis=0), std, num[0], num[-1],
                                   Zi[-1, n_numeric:],   # one-hots: latest value
                                   [len(num)]])

        seqs, rows, rows_tree, meta = [], [], [], []
        for j, i in enumerate(idx):
            sl = slice(offsets[j], offsets[j + 1])
            seqs.append(pd.DataFrame(Z[sl], columns=sequence_columns))
            rows.append(_agg(Z[sl]))
            rows_tree.append(_agg(Z_tree[sl]))
            ev = inputs[i]
            meta.append({"event_id": ids[i], "n_cdms": len(ev),
                         "last_risk": float(ev.iloc[-1]["risk"]), "final_risk": y_reg[i]})

        sequences[name] = seqs
        aggregate[name] = pd.DataFrame(np.vstack(rows), columns=aggregate_columns).astype(np.float32)
        aggregate_trees[name] = pd.DataFrame(np.vstack(rows_tree), columns=aggregate_columns).astype(np.float32)
        event_meta[name] = pd.DataFrame(meta)

    # 5. shared cross-validation folds over the development set (train + val,
    #    in that order). Everyone who compares models should use these.
    y_dev = np.concatenate([y_class[splits["train"]], y_class[splits["val"]]])
    cv_folds = shared_cv_folds(y_dev, n_splits=cv_splits, seed=seed)

    return {
        "splits": splits,
        "cv_folds": cv_folds,
        "aggregate_trees": aggregate_trees,
        "event_ids": {k: ids[v] for k, v in splits.items()},
        "y_class": {k: y_class[v] for k, v in splits.items()},
        "y_reg": {k: y_reg[v] for k, v in splits.items()},
        "sequences": sequences,
        "sequence_columns": sequence_columns,
        "aggregate": aggregate,
        "event_meta": event_meta,
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "log_columns": log_columns,
        "pca_columns": pca_cols,
        "passthrough_columns": passthrough,
        "transformer": transformer,
    }


def shared_cv_folds(y_dev, n_splits: int = 5, seed: int = 3155):
    """Stratified K-fold indices over the development set (train rows, then val rows).

    Everyone comparing models must use the same folds, otherwise differences
    between models are partly differences between folds -- and with ~11
    positives per fold that noise is as large as the effects being measured.
    Indices refer to ``np.concatenate([X_train, X_val])`` in that order.
    """
    from sklearn.model_selection import StratifiedKFold

    y_dev = np.asarray(y_dev)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y_dev)), y_dev)]


def ImportSharedSequences(split: str = "train", **kwargs):
    """RNN-friendly accessor: ``X_events, y = ImportSharedSequences("train")``.

    Same shape as :func:`ImportTrainingData` -- a list of time-ordered
    per-event DataFrames and a list of binary labels -- but PCA-reduced,
    imputed, scaled, and restricted to one split of the shared event-level
    split, so that train / validation / test events agree with the other
    models. Pass ``split="val"`` or ``"test"`` for the others. Building the
    datasets takes ~30 s; call :func:`build_shared_datasets` once instead if
    several splits are needed.
    """
    data = build_shared_datasets(**kwargs)
    return data["sequences"][split], list(data["y_class"][split])
