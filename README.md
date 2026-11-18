# FYS-STK3155 — Project 3: predicting satellite collision risk

Models for the [ESA Collision Avoidance Challenge](https://kelvins.esa.int/collision-avoidance-challenge/)
data: sequences of *conjunction data messages* (CDMs) issued in the days before a
predicted close approach between a satellite and another object. Each CDM carries
a collision risk `risk = log10(Pc)`. The task is to say, **two days before closest
approach**, what the final risk will be.

## Setup (reproducible with `uv`)

```bash
# install uv once: https://docs.astral.sh/uv/getting-started/installation/
uv sync                       # creates .venv with the pinned versions in uv.lock
uv run jupyter lab            # or open the notebooks in VS Code with the .venv kernel
```

Linux/Windows get the CPU-only PyTorch build (the models train in minutes on
CPU; no GPU needed). Run all notebooks from inside `code/` — they read
`../data/train_data.zip` and write to `../results/`.

To re-execute a notebook headlessly:

```bash
cd code
uv run jupyter nbconvert --to notebook --execute --inplace FFNN_classification.ipynb
```

## Layout

```
data/      train_data.zip (labelled CDMs), test_data.csv (competition set, NO labels — never used for selection)
code/
  read_data.py            event filtering, two importers (binary / continuous target),
                          latest-CDM vs aggregated features, leakage-safe preprocessing,
                          event-level splitting, corrected ESA MSE/F2 score
  ffnn_core.py            PyTorch FFNN, LeCun/Xavier/He init, training loop with early
                          stopping, weighted BCE, BCE + soft-F2, focal loss, L1/L2,
                          sklearn wrappers
  plot_style.py           shared colour-blind-safe figure style for the report
  FFNN_classification.ipynb
  XGBoost.ipynb, RNN_model.ipynb   (other members' models)
results/
  classification/{figures,tables}/   PNGs and CSV twins of every figure
bin/
  FFNN_regression.ipynb, results/regression/   the regression study, archived after the
                                               group chose classification (still runnable
                                               from code/ if copied back)
```

## Shared data pipeline (all three models)

`read_data.build_shared_datasets()` builds every representation the group uses
from **one** event-level split, so the RNN, FFNN and XGBoost are trained and
tested on the same events:

```python
import numpy as np, read_data as rd
shared = rd.build_shared_datasets()     # ~20 s; same split, PCA and folds for everyone
y = shared["y_class"]                   # {"train","val","test"} -> 0/1 labels
```

| Who | Import | What it is |
|---|---|---|
| **Jon (RNN)** | `shared["sequences"]["train"]` (also `"val"`, `"test"`) | list of per-event DataFrames, CDM level, PCA-reduced. Columns: `shared["sequence_columns"]`. Drop-in for the old `X_events, y`: `rd.ImportSharedSequences("train")` |
| **Kevin (FFNN)** | `shared["aggregate"]["train"]` | one row per event, 233 features, centred; fit a `StandardScaler` on the train rows |
| **Emil (XGBoost)** | `shared["aggregate_trees"]["train"]` | same events, columns and PCA as Kevin's, but pass-through columns (`risk`, `time_to_tca`, ...) in real units, not centred or scaled. PC scores are centred by construction |

**Same validation set and folds for everyone.** The train / val / test events are
identical in all three datasets, and model selection uses the same 5-fold
stratified CV over the development set (train rows followed by val rows):

```python
folds = shared["cv_folds"]                          # list of (fit_idx, val_idx)
X_dev = np.vstack([X_train, X_val]); y_dev = np.concatenate([y["train"], y["val"]])
GridSearchCV(model, grid, cv=folds, ...)            # or: for fit_idx, val_idx in folds: ...
```

Each fold holds ~11 positives, so fold-to-fold noise is as large as most model
differences; scoring on different folds would make results incomparable. Touch
the test split once, at the end.

Steps, each fitted on training events' CDMs only, then applied to the other
splits:

1. **Event filtering** — inputs are CDMs with `time_to_tca >= 2`, the target is
   read from the CDM closest to TCA (`time_to_tca < 1`); events lacking either
   side are dropped. 8 293 events survive, 66 positive (0.8 %).
2. **Stratified 70/15/15 event-level split** (seed 3155). No event appears in
   two splits.
3. **log10** on the 15 non-negative heavy-tailed columns (covariance sigmas
   and determinants; values reach 1e46 and several columns share sentinel
   maxima such as 6.38e+07).
4. **Median imputation** and **standardisation**.
5. **PCA on the 96 orbital-geometry / covariance columns**, keeping 95 % of
   the variance → 48 components. PC 1 carries only 11 %, so the variance is
   spread thin and the PCA roughly halves this block rather than collapsing
   it. `time_to_tca`, `risk`, `max_risk_estimate`, `max_risk_scaling` and the
   two categoricals (`mission_id`, `c_object_type`, one-hot) bypass the PCA:
   PCA keeps high-*variance* directions, not high-*predictive* ones, and
   `risk` is the strongest predictor there is.
6. **Aggregation** per event (mean / std / first / last of every numeric
   sequence column, latest one-hot values, `n_cdms`) for the FFNN and XGBoost.
   There is deliberately **no second PCA** after aggregation — a second
   unsupervised reduction only compounds the information loss of the first.

PCA must come *after* imputation and scaling (it cannot take NaNs, and the
1e46 covariance determinants would otherwise be PC 1 on their own), and must be
fitted on training events only — both are enforced inside the function.

`include_current_risk=False` removes the three risk-derived columns from the
pass-through block (they never enter the PCA) for the ablation.

## The FFNN study

* **Model selection by 5-fold stratified CV** on train+validation, because only
  66 events (0.8 %) are positive and a single validation split holds ~10 of them.
  The test split is opened once, at the end.

### Classification — `FFNN_classification.ipynb`

Predict whether the final risk is at least −6 (`Pc >= 1e-6`). Compares
activation functions × initialisers (3×3); loss functions — weighted BCE,
BCE + λ(1 − soft-F2) over a λ sweep, focal loss, and the MSE/F2 ratio; four
architectures; learning rate × weight decay via `GridSearchCV`; and selects the
decision threshold from out-of-fold predictions (max-F2 and 95 %-recall rules). Reports F2, recall, precision, PR-AUC, ROC-AUC,
confusion matrix and review workload, and benchmarks against the untrained rule
"flag if current risk ≥ −6", logistic regression, random forest and gradient
boosting.

### Regression — `bin/FFNN_regression.ipynb` (archived)

Predict the continuous final `log10(Pc)`. Same activation / initialiser /
architecture comparison, then L1 vs. L2 regularisation (weight norms and
sparsity recorded), a bootstrap bias–variance decomposition across model
complexities, and MSE / RMSE / MAE / R² / Spearman on all events, on events
above the −30 reporting floor, and on the high-risk region. Benchmarked against
the persistence rule "risk stays unchanged", OLS, Ridge, Lasso, random forest
and gradient boosting.

## Changes made to `read_data.py`

* `ImportTrainingData()` kept for backwards compatibility (binary target, risk
  excluded, two-element return). New `ImportTrainingDataClassification()` and
  `ImportTrainingDataRegression()` share the filtering, return `event_ids`, and
  expose `include_current_risk`.
* `loss()` had `fbeta_score(y_pred, y_true)` — argument order swapped, which
  silently swaps precision and recall. Replaced by `kelvins_loss(y_true, y_pred)`
  with explicit thresholding; `loss()` remains as a deprecated alias.
* Teammates' additions from upstream are kept: `ImportTestData()` (public + private test
  files, binary target), `cost()` / `evaluate()` (MSE/F2 score) and the log of the
  covariance determinants inside `aggregate()`.
* Target-row selection is documented: rows are sorted by `time_to_tca`
  descending, so `targets.iloc[-1]` is the observation closest to TCA.
* `aggregate()` kept, but new code should use `build_shared_datasets()` (above),
  which replaces the per-split `pd.get_dummies` calls with one fitted
  `OneHotEncoder` and adds the shared split and PCA. `fit_feature_spec` /
  `build_event_table` / `make_preprocessor` / `prepare_design_matrices` are the
  earlier, PCA-free pipeline and are still used by the archived regression
  notebook.
