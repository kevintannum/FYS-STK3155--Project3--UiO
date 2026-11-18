"""Reusable feed-forward neural-network machinery for FYS-STK3155 Project 3.

Everything that both the classification and the regression notebook need lives
here: seeding, the configurable :class:`FFNN`, the weight initialisers, the
training loop with early stopping, and thin scikit-learn wrappers so that
``GridSearchCV`` can drive PyTorch models.

The task-specific experiments (what is compared against what, which metrics are
reported, which plots are drawn) stay in the notebooks.
"""

from __future__ import annotations

import copy
import random
import time
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import average_precision_score, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------------------------


def set_seed(seed: int = 3155) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    Called before *every* model fit in the comparison experiments, so that two
    architectures are not separated merely by their random draw.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Activations and initialisation
# ---------------------------------------------------------------------------

#: Negative slope used wherever Leaky ReLU appears, including in the He/Kaiming
#: gain calculation.
LEAKY_SLOPE = 0.01

ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=LEAKY_SLOPE),
}

INITIALIZATIONS = ("lecun", "xavier", "he")


def lecun_normal_(tensor: torch.Tensor) -> torch.Tensor:
    """LeCun normal: ``W ~ N(0, 1 / fan_in)``.

    Preserves the forward-pass variance for a roughly linear activation.
    PyTorch has no built-in for this, so it is written out.
    """
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = np.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0.0, std)


def init_layer_(layer: nn.Linear, method: str, activation: str, *, output_layer: bool) -> None:
    """Initialise one linear layer in place; biases are always zero.

    * ``lecun``  -- ``N(0, 1/fan_in)``.
    * ``xavier`` -- ``N(0, 2/(fan_in+fan_out))``; compromises between preserving
      forward and backward variance.
    * ``he``     -- ``N(0, 2/fan_in)`` for ReLU/sigmoid, and the slope-corrected
      ``N(0, 2/((1+a^2) fan_in))`` for Leaky ReLU, compensating for ReLU zeroing
      roughly half of its inputs.

    The **output layer always uses Xavier**, regardless of ``method``: it feeds a
    logit or a linear output, not a hidden activation, so the ReLU correction
    does not apply there. Keeping it fixed also means the 3x3 grid varies only
    the hidden layers.
    """
    if output_layer:
        nn.init.xavier_normal_(layer.weight)
    elif method == "lecun":
        lecun_normal_(layer.weight)
    elif method == "xavier":
        nn.init.xavier_normal_(layer.weight)
    elif method == "he":
        if activation == "leaky_relu":
            nn.init.kaiming_normal_(
                layer.weight, a=LEAKY_SLOPE, mode="fan_in", nonlinearity="leaky_relu"
            )
        else:
            # gain = sqrt(2) -> std = sqrt(2 / fan_in), the textbook He variance.
            # Applied to sigmoid too, so that "He" means the same formula in
            # every cell of the grid rather than silently collapsing to LeCun.
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    else:
        raise ValueError(f"unknown initialization {method!r}; expected one of {INITIALIZATIONS}")

    nn.init.zeros_(layer.bias)


class FFNN(nn.Module):
    """A configurable fully-connected feed-forward network.

    Parameters
    ----------
    input_dim : int
    hidden_dims : sequence of int
        Width of each hidden layer, e.g. ``[100, 100]`` or ``[128, 64, 32]``.
        An empty sequence gives a plain linear model (useful as a baseline).
    activation : {"sigmoid", "relu", "leaky_relu"}
        Applied after every hidden layer.
    initialization : {"lecun", "xavier", "he"}
        Applied to the hidden layers; the output layer always uses Xavier.
    output_dim : int
        1 for binary classification (a single logit) and for scalar regression.

    Notes
    -----
    ``forward`` returns the **raw** output: a logit for classification (no
    sigmoid) and an unconstrained real number for regression (no squashing).
    Keeping the sigmoid out of the graph lets training use
    ``BCEWithLogitsLoss``, which is numerically stable because it fuses the
    log-sum-exp.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (100, 100),
        activation: str = "relu",
        initialization: str = "he",
        output_dim: int = 1,
    ):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; expected {list(ACTIVATIONS)}")

        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.initialization = initialization

        layers: list[nn.Module] = []
        prev = input_dim
        for width in self.hidden_dims:
            linear = nn.Linear(prev, width)
            init_layer_(linear, initialization, activation, output_layer=False)
            layers += [linear, ACTIVATIONS[activation]()]
            prev = width

        out = nn.Linear(prev, output_dim)
        init_layer_(out, initialization, activation, output_layer=True)
        layers.append(out)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (the network's degrees of freedom)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weight_matrices(model: nn.Module):
    """Yield the weight tensors only, skipping biases.

    Used by the L1/L2 penalties and by the sparsity diagnostics: biases are
    deliberately left unregularised, as in the Project 2 derivations.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            yield module.weight


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def make_bce_loss(pos_weight: float | None, device: torch.device) -> nn.Module:
    """Weighted binary cross-entropy on logits.

    ``pos_weight`` multiplies the positive term of the loss, so a value of
    ``n_negative / n_positive`` makes a missed high-risk event about as
    expensive as mislabelling every negative. It must be estimated from the
    **training fold only**.
    """
    kwargs = {}
    if pos_weight is not None:
        kwargs["pos_weight"] = torch.tensor([float(pos_weight)], device=device)
    return nn.BCEWithLogitsLoss(**kwargs)


def soft_f2_mse_loss(logits: torch.Tensor, targets: torch.Tensor, *, beta: float = 2.0,
                     eps: float = 1e-7) -> torch.Tensor:
    """Differentiable stand-in for the ESA ``MSE / F2`` score.

    ``fbeta_score`` needs hard 0/1 labels, so it has zero gradient almost
    everywhere and cannot be optimised directly. Replacing the counts by their
    *expected* values under the predicted probabilities ``p`` gives a smooth
    surrogate:

    ``TP = sum(p*y)``, ``FP = sum(p*(1-y))``, ``FN = sum((1-p)*y)``

    ``F_beta = (1+b^2) TP / ((1+b^2) TP + b^2 FN + FP)``

    and the loss is ``MSE(p, y) / F_beta``, mirroring the competition score
    (low MSE, high F2 -> low loss). Reported in the notebook as a
    *loss-function comparison* against weighted BCE, not as the default.
    """
    p = torch.sigmoid(logits).view(-1)
    y = targets.view(-1)

    tp = (p * y).sum()
    fp = (p * (1.0 - y)).sum()
    fn = ((1.0 - p) * y).sum()

    b2 = beta * beta
    f_beta = ((1.0 + b2) * tp) / ((1.0 + b2) * tp + b2 * fn + fp + eps)

    mse = torch.mean((p - y) ** 2)
    return mse / (f_beta + eps)


def soft_f_beta(logits: torch.Tensor, targets: torch.Tensor, *, beta: float = 2.0,
                eps: float = 1e-7) -> torch.Tensor:
    """Expected F-beta under the predicted probabilities (Tversky-type surrogate).

    With ``p = sigmoid(logits)`` the expected counts are ``TP = sum(p y)``,
    ``FP = sum(p (1-y))``, ``FN = sum((1-p) y)``, giving

    ``F_beta = (1+b^2) TP / ((1+b^2) TP + b^2 FN + FP)``.

    This is the Tversky index of Salehi et al. (2017) with
    ``alpha = 1/(1+b^2)`` on FP and ``beta = b^2/(1+b^2)`` on FN -- for b = 2,
    false negatives are weighted 0.8 against 0.2 for false positives.

    The quantity is *non-decomposable*: it is a property of the whole batch, not
    a sum over samples. A batch without any positive has ``TP = FN = 0`` and the
    term is identically zero with zero gradient, so it simply contributes
    nothing on that step. With under 1 % positives, batches must therefore be
    large enough to contain positives most of the time.
    """
    p = torch.sigmoid(logits).view(-1)
    y = targets.view(-1)
    tp = (p * y).sum()
    fp = (p * (1.0 - y)).sum()
    fn = ((1.0 - p) * y).sum()
    b2 = beta * beta
    return ((1.0 + b2) * tp) / ((1.0 + b2) * tp + b2 * fn + fp + eps)


class BCESoftFBetaLoss(nn.Module):
    """Weighted BCE plus a soft-F-beta penalty: ``BCE + lam * (1 - F_beta_soft)``.

    The "combo" form used in segmentation (BCE + Dice/Tversky; Taghanaki et al.
    2019): the cross-entropy term is a proper scoring rule that keeps the
    probabilities on a meaningful scale and the gradients well-behaved, while
    the F-beta term adds direct pressure on the recall-weighted objective. With
    ``lam = 0`` it reduces to weighted BCE.
    """

    def __init__(self, pos_weight: float | None, lam: float, beta: float,
                 device: torch.device):
        super().__init__()
        self.bce = make_bce_loss(pos_weight, device)
        self.lam = float(lam)
        self.beta = float(beta)

    def forward(self, logits, targets):
        loss = self.bce(logits, targets)
        if self.lam:
            loss = loss + self.lam * (1.0 - soft_f_beta(logits, targets, beta=self.beta))
        return loss


class FocalLoss(nn.Module):
    """Alpha-balanced focal loss on logits (Lin et al., ICCV 2017).

    ``FL = alpha_t (1 - p_t)^gamma * (-log p_t)``, where ``p_t`` is the
    probability assigned to the true class. ``(1 - p_t)^gamma`` shrinks the
    contribution of well-classified examples -- here the thousands of easy
    negatives -- so the rare, hard positives dominate the gradient.
    ``alpha_t`` is ``pos_weight`` for positives and 1 for negatives, i.e. the
    same class weighting as the BCE baseline, so the two differ only in the
    focusing term. ``gamma = 0`` recovers weighted BCE exactly.
    """

    def __init__(self, pos_weight: float | None, gamma: float, device: torch.device):
        super().__init__()
        self.gamma = float(gamma)
        self.pos_weight = float(pos_weight) if pos_weight is not None else 1.0

    def forward(self, logits, targets):
        logits = logits.view(-1)
        y = targets.view(-1)
        # -log p_t, computed stably from the logits.
        ce = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p_t = torch.exp(-ce)
        alpha_t = torch.where(y > 0.5, torch.full_like(y, self.pos_weight), torch.ones_like(y))
        return (alpha_t * (1.0 - p_t) ** self.gamma * ce).mean()


def l1_penalty(model: nn.Module) -> torch.Tensor:
    """``sum |W|`` over weight matrices only (biases excluded)."""
    return sum(w.abs().sum() for w in weight_matrices(model))


def l2_penalty(model: nn.Module) -> torch.Tensor:
    """``sum W^2`` over weight matrices only (biases excluded).

    Used when L2 is added explicitly to the loss. The alternative is AdamW's
    ``weight_decay``, which is *decoupled* -- it shrinks the weights directly
    rather than through the gradient -- and which here is also restricted to
    weights via parameter groups (see :func:`make_optimizer`).
    """
    return sum((w ** 2).sum() for w in weight_matrices(model))


def make_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0):
    """AdamW with weight decay applied to weights but **not** biases.

    PyTorch applies ``weight_decay`` to every parameter by default, biases
    included. Splitting the parameters into two groups keeps the implementation
    consistent with the theory, where the penalty term sums over ``W`` only.
    """
    decay, no_decay = [], []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            decay.append(module.weight)
            no_decay.append(module.bias)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _to_tensor(a, device: torch.device) -> torch.Tensor:
    """Dense float32 tensor on ``device`` (float32 everywhere, as specified)."""
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)


def make_loader(X, y, batch_size: int, shuffle: bool, device: torch.device,
                generator: torch.Generator | None = None) -> DataLoader:
    """``DataLoader`` over a ``TensorDataset`` of float32 tensors."""
    ds = TensorDataset(_to_tensor(X, device), _to_tensor(y, device).view(-1, 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_model(
    model: nn.Module,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    *,
    task: str = "classification",
    loss_name: str = "bce",
    pos_weight: float | None = None,
    f2_lambda: float = 0.0,
    focal_gamma: float = 2.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    l1_lambda: float = 0.0,
    l2_lambda: float = 0.0,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 20,
    grad_clip: float | None = 5.0,
    device: torch.device | None = None,
    verbose: bool = False,
):
    """Train with mini-batch AdamW and early stopping; return ``(model, history)``.

    Parameters
    ----------
    task : {"classification", "regression"}
        Selects the default loss and the quantity that early stopping watches:
        validation **average precision** (PR-AUC) for classification -- a
        threshold-free summary that behaves sensibly when under 1 % of events
        are positive -- and validation **MSE** for regression.
    loss_name : {"bce", "bce_soft_f2", "focal", "soft_f2_mse", "mse"}
        ``bce_soft_f2`` is ``BCE + f2_lambda * (1 - soft F2)``; ``focal`` uses
        ``focal_gamma``. Both share ``pos_weight`` with plain BCE.
    l1_lambda, l2_lambda : float
        Penalties added explicitly to the loss, over weights only. ``l2_lambda``
        is an alternative to ``weight_decay``; use one or the other.
    grad_clip : float or None
        Max gradient norm. The soft-F2 loss has a division in it and can spike,
        so clipping is on by default.

    Returns
    -------
    model : nn.Module
        With the **best** validation state restored (deep-copied at the best
        epoch), not the final-epoch state.
    history : dict
        Per-epoch ``train_loss``, ``val_loss``, ``val_score``, ``epoch_time``,
        ``cum_time``, plus ``best_epoch``, ``n_epochs``, ``total_time`` and
        ``best_score``.
    """
    device = device or get_device()
    model = model.to(device)

    has_val = X_val is not None and y_val is not None

    if loss_name == "bce":
        criterion = make_bce_loss(pos_weight, device)
    elif loss_name == "bce_soft_f2":
        criterion = BCESoftFBetaLoss(pos_weight, lam=f2_lambda, beta=2.0, device=device)
    elif loss_name == "focal":
        criterion = FocalLoss(pos_weight, gamma=focal_gamma, device=device)
    elif loss_name == "mse":
        criterion = nn.MSELoss()
    elif loss_name == "soft_f2_mse":
        criterion = soft_f2_mse_loss
    else:
        raise ValueError(f"unknown loss_name {loss_name!r}")

    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True, device=device)

    if has_val:
        Xv = _to_tensor(X_val, device)
        yv = _to_tensor(y_val, device).view(-1, 1)
        yv_np = np.asarray(y_val, dtype=float).ravel()

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_score": [],
        "epoch_time": [], "cum_time": [],
    }

    # Higher-is-better for classification (PR-AUC), lower-is-better for MSE.
    better = (lambda new, old: new > old) if task == "classification" else (lambda new, old: new < old)
    best_score = -np.inf if task == "classification" else np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    t_start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        t_epoch = time.perf_counter()

        model.train()
        running, n_seen = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            if l1_lambda:
                loss = loss + l1_lambda * l1_penalty(model)
            if l2_lambda:
                loss = loss + l2_lambda * l2_penalty(model)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item() * xb.shape[0]
            n_seen += xb.shape[0]

        train_loss = running / max(n_seen, 1)

        if has_val:
            model.eval()
            with torch.no_grad():
                out_v = model(Xv)
                val_loss = float(criterion(out_v, yv).item())
                if task == "classification":
                    probs = torch.sigmoid(out_v).cpu().numpy().ravel()
                    # Undefined with a single class present in the fold.
                    val_score = (
                        float(average_precision_score(yv_np, probs))
                        if len(np.unique(yv_np)) > 1 else float("nan")
                    )
                else:
                    preds = out_v.cpu().numpy().ravel()
                    val_score = float(mean_squared_error(yv_np, preds))
        else:
            val_loss, val_score = float("nan"), float("nan")

        dt = time.perf_counter() - t_epoch
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_score"].append(val_score)
        history["epoch_time"].append(dt)
        history["cum_time"].append(time.perf_counter() - t_start)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | score {val_score:.4f}")

        if has_val and np.isfinite(val_score) and better(val_score, best_score):
            best_score, best_epoch = val_score, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if has_val and epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)

    history.update(
        best_epoch=best_epoch,
        best_score=best_score,
        n_epochs=len(history["train_loss"]),
        total_time=time.perf_counter() - t_start,
    )
    return model, history


@torch.no_grad()
def predict_raw(model: nn.Module, X, device: torch.device | None = None,
                batch_size: int = 4096) -> np.ndarray:
    """Model output as a 1-D NumPy array (logits for classification)."""
    device = device or get_device()
    model = model.to(device).eval()
    X = _to_tensor(X, device)
    chunks = [model(X[i:i + batch_size]).cpu().numpy().ravel()
              for i in range(0, X.shape[0], batch_size)]
    return np.concatenate(chunks) if chunks else np.empty(0)


def predict_proba_ffnn(model: nn.Module, X, device=None) -> np.ndarray:
    """Positive-class probabilities, i.e. ``sigmoid`` applied outside the graph."""
    from scipy.special import expit
    return expit(predict_raw(model, X, device))


# ---------------------------------------------------------------------------
# scikit-learn wrappers
# ---------------------------------------------------------------------------
#
# These exist so that ``GridSearchCV`` / ``cross_validate`` can drive the
# PyTorch models directly: scikit-learn handles the stratified folds, the
# parallelism and the score bookkeeping, while the actual fitting is still the
# ``train_model`` loop above.
#
# Early stopping needs a validation set that the *search* does not know about,
# so each ``fit`` carves an internal stratified slice out of its own training
# fold. The outer CV fold therefore stays genuinely held out within the search.
#
# Only the two wrappers below are classes; everything else in this project is a
# plain function.
# ---------------------------------------------------------------------------


class TorchFFNNClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn estimator wrapping :class:`FFNN` for binary classification.

    ``pos_weight="auto"`` recomputes ``n_negative / n_positive`` inside every
    fold from that fold's training rows only.

    ``threshold`` affects ``predict`` only; ``predict_proba`` and
    ``decision_function`` are untouched, so threshold-free scorers such as
    ``average_precision`` are unaffected by it.
    """

    def __init__(
        self,
        hidden_dims=(100, 100),
        activation: str = "relu",
        initialization: str = "he",
        loss_name: str = "bce",
        pos_weight="auto",
        f2_lambda: float = 0.0,
        focal_gamma: float = 2.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        l1_lambda: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        grad_clip: float | None = 5.0,
        validation_fraction: float = 0.2,
        threshold: float = 0.5,
        seed: int = 3155,
        device=None,
    ):
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.initialization = initialization
        self.loss_name = loss_name
        self.pos_weight = pos_weight
        self.f2_lambda = f2_lambda
        self.focal_gamma = focal_gamma
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.validation_fraction = validation_fraction
        self.threshold = threshold
        self.seed = seed
        self.device = device

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32).ravel()

        set_seed(self.seed)
        device = self.device or get_device()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]

        # Internal early-stopping split, stratified so the few positives are
        # present on both sides. Falls back to no validation if that is
        # impossible (e.g. fewer than 2 positives in this fold).
        stratify = y if np.bincount(y.astype(int)).min() >= 2 else None
        if self.validation_fraction and stratify is not None:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=self.validation_fraction,
                stratify=stratify, random_state=self.seed,
            )
        else:
            X_tr, y_tr, X_va, y_va = X, y, None, None

        if self.pos_weight == "auto":
            n_pos = float(y_tr.sum())
            pw = (len(y_tr) - n_pos) / n_pos if n_pos > 0 else 1.0
        else:
            pw = self.pos_weight

        model = FFNN(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            initialization=self.initialization,
            output_dim=1,
        )
        self.model_, self.history_ = train_model(
            model, X_tr, y_tr, X_va, y_va,
            task="classification", loss_name=self.loss_name, pos_weight=pw,
            f2_lambda=self.f2_lambda, focal_gamma=self.focal_gamma,
            lr=self.lr, weight_decay=self.weight_decay, l1_lambda=self.l1_lambda,
            batch_size=self.batch_size, max_epochs=self.max_epochs,
            patience=self.patience, grad_clip=self.grad_clip, device=device,
        )
        self.pos_weight_ = pw
        self.n_epochs_ = self.history_["n_epochs"]
        self.train_time_ = self.history_["total_time"]
        return self

    def decision_function(self, X):
        """Raw logits -- what threshold-free scorers should see."""
        return predict_raw(self.model_, np.asarray(X, dtype=np.float32), self.device)

    def predict_proba(self, X):
        p = predict_proba_ffnn(self.model_, np.asarray(X, dtype=np.float32), self.device)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


class TorchFFNNRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn estimator wrapping :class:`FFNN` for scalar regression.

    The output layer is linear -- no sigmoid -- because the target is log10 Pc,
    an unbounded negative quantity.
    """

    def __init__(
        self,
        hidden_dims=(100, 100),
        activation: str = "relu",
        initialization: str = "he",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        grad_clip: float | None = 5.0,
        validation_fraction: float = 0.2,
        seed: int = 3155,
        device=None,
    ):
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.initialization = initialization
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.validation_fraction = validation_fraction
        self.seed = seed
        self.device = device

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        set_seed(self.seed)
        device = self.device or get_device()
        self.n_features_in_ = X.shape[1]

        if self.validation_fraction:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.seed
            )
        else:
            X_tr, y_tr, X_va, y_va = X, y, None, None

        model = FFNN(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            initialization=self.initialization,
            output_dim=1,
        )
        self.model_, self.history_ = train_model(
            model, X_tr, y_tr, X_va, y_va,
            task="regression", loss_name="mse",
            lr=self.lr, weight_decay=self.weight_decay,
            l1_lambda=self.l1_lambda, l2_lambda=self.l2_lambda,
            batch_size=self.batch_size, max_epochs=self.max_epochs,
            patience=self.patience, grad_clip=self.grad_clip, device=device,
        )
        self.n_epochs_ = self.history_["n_epochs"]
        self.train_time_ = self.history_["total_time"]
        return self

    def predict(self, X):
        return predict_raw(self.model_, np.asarray(X, dtype=np.float32), self.device)
