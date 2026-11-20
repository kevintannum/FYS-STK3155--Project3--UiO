import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.functional import binary_cross_entropy_with_logits
from sklearn.model_selection import train_test_split
import pandas as pd
from read_data import build_shared_datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, log_loss
import random

from sklearn.metrics import roc_auc_score

class F2VektetMSELoss(nn.Module):
    """LLM assisted (UiOGPT)"""
    def __init__(self, threshold=1e-6, beta=2.0, eps=1e-8):
        """LLM assisted (UiOGPT)"""
        super().__init__()
        self.threshold = threshold
        self.beta = beta
        self.eps = eps

    def beregn_f2(self, y_true, y_pred):
        """LLM assisted (UiOGPT)"""
        """
        Beregner F2-score basert på hele inputen.

        Positiv klasse:
            risiko >= threshold
        """
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        ekte_positiv = y_true >= self.threshold
        predikert_positiv = y_pred >= self.threshold

        tp = (ekte_positiv & predikert_positiv).sum().float()
        fp = (~ekte_positiv & predikert_positiv).sum().float()
        fn = (ekte_positiv & ~predikert_positiv).sum().float()

        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)

        beta_kvadrat = self.beta ** 2

        f2 = (
            (1 + beta_kvadrat) * precision * recall
            / (beta_kvadrat * precision + recall + self.eps)
        )

        return f2

    def forward(self, y_pred, y_true, f2_score=None):
        """LLM assisted (UiOGPT)"""
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            Predikert risiko, r_hat.

        y_true : torch.Tensor
            Fasiten / faktisk risiko, r.

        f2_score : float eller torch.Tensor, optional
            F2 beregnet over hele datasettet. Hvis ikke gitt,
            beregnes F2 fra batchen. For trening bør du helst
            sende inn en F2-score fra hele treningssettet.
        """
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        # MSE skal bare gjelde den høye risikoklassen
        high_risk_mask = y_true >= self.threshold

        # Dersom batchen ikke har noen positive eksempler
        if not high_risk_mask.any():
            return y_pred.sum() * 0.0

        mse = torch.mean(
            (y_true[high_risk_mask] - y_pred[high_risk_mask]) ** 2
        )

        if f2_score is None:
            # detach fordi F2 bruker hard terskel og ikke er deriverbar
            with torch.no_grad():
                f2_score = self.beregn_f2(y_true, y_pred)

        if not torch.is_tensor(f2_score):
            f2_score = torch.tensor(
                f2_score,
                dtype=y_pred.dtype,
                device=y_pred.device,
            )
        else:
            f2_score = f2_score.to(
                dtype=y_pred.dtype,
                device=y_pred.device,
            )

        # Unngår divisjon på null
        f2_score = torch.clamp(f2_score, min=self.eps)

        loss = mse / f2_score

        return loss



class SequenceDataset(Dataset):
    """
    Dataset for sekvenser med ulik lengde.

    X må være en liste der hvert element er en sekvens med shape:
        (sequence_length, n_features)

    Sekvensene kan være pandas DataFrames, NumPy-arrays eller torch-tensorer.
    """

    def __init__(self, X, y=None):
        self.X = [self._to_tensor(sequence) for sequence in X]

        if y is None:
            self.y = None
        else:
            y = np.asarray(y, dtype=np.float32)

            if len(self.X) != len(y):
                raise ValueError(
                    f"X og y må ha samme lengde. Fikk {len(self.X)} sekvenser "
                    f"og {len(y)} labels."
                )

            self.y = torch.tensor(y, dtype=torch.float32)

    @staticmethod
    def _to_tensor(sequence):
        """Konverterer én sekvens til float32-tensor."""
        if isinstance(sequence, pd.DataFrame):
            sequence = sequence.to_numpy(dtype=np.float32)

        if isinstance(sequence, np.ndarray):
            sequence = torch.tensor(sequence, dtype=torch.float32)

        if not isinstance(sequence, torch.Tensor):
            raise TypeError(
                "Hver sekvens må være en pandas DataFrame, NumPy-array "
                "eller torch.Tensor."
            )

        if sequence.ndim != 2:
            raise ValueError(
                "Hver sekvens må ha shape (sequence_length, n_features)."
            )

        if len(sequence) == 0:
            raise ValueError("En sekvens kan ikke være tom.")

        return sequence.to(dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]

        return self.X[idx], self.y[idx]


def training_collate_fn(batch):
    """
    Padder en batch for trening.

    Returns
    -------
    X_padded : Tensor
        Shape: (batch_size, lengste_sekvens_i_batch, n_features).
    lengths : Tensor
        Opprinnelige sekvenslengder, shape (batch_size,).
    y : Tensor
        Binære targets, shape (batch_size,).
    """
    sequences, targets = zip(*batch)

    lengths = torch.tensor(
        [sequence.shape[0] for sequence in sequences],
        dtype=torch.long,
    )

    X_padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )

    y = torch.stack(targets)

    return X_padded, lengths, y


def prediction_collate_fn(batch):
    """
    Padder en batch for prediksjon uten targets.

    Returns
    -------
    X_padded : Tensor
        Shape: (batch_size, lengste_sekvens_i_batch, n_features).
    lengths : Tensor
        Opprinnelige sekvenslengder.
    """
    lengths = torch.tensor(
        [sequence.shape[0] for sequence in batch],
        dtype=torch.long,
    )

    X_padded = pad_sequence(
        batch,
        batch_first=True,
        padding_value=0.0,
    )

    return X_padded, lengths


class RNNClassifier(nn.Module):
    """
    RNN-klassifikator for sekvenser med ulik lengde.

    Modellen predikerer sannsynligheten for klasse 1, for eksempel
    "high-risk event".

    Du kan angi hidden layers på to måter:

    1. Samme antall noder i alle lag:

        model = RNNClassifier(
            input_size=60,
            hidden_layers=3,
            nodes_per_layer=64,
        )

        Dette gir hidden sizes: [64, 64, 64].

    2. Ulikt antall noder per lag:

        model = RNNClassifier(
            input_size=60,
            hidden_layers=[128, 64, 32],
        )

        Dette gir hidden sizes: [128, 64, 32].

    Parameters
    ----------
    input_size : int
        Antall features ved hvert tidssteg / CDM.
    hidden_layers : int eller list[int]
        Antall hidden layers, eller eksplisitt liste med noder per lag.
    nodes_per_layer : int
        Antall noder i hvert hidden layer når `hidden_layers` er et heltall.
    rnn_type : {"gru", "lstm", "rnn"}
        Type rekurrentt lag.
    dropout : float
        Dropout etter hvert RNN-lag og før klassifikasjonshodet.
    bidirectional : bool
        Dersom True brukes en bidirectional RNN.
    device : str eller None
        For eksempel "cpu", "cuda" eller "mps". Hvis None velges CUDA dersom
        det er tilgjengelig, ellers CPU.
    """

    def __init__(
        self,
        input_size,
        hidden_layers=1,
        nodes_per_layer=64,
        rnn_type="gru",
        dropout=0.0,
        bidirectional=False,
        device=None,
    ):
        super().__init__()

        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError("input_size må være et positivt heltall.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout må være i intervallet [0, 1).")

        self.input_size = input_size
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional

        # Alternativ A:
        # hidden_layers=3, nodes_per_layer=64 -> [64, 64, 64]
        if isinstance(hidden_layers, int):
            if hidden_layers < 1:
                raise ValueError("hidden_layers må være minst 1.")

            if not isinstance(nodes_per_layer, int) or nodes_per_layer < 1:
                raise ValueError(
                    "nodes_per_layer må være et positivt heltall."
                )

            self.hidden_sizes = [nodes_per_layer] * hidden_layers

        # Alternativ B:
        # hidden_layers=[128, 64, 32] -> [128, 64, 32]
        elif isinstance(hidden_layers, (list, tuple)):
            if len(hidden_layers) == 0:
                raise ValueError(
                    "hidden_layers-listen må inneholde minst ett lag."
                )

            if not all(
                isinstance(nodes, int) and nodes > 0
                for nodes in hidden_layers
            ):
                raise ValueError(
                    "Alle verdier i hidden_layers-listen må være "
                    "positive heltall."
                )

            self.hidden_sizes = list(hidden_layers)

        else:
            raise TypeError(
                "hidden_layers må være enten et heltall eller en liste "
                "med positive heltall."
            )

        rnn_classes = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }

        if self.rnn_type not in rnn_classes:
            raise ValueError(
                "rnn_type må være 'rnn', 'gru' eller 'lstm'."
            )

        rnn_class = rnn_classes[self.rnn_type]

        # Ett lag bygges om gangen, slik at hvert lag kan ha ulik hidden size.
        self.rnn_layers = nn.ModuleList()

        current_input_size = input_size
        direction_factor = 2 if bidirectional else 1

        for hidden_size in self.hidden_sizes:
            layer = rnn_class(
                input_size=current_input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )

            self.rnn_layers.append(layer)

            # Output fra et bidirectional lag har dobbelt så mange features.
            current_input_size = hidden_size * direction_factor

        self.dropout_layer = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            in_features=self.hidden_sizes[-1] * direction_factor,
            out_features=1,
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.to(self.device)

        self.history = {
            "train_data_loss": [],
            "train_penalty_loss": [],
            "train_total_loss": [],
            "val_loss": [],
        }

    def forward(self, X_padded, lengths):
        """
        Kjører RNN-en og returnerer logits.

        Parameters
        ----------
        X_padded : torch.Tensor
            Shape: (batch_size, max_sequence_length, input_size).
        lengths : torch.Tensor
            Originale lengder før padding, shape: (batch_size,).

        Returns
        -------
        logits : torch.Tensor
            Én logit per event, shape: (batch_size,).

        Notes
        -----
        Sigmoid brukes ikke her. `BCEWithLogitsLoss` forventer logits.
        Sannsynligheter fås med `predict_proba()`.
        """
        x = X_padded

        for layer_index, rnn_layer in enumerate(self.rnn_layers):
            packed_x = pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )

            packed_output, hidden = rnn_layer(packed_x)

            # LSTM returnerer (hidden_state, cell_state).
            if self.rnn_type == "lstm":
                hidden = hidden[0]

            # hidden har shape:
            # (n_directions, batch_size, hidden_size), siden hvert rnn_layer
            # har num_layers=1.
            if self.bidirectional:
                final_hidden = torch.cat(
                    [hidden[-2], hidden[-1]],
                    dim=1,
                )
            else:
                final_hidden = hidden[-1]

            # Neste RNN-lag trenger sekvensoutput, ikke bare final hidden state.
            if layer_index < len(self.rnn_layers) - 1:
                x, _ = pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                )

                x = self.dropout_layer(x)

        final_hidden = self.dropout_layer(final_hidden)
        logits = self.classifier(final_hidden).squeeze(1)

        return logits

    def _make_optimizer(
        self,
        optimizer,
        learning_rate,
        optimizer_kwargs=None,
    ):
        """
        Oppretter optimizer.

        optimizer kan være:
        - streng: "adam", "adamw", "sgd" eller "rmsprop";
        - en PyTorch optimizer-klasse, f.eks. torch.optim.Adam;
        - en allerede opprettet PyTorch optimizer.
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if isinstance(optimizer, torch.optim.Optimizer):
            return optimizer

        if isinstance(optimizer, str):
            optimizer_map = {
                "adam": torch.optim.Adam,
                "adamw": torch.optim.AdamW,
                "sgd": torch.optim.SGD,
                "rmsprop": torch.optim.RMSprop,
            }

            optimizer_name = optimizer.lower()

            if optimizer_name not in optimizer_map:
                raise ValueError(
                    f"Ukjent optimizer {optimizer!r}. Velg en av "
                    f"{list(optimizer_map)}."
                )

            optimizer_class = optimizer_map[optimizer_name]

        elif (
            isinstance(optimizer, type)
            and issubclass(optimizer, torch.optim.Optimizer)
        ):
            optimizer_class = optimizer

        else:
            raise TypeError(
                "optimizer må være en streng som 'adam', en optimizer-klasse "
                "som torch.optim.Adam, eller et optimizer-objekt."
            )

        return optimizer_class(
            self.parameters(),
            lr=learning_rate,
            **optimizer_kwargs,
        )

    def _weight_penalty(self, penalty, penalty_strength):
        """
        Beregner L1- eller L2-regularisering av vektmatrisene.

        Bias-parametere utelates. I PyTorch er bias normalt endimensjonal,
        mens vekter er todimensjonale eller høyere.

        Parameters
        ----------
        penalty : None, "l1" eller "l2"
            Type regularisering.
        penalty_strength : float
            Regulariseringsstyrken, $\lambda$.

        Returns
        -------
        torch.Tensor
            Regulariseringsleddet som skal legges til datatapet.
        """
        if penalty is None or penalty_strength == 0.0:
            return torch.zeros((), device=self.device)

        weight_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.ndim > 1
        ]

        if penalty == "l1":
            regularization = sum(
                torch.sum(torch.abs(parameter))
                for parameter in weight_parameters
            )

        elif penalty == "l2":
            regularization = sum(
                torch.sum(parameter.pow(2))
                for parameter in weight_parameters
            )

        else:
            # Dette bør normalt fanges av valideringen i fit().
            raise ValueError("penalty må være None, 'l1' eller 'l2'.")

        return penalty_strength * regularization

    def fit(
        self,
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        optimizer="adam",
        learning_rate=1e-3,
        optimizer_kwargs=None,
        X_val=None,
        y_val=None,
        class_weight=None,
        penalty=None,
        penalty_strength=0.0,
        shuffle=True,
        verbose=True,
    ):
        """
        Trener RNN-modellen.

        Parameters
        ----------
        X_train : list
            Liste med sekvenser. Hvert element må ha shape
            (sequence_length, n_features).
        y_train : array-like
            Binære targets, én verdi per sekvens.
        epochs : int
            Antall epoker.
        batch_size : int
            Antall events per batch.
        optimizer : str, optimizer-klasse eller optimizer-objekt
            Eksempel: "adam", "adamw", "sgd", torch.optim.AdamW.
        learning_rate : float
            Learning rate, dersom optimizeren opprettes her.
        optimizer_kwargs : dict eller None
            Ekstra optimizer-parametere, f.eks. {"momentum": 0.9} for SGD.

            Ikke bruk {"weight_decay": ...} sammen med penalty="l2",
            fordi dette da gir to L2-straffeledd.
        X_val, y_val : optional
            Valideringsdata. Valideringstapet rapporteres uten penalty-leddet.
        class_weight : float eller None
            Positiv klassevekt i BCEWithLogitsLoss. Kan settes til:

                n_negative / n_positive

            ved sterk klasseubalanse.
        penalty : None, "l1" eller "l2"
            Regulariseringstype:

            - None: ingen vektregularisering;
            - "l1": $\lambda \sum_i |w_i|$;
            - "l2": $\lambda \sum_i w_i^2$.

            Bias-parametere regulariseres ikke.
        penalty_strength : float
            Verdien av $\lambda$.
        shuffle : bool
            Om treningsdataene skal stokkes ved hver epoch.
        verbose : bool
            Skriver ut loss per epoch dersom True.

        Returns
        -------
        self
        """
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs må være et positivt heltall.")

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size må være et positivt heltall.")

        if learning_rate <= 0:
            raise ValueError("learning_rate må være større enn 0.")

        if penalty is not None:
            penalty = penalty.lower()

            if penalty not in {"l1", "l2"}:
                raise ValueError(
                    "penalty må være None, 'l1' eller 'l2'."
                )

        if penalty_strength < 0:
            raise ValueError(
                "penalty_strength må være større enn eller lik 0."
            )

        if (X_val is None) != (y_val is None):
            raise ValueError(
                "Oppgi både X_val og y_val, eller ingen av dem."
            )

        train_dataset = SequenceDataset(X_train, y_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=training_collate_fn,
        )

        if class_weight is None:
            criterion = nn.BCEWithLogitsLoss()

        else:
            if class_weight <= 0:
                raise ValueError("class_weight må være større enn 0.")

            pos_weight = torch.tensor(
                [class_weight],
                dtype=torch.float32,
                device=self.device,
            )

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer_object = self._make_optimizer(
            optimizer=optimizer,
            learning_rate=learning_rate,
            optimizer_kwargs=optimizer_kwargs,
        )

        if X_val is not None:
            val_dataset = SequenceDataset(X_val, y_val)

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=training_collate_fn,
            )
        else:
            val_loader = None

        self.history = {
            "train_data_loss": [],
            "train_penalty_loss": [],
            "train_total_loss": [],
            "val_loss": [],
        }

        for epoch in range(1, epochs + 1):
            self.train()

            total_data_loss = 0.0
            total_penalty_loss = 0.0
            total_total_loss = 0.0
            n_train_events = 0

            for X_batch, lengths, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                lengths = lengths.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer_object.zero_grad()

                logits = self(X_batch, lengths)

                # Klassifikasjonstapet.
                data_loss = criterion(logits, y_batch)

                # L1- eller L2-straffeleddet.
                regularization_loss = self._weight_penalty(
                    penalty=penalty,
                    penalty_strength=penalty_strength,
                )

                # Tapet som faktisk optimaliseres.
                total_loss = data_loss + regularization_loss

                total_loss.backward()
                optimizer_object.step()

                batch_n = len(y_batch)

                total_data_loss += data_loss.item() * batch_n
                total_penalty_loss += regularization_loss.item() * batch_n
                total_total_loss += total_loss.item() * batch_n
                n_train_events += batch_n

            mean_data_loss = total_data_loss / n_train_events
            mean_penalty_loss = total_penalty_loss / n_train_events
            mean_total_loss = total_total_loss / n_train_events

            self.history["train_data_loss"].append(mean_data_loss)
            self.history["train_penalty_loss"].append(mean_penalty_loss)
            self.history["train_total_loss"].append(mean_total_loss)

            # Validation rapporteres uten penalty-leddet, siden dette måler
            # prediksjonsfeil på usette data og ikke størrelsen på vektene.
            if val_loader is not None:
                self.eval()

                total_val_loss = 0.0
                n_val_events = 0

                with torch.no_grad():
                    for X_batch, lengths, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        lengths = lengths.to(self.device)
                        y_batch = y_batch.to(self.device)

                        logits = self(X_batch, lengths)
                        val_loss = criterion(logits, y_batch)

                        batch_n = len(y_batch)
                        total_val_loss += val_loss.item() * batch_n
                        n_val_events += batch_n

                mean_val_loss = total_val_loss / n_val_events
                self.history["val_loss"].append(mean_val_loss)

                if verbose:
                    print(
                        f"Epoch {epoch:3d}/{epochs} | "
                        f"train data loss: {mean_data_loss:.6f} | "
                        f"penalty: {mean_penalty_loss:.6f} | "
                        f"train total loss: {mean_total_loss:.6f} | "
                        f"val loss: {mean_val_loss:.6f}"
                    )

            elif verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train data loss: {mean_data_loss:.6f} | "
                    f"penalty: {mean_penalty_loss:.6f} | "
                    f"train total loss: {mean_total_loss:.6f}"
                )

        return self

    @torch.no_grad()
    def predict_proba(self, X, batch_size=128):
        """
        Predikerer sannsynligheten for klasse 1.

        Parameters
        ----------
        X : list
            Liste med sekvenser.
        batch_size : int
            Antall events per prediksjonsbatch.

        Returns
        -------
        np.ndarray
            Shape: (n_events,). Hvert element er $P(y = 1)$.
        """
        if len(X) == 0:
            return np.array([], dtype=np.float32)

        self.eval()

        dataset = SequenceDataset(X)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=prediction_collate_fn,
        )

        probabilities = []

        for X_batch, lengths in loader:
            X_batch = X_batch.to(self.device)
            lengths = lengths.to(self.device)

            logits = self(X_batch, lengths)
            batch_probabilities = torch.sigmoid(logits)

            probabilities.append(batch_probabilities.cpu().numpy())

        return np.concatenate(probabilities)

    @torch.no_grad()
    def predict(self, X, threshold=0.5, batch_size=128):
        """
        Predikerer binære klasser.

        Parameters
        ----------
        X : list
            Liste med sekvenser.
        threshold : float
            Klasse 1 hvis sannsynligheten er minst threshold.
        batch_size : int
            Antall events per batch.

        Returns
        -------
        np.ndarray
            Binære prediksjoner med shape (n_events,).
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold må være mellom 0 og 1.")

        probabilities = self.predict_proba(
            X,
            batch_size=batch_size,
        )

        return (probabilities >= threshold).astype(int)

def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    threshold=0.5,
    normalize=None,
    batch_size=128,
    title=None,
):
    """
    Plotter confusion matrix for en trent RNNClassifier.

    Parameters
    ----------
    model : RNNClassifier
        Ferdig trent modell med predict_proba().
    X_test : list
        Liste med testsekvenser.
    y_test : array-like
        Sanne binære labels.
    threshold : float
        Klassifiserer som positiv dersom P(y=1) >= threshold.
    normalize : {None, "true", "pred", "all"}
        None   : viser rå antall.
        "true" : normaliserer hver rad; viser recall-andeler.
        "pred" : normaliserer hver kolonne; viser precision-andeler.
        "all"  : normaliserer med totalt antall observasjoner.
    batch_size : int
        Batch-størrelse ved prediksjon.
    title : str eller None
        Valgfri plottittel.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix.
    y_prob : np.ndarray
        Predikerte sannsynligheter.
    y_pred : np.ndarray
        Binære prediksjoner.
    """
    y_test = np.asarray(y_test).astype(int)

    y_prob = model.predict_proba(
        X_test,
        batch_size=batch_size,
    )

    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
        normalize=normalize,
    )

    if normalize is None:
        fmt = "d"
        colorbar_label = "Antall events"
    else:
        fmt = ".2f"
        colorbar_label = "Andel"

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        square=True,
        cbar_kws={"label": colorbar_label},
        xticklabels=["Lav risiko (0)", "Høy risiko (1)"],
        yticklabels=["Lav risiko (0)", "Høy risiko (1)"],
        ax=ax,
    )

    ax.set_xlabel("Predikert klasse")
    ax.set_ylabel("Sann klasse")

    if title is None:
        title = f"Confusion matrix, threshold = {threshold:.4f}"

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return cm, y_prob, y_pred


def plot_roc_curve(
    model,
    X_test,
    y_test,
    batch_size=128,
    title="ROC-kurve",
):
    """
    Plotter ROC-kurve for en trent RNNClassifier.

    Parameters
    ----------
    model : RNNClassifier
        Trent modell med predict_proba().
    X_test : list
        Liste av testsekvenser.
    y_test : array-like
        Sanne binære labels.
    batch_size : int
        Batch-størrelse ved prediksjon.
    title : str
        Tittel på plottet.

    Returns
    -------
    auc : float
        Arealet under ROC-kurven.
    y_prob : np.ndarray
        Predikerte sannsynligheter for klasse 1.
    """
    y_test = np.asarray(y_test).astype(int)

    # Sannsynlighet for klasse 1 / høy risiko.
    y_prob = model.predict_proba(
        X_test,
        batch_size=batch_size,
    )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob,
    )

    auc = roc_auc_score(
        y_test,
        y_prob,
    )

    plt.figure(figsize=(6, 5))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"RNN (AUC = {auc:.3f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Tilfeldig klassifikator",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return auc, y_prob, fpr, tpr, thresholds

import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score


def set_seed(seed):
    """Setter tilfeldighetsfrø for mer reproduserbare RNN-forsøk."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_rnn_architecture_heatmaps(
    data,
    hidden_layer_values=(1, 2, 3),
    nodes_per_layer_values=(25, 50, 75),
    rnn_type="gru",
    dropout=0.2,
    epochs=20,
    batch_size=32,
    optimizer="adam",
    learning_rate=1e-4,
    class_weight=None,
    penalty=None,
    penalty_strength=0.0,
    device=None,
    seed=3155,
):
    """
    Kryssvaliderer RNN-arkitekturer og plotter heatmaps for loss og ROC-AUC.

    Parameters
    ----------
    data : dict
        Output fra build_shared_datasets().
    hidden_layer_values : iterable
        Antall hidden layers, f.eks. [1, 2, 3].
    nodes_per_layer_values : iterable
        Antall noder per hidden layer, f.eks. [25, 50, 75].
    rnn_type : str
        "gru", "lstm" eller "rnn".
    dropout : float
        Dropout-sannsynlighet.
    epochs : int
        Antall treningsepoker per CV-fold.
    batch_size : int
        Batch-størrelse.
    optimizer : str
        For eksempel "adam", "adamw" eller "sgd".
    learning_rate : float
        Learning rate.
    class_weight : float eller None
        Positiv klassevekt. Hvis None, beregnes vekten separat i hver fold
        som antall negative delt på antall positive trenings-events.
    penalty : None, "l1" eller "l2"
        Regulariseringstype.
    penalty_strength : float
        Styrken på regulariseringen.
    device : str eller None
        F.eks. "cpu", "cuda" eller None.
    seed : int
        Tilfeldighetsfrø.

    Returns
    -------
    results : pd.DataFrame
        Resultater for hver arkitektur og CV-fold.
    loss_heatmap : pd.DataFrame
        Gjennomsnittlig validation loss.
    auc_heatmap : pd.DataFrame
        Gjennomsnittlig validation ROC-AUC.
    """
    X_train = data["sequences"]["train"]
    y_train = np.asarray(data["y_class"]["train"])
    cv_folds = data["cv_folds"]
    input_size = len(data["sequence_columns"])

    results_rows = []

    for n_layers in hidden_layer_values:
        for n_nodes in nodes_per_layer_values:
            print(
                f"Tester: {n_layers} hidden layer(s), "
                f"{n_nodes} nodes per layer"
            )

            for fold_number, (fit_idx, val_idx) in enumerate(
                cv_folds,
                start=1,
            ):
                # Hent sekvensene til denne folden.
                X_fit = [X_train[i] for i in fit_idx]
                X_val = [X_train[i] for i in val_idx]

                y_fit = y_train[fit_idx]
                y_val = y_train[val_idx]

                # Beregn positiv klassevekt fra kun treningsdelen av folden.
                if class_weight is None:
                    n_positive = np.sum(y_fit == 1)
                    n_negative = np.sum(y_fit == 0)

                    if n_positive == 0:
                        current_class_weight = None
                    else:
                        current_class_weight = n_negative / n_positive
                else:
                    current_class_weight = class_weight

                # Ulikt seed per fold/arkitektur, men reproduserbart.
                current_seed = (
                    seed
                    + 1000 * n_layers
                    + 100 * n_nodes
                    + fold_number
                )
                set_seed(current_seed)

                # Viktig: ny modell for hver fold.
                model = RNNClassifier(
                    input_size=input_size,
                    hidden_layers=n_layers,
                    nodes_per_layer=n_nodes,
                    rnn_type=rnn_type,
                    dropout=dropout,
                    device=device,
                )

                model.fit(
                    X_train=X_fit,
                    y_train=y_fit,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    class_weight=current_class_weight,
                    penalty=penalty,
                    penalty_strength=penalty_strength,
                    verbose=False,
                )

                # Loss etter siste training epoch.
                final_val_loss = model.history["val_loss"][-1]

                # ROC-AUC på validation-delen etter siste training epoch.
                y_val_prob = model.predict_proba(
                    X_val,
                    batch_size=batch_size,
                )
                val_auc = roc_auc_score(y_val, y_val_prob)

                results_rows.append(
                    {
                        "hidden_layers": n_layers,
                        "nodes_per_layer": n_nodes,
                        "fold": fold_number,
                        "final_val_loss": final_val_loss,
                        "val_auc": val_auc,
                    }
                )

                print(
                    f"  Fold {fold_number}/{len(cv_folds)} | "
                    f"loss: {final_val_loss:.4f} | "
                    f"AUC: {val_auc:.4f}"
                )

    results = pd.DataFrame(results_rows)

    # Gjennomsnitt over CV-foldene.
    loss_heatmap = results.pivot_table(
        index="hidden_layers",
        columns="nodes_per_layer",
        values="final_val_loss",
        aggfunc="mean",
    ).reindex(
        index=hidden_layer_values,
        columns=nodes_per_layer_values,
    )

    auc_heatmap = results.pivot_table(
        index="hidden_layers",
        columns="nodes_per_layer",
        values="val_auc",
        aggfunc="mean",
    ).reindex(
        index=hidden_layer_values,
        columns=nodes_per_layer_values,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    sns.heatmap(
        loss_heatmap,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Gjennomsnittlig validation loss"},
        ax=axes[0],
    )
    axes[0].set_title("CV validation loss")
    axes[0].set_xlabel("Nodes per hidden layer")
    axes[0].set_ylabel("Antall hidden layers")

    sns.heatmap(
        auc_heatmap,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Gjennomsnittlig validation ROC-AUC"},
        ax=axes[1],
    )
    axes[1].set_title("CV ROC-AUC")
    axes[1].set_xlabel("Nodes per hidden layer")
    axes[1].set_ylabel("Antall hidden layers")

    plt.tight_layout()
    plt.show()

    return results, loss_heatmap, auc_heatmap


def compare_rnn_optimizers_cv(
    data,
    hidden_layers=1,
    nodes_per_layer=25,
    rnn_type="gru",
    dropout=0.2,
    epochs=10,
    batch_size=32,
    learning_rates=None,
    class_weight=10,
    penalty="l2",
    penalty_strength=1e-5,
    device=None,
    seed=3155,
):
    """
    Sammenligner Adam, RMSprop og plain SGD med kryssvalidering.

    Parameters
    ----------
    data : dict
        Output fra build_shared_datasets().
    hidden_layers : int eller list[int]
        Arkitekturargument til RNNClassifier.
    nodes_per_layer : int
        Noder per hidden layer dersom hidden_layers er et heltall.
    rnn_type : str
        "gru", "lstm" eller "rnn".
    dropout : float
        Dropout-rate.
    epochs : int
        Antall treningsepoker per fold.
    batch_size : int
        Batch-størrelse.
    learning_rates : dict eller None
        Learning rate per optimizer. Hvis None brukes anbefalte startverdier.
    class_weight : float eller None
        Positiv klassevekt. Hvis None beregnes den separat fra treningsdelen
        av hver fold.
    penalty : None, "l1" eller "l2"
        Type regularisering.
    penalty_strength : float
        Regulariseringsstyrke.
    device : str eller None
        Eksempel: "cpu", "cuda" eller None.
    seed : int
        Tilfeldighetsfrø.

    Returns
    -------
    results : pd.DataFrame
        Ett resultat per optimizer og CV-fold.
    summary : pd.DataFrame
        Gjennomsnitt og standardavvik for hver optimizer.
    """
    if learning_rates is None:
        # SGD trenger ofte en større learning rate enn Adam/RMSprop.
        learning_rates = {
            "adam": 1e-4,
            "rmsprop": 1e-4,
            "sgd": 1e-2,
        }

    optimizers = ["adam", "rmsprop", "sgd"]

    missing = set(optimizers) - set(learning_rates)
    if missing:
        raise ValueError(
            f"learning_rates mangler verdier for: {sorted(missing)}"
        )

    X_train = data["sequences"]["train"]
    y_train = np.asarray(data["y_class"]["train"])
    cv_folds = data["cv_folds"]
    input_size = len(data["sequence_columns"])

    results_rows = []

    for optimizer_number, optimizer_name in enumerate(optimizers):
        learning_rate = learning_rates[optimizer_name]

        print(
            f"\n{'=' * 60}\n"
            f"Tester optimizer: {optimizer_name.upper()} | "
            f"learning rate: {learning_rate}\n"
            f"{'=' * 60}"
        )

        for fold_number, (fit_idx, val_idx) in enumerate(cv_folds, start=1):
            X_fit = [X_train[i] for i in fit_idx]
            X_val = [X_train[i] for i in val_idx]

            y_fit = y_train[fit_idx]
            y_val = y_train[val_idx]

            # Hvis vekten ikke er oppgitt, estimeres den fra kun denne
            # foldens treningsdata.
            if class_weight is None:
                n_positive = np.sum(y_fit == 1)
                n_negative = np.sum(y_fit == 0)

                current_class_weight = (
                    n_negative / n_positive
                    if n_positive > 0
                    else None
                )
            else:
                current_class_weight = class_weight

            current_seed = seed + optimizer_number * 100 + fold_number
            set_seed(current_seed)

            model = RNNClassifier(
                input_size=input_size,
                hidden_layers=hidden_layers,
                nodes_per_layer=nodes_per_layer,
                rnn_type=rnn_type,
                dropout=dropout,
                device=device,
            )

            # Plain SGD: ingen momentum, ingen weight decay.
            # Adam og RMSprop bruker også standardinnstillinger.
            model.fit(
                X_train=X_fit,
                y_train=y_fit,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                optimizer=optimizer_name,
                learning_rate=learning_rate,
                optimizer_kwargs={},
                class_weight=current_class_weight,
                penalty=penalty,
                penalty_strength=penalty_strength,
                verbose=False,
            )

            final_val_loss = model.history["val_loss"][-1]

            y_val_prob = model.predict_proba(
                X_val,
                batch_size=batch_size,
            )

            val_auc = roc_auc_score(y_val, y_val_prob)

            results_rows.append(
                {
                    "optimizer": optimizer_name,
                    "learning_rate": learning_rate,
                    "fold": fold_number,
                    "final_val_loss": final_val_loss,
                    "val_auc": val_auc,
                }
            )

            print(
                f"Fold {fold_number}/{len(cv_folds)} | "
                f"loss: {final_val_loss:.4f} | "
                f"AUC: {val_auc:.4f}"
            )

    results = pd.DataFrame(results_rows)

    summary = (
        results
        .groupby(["optimizer", "learning_rate"], as_index=False)
        .agg(
            mean_val_loss=("final_val_loss", "mean"),
            std_val_loss=("final_val_loss", "std"),
            mean_auc=("val_auc", "mean"),
            std_auc=("val_auc", "std"),
        )
    )

    # Få en fast og pen rekkefølge i tabeller og plott.
    optimizer_order = ["adam", "rmsprop", "sgd"]

    summary["optimizer"] = pd.Categorical(
        summary["optimizer"],
        categories=optimizer_order,
        ordered=True,
    )

    summary = summary.sort_values("optimizer").reset_index(drop=True)

    # Plot gjennomsnitt med standardavvik mellom foldene.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
    )

    sns.barplot(
        data=summary,
        x="optimizer",
        y="mean_val_loss",
        hue="optimizer",
        palette="YlOrRd",
        legend=False,
        ax=axes[0],
    )

    axes[0].errorbar(
        x=np.arange(len(summary)),
        y=summary["mean_val_loss"],
        yerr=summary["std_val_loss"],
        fmt="none",
        color="black",
        capsize=5,
    )

    axes[0].set_title("Kryss-validert validation loss")
    axes[0].set_xlabel("Optimizer")
    axes[0].set_ylabel("Gjennomsnittlig validation loss")

    sns.barplot(
        data=summary,
        x="optimizer",
        y="mean_auc",
        hue="optimizer",
        palette="YlGnBu",
        legend=False,
        ax=axes[1],
    )

    axes[1].errorbar(
        x=np.arange(len(summary)),
        y=summary["mean_auc"],
        yerr=summary["std_auc"],
        fmt="none",
        color="black",
        capsize=5,
    )

    axes[1].set_title("Kryss-validert ROC-AUC")
    axes[1].set_xlabel("Optimizer")
    axes[1].set_ylabel("Gjennomsnittlig ROC-AUC")
    axes[1].set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.show()

    return results, summary


def plot_learning_rate_l2_heatmaps(
    data,
    learning_rates=(1e-4, 1e-3, 1e-2),
    penalty_strengths=(1e-5, 1e-4, 1e-3),
    hidden_layers=1,
    nodes_per_layer=25,
    rnn_type="gru",
    dropout=0.2,
    epochs=10,
    batch_size=32,
    optimizer="adam",
    class_weight=10,
    device=None,
    seed=3155,
):
    """
    Kryssvaliderer kombinasjoner av learning rate og L2-regularisering.

    Heatmaps:
        - rader: L2 penalty strength
        - kolonner: learning rate
        - venstre: gjennomsnittlig validation loss
        - høyre: gjennomsnittlig validation ROC-AUC

    Parameters
    ----------
    data : dict
        Output fra build_shared_datasets().
    learning_rates : iterable
        Learning rates, f.eks. [1e-4, 1e-3, 1e-2].
    penalty_strengths : iterable
        L2-verdier, f.eks. [1e-5, 1e-4, 1e-3].
    hidden_layers : int eller list[int]
        Argument som sendes til RNNClassifier.
    nodes_per_layer : int
        Noder per hidden layer dersom hidden_layers er et heltall.
    rnn_type : str
        "gru", "lstm" eller "rnn".
    dropout : float
        Dropout-sannsynlighet.
    epochs : int
        Antall treningsepoker per fold.
    batch_size : int
        Batch-størrelse.
    optimizer : str
        "adam", "adamw", "rmsprop" eller "sgd".
    class_weight : float eller None
        Positiv klassevekt. Ved None beregnes vekten per fold som
        antall negative / antall positive i foldens treningsdata.
    device : str eller None
        For eksempel "cpu", "cuda" eller None.
    seed : int
        Tilfeldighetsfrø.

    Returns
    -------
    results : pd.DataFrame
        Resultater for hver learning rate, penalty og CV-fold.
    loss_heatmap : pd.DataFrame
        Gjennomsnittlig validation loss.
    auc_heatmap : pd.DataFrame
        Gjennomsnittlig validation ROC-AUC.
    """
    X_train = data["sequences"]["train"]
    y_train = np.asarray(data["y_class"]["train"])
    cv_folds = data["cv_folds"]
    input_size = len(data["sequence_columns"])

    learning_rates = list(learning_rates)
    penalty_strengths = list(penalty_strengths)

    results_rows = []

    for penalty_index, penalty_strength in enumerate(penalty_strengths):
        for lr_index, learning_rate in enumerate(learning_rates):
            print(
                f"\nTester learning_rate={learning_rate:.0e}, "
                f"L2 penalty={penalty_strength:.0e}"
            )

            for fold_number, (fit_idx, val_idx) in enumerate(
                cv_folds,
                start=1,
            ):
                X_fit = [X_train[i] for i in fit_idx]
                X_val = [X_train[i] for i in val_idx]

                y_fit = y_train[fit_idx]
                y_val = y_train[val_idx]

                if class_weight is None:
                    n_positive = np.sum(y_fit == 1)
                    n_negative = np.sum(y_fit == 0)

                    current_class_weight = (
                        n_negative / n_positive
                        if n_positive > 0
                        else None
                    )
                else:
                    current_class_weight = class_weight

                # Samme seed-logikk for reproduserbare, men ulike, forsøk.
                current_seed = (
                    seed
                    + 10_000 * penalty_index
                    + 1_000 * lr_index
                    + fold_number
                )
                set_seed(current_seed)

                # Ny modell per fold: aldri gjenbruk vektene fra en annen fold.
                model = RNNClassifier(
                    input_size=input_size,
                    hidden_layers=hidden_layers,
                    nodes_per_layer=nodes_per_layer,
                    rnn_type=rnn_type,
                    dropout=dropout,
                    device=device,
                )

                model.fit(
                    X_train=X_fit,
                    y_train=y_fit,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=epochs,
                    batch_size=batch_size,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                    class_weight=current_class_weight,
                    penalty="l2",
                    penalty_strength=penalty_strength,
                    verbose=False,
                )

                # Klassen har ikke early stopping. Derfor brukes tapet fra
                # den siste epoken.
                final_val_loss = model.history["val_loss"][-1]

                y_val_prob = model.predict_proba(
                    X_val,
                    batch_size=batch_size,
                )
                val_auc = roc_auc_score(y_val, y_val_prob)

                results_rows.append(
                    {
                        "learning_rate": learning_rate,
                        "penalty_strength": penalty_strength,
                        "fold": fold_number,
                        "final_val_loss": final_val_loss,
                        "val_auc": val_auc,
                    }
                )

                print(
                    f"  Fold {fold_number}/{len(cv_folds)} | "
                    f"loss: {final_val_loss:.4f} | "
                    f"AUC: {val_auc:.4f}"
                )

    results = pd.DataFrame(results_rows)

    loss_heatmap = (
        results
        .pivot_table(
            index="penalty_strength",
            columns="learning_rate",
            values="final_val_loss",
            aggfunc="mean",
        )
        .reindex(
            index=penalty_strengths,
            columns=learning_rates,
        )
    )

    auc_heatmap = (
        results
        .pivot_table(
            index="penalty_strength",
            columns="learning_rate",
            values="val_auc",
            aggfunc="mean",
        )
        .reindex(
            index=penalty_strengths,
            columns=learning_rates,
        )
    )

    # Mer lesbare etiketter enn 0.0001 og 0.00001.
    lr_labels = [f"{lr:.0e}" for lr in learning_rates]
    penalty_labels = [f"{penalty:.0e}" for penalty in penalty_strengths]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
    )

    sns.heatmap(
        loss_heatmap,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        xticklabels=lr_labels,
        yticklabels=penalty_labels,
        cbar_kws={"label": "Gjennomsnittlig validation loss"},
        ax=axes[0],
    )

    axes[0].set_title("CV validation loss")
    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylabel("L2 penalty strength")

    sns.heatmap(
        auc_heatmap,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        xticklabels=lr_labels,
        yticklabels=penalty_labels,
        cbar_kws={"label": "Gjennomsnittlig validation ROC-AUC"},
        ax=axes[1],
    )

    axes[1].set_title("CV ROC-AUC")
    axes[1].set_xlabel("Learning rate")
    axes[1].set_ylabel("L2 penalty strength")

    plt.tight_layout()
    plt.show()

    return results, loss_heatmap, auc_heatmap


import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, log_loss


def set_seed(seed):
    """Gjør PyTorch/Numpy-resultater mer reproduserbare."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_sequence_length_vs_metrics(
    data,
    hidden_layers=1,
    nodes_per_layer=25,
    rnn_type="gru",
    dropout=0.2,
    epochs=10,
    batch_size=32,
    optimizer="adam",
    learning_rate=1e-4,
    class_weight=10,
    penalty="l2",
    penalty_strength=1e-5,
    n_bins=5,
    binning="quantile",
    device=None,
    seed=3155,
):
    """
    Plotter out-of-fold ROC-AUC og BCE-loss gruppert etter sekvenslengde.

    En sekvenslengde er antall CDMer (rader) i ett event før 2-dagers-cutoff.
    Hver prediksjon er out-of-fold: modellen som predikerer et event har ikke
    sett det eventet under trening.

    Parameters
    ----------
    data : dict
        Output fra build_shared_datasets().
    hidden_layers, nodes_per_layer, rnn_type, dropout :
        RNN-arkitektur.
    epochs, batch_size, optimizer, learning_rate :
        Treningsinnstillinger.
    class_weight : float eller None
        Positiv klassevekt. Hvis None estimeres den separat i hver CV-fold.
    penalty, penalty_strength :
        Regularisering.
    n_bins : int
        Antall grupper av sekvenslengder.
    binning : {"quantile", "equal_width"}
        "quantile" gir omtrent like mange events per gruppe og er normalt best.
        "equal_width" deler det numeriske området i like brede intervaller.
    device : str eller None
        For eksempel "cpu", "cuda" eller None.
    seed : int
        Tilfeldighetsfrø.

    Returns
    -------
    event_results : pd.DataFrame
        Én rad per training-event, med OOF-prediksjon og sekvenslengde.
    summary : pd.DataFrame
        AUC, loss, antall events og antall positive per lengdegruppe.
    """
    if binning not in {"quantile", "equal_width"}:
        raise ValueError("binning må være 'quantile' eller 'equal_width'.")

    X_train = data["sequences"]["train"]
    y_train = np.asarray(data["y_class"]["train"])
    cv_folds = data["cv_folds"]
    input_size = len(data["sequence_columns"])

    n_events = len(X_train)

    # Hver event fylles inn én gang, når den er validation-data i sin fold.
    oof_probabilities = np.full(n_events, np.nan, dtype=float)
    fold_numbers = np.full(n_events, -1, dtype=int)

    # Sekvenslengde = antall CDMs/rader i eventet.
    sequence_lengths = np.asarray(
        [len(sequence) for sequence in X_train],
        dtype=int,
    )

    for fold_number, (fit_idx, val_idx) in enumerate(cv_folds, start=1):
        print(f"Trener CV-fold {fold_number}/{len(cv_folds)}")

        X_fit = [X_train[i] for i in fit_idx]
        X_val = [X_train[i] for i in val_idx]

        y_fit = y_train[fit_idx]

        if class_weight is None:
            n_positive = np.sum(y_fit == 1)
            n_negative = np.sum(y_fit == 0)

            current_class_weight = (
                n_negative / n_positive
                if n_positive > 0
                else None
            )
        else:
            current_class_weight = class_weight

        set_seed(seed + fold_number)

        model = RNNClassifier(
            input_size=input_size,
            hidden_layers=hidden_layers,
            nodes_per_layer=nodes_per_layer,
            rnn_type=rnn_type,
            dropout=dropout,
            device=device,
        )

        model.fit(
            X_train=X_fit,
            y_train=y_fit,
            X_val=X_val,
            y_val=y_train[val_idx],
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            class_weight=current_class_weight,
            penalty=penalty,
            penalty_strength=penalty_strength,
            verbose=False,
        )

        oof_probabilities[val_idx] = model.predict_proba(
            X_val,
            batch_size=batch_size,
        )
        fold_numbers[val_idx] = fold_number

    if np.isnan(oof_probabilities).any():
        raise RuntimeError(
            "Noen events mangler out-of-fold-prediksjon. "
            "Sjekk at CV-foldene dekker alle training-events."
        )

    event_results = pd.DataFrame(
        {
            "sequence_length": sequence_lengths,
            "y_true": y_train,
            "y_probability": oof_probabilities,
            "fold": fold_numbers,
        }
    )

    # Gruppér sekvenslengdene.
    if binning == "quantile":
        # duplicates="drop" håndterer dersom mange events har samme lengde.
        event_results["length_bin"] = pd.qcut(
            event_results["sequence_length"],
            q=n_bins,
            duplicates="drop",
        )
    else:
        event_results["length_bin"] = pd.cut(
            event_results["sequence_length"],
            bins=n_bins,
            include_lowest=True,
        )

    rows = []

    for length_bin, group in event_results.groupby(
        "length_bin",
        observed=True,
    ):
        y_group = group["y_true"].to_numpy()
        p_group = group["y_probability"].to_numpy()

        # AUC er ikke definert dersom gruppen bare har én klasse.
        if len(np.unique(y_group)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_group, p_group)

        # Unweighted BCE / log loss. clip beskytter mot log(0).
        probabilities_clipped = np.clip(
            p_group,
            1e-7,
            1 - 1e-7,
        )

        bce_loss = log_loss(
            y_group,
            probabilities_clipped,
            labels=[0, 1],
        )

        rows.append(
            {
                "length_bin": str(length_bin),
                "mean_sequence_length": group["sequence_length"].mean(),
                "min_sequence_length": group["sequence_length"].min(),
                "max_sequence_length": group["sequence_length"].max(),
                "n_events": len(group),
                "n_positive": int(y_group.sum()),
                "auc": auc,
                "bce_loss": bce_loss,
            }
        )

    summary = pd.DataFrame(rows)

    # Sorterer intervallene etter faktisk gjennomsnittlig sekvenslengde.
    summary = summary.sort_values(
        "mean_sequence_length",
    ).reset_index(drop=True)

    x = np.arange(len(summary))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 5),
    )

    # ROC-AUC
    axes[0].bar(
        x,
        summary["auc"],
        color="steelblue",
        edgecolor="black",
    )
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Out-of-fold ROC-AUC etter sekvenslengde")
    axes[0].set_xlabel("Sekvenslengdegruppe, antall CDMs")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        summary["length_bin"],
        rotation=30,
        ha="right",
    )
    axes[0].grid(axis="y", alpha=0.3)

    # Binary cross-entropy
    axes[1].bar(
        x,
        summary["bce_loss"],
        color="indianred",
        edgecolor="black",
    )
    axes[1].set_title("Out-of-fold BCE-loss etter sekvenslengde")
    axes[1].set_xlabel("Sekvenslengdegruppe, antall CDMs")
    axes[1].set_ylabel("Binary cross-entropy loss")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        summary["length_bin"],
        rotation=30,
        ha="right",
    )
    axes[1].grid(axis="y", alpha=0.3)

    # Vis antall events og antall positive over hver søyle.
    for axis, values in zip(
        axes,
        [summary["auc"], summary["bce_loss"]],
    ):
        for i, value in enumerate(values):
            if not np.isnan(value):
                axis.text(
                    i,
                    value,
                    f"n={summary.loc[i, 'n_events']}\n"
                    f"pos={summary.loc[i, 'n_positive']}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.show()

    return event_results, summary
