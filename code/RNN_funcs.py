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

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)


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

    initialization : {"default", "xavier", "he", "lecun"}
        Initialiseringsmetode for modellens vekter.

        - "default": PyTorch sin standardinitialisering.
        - "xavier": Xavier uniform for input-vekter og classifier.
        - "he": He/Kaiming uniform for input-vekter og classifier.
        - "lecun": LeCun uniform for input-vekter og classifier.

        For ikke-default valg initialiseres recurrent-to-recurrent-vekter
        orthogonalt, og bias settes til null.

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
        initialization="default",
    ):
        super().__init__()

        if not isinstance(input_size, int) or input_size < 1:
            raise ValueError("input_size må være et positivt heltall.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout må være i intervallet [0, 1).")

        self.input_size = input_size
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional

        self.initialization = initialization.lower()

        valid_initializations = {
            "default",
            "xavier",
            "he",
            "lecun",
        }

        if self.initialization not in valid_initializations:
            raise ValueError(
                "initialization må være en av: "
                "'default', 'xavier', 'he' eller 'lecun'."
            )


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

        self._initialize_weights()

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
    
    def _initialize_weights(self):
        """
        Initialiserer RNN-vekter og klassifikasjonshodet.

        - weight_ih_*: valgt metode.
        - weight_hh_*: orthogonal initialisering.
        - bias_*: null.
        """
        if self.initialization == "default":
            return

        for name, parameter in self.named_parameters():
            if "weight_ih" in name:
                if self.initialization == "xavier":
                    nn.init.xavier_uniform_(parameter)

                elif self.initialization == "he":
                    nn.init.kaiming_uniform_(
                        parameter,
                        nonlinearity="relu",
                    )

                elif self.initialization == "lecun":
                    # LeCun uniform:
                    # U(-sqrt(3/fan_in), sqrt(3/fan_in))
                    fan_in = parameter.size(1)
                    bound = (3.0 / fan_in) ** 0.5
                    nn.init.uniform_(
                        parameter,
                        -bound,
                        bound,
                    )

            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)

            elif "bias" in name:
                nn.init.zeros_(parameter)

        # Klassifikasjonshodet er ikke navngitt weight_ih / weight_hh.
        if self.initialization == "xavier":
            nn.init.xavier_uniform_(self.classifier.weight)

        elif self.initialization == "he":
            nn.init.kaiming_uniform_(
                self.classifier.weight,
                nonlinearity="relu",
            )

        elif self.initialization == "lecun":
            fan_in = self.classifier.weight.size(1)
            bound = (3.0 / fan_in) ** 0.5
            nn.init.uniform_(
                self.classifier.weight,
                -bound,
                bound,
            )

        nn.init.zeros_(self.classifier.bias)


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
        label="Random classifier",
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
        cbar_kws={"label": "Average validation loss"},
        ax=axes[0],
    )
    axes[0].set_title("CV validation loss")
    axes[0].set_xlabel("Nodes per hidden layer")
    axes[0].set_ylabel("Number of hidden layers")

    sns.heatmap(
        auc_heatmap,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Average validation ROC-AUC"},
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
    class_weight=None,
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
    # Angi ønsket rekkefølge.
    optimizer_order = ["adam", "rmsprop", "sgd"]

    summary["optimizer"] = pd.Categorical(
        summary["optimizer"],
        categories=optimizer_order,
        ordered=True,
    )

    summary = (
        summary
        .sort_values("optimizer")
        .reset_index(drop=True)
    )

    # Én x-posisjon per optimizer.
    x_positions = np.arange(len(summary))

    # Standardavvik mellom CV-foldene.
    loss_error = summary["std_val_loss"].fillna(0).to_numpy()
    auc_error = summary["std_auc"].fillna(0).to_numpy()

    # Én farge per optimizer.
    optimizer_colors = {
        "adam": "#EDD27D",
        "rmsprop": "#E98B4A",
        "sgd": "#CE3434",
    }

    colors = [
        optimizer_colors[optimizer]
        for optimizer in summary["optimizer"]
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13, 5),
    )

    # -------------------------------------------------
    # Kryss-validert validation loss
    # -------------------------------------------------
    axes[0].bar(
        x_positions,
        summary["mean_val_loss"],
        yerr=loss_error,
        color=colors,
        edgecolor="black",
        width=0.45,
        capsize=5,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.5,
            "capthick": 1.5,
        },
    )

    axes[0].set_title("Cross-validated validation loss")
    axes[0].set_xlabel("Optimizer")
    axes[0].set_ylabel("Average validation loss")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(summary["optimizer"])
    axes[0].grid(axis="y", alpha=0.3)

    # -------------------------------------------------
    # Kryss-validert ROC-AUC
    # -------------------------------------------------
    axes[1].bar(
        x_positions,
        summary["mean_auc"],
        yerr=auc_error,
        color=colors,
        edgecolor="black",
        width=0.45,
        capsize=5,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.5,
            "capthick": 1.5,
        },
    )

    axes[1].set_title("Cross-validated ROC-AUC")
    axes[1].set_xlabel("Optimizer")
    axes[1].set_ylabel("Average ROC-AUC")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(summary["optimizer"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

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
        cbar_kws={"label": "Average validation loss"},
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
        cbar_kws={"label": "Average validation ROC-AUC"},
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
    device=None,
    seed=3155,
):
    """
    Plot out-of-fold ROC-AUC and binary cross-entropy loss by sequence length.

    Sequence length is defined as the number of CDMs (rows) in an event
    before the two-day cutoff.

    Each event receives an out-of-fold prediction. This means that the model
    predicting an event has not seen that event during training.

    The range from the shortest to the longest observed sequence is divided
    into `n_bins` integer intervals. The intervals are made as equally sized
    as possible. If the integer range cannot be divided evenly, the first
    intervals receive one additional sequence length.

    Example: if lengths range from 1 to 23 and n_bins=5, the groups are:

    1–5, 6–10, 11–15, 16–19, 20–23

    Parameters
    ----------
    data : dict
        Output from build_shared_datasets().
    hidden_layers : int or list[int]
        Number of recurrent layers, or a list of hidden units per layer.
    nodes_per_layer : int
        Number of hidden units per layer when hidden_layers is an integer.
    rnn_type : {"rnn", "gru", "lstm"}
        Type of recurrent neural network.
    dropout : float
        Dropout probability.
    epochs : int
        Number of training epochs per cross-validation fold.
    batch_size : int
        Batch size used during training and prediction.
    optimizer : str
        Optimizer name, for example "adam", "adamw", "sgd", or "rmsprop".
    learning_rate : float
        Learning rate used by the optimizer.
    class_weight : float or None
        Positive class weight. If None, it is estimated separately from the
        training partition in each cross-validation fold.
    penalty : {None, "l1", "l2"}
        Weight regularization type.
    penalty_strength : float
        Regularization strength.
    n_bins : int
        Number of integer sequence-length intervals.
    device : str or None
        Device used for training, for example "cpu", "cuda", or None.
    seed : int
        Base random seed.

    Returns
    -------
    event_results : pd.DataFrame
        One row per training event, including out-of-fold prediction,
        sequence length, and sequence-length group.
    summary : pd.DataFrame
        ROC-AUC, binary cross-entropy loss, event counts, and positive-case
        counts for each sequence-length group.
    """
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    X_train = data["sequences"]["train"]
    y_train = np.asarray(data["y_class"]["train"])
    cv_folds = data["cv_folds"]
    input_size = len(data["sequence_columns"])

    n_events = len(X_train)

    # Each event receives one prediction when it is validation data.
    oof_probabilities = np.full(
        n_events,
        np.nan,
        dtype=float,
    )

    fold_numbers = np.full(
        n_events,
        -1,
        dtype=int,
    )

    # Sequence length equals the number of CDMs/rows in each event.
    sequence_lengths = np.asarray(
        [len(sequence) for sequence in X_train],
        dtype=int,
    )

    minimum_length = int(sequence_lengths.min())
    maximum_length = int(sequence_lengths.max())

    # Include both endpoints. For example, 1 through 23 contains 23 lengths.
    number_of_possible_lengths = (
        maximum_length - minimum_length + 1
    )

    if n_bins > number_of_possible_lengths:
        raise ValueError(
            "n_bins cannot be larger than the number of possible "
            "integer sequence lengths. "
            f"Observed sequence-length range is "
            f"{minimum_length}–{maximum_length}."
        )

    for fold_number, (fit_idx, val_idx) in enumerate(
        cv_folds,
        start=1,
    ):
        print(f"Training CV fold {fold_number}/{len(cv_folds)}")

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

        oof_probabilities[val_idx] = model.predict_proba(
            X_val,
            batch_size=batch_size,
        )

        fold_numbers[val_idx] = fold_number

    if np.isnan(oof_probabilities).any():
        raise RuntimeError(
            "Some events are missing out-of-fold predictions. "
            "Check that the CV folds cover all training events."
        )

    event_results = pd.DataFrame(
        {
            "sequence_length": sequence_lengths,
            "y_true": y_train,
            "y_probability": oof_probabilities,
            "fold": fold_numbers,
        }
    )

    # Divide possible integer lengths as evenly as possible.
    #
    # Example:
    # 23 possible lengths and 5 bins:
    # base_width = 4, remainder = 3
    # bin widths = [5, 5, 5, 4, 4]
    base_width, remainder = divmod(
        number_of_possible_lengths,
        n_bins,
    )

    bin_widths = [
        base_width + 1 if bin_index < remainder else base_width
        for bin_index in range(n_bins)
    ]

    bin_starts = []
    bin_ends = []

    current_start = minimum_length

    for width in bin_widths:
        current_end = current_start + width - 1

        bin_starts.append(current_start)
        bin_ends.append(current_end)

        current_start = current_end + 1

    bin_labels = [
        f"{start}–{end}"
        for start, end in zip(bin_starts, bin_ends)
    ]

    # Half-integer edges assign integer sequence lengths to inclusive bins.
    # Example for 1–5 and 6–10: 0.5, 5.5, 10.5.
    bin_edges = [minimum_length - 0.5] + [
        end + 0.5
        for end in bin_ends
    ]

    event_results["length_bin"] = pd.cut(
        event_results["sequence_length"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        ordered=True,
    )

    rows = []

    for length_bin, group in event_results.groupby(
        "length_bin",
        observed=True,
    ):
        y_group = group["y_true"].to_numpy()
        p_group = group["y_probability"].to_numpy()

        # ROC-AUC requires at least one observation from each class.
        if len(np.unique(y_group)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(
                y_group,
                p_group,
            )

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
                "mean_sequence_length": (
                    group["sequence_length"].mean()
                ),
                "min_sequence_length": (
                    group["sequence_length"].min()
                ),
                "max_sequence_length": (
                    group["sequence_length"].max()
                ),
                "n_events": len(group),
                "n_positive_cases": int(y_group.sum()),
                "auc": auc,
                "bce_loss": bce_loss,
            }
        )

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "min_sequence_length",
    ).reset_index(drop=True)

    x_positions = np.arange(len(summary))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 5),
    )

    # ROC-AUC plot.
    axes[0].bar(
        x_positions,
        summary["auc"],
        color="steelblue",
        edgecolor="black",
    )

    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Out-of-fold ROC-AUC by sequence length")
    axes[0].set_xlabel("Sequence length group (number of CDMs)")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(
        summary["length_bin"],
        rotation=30,
        ha="right",
    )
    axes[0].grid(axis="y", alpha=0.3)

    # Binary cross-entropy plot.
    axes[1].bar(
        x_positions,
        summary["bce_loss"],
        color="indianred",
        edgecolor="black",
    )

    axes[1].set_title(
        "Out-of-fold binary cross-entropy loss by sequence length"
    )
    axes[1].set_xlabel("Sequence length group (number of CDMs)")
    axes[1].set_ylabel("Binary cross-entropy loss")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(
        summary["length_bin"],
        rotation=30,
        ha="right",
    )
    axes[1].grid(axis="y", alpha=0.3)

    # Add event counts and positive-case counts above the bars.
    for axis, values in zip(
        axes,
        [summary["auc"], summary["bce_loss"]],
    ):
        for index, value in enumerate(values):
            if not np.isnan(value):
                axis.text(
                    index,
                    value,
                    f"Events: {summary.loc[index, 'n_events']}\n"
                    f"Positive cases: "
                    f"{summary.loc[index, 'n_positive_cases']}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.show()

    return event_results, summary


    import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


def set_seed(seed):
    """
    Set random seeds for more reproducible experiments.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compare_initializations_cv(
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
    device=None,
    seed=3155,
):
    """
    Compare weight initialization methods using cross-validation.

    The function evaluates the following initialization methods:

    - "default": PyTorch's built-in initialization.
    - "xavier": Xavier/Glorot uniform initialization.
    - "he": He/Kaiming uniform initialization.
    - "lecun": LeCun uniform initialization.

    A new model is created and trained for every initialization method
    and cross-validation fold.

    Parameters
    ----------
    data : dict
        Output from build_shared_datasets().
    hidden_layers : int or list[int]
        Number of recurrent layers, or a list defining hidden units
        in each layer.
    nodes_per_layer : int
        Number of hidden units per layer when hidden_layers is an integer.
    rnn_type : {"rnn", "gru", "lstm"}
        Type of recurrent neural network.
    dropout : float
        Dropout probability.
    epochs : int
        Number of training epochs per fold.
    batch_size : int
        Training batch size.
    optimizer : str
        Optimizer name, for example "adam", "rmsprop", or "sgd".
    learning_rate : float
        Optimizer learning rate.
    class_weight : float or None
        Positive class weight for BCEWithLogitsLoss. If None, it is
        calculated separately from the training partition in each fold.
    penalty : {None, "l1", "l2"}
        Type of weight regularization.
    penalty_strength : float
        Regularization strength.
    device : str or None
        Device used for training, for example "cpu", "cuda", or None.
    seed : int
        Base random seed.

    Returns
    -------
    results : pd.DataFrame
        One row per initialization method and CV fold.
    summary : pd.DataFrame
        Mean and standard deviation of validation loss and ROC-AUC
        for each initialization method.
    """
    initialization_methods = [
        "default",
        "xavier",
        "he",
        "lecun",
    ]

    X_train = data["sequences"]["train"]
    y_train = np.asarray(data["y_class"]["train"])
    cv_folds = data["cv_folds"]
    input_size = len(data["sequence_columns"])

    results_rows = []

    for initialization_number, initialization in enumerate(
        initialization_methods
    ):
        print(
            f"\n{'=' * 60}\n"
            f"Testing initialization: {initialization.upper()}\n"
            f"{'=' * 60}"
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

            # Each initialization/fold combination receives a reproducible seed.
            current_seed = (
                seed
                + initialization_number * 100
                + fold_number
            )
            set_seed(current_seed)

            model = RNNClassifier(
                input_size=input_size,
                hidden_layers=hidden_layers,
                nodes_per_layer=nodes_per_layer,
                rnn_type=rnn_type,
                dropout=dropout,
                initialization=initialization,
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

            final_val_loss = model.history["val_loss"][-1]

            y_val_probabilities = model.predict_proba(
                X_val,
                batch_size=batch_size,
            )

            val_auc = roc_auc_score(
                y_val,
                y_val_probabilities,
            )

            results_rows.append(
                {
                    "initialization": initialization,
                    "fold": fold_number,
                    "final_val_loss": final_val_loss,
                    "val_auc": val_auc,
                }
            )

            print(
                f"Fold {fold_number}/{len(cv_folds)} | "
                f"Validation loss: {final_val_loss:.4f} | "
                f"Validation AUC: {val_auc:.4f}"
            )

    results = pd.DataFrame(results_rows)

    summary = (
        results
        .groupby("initialization", as_index=False)
        .agg(
            mean_val_loss=("final_val_loss", "mean"),
            std_val_loss=("final_val_loss", "std"),
            mean_auc=("val_auc", "mean"),
            std_auc=("val_auc", "std"),
        )
    )

    # Keep a consistent and meaningful order in the table and plots.
    summary["initialization"] = pd.Categorical(
        summary["initialization"],
        categories=initialization_methods,
        ordered=True,
    )

    summary = (
        summary
        .sort_values("initialization")
        .reset_index(drop=True)
    )

    # One explicit x-position per bar. This prevents misalignment between
    # bars, error bars, and x-axis labels.
    x_positions = np.arange(len(summary))

    loss_error = summary["std_val_loss"].fillna(0).to_numpy()
    auc_error = summary["std_auc"].fillna(0).to_numpy()

    colors = {
        "default": "#7B9ACC",
        "xavier": "#74B49B",
        "he": "#F2B84B",
        "lecun": "#D97B66",
    }

    bar_colors = [
        colors[initialization]
        for initialization in summary["initialization"]
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 5),
    )

    # Cross-validated validation loss.
    axes[0].bar(
        x_positions,
        summary["mean_val_loss"],
        yerr=loss_error,
        color=bar_colors,
        edgecolor="black",
        width=0.55,
        capsize=5,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.5,
            "capthick": 1.5,
        },
    )

    axes[0].set_title("Cross-validated validation loss")
    axes[0].set_xlabel("Initialization method")
    axes[0].set_ylabel("Mean validation loss")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(summary["initialization"])
    axes[0].grid(axis="y", alpha=0.3)

    # Cross-validated ROC-AUC.
    axes[1].bar(
        x_positions,
        summary["mean_auc"],
        yerr=auc_error,
        color=bar_colors,
        edgecolor="black",
        width=0.55,
        capsize=5,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.5,
            "capthick": 1.5,
        },
    )

    axes[1].set_title("Cross-validated ROC-AUC")
    axes[1].set_xlabel("Initialization method")
    axes[1].set_ylabel("Mean ROC-AUC")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(summary["initialization"])
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results, summary

@torch.no_grad()
def evaluate_on_test_set(
    model,
    X_test,
    y_test,
    loss_function,
    batch_size=128,
):
    """
    Evaluate a trained sequence regression model on the held-out test set.

    The loss is calculated once across all test observations so that the
    F2 score is based on the entire test set rather than individual batches.

    Parameters
    ----------
    model : torch.nn.Module
        A trained model.
    X_test : list
        List of test sequences.
    y_test : array-like
        True continuous risk values for the test sequences.
    loss_function : F2VektetMSELoss
        Loss function used for evaluation.
    batch_size : int
        Number of sequences evaluated per batch.

    Returns
    -------
    test_loss : float
        F2-weighted MSE loss on the complete test set.
    test_f2 : float
        F2 score on the complete test set.
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Model predictions.
    """
    model.eval()

    test_dataset = SequenceDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=training_collate_fn,
    )

    all_predictions = []
    all_targets = []

    for X_batch, lengths, y_batch in test_loader:
        X_batch = X_batch.to(model.device)
        lengths = lengths.to(model.device)
        y_batch = y_batch.to(model.device)

        predictions = model(X_batch, lengths)

        all_predictions.append(predictions.reshape(-1))
        all_targets.append(y_batch.reshape(-1))

    # Combine all batches before calculating F2 and loss.
    y_pred = torch.cat(all_predictions)
    y_true = torch.cat(all_targets)

    test_f2 = loss_function.beregn_f2(
        y_true=y_true,
        y_pred=y_pred,
    )

    test_loss = loss_function(
        y_pred=y_pred,
        y_true=y_true,
        f2_score=test_f2,
    )

    return (
        test_loss.item(),
        test_f2.item(),
        y_true.cpu().numpy(),
        y_pred.cpu().numpy(),
    )




def plot_precision_recall_curve(
    model,
    X,
    y_true,
    batch_size=128,
    title="Precision–Recall Curve",
):
    """
    Plot a precision-recall curve for a trained binary classification model.

    The model must implement a `predict_proba()` method that returns the
    predicted probability of the positive class for each input sequence.

    Parameters
    ----------
    model : RNNClassifier
        A trained binary classification model with a `predict_proba` method.
    X : list
        Input sequences to evaluate, for example `data["sequences"]["test"]`.
    y_true : array-like
        True binary labels. Values must be 0 or 1.
    batch_size : int, default=128
        Prediction batch size.
    title : str, default="Precision–Recall Curve"
        Title shown above the plot.

    Returns
    -------
    precision : np.ndarray
        Precision values for different probability thresholds.
    recall : np.ndarray
        Recall values for different probability thresholds.
    thresholds : np.ndarray
        Probability thresholds. This array has one fewer element than
        precision and recall.
    average_precision : float
        Average precision score.
    """
    y_true = np.asarray(y_true).reshape(-1)

    if len(X) != len(y_true):
        raise ValueError(
            "X and y_true must contain the same number of observations."
        )

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary values: 0 and 1.")

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "A precision-recall curve requires both classes in y_true."
        )

    y_probability = model.predict_proba(
        X,
        batch_size=batch_size,
    )

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_probability,
    )

    average_precision = average_precision_score(
        y_true,
        y_probability,
    )

    positive_prevalence = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        recall,
        precision,
        color="steelblue",
        linewidth=2,
        label=f"Model (average precision = {average_precision:.3f})",
    )

    # Baseline precision equals the proportion of positive cases.
    ax.axhline(
        positive_prevalence,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=(
            "No-skill baseline "
            f"(positive prevalence = {positive_prevalence:.3f})"
        ),
    )

    ax.set_title(title)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()

    return precision, recall, thresholds, average_precision