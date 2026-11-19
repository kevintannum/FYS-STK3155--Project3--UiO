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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

