#TODO: Ähnlich wie in naive_bayes.py eine LSTM Pipeline bauen
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

from .config import NUM_CLASSES, EMBED_DIM, LSTM_UNITS, MAX_SEQ_LEN


def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = EMBED_DIM,
    lstm_units: int = LSTM_UNITS,
    input_length: int = MAX_SEQ_LEN,
    num_classes: int = NUM_CLASSES,
):
    """
    Baut ein etwas stärkeres LSTM-Modell für Textklassifikation.
    """
    model = Sequential()
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,   # <-- wichtig, Padding ignorieren
    ))
    # Bidirektionales LSTM hilft oft stark bei Text
    model.add(Bidirectional(LSTM(lstm_units)))
    model.add(Dropout(0.5)) #<--- wurde erhöht um Overfitting zu verhindern
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model
