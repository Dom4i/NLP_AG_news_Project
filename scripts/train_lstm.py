# scripts/train_lstm.py

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.data_loading import load_train_test
from src.preprocessing import get_text_and_labels
from src.lstm import build_lstm_model
from src.evaluation import evaluate_model
from src.config import MAX_VOCAB_SIZE, MAX_SEQ_LEN, BATCH_SIZE, EPOCHS
import matplotlib.pyplot as plt


class KerasPredictWrapper:
    """
    Wrapper, damit unser Keras-Modell mit evaluate_model() kompatibel ist.
    evaluate_model() erwartet ein Objekt mit .predict(X), das Klassenlabels (1..4) zurückgibt.
    """
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        proba = self.model.predict(X, verbose=0)
        # argmax -> 0..3, +1 -> 1..4 (wie im Dataset)
        return np.argmax(proba, axis=1) + 1


def main():
    # 1. Daten laden
    train_df, test_df = load_train_test()

    # 2. Text & Labels extrahieren
    X_train_text, y_train = get_text_and_labels(train_df)
    X_test_text, y_test = get_text_and_labels(test_df)

    print("X_train_text sample:", X_train_text.iloc[0][:200])
    print("Length of first text:", len(X_train_text.iloc[0].split()))

    print("Unique labels:", sorted(y_train.unique()))
    print("Unique labels after mapping:", set((y_train - 1)))

    # 3. Tokenizer fitten (nur auf Trainingsdaten)
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    # 4. Texte in Sequenzen umwandeln
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    print("Vocab size:", len(tokenizer.word_index))
    print("Example tokenized:", X_train_seq[0][:20])

    # 5. Padding/Truncating auf feste Länge
    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post",
    )
    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post",
    )

    # 6. Vokabulargröße bestimmen
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)

    # 7. LSTM-Modell bauen
    model = build_lstm_model(vocab_size=vocab_size)
    model.summary()

    # 🔴 Labels für Training auf 0..3 mappen (Keras braucht 0-based Klassen)
    y_train_0based = y_train.values - 1

    history = model.fit(
        X_train_pad,
        y_train_0based,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        shuffle=True,  # zur Sicherheit explizit
        verbose=1,
    )

    # 9. Evaluation mit der bestehenden evaluate_model-Funktion
    wrapped_model = KerasPredictWrapper(model)
    evaluate_model(wrapped_model, X_test_pad, y_test, title="LSTM (Keras)")

    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
