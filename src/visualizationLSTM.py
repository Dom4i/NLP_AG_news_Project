# src/visualizationLSTM.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


def plot_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Training & Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels, target_names, title, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_prediction_confidence(proba, title="Prediction Confidence Histogram"):
    max_conf = np.max(proba, axis=1)

    plt.figure(figsize=(6, 4))
    plt.hist(max_conf, bins=20, alpha=0.7)
    plt.title(title)
    plt.xlabel("Confidence (max softmax probability)")
    plt.ylabel("Count")
    plt.show()

def show_misclassified_examples(texts, y_true, y_pred, class_names, max_examples=5):
    """Gibt ein paar falsch klassifizierte Beispiele aus."""
    mis = np.where(y_true != y_pred)[0]

    print(f"\n=== Falsch klassifizierte Beispiele: {len(mis)} ===\n")

    for i in mis[:max_examples]:
        print(f"Text {i}:")
        print(texts.iloc[i][:300].replace("\n", " ") + "...")
        print(f"True: {class_names[y_true.iloc[i]]}, Pred: {class_names[y_pred[i]]}")
        print("-" * 60)


def plot_tsne_embeddings(model, X_pad, y_true, class_names, max_samples=2000):
    """
    Holt Embedding-Layer-Ausgaben und projiziert sie via t-SNE auf 2D.
    Funktioniert NUR für Keras LSTM Modelle.
    """

    # Falls der Wrapper übergeben wurde: echtes Keras-Modell holen
    if hasattr(model, "model"):
        keras_model = model.model
    else:
        keras_model = model   # Direktes Sequential-Modell

    # Embedding-Layer extrahieren
    embedding_model = None
    for layer in keras_model.layers:
        if "embedding" in layer.name.lower():
            embedding_model = layer
            break

    if embedding_model is None:
        print("[WARN] Kein Embedding-Layer gefunden – t-SNE wird übersprungen.")
        return

    # Embeddings berechnen
    X_embed = embedding_model(X_pad).numpy()

    # Durchschnitt pro Sequenz (mean pooling)
    X_embed_mean = X_embed.mean(axis=1)

    # Subsampling
    if len(X_embed_mean) > max_samples:
        idx = np.random.choice(len(X_embed_mean), max_samples, replace=False)
        X_emb = X_embed_mean[idx]
        y_small = y_true.iloc[idx]
    else:
        X_emb = X_embed_mean
        y_small = y_true

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    X_2d = tsne.fit_transform(X_emb)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y_small,
        cmap="tab10",
        alpha=0.7
    )

    handles, _ = scatter.legend_elements()
    plt.legend(handles, list(class_names.values()))
    plt.title("t-SNE Sentence Embedding Visualization")
    plt.show()

def plot_class_distribution(y_train, y_test, class_names):
    import matplotlib.pyplot as plt
    import numpy as np

    counts_train = y_train.value_counts().sort_index()
    counts_test = y_test.value_counts().sort_index()

    x = np.arange(len(class_names))

    plt.figure(figsize=(8,5))
    plt.bar(x - 0.2, counts_train, width=0.4, label="Train")
    plt.bar(x + 0.2, counts_test, width=0.4, label="Test")
    plt.xticks(x, [class_names[i+1] for i in x])
    plt.ylabel("Anzahl Samples")
    plt.title("Klassenverteilung")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_error_vs_length(X_text, y_true, y_pred):
    import matplotlib.pyplot as plt
    import numpy as np

    lengths = X_text.apply(lambda x: len(x.split()))
    errors = (y_true != y_pred).astype(int)

    plt.figure(figsize=(8,5))
    plt.scatter(lengths, errors, alpha=0.3)
    plt.xlabel("Sequenzlänge (Tokens)")
    plt.ylabel("Fehler (1=Falsch, 0=Richtig)")
    plt.title("Fehlerquote vs Sequenzlänge")
    plt.tight_layout()
    plt.show()


def plot_confidence_per_class(proba, y_true, class_names):
    import matplotlib.pyplot as plt
    import numpy as np

    num_classes = len(class_names)

    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        class_idx = np.where(y_true == (i + 1))[0]
        class_probs = proba[class_idx, i]
        plt.hist(class_probs, bins=20, alpha=0.5, label=class_names[i + 1])

    plt.xlabel("Softmax-Wahrscheinlichkeit")
    plt.ylabel("Anzahl")
    plt.title("Vorhersage-Sicherheit pro Klasse")
    plt.legend()
    plt.tight_layout()
    plt.show()

