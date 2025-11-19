# src/evaluation.py
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CLASS_NAMES


def evaluate_model(model, X_test, y_test, title: str = "Model"):
    y_pred = model.predict(X_test)

    # Klassen sortiert nach Index (1,2,3,4)
    labels = sorted(CLASS_NAMES.keys())
    target_names = [CLASS_NAMES[l] for l in labels]

    print("=== Klassen / Topics ===")
    for l in labels:
        print(f"{l} -> {CLASS_NAMES[l]}")
    print()

    acc = accuracy_score(y_test, y_pred)
    print(f"=== {title} ===")
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names
    ))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
