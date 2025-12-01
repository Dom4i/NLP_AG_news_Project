# scripts/train_naive_bayes.py
from src.data_loading import load_train_test
from src.preprocessing import get_text_and_labels
from src.naive_bayes import build_naive_bayes_pipeline
from src.evaluation import evaluate_model

from src.visualizationNB import (
    plot_top_words,
    plot_wordclouds,
    plot_roc_curves,
    plot_precision_recall,
    plot_feature_weights_heatmap,
    plot_class_distribution

)
from src.config import CLASS_NAMES


def main():
    # 1. Daten laden
    train_df, test_df = load_train_test()

    # 2. Text & Labels extrahieren
    X_train, y_train = get_text_and_labels(train_df)
    X_test, y_test = get_text_and_labels(test_df)

    # 3. Modell bauen
    model = build_naive_bayes_pipeline()

    # 4. Trainieren
    print("Training Naive Bayes...")
    model.fit(X_train, y_train)

    # 5. Evaluieren
    evaluate_model(model, X_test, y_test, title="Naive Bayes (TF-IDF)")

    print("\n=== Zusätzliche Visualisierungen ===")
    plot_top_words(model, n=20, class_names=list(CLASS_NAMES.values()))
    plot_wordclouds(model, class_names=list(CLASS_NAMES.values()))
    plot_roc_curves(model, X_test, y_test, class_names=list(CLASS_NAMES.values()))
    plot_precision_recall(model, X_test, y_test, class_names=list(CLASS_NAMES.values()))
    plot_feature_weights_heatmap(model, class_names=list(CLASS_NAMES.values()), top_n=30)
    plot_class_distribution(y_train, y_test, class_names=list(CLASS_NAMES.values()))

    print("Klassen im Testset:", sorted(y_test.unique()))
    print("Alle Klassen:", list(CLASS_NAMES.keys()))


if __name__ == "__main__":
    main()
