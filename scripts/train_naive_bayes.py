# scripts/train_naive_bayes.py
from src.data_loading import load_train_test
from src.preprocessing import get_text_and_labels
from src.naive_bayes import build_naive_bayes_pipeline
from src.evaluation import evaluate_model

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

if __name__ == "__main__":
    main()
