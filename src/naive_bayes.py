# src/naive_bayes.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_naive_bayes_pipeline(
    max_features: int = 20000,
    ngram_range=(1, 2)
) -> Pipeline:
    """
    Erstellt eine Textklassifikations-Pipeline mit Tfidf + Multinomial Naive Bayes.
    """
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )),
        ("clf", MultinomialNB())
    ])
    return pipe
