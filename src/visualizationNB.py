# src/visualizations.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_top_words(model, n=20, class_names=None):

    nb = model.named_steps["clf"]  # Dein MultinomialNB
    vectorizer = model.named_steps["tfidf"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    class_log_probs = nb.feature_log_prob_

    num_classes = class_log_probs.shape[0]

    for idx in range(num_classes):
        topn = np.argsort(class_log_probs[idx])[::-1][:n]
        plt.figure(figsize=(10, 4))
        plt.barh(feature_names[topn][::-1], class_log_probs[idx][topn][::-1])
        plt.title(f"Top {n} Wörter – Klasse {class_names[idx] if class_names else idx}")
        plt.xlabel("Log Wahrscheinlichkeit")
        plt.tight_layout()
        plt.show()


def plot_wordclouds(model, class_names=None):

    nb = model.named_steps["clf"]
    vectorizer = model.named_steps["tfidf"]

    feature_names = vectorizer.get_feature_names_out()
    class_log_probs = nb.feature_log_prob_
    num_classes = class_log_probs.shape[0]

    for idx in range(num_classes):
        weights = dict(zip(feature_names, class_log_probs[idx]))
        wc = WordCloud(width=900, height=500, background_color="white")
        wc.generate_from_frequencies(weights)

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Wordcloud – Klasse {class_names[idx] if class_names else idx}")
        plt.show()


def plot_roc_curves(model, X_test, y_test, class_names):

    y_score = model.predict_proba(X_test)
    classes_labels = sorted([1, 2, 3, 4])
    y_binarized = label_binarize(y_test, classes=classes_labels)
    num_classes = y_binarized.shape[1]

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_binarized[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()


def plot_precision_recall(model, X_test, y_test, class_names):

    y_score = model.predict_proba(X_test)
    classes_labels = sorted([1, 2, 3, 4])
    y_binarized = label_binarize(y_test, classes=classes_labels)
    num_classes = y_binarized.shape[1]

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_binarized[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.show()


def plot_feature_weights_heatmap(model, class_names=None, top_n=30):
    import pandas as pd
    nb = model.named_steps["clf"]
    vectorizer = model.named_steps["tfidf"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    class_log_probs = nb.feature_log_prob_

    data = {}
    for idx, cname in enumerate(class_names):
        top_idx = np.argsort(class_log_probs[idx])[-top_n:]
        top_features = feature_names[top_idx]
        top_probs = class_log_probs[idx][top_idx]
        data[cname] = pd.Series(top_probs, index=top_features)

    df = pd.DataFrame(data)

    # Heatmap Größe anpassen je nach Anzahl Features
    plt.figure(figsize=(len(class_names) * 3, top_n * 0.4 + 4))

    # Heatmap mit anderen parameter damit man lesen kann
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={'label': 'Log Wahrscheinlichkeit'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title("Top Feature Log-Probabilities pro Klasse", fontsize=16)
    plt.xlabel("Klasse", fontsize=14)
    plt.ylabel("Feature", fontsize=14)

    # Lesbare Schriftgröße für Y-Ticks
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=12, rotation=45)

    plt.tight_layout()
    plt.show()


def plot_class_distribution(y_train, y_test, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.countplot(x=y_train, ax=axes[0])
    axes[0].set_title("Trainingsset")
    axes[0].set_xticks(np.arange(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45)

    sns.countplot(x=y_test, ax=axes[1])
    axes[1].set_xticks(np.arange(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45)

    plt.tight_layout()
    plt.show()
