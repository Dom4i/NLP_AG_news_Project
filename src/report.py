"""
===========================
Project Report
===========================

Authors: Gössl Marcel, Marek Simon, Schrenk Dominik, Unger Miriam
Date: 06.11.2025
Course: Natural Language Processing

----------------------------------------
1. Dataset Description
----------------------------------------
• What dataset did you use?
    https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download&select=train.csv
    Diese Dateien bestehen aus mehr als  120 000 Proben von Nachrichtenartikeln und enthält 3 Spalten. Die erste Spalte ist die Klassen-ID, die zweite Spalte der Titel und die dritte Spalte die Beschreibung. Die Klassen-IDs sind von 1 bis 4 nummeriert, wobei 1 für Welt, 2 für Sport, 3 für Wirtschaft und 4 für Wissenschaft/Technik steht.
• How many images/classes does it contain?
    Gesamtanzahl der Zeilen: 127 600
    Klassen:

• Was the dataset balanced or imbalanced?
    Gebalenced welches man an einer späteren Visualization gut erkennen kann.

• How did you preprocess or split the data?
    Netterweise war der Datensatz schon in train und test gesplittet, dies haben wir einfach übernommen.

----------------------------------------
2. Relation to Real Applications
----------------------------------------
• How does this project relate to real-world use cases?
• In what kind of system could your model be applied?
• What practical benefits could it have?

    Dieses Projekt zeigt, wie man Texte automatisch in verschiedene Kategorien einordnen kann. Solche Modelle werden in der echten Welt zum Beispiel eingesetzt, um Nachrichtenartikel,
    E-Mails oder Kundenbewertungen automatisch zu sortieren. Man könnte sie auch nutzen, um Spam von wichtigen Nachrichten zu unterscheiden oder grosse Textmengen zu analysieren, ohne dass Menschen alles lesen müssen.
    Der praktische Nutzen ist, dass man Zeit spart, Fehler reduziert und schneller einen Überblick bekommt, wenn man viele Texte hat.


----------------------------------------
3. Problems and Solutions
----------------------------------------
• Which problems or challenges did you encounter?
   Problem mit #39
   Und AP APF Reuter New York


• How did you solve or work around them?

----------------------------------------
4. Implementation Details
----------------------------------------
    Für dieses Projekt haben wir Python verwendet und verschiedene Bibliotheken importiert, die uns das Arbeiten mit Texten und maschinellem Lernen erleichtern. Dazu gehören Pandas und NumPy für die Datenverarbeitung,
    scikit-learn für klassische Machine-Learning-Modelle wie Naive Bayes, und TensorFlow/Keras für das tiefere Lernen mit LSTM-Netzen. Wir haben zwei unterschiedliche Modelltypen eingesetzt: Zum einen Naive Bayes,
    das schnell trainiert werden kann und gut bei klar trennbaren Textkategorien funktioniert, und zum anderen LSTM, ein neuronales Netzwerk, das den Text als Sequenz verarbeitet und dadurch zusammenhängende Muster in den Sätzen erkennen kann.
    So können wir vergleichen, welches Modell für unsere Textklassifizierungsaufgabe besser geeignet ist, und unterschiedliche Ansätze für die Analyse ausprobieren.
----------------------------------------
5. Results and Evaluation
----------------------------------------
    Wir haben zwei Modelle getestet: Naive Bayes und ein LSTM. Beide arbeiten sehr gut und erreichen über 90 % Genauigkeit. Besonders einfach zu erkennen sind die Kategorien Sports und World, etwas schwieriger sind Business und Science,
    weil sich die Themen überschneiden. LSTM ist insgesamt ein bisschen besser, besonders bei längeren oder komplexeren Texten. Die zusätzlichen Plots wie Confusion Matrix, t-SNE oder Feature-Heatmaps helfen uns zu sehen,
    welche Texte das Modell gut oder schlecht versteht und welche Wörter oder Muster besonders wichtig sind.

----------------------------------------
6. Discussion and Learnings
----------------------------------------
    Mit dem Naive-Bayes-Modell lief’s am Anfang echt entspannt. Das hatten wir ja schon zig Mal gemacht, alles fühlte sich vertraut an und ging relativ schnell. Beim LSTM dagegen haben wir echt mehr gekämpft. Da mussten wir
    bei den Sachen wie Tokenizer, Embeddings schon eher aufpassen und schauen das wir da keinen Overfitting reinbringen.

    Ansich war es aber eine wirklich mega entspannte Aufgabe und manches aus dem Unterricht macht jetzt schon mehr Sinn.
"""

