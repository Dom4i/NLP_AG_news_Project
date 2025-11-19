# Trash Type Image Classification
# Python-Version

Python 3.12.9

# Umgebung einrichten

Virtuelle Umgebung erstellen:
python -m venv .venv

Aktivieren:

Windows (PowerShell):
.venv\Scripts\Activate.ps1

macOS/Linux:
source .venv/bin/activate

# Abhängigkeiten installieren:
pip install -r requirements.txt

# Neue Abhängigkeiten hinzufügen

Wenn du ein neues Paket installierst, z. B.:
pip install opencv-python

Danach immer:
pip freeze > requirements.txt
Damit wird die aktuelle Paketliste für alle aktualisiert.

# Kaggle Dataset

Dataset bereits in Github hochgeladen, hier der Link zum Original:
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download

data
├── test.csv
└── train.csv

# Projekt starten
Scripts befinden sich im Ordner `scripts`.