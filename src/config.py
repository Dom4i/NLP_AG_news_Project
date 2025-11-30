from pathlib import Path

# Projektwurzel = Ordner, der dieses File enthält, 2 Ebenen hoch
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"



LABEL_COLUMN = "Class Index"
TEXT_COLUMNS = ["Title", "Description"]


# AG News Klassenmapping :
CLASS_NAMES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Science",
}

# Anzahl Klassen (praktisch fürs LSTM)
NUM_CLASSES = len(CLASS_NAMES)

# LSTM / Tokenizer Settings
MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 50

EMBED_DIM = 128
LSTM_UNITS = 128

BATCH_SIZE = 128
EPOCHS = 5