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