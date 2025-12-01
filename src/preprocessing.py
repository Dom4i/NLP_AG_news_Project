import html
import re
from typing import List, Tuple
import pandas as pd
from .config import TEXT_COLUMNS, LABEL_COLUMN


def clean_text(text: str) -> str:
    """Bereinigt Text von HTML-Entities wie #39."""
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    text = re.sub(r'#\d+;', "'", text)  # z.B. #39; -> '
    text = re.sub(r'#\d+', "'", text)  # z.B. #39 -> '
    text = re.sub(r'\b(AP|AFP|Reuters)\b', '', text) # Nachrichtenagenturen

    text = " ".join(text.split())
    return text

def combine_text_columns(df: pd.DataFrame,
                         text_cols: List[str] = None) -> pd.Series:
    """Führt mehrere Textspalten in eine zusammen und bereinigt HTML-Entities."""
    if text_cols is None:
        text_cols = TEXT_COLUMNS

    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        raise ValueError(f"Keine der TEXT_COLUMNS {text_cols} in DataFrame gefunden.")

    combined = df[cols].fillna("").agg(" ".join, axis=1)
    combined_clean = combined.apply(clean_text)

    return combined_clean

def get_text_and_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Gibt Textserie X und Labelserie y zurück."""
    X = combine_text_columns(df)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label-Spalte '{LABEL_COLUMN}' nicht im DataFrame gefunden.")
    y = df[LABEL_COLUMN]
    return X, y
