from typing import List, Tuple
import pandas as pd
from .config import TEXT_COLUMNS, LABEL_COLUMN

def combine_text_columns(df: pd.DataFrame,
                         text_cols: List[str] = None) -> pd.Series:
    """Führt mehrere Textspalten in eine zusammen."""
    if text_cols is None:
        text_cols = TEXT_COLUMNS

    # Nur existierende Spalten verwenden
    cols = [c for c in text_cols if c in df.columns]
    if not cols:
        raise ValueError(f"Keine der TEXT_COLUMNS {text_cols} in DataFrame gefunden.")

    # Fehlende Werte zu leeren Strings machen, dann zusammenfügen
    return df[cols].fillna("").agg(" ".join, axis=1)


def get_text_and_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Gibt Textserie X und Labelserie y zurück."""
    X = combine_text_columns(df)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Label-Spalte '{LABEL_COLUMN}' nicht im DataFrame gefunden.")
    y = df[LABEL_COLUMN]
    return X, y
