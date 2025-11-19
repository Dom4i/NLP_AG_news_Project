import pandas as pd
from .config import TRAIN_PATH, TEST_PATH

def load_train_test():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df
