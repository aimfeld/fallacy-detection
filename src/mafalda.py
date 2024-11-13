"""
This module functions for dealing with the MAFALDA dataset by Helwe et al. (2024).
"""
import pandas as pd
import json
from .utils import log

# Columns that contain dictionaries that need to be converted to/from JSON
DICT_COLS = ['sentences_with_labels']

def create_mafalda_df() -> pd.DataFrame:
    df = pd.read_json('datasets/MAFALDA/gold_standard_dataset.jsonl', lines=True)

    return df


def get_mafalda_df(filename: str) -> pd.DataFrame:
    """
    Load the mafalda dataframe from a CSV file, or create a new one if the file doesn't exist.
    """
    try:
        df = pd.read_csv(filename)

        log(f"Loaded existing mafalda dataframe from {filename}.")
    except FileNotFoundError:
        df = create_mafalda_df()

        log("Created new mafalda dataframe.")

    for col in DICT_COLS:
        df[col] = df[col].apply(json.loads)

    return df


def save_mafalda_df(df: pd.DataFrame, filename: str):
    """
    Save the mafalda dataframe to a CSV file.
    """
    df = df.copy()
    for col in DICT_COLS:
        df[col] = df[col].apply(json.dumps)

    df.to_csv(filename, index=False)