"""
This module functions for dealing with the MAFALDA dataset by Helwe et al. (2024).
"""
import pandas as pd
import json
from .utils import log
import os.path


def create_mafalda_df() -> pd.DataFrame:
    df = pd.read_json('datasets/MAFALDA/gold_standard_dataset.jsonl', lines=True)
    df['sentences_with_labels'] = df['sentences_with_labels'].apply(json.loads)

    return df


def get_mafalda_df(filename: str) -> pd.DataFrame:
    """
    Load the mafalda dataframe from a JSONL file, or create a new one if the file doesn't exist.
    """
    if os.path.isfile(filename):
        df = pd.read_json(filename, lines=True)

        log(f"Loaded existing mafalda dataframe from {filename}.")
    else:
        df = create_mafalda_df()

        log("Created new mafalda dataframe.")

    return df


def save_mafalda_df(df: pd.DataFrame, filename: str):
    """
    Save the mafalda dataframe to a JSON file.
    """
    df.to_json(filename, index=False, orient='records', lines=True)