"""
This module functions for dealing with the FALLACIES dataset by Hong et al. (2024).
"""
import pandas as pd
import json
from .utils import log


def create_fallacy_df() -> pd.DataFrame:
    df = pd.read_json('datasets/FALLACIES/step_fallacy.test.jsonl', lines=True)
    df = add_taxonomy(df)

    df['step'] = df['step'].apply(_remove_square_brackets)

    return df


def get_fallacy_df(filename: str, only_incorrect: bool = False) -> pd.DataFrame:
    """
    Load the fallacy dataframe from a CSV file, or create a new one if the file doesn't exist.
    """
    try:
        df = pd.read_csv(filename)
        df = df.fillna('')

        log(f"Loaded existing fallacy dataframe from {filename}.")
    except FileNotFoundError:
        df = create_fallacy_df()

        log("Created new fallacy identification dataframe.")

    if only_incorrect:
        # Select only incorrect reasoning steps
        df = df[df['label'] == 1]

    df['label'] = pd.Categorical(df['label'], categories=[1, 0])
    df['fallacy'] = pd.Categorical(df['fallacy'], categories=get_fallacy_list())
    df['category'] = pd.Categorical(df['category'], categories=['formal', 'informal'])
    df['subcategory'] = pd.Categorical(df['subcategory'])

    return df


def save_fallacy_df(df_fallacies: pd.DataFrame, filename: str):
    """
    Save the fallacy dataframe to a CSV file.
    """
    df_fallacies.to_csv(filename, index=False)


def get_fallacy_list() -> list[str]:
    with open('datasets/FALLACIES/fallacy_taxonomy.json') as f:
        taxonomy = json.load(f)

    return taxonomy['all']


def _remove_square_brackets(string: str):
    return string.replace("[", "").replace("]", "")


def add_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns for the fallacy category and subcategory to the dataframe.
    """
    df = df.copy()
    with open('datasets/FALLACIES/fallacy_taxonomy.json') as f:
        taxonomy = json.load(f)

    # If dataframe has no 'fallacy' column, add it temporarily from index, assuming index is the fallacy
    has_fallacy_col = 'fallacy' in df.columns
    if not has_fallacy_col:
        df['fallacy'] = df.index

    df['category'] = df.apply(lambda row: 'formal' if row['fallacy'] in taxonomy['formal'] else 'informal', axis=1)

    subcategory_map = {}
    for subcategory, fallacies in taxonomy.items():
        if subcategory in ['all', 'formal', 'informal']:
            continue
        for fallacy in fallacies:
            subcategory_map[fallacy] = subcategory

    df['subcategory'] = df.apply(lambda row: subcategory_map[row['fallacy']], axis=1)

    if not has_fallacy_col:
        df.drop(columns='fallacy', inplace=True)

    return df
