"""
This module functions for dealing with the FALLACIES dataset by Hong et al. (2024).
"""
import pandas as pd
import json


def create_fallacy_df() -> pd.DataFrame:
    df = pd.read_json('datasets/FALLACIES/step_fallacy.test.jsonl', lines=True)
    df = add_taxonomy(df)

    df['step'] = df['step'].apply(_remove_square_brackets)

    return df


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
