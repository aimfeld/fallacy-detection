"""
This module functions for dealing with the FALLACIES dataset by Hong et al. (2024).
"""
import pandas as pd
import json


def create_fallacy_df() -> pd.DataFrame:
    df = pd.read_json('fallacies/step_fallacy.test.jsonl', lines=True)
    df = add_taxonomy(df)

    df['step'] = df['step'].apply(_remove_square_brackets)

    return df


def get_fallacy_list() -> list[str]:
    with open('fallacies/fallacy_taxonomy.json') as f:
        taxonomy = json.load(f)

    return taxonomy['all']


def _remove_square_brackets(string: str):
    return string.replace("[", "").replace("]", "")


def add_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns for the fallacy category and subcategory to the dataframe.
    """
    df = df.copy()
    with open('fallacies/fallacy_taxonomy.json') as f:
        taxonomy = json.load(f)

    df['category'] = df.apply(lambda row: 'formal' if row['fallacy'] in taxonomy['formal'] else 'informal', axis=1)

    subcategory_map = {}
    for subcategory, fallacies in taxonomy.items():
        if subcategory in ['all', 'formal', 'informal']:
            continue
        for fallacy in fallacies:
            subcategory_map[fallacy] = subcategory

    df['subcategory'] = df.apply(lambda row: subcategory_map[row['fallacy']], axis=1)

    return df
