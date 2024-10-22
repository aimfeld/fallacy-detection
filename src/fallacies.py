"""
This module provides a function to load the fallacy dataset and add the taxonomy to it.
"""
import pandas as pd
import json


def create_fallacy_df() -> pd.DataFrame:
    df = pd.read_json('fallacies/step_fallacy.test.jsonl', lines=True)
    _add_taxonomy(df)

    df['step'] = df['step'].apply(_remove_square_brackets)

    return df

def get_fallacy_list() -> list[str]:
    with open('fallacies/fallacy_taxonomy.json') as f:
        taxonomy = json.load(f)

    return taxonomy['all']

def _remove_square_brackets(string):
    return string.replace("[", "").replace("]", "")


def _add_taxonomy(df_fallacies):
    with open('fallacies/fallacy_taxonomy.json') as f:
        taxonomy = json.load(f)

    df_fallacies['category'] = df_fallacies.apply(lambda row: 'formal' if row['fallacy'] in taxonomy['formal'] else 'informal', axis=1)

    subcategory_map = {}
    for subcategory, fallacies in taxonomy.items():
        if subcategory in ['all', 'formal', 'informal']:
            continue
        for fallacy in fallacies:
            subcategory_map[fallacy] = subcategory

    df_fallacies['subcategory'] = df_fallacies.apply(lambda row: subcategory_map[row['fallacy']], axis=1)
