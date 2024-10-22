"""
This module contains functions for analyzing the fallacy experiments.
"""
import pandas as pd
import re
from .llms import LLM


def score_fallacy_identification(df_fallacies: pd.DataFrame):
    for llm_label in LLM:
        response_column = f"{llm_label.value}_response"
        if response_column not in df_fallacies.columns:
            continue

        score_column = f"{llm_label.value}_score"
        df_fallacies[score_column] = df_fallacies.apply(
            lambda row: _get_fallacy_identification_score(row["label"], row[response_column]), axis=1
        ).astype('Int64')


def _get_fallacy_identification_score(label: int, response: str):
    # The model don't seem to respond in lowercase yes or no. But in chain-of-thought prompt
    # responses, the reasoning sometimes contains the word "no", when the final answer is "Yes".
    contains_yes = bool(re.search(r'\bYes\b', response))
    contains_no = bool(re.search(r'\bNo\b', response))

    if contains_yes and not contains_no:
        response_label = 0  # Reasoning step is evaluated as valid
    elif contains_no and not contains_yes:
        response_label = 1  # Reasoning step is evaluated as invalid
    else:
        return pd.NA  # Response does not contain a valid answer

    return 1 if response_label == label else 0


def get_accuracies(df_fallacies: pd.DataFrame):
    group_columns = ['category', 'subcategory', 'fallacy']
    score_columns = group_columns + [llm.value + '_score' for llm in LLM if
                                     llm.value + '_score' in df_fallacies.columns]

    # Rename LLM columns
    df_scores = df_fallacies[score_columns]
    df_scores.columns = group_columns + [llm.value for llm in LLM if llm.value + '_score' in df_scores.columns]

    # Calculate macro-averages, giving equal weight to each category, subcategory, and fallacy
    df_type_accuracies = df_scores.groupby(['category', 'subcategory', 'fallacy']).mean() * 100
    df_subcategory_accuracies = df_type_accuracies.groupby(['category', 'subcategory']).mean()
    df_category_accuracies = df_subcategory_accuracies.groupby(['category']).mean()
    df_global_accuracies = df_category_accuracies.mean().to_frame().T
    df_global_accuracies.index = ['accuracy']

    return (
        df_type_accuracies.T,
        df_subcategory_accuracies.T,
        df_category_accuracies.T,
        df_global_accuracies.T
    )

# Add LLM info based on the dataframe index which contains LLM keys
def add_llm_info(df: pd.DataFrame, label = False, group = False, provider = False):
    df_info = df.copy()
    add_all = not label and not provider and not group

    if label or add_all:
        df_info['llm_label'] = df_info.apply(lambda row: LLM(row.name).label, axis=1)
    if group or add_all:
        df_info['llm_group'] = df_info.apply(lambda row: LLM(row.name).group, axis=1)
    if provider or add_all:
        df_info['llm_provider'] = df_info.apply(lambda row: LLM(row.name).provider, axis=1)


    return df_info