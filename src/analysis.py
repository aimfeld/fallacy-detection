"""
This module contains functions for analyzing the fallacy experiments.
"""
import pandas as pd
import re
from .llms import LLM
from .experiment import RESPONSE_ERROR


def score_fallacy_identification(df_fallacies: pd.DataFrame):
    for llm in LLM:
        response_column = f"{llm.key}_response"
        if response_column not in df_fallacies.columns:
            continue

        score_column = f"{llm.key}_score"
        df_fallacies[score_column] = df_fallacies.apply(
            lambda row: _get_fallacy_identification_score(row["label"], row[response_column]), axis=1
        ).astype('UInt8')


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


def score_fallacy_classification(df_fallacies: pd.DataFrame):
    for llm in LLM:
        response_column = f"{llm.key}_response"
        if response_column not in df_fallacies.columns:
            continue

        score_column = f"{llm.key}_score"
        df_fallacies[score_column] = df_fallacies.apply(
            lambda row: _get_fallacy_classification_score(row["fallacy"], row[response_column]), axis=1
        ).astype('Int64')


def _get_fallacy_classification_score(fallacy: str, response: str):
    if response == '' or response == RESPONSE_ERROR:
        return pd.NA

    contains_fallacy = bool(re.search(fallacy, response, re.IGNORECASE))

    return 1 if contains_fallacy else 0


def get_accuracies(df_fallacies: pd.DataFrame):
    group_columns = ['category', 'subcategory', 'fallacy']
    score_columns = group_columns + [llm.key + '_score' for llm in LLM if
                                     llm.key + '_score' in df_fallacies.columns]

    # Rename LLM columns
    df_scores = df_fallacies[score_columns]
    df_scores.columns = group_columns + [llm.key for llm in LLM if llm.key + '_score' in df_scores.columns]

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


def add_llm_info(df: pd.DataFrame, label = False, group = False, provider = False):
    """
    Add LLM info based on the dataframe index which contains LLM keys.
    """
    df_info = df.copy()
    add_all = not label and not provider and not group
    llms = {llm.key: llm for llm in LLM}

    if label or add_all:
        df_info['llm_label'] = df_info.apply(lambda row: llms[row.name].label, axis=1)
    if group or add_all:
        df_info['llm_group'] = df_info.apply(lambda row: llms[row.name].group.value, axis=1)
    if provider or add_all:
        df_info['llm_provider'] = df_info.apply(lambda row: llms[row.name].provider.value, axis=1)


    return df_info