"""
This module contains functions for analyzing the fallacy experiments.
"""
import numpy as np
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
    filtered_response = response.lower()
    contains_yes = bool(re.search(r'\byes\b', filtered_response))
    contains_no = bool(re.search(r'\bno\b', filtered_response))

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
    df_scores.columns = group_columns + [llm.label for llm in LLM if llm.value + '_score' in df_scores.columns]

    # Calculate macro-averages, giving equal weight to each category, subcategory, and fallacy
    df_type_accuracies = df_scores.groupby(['category', 'subcategory', 'fallacy']).mean()
    df_subcategory_accuracies = df_type_accuracies.groupby(['category', 'subcategory']).mean()
    df_category_accuracies = df_subcategory_accuracies.groupby(['category']).mean()
    df_global_accuracies = df_category_accuracies.mean().to_frame().T
    df_global_accuracies.index = ['Accuracy']

    # Round at the end
    precision = 1
    return (
        (df_type_accuracies * 100).round(precision).T,
        (df_subcategory_accuracies * 100).round(precision).T,
        (df_category_accuracies * 100).round(precision).T,
        (df_global_accuracies * 100).round(precision).T
    )
