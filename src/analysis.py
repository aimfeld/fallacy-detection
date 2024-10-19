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


def get_fallacy_identification_accuracies(df_fallacies: pd.DataFrame, groupby: list[str] = None):
    accuracies = {}
    for llm in LLM:
        score_column = llm.value + '_score'
        if score_column in df_fallacies.columns:
            # Invalid responses (NA) are not included in the mean calculation
            # There are very few invalid responses, e.g. due to politically incorrect language in the prompt.
            # In rare cases, the models refuse to answer with "Yes" or "No", e.g. when the prompt is a question.
            # The models are not penalized for these invalid responses.
            mean_score = df_fallacies[score_column].mean() if groupby is None \
                else df_fallacies.groupby(groupby)[score_column].mean()
            accuracies[llm.label] = np.round(mean_score * 100, 1)

    columns = ['Accuracy'] if groupby is None else None
    return pd.DataFrame.from_dict(accuracies, orient='index', columns=columns)
