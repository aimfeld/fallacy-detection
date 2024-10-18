"""
This module contains functions for analyzing the fallacy experiments.
"""
import numpy as np
import pandas as pd
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
    if len(filtered_response) > 5:
        # Answers like "I don't know" or "Yes. No. No. Yes. Yes." are invalid
        return pd.NA
    elif "no" in filtered_response:
        response_label = 1
    elif "yes" in filtered_response:
        response_label = 0
    else:
        return pd.NA

    return 1 if response_label == label else 0


def get_fallacy_identification_accuracies(df_fallacies: pd.DataFrame, groupby: list[str] = None):
    accuracies = {}
    for llm in LLM:
        score_column = llm.value + '_score'
        if score_column in df_fallacies.columns:
            mean_score = df_fallacies[score_column].mean() if groupby is None else df_fallacies.groupby(groupby)[score_column].mean()
            accuracies[llm.label] = np.round(mean_score * 100, 1)

    columns = ['Accuracy'] if groupby is None else None
    return pd.DataFrame.from_dict(accuracies, orient='index', columns=columns)


