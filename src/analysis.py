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
    if "no" in filtered_response:
        response_label = 1
    elif "yes" in filtered_response:
        response_label = 0
    else:
        return pd.NA

    return 1 if response_label == label else 0


def get_fallacy_identification_accuracies(df_fallacies: pd.DataFrame) -> pd.DataFrame:
    accuracies = {}
    for llm in LLM:
        score_column = llm.value + '_score'
        if score_column in df_fallacies.columns:
            accuracies[llm.label] = np.round(df_fallacies[score_column].mean() * 100, 1)

    return pd.DataFrame(accuracies.items(), columns=['LLM', 'Accuracy'])


def get_fallacy_identification_grouped_accuracies(df_fallacies: pd.DataFrame, groupby: list[str]):
    accuracies = {}
    for llm in LLM:
        score_column = llm.value + '_score'
        if score_column in df_fallacies.columns:
            accuracies[llm.label] = np.round(df_fallacies.groupby(groupby)[score_column].mean() * 100, 1)

    return pd.DataFrame.from_dict(accuracies, orient='index')


