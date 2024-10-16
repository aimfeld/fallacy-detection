import numpy as np
import pandas as pd
from .llms import LLMLabel


def score_fallacy_identification(df_fallacies: pd.DataFrame):
    for llm_label in LLMLabel:
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