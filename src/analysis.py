"""
This module contains functions for analyzing the fallacy experiments.
"""
import pandas as pd
import re
from .llms import LLM
from .experiment import RESPONSE_ERROR


def add_identification_scores(df_fallacies: pd.DataFrame):
    for llm in LLM:
        response_column = f"{llm.key}_response"
        if response_column not in df_fallacies.columns:
            continue

        pred_column = f"{llm.key}_pred"
        score_column = f"{llm.key}_score"
        df_fallacies[pred_column] = df_fallacies.apply(
            lambda row: _get_identification_prediction(row["label"], row[response_column]), axis=1
        ).astype('UInt8')
        df_fallacies[score_column] = (
                ~df_fallacies[pred_column].isna() & (df_fallacies['label'] == df_fallacies[pred_column])
        ).astype('UInt8')


def _get_identification_prediction(label: int, response: str) -> int | None:
    # The model don't seem to respond in lowercase yes or no. But in chain-of-thought prompt
    # responses, the reasoning sometimes contains the word "no", when the final answer is "Yes".
    contains_yes = bool(re.search(r'\bYes\b', response))
    contains_no = bool(re.search(r'\bNo\b', response))

    if contains_yes == contains_no:
        return None  # Contains neither "Yes" nor "No", or contains both (e.g. "Yes. No. No. Yes. Yes."

    return 0 if contains_yes else 1 # Yes means reasoning step is correct, therefore 0, not a fallacy


def add_classification_scores(df_fallacies: pd.DataFrame):
    for llm in LLM:
        response_column = f"{llm.key}_response"
        if response_column not in df_fallacies.columns:
            continue

        score_column = f"{llm.key}_score"
        df_fallacies[score_column] = df_fallacies.apply(
            lambda row: _get_classification_score(row["fallacy"], row[response_column]), axis=1
        ).astype('UInt8')


def _get_classification_score(fallacy: str, response: str):
    if response == '' or response == RESPONSE_ERROR:
        return pd.NA

    contains_fallacy = bool(re.search(fallacy, response, re.IGNORECASE))

    return 1 if contains_fallacy else 0


def get_macro_accuracies(df_fallacies: pd.DataFrame):
    """
    Calculate accuracies for each LLM. The accuracies are macro-averages, giving equal weight to each
    subcategory and category, as in Hong et al. (2024).
    """
    group_columns = ['category', 'subcategory', 'fallacy']
    score_columns =  df_fallacies.columns[df_fallacies.columns.str.endswith('_score')].tolist()

    # Rename LLM columns
    df_scores = df_fallacies[group_columns + score_columns]
    df_scores.columns = df_scores.columns.str.removesuffix('_score')

    # Calculate macro-averages, giving equal weight to each category and subcategory
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


def add_llm_info(df: pd.DataFrame, label=False, group=False, provider=False):
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


def get_identification_confusion_metrics(df_fallacies: pd.DataFrame) -> pd.DataFrame:
    # Get all model columns
    pred_cols = [col for col in df_fallacies.columns if col.endswith('_pred')]

    # Calculate metrics for each model
    df_confusion = pd.DataFrame(
        [_get_confusion_metrics(df_fallacies, 'label', pred_col) for pred_col in pred_cols],
        index=[col.replace('_pred', '') for col in pred_cols]
    )

    int_cols = ['TP', 'TN', 'FP', 'FN']
    df_confusion[int_cols] = df_confusion[int_cols].astype('UInt16')

    return df_confusion


def _get_confusion_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> pd.Series:
    """Calculate basic confusion matrix values"""

    tp = ((df[true_col] == 1) & (df[pred_col] == 1)).sum()
    tn = ((df[true_col] == 0) & (df[pred_col] == 0)).sum()
    fp = ((df[true_col] == 0) & (df[pred_col] == 1)).sum()
    fn = ((df[true_col] == 1) & (df[pred_col] == 0)).sum()

    return pd.Series({
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
    })

def add_confusion_scores(df_confusion: pd.DataFrame) -> pd.DataFrame:
    df = df_confusion.copy()
    for index, row in df.iterrows():
        accuracy, precision, recall, f1 = get_confusion_scores(row['TP'], row['TN'], row['FP'], row['FN'])
        df.loc[index, 'Accuracy'] = accuracy
        df.loc[index, 'Precision'] = precision
        df.loc[index, 'Recall'] = recall
        df.loc[index, 'F1'] = f1

    return df


def get_confusion_scores(tp: int, tn: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1