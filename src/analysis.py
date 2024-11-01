"""
This module contains functions for analyzing the fallacy experiments.
"""
import pandas as pd
import numpy as np
import re
from .llms import LLM
from .experiment import RESPONSE_ERROR
from .fallacies import get_fallacy_list
from statsmodels.stats.contingency_tables import mcnemar


def get_sanity_check(df_fallacies: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of missing responses and invalid predictions for each LLM.
    Invalid predictions result from missing responses, response errors, or failed extraction of Yes/No or fallacy type.
    """
    response_cols = [col for col in df_fallacies if col.endswith('_response')]
    response_lengths: pd.Series = df_fallacies[response_cols].apply(lambda x: x.str.len().mean().round(1))
    response_lengths.index = response_lengths.index.str.removesuffix('_response')

    missing_responses: pd.Series = df_fallacies[response_cols].isin(['', RESPONSE_ERROR]).sum()
    missing_responses.index = missing_responses.index.str.removesuffix('_response')

    prediction_cols = [col for col in df_fallacies if col.endswith('_pred')]
    invalid_predictions: pd.Series = df_fallacies[prediction_cols].isna().sum()
    invalid_predictions.index = invalid_predictions.index.str.removesuffix('_pred')

    types = { 'response_length_mean': 'float64', 'missing_responses': 'int16', 'invalid_predictions': 'int16'}
    return pd.DataFrame([response_lengths, missing_responses, invalid_predictions],
                        index=['response_length_mean', 'missing_responses', 'invalid_predictions'], ).T.astype(types)


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

    return 0 if contains_yes else 1  # Yes means reasoning step is correct, therefore 0, not a fallacy


def add_classification_scores(df_fallacies: pd.DataFrame):
    fallacies = get_fallacy_list()
    for llm in LLM:
        response_column = f"{llm.key}_response"
        if response_column not in df_fallacies.columns:
            continue

        pred_column = f"{llm.key}_pred"
        score_column = f"{llm.key}_score"
        df_fallacies[pred_column] = df_fallacies.apply(
            lambda row: _get_classification_prediction(fallacies, row[response_column]), axis=1
        )
        df_fallacies[score_column] = (
                ~df_fallacies[pred_column].isna() & (df_fallacies['fallacy'] == df_fallacies[pred_column])
        ).astype('UInt8')


def _get_classification_prediction(fallacies: list[str], response: str) -> str | None:
    if response == '' or response == RESPONSE_ERROR:
        return None

    for fallacy in fallacies:
        if bool(re.search(fallacy, response, re.IGNORECASE)):
            return fallacy

    return None


def get_macro_accuracies(df_fallacies: pd.DataFrame):
    """
    Calculate accuracies for each LLM. The accuracies are macro-averages, giving equal weight to each
    subcategory and category, as in Hong et al. (2024).
    """
    group_columns = ['category', 'subcategory', 'fallacy']
    score_columns = df_fallacies.columns[df_fallacies.columns.str.endswith('_score')].tolist()

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
    """
    Returns a multi-index DataFrame with confusion metrics for each LLM and fallacy.
    """
    pred_cols = [col for col in df_fallacies.columns if col.endswith('_pred')]
    llms = {llm.key: llm for llm in LLM}
    rows = []
    for pred_col in pred_cols:
        llm_key = pred_col.replace('_pred', '')
        for fallacy in get_fallacy_list():
            df_fallacy = df_fallacies[df_fallacies['fallacy'] == fallacy]
            metrics = _get_confusion_metrics(df_fallacy, 'label', pred_col)

            metrics['llm'] = llm_key
            metrics['llm_group'] = llms[llm_key].group.value
            metrics['fallacy'] = fallacy
            metrics['category'] = df_fallacy['category'].iloc[0]
            metrics['subcategory'] = df_fallacy['subcategory'].iloc[0]

            rows.append(metrics)

    # Create multi-index DataFrame
    df_result = pd.DataFrame(rows)
    df_result = df_result.set_index(['llm', 'llm_group', 'category', 'subcategory', 'fallacy'])

    return df_result


# noinspection PyUnresolvedReferences
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


def mcnemar_test(fp: int, fn: int) -> float:
    """
    Test whether the false positives (FP) and false negatives (FN) are significantly different using McNemar's test.
    McNemar's test is the most appropriate choice because:
    - It's specifically designed for comparing paired nominal data in a confusion matrix
    - It's particularly suitable when you want to compare the off-diagonal elements of a 2x2 confusion matrix
    """
    # Create the contingency table for McNemar's test
    # Note: Only the off-diagonal elements (FP and FN) are used
    contingency_table = np.array([[0, fp],
                                  [fn, 0]])

    # Perform McNemar's test
    return mcnemar(contingency_table).pvalue
