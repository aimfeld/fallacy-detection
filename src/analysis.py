"""
This module contains functions for analyzing the fallacy experiments.
"""
import pandas as pd
import numpy as np
import re
from typing import Union
from .llms import LLM, LLMGroup, LLMProvider
from .constants import RESPONSE_ERROR
from .fallacies import get_fallacy_list, add_taxonomy
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

    types = {'response_length_mean': 'float64', 'missing_responses': 'int16', 'invalid_predictions': 'int16'}
    return pd.DataFrame([response_lengths, missing_responses, invalid_predictions],
                        index=['response_length_mean', 'missing_responses', 'invalid_predictions'], ).T.astype(types)


def add_identification_scores(df_fallacies: pd.DataFrame, punish_missing: bool = True, flip: bool = False):
    """
    Add identification predictions and scores (0 or 1) to the DataFrame.

    Args:
        df_fallacies: DataFrame with fallacy predictions
        punish_missing: If True, missing/invalid predictions are penalized with a score of 0.
                        If False, missing predictions are ignored and do not affect the accuracy.
        flip: If True, flip the identification prediction (0=No, 1=Yes) to account for prompt variation.
    """
    response_cols = [col for col in df_fallacies.columns if col.endswith('_response')]
    for response_col in response_cols:
        llm_key = response_col.removesuffix('_response')
        pred_column = f"{llm_key}_pred"
        score_column = f"{llm_key}_score"
        df_fallacies[pred_column] = df_fallacies.apply(
            lambda row: _get_identification_prediction(row["label"], row[response_col]), axis=1
        )
        if flip:
            df_fallacies[pred_column] = df_fallacies[pred_column].apply(lambda x: 1 - x)

        df_fallacies[pred_column] = pd.Categorical(df_fallacies[pred_column], categories=[1, 0])

        df_fallacies[score_column] = (df_fallacies['label'] == df_fallacies[pred_column]).astype('UInt8')

        if not punish_missing:
            # Set score to NA if prediction is missing so that it doesn't affect the accuracy
            df_fallacies.loc[df_fallacies[pred_column].isna(), score_column] = pd.NA


def _get_identification_prediction(label: int, response: str) -> Union[int, pd.NA]:
    """
    Return the identification prediction (0=Yes, 1=No) based on the response.
    """

    # The model don't seem to respond in lowercase yes or no. But in chain-of-thought prompt
    # responses, the reasoning sometimes contains the word "no", when the final answer is "Yes".
    contains_yes = bool(re.search(r'\bYes\b', response))
    contains_no = bool(re.search(r'\bNo\b', response))

    if contains_yes == contains_no:
        return pd.NA  # Contains neither "Yes" nor "No", or contains both (e.g. "Yes. No. No. Yes. Yes."

    return 0 if contains_yes else 1  # Yes means reasoning step is correct, therefore 0, not a fallacy


def add_classification_scores(df_fallacies: pd.DataFrame, punish_missing: bool = True):
    """
    Add classification predictions (fallacy type) and scores (0 or 1) to the DataFrame.
    """
    fallacies = get_fallacy_list()
    response_cols = [col for col in df_fallacies.columns if col.endswith('_response')]
    for response_col in response_cols:
        llm_key = response_col.removesuffix('_response')
        pred_column = f"{llm_key}_pred"
        score_column = f"{llm_key}_score"
        df_fallacies[pred_column] = df_fallacies.apply(
            lambda row: _get_classification_prediction(fallacies, row[response_col]), axis=1
        )
        df_fallacies[pred_column] = pd.Categorical(df_fallacies[pred_column], categories=fallacies)
        df_fallacies[score_column] = (df_fallacies['fallacy'] == df_fallacies[pred_column]).astype('UInt8')

        if not punish_missing:
            # Set score to NA if prediction is missing so that it doesn't affect the accuracy
            df_fallacies.loc[df_fallacies[pred_column].isna(), score_column] = pd.NA


def _get_classification_prediction(fallacies: list[str], response: str) -> Union[str, pd.NA]:
    """Return the classification prediction (fallacy type) based on the response."""
    if response == '' or response == RESPONSE_ERROR:
        return pd.NA

    for fallacy in fallacies:
        # Check if the response contains the fallacy name, ignoring case
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
    df_type_accuracies = df_scores.groupby(['category', 'subcategory', 'fallacy'], observed=True).mean() * 100
    df_subcategory_accuracies = df_type_accuracies.groupby(['category', 'subcategory'], observed=True).mean()
    df_category_accuracies = df_subcategory_accuracies.groupby(['category'], observed=True).mean()
    df_overall_accuracies = df_category_accuracies.mean().to_frame().T
    df_overall_accuracies.index = ['accuracy']

    return (
        df_type_accuracies.T,
        df_subcategory_accuracies.T,
        df_category_accuracies.T,
        df_overall_accuracies.T
    )


def add_llm_info(df: pd.DataFrame, label=False, group=False, provider=False):
    """
    Add LLM info based on the dataframe index which contains LLM keys.
    """
    df_info = df.copy()
    add_all = not label and not provider and not group
    llms = {llm.key: llm for llm in LLM}

    # If dataframe has no 'llm' column, add it temporarily from index, assuming index is llm key
    has_llm_col = 'llm' in df_info.columns
    if not has_llm_col:
        df_info['llm'] = df_info.index

    if label or add_all:
        df_info['llm_label'] = df_info.apply(lambda row: llms[row['llm']].label, axis=1)
    if group or add_all:
        categories = [group.value for group in LLMGroup]
        df_info['llm_group'] = df_info.apply(lambda row: llms[row['llm']].group.value, axis=1)
        df_info['llm_group'] = pd.Categorical(df_info['llm_group'], categories=categories, ordered=True)
    if provider or add_all:
        categories = [provider.value for provider in LLMProvider]
        df_info['llm_provider'] = df_info.apply(lambda row: llms[row['llm']].provider.value, axis=1)
        df_info['llm_provider'] = pd.Categorical(df_info['llm_provider'], categories=categories, ordered=True)

    if not has_llm_col:
        df_info.drop(columns='llm', inplace=True)

    return df_info


def get_confusion_matrices(df_fallacies: pd.DataFrame, actual_col: str) -> pd.DataFrame:
    """
    Returns a multi-index DataFrame with n*n confusion matrices for each LLM.

    Args:
        df_fallacies: DataFrame with fallacy predictions
        actual_col: Column name for the actual labels ('label' for identification, 'fallacy' for classification)

    Returns:
        Multi-index DataFrame with n*n confusion matrices for each LLM. Columns are actual labels, rows are predicted
        labels.
    """
    assert actual_col in ['label', 'fallacy'], "actual_col must be 'label' or 'fallacy'"

    pred_cols = [col for col in df_fallacies.columns if col.endswith('_pred')]
    confusion_matrices: list[pd.DataFrame] = []

    for pred_col in pred_cols:
        # Fallacy identification (Yes/No): one confusion matrix per LLM and fallacy
        if actual_col == 'label':
            for fallacy in get_fallacy_list():
                df_confusion_matrix = _get_crosstab(df_fallacies[df_fallacies['fallacy'] == fallacy], actual_col,
                                                    pred_col)
                df_confusion_matrix['fallacy'] = fallacy

                confusion_matrices.append(df_confusion_matrix)

        # Fallacy classification (fallacy type): one confusion matrix per LLM
        elif actual_col == 'fallacy':
            df_confusion_matrix = _get_crosstab(df_fallacies, actual_col, pred_col)

            confusion_matrices.append(df_confusion_matrix)

    df_confusion_matrices = pd.concat(confusion_matrices)
    df_confusion_matrices.index.name = actual_col
    df_confusion_matrices = df_confusion_matrices.reset_index()

    df_confusion_matrices = add_llm_info(df_confusion_matrices, group=True)
    df_confusion_matrices = add_taxonomy(df_confusion_matrices)

    # Create multi-index DataFrame
    multi_index = ['llm', 'llm_group', 'category', 'subcategory']
    multi_index += ['fallacy'] if actual_col == 'label' else []
    multi_index += [actual_col]
    df_confusion_matrices = df_confusion_matrices.set_index(multi_index)

    return df_confusion_matrices


def _get_crosstab(df_fallacies: pd.DataFrame, actual_col: str, pred_col: str) -> pd.DataFrame:
    """
    Returns a confusion matrix (cross-tabulation) for the given actual and predicted columns.
    """
    df_crosstab = pd.crosstab(df_fallacies[pred_col], df_fallacies[actual_col],
                              rownames=['predicted'], colnames=['actual'], dropna=False)

    # Drop the row containing invalid predictions to get an n*n confusion matrix
    df_crosstab = df_crosstab[df_crosstab.index.notna()]

    df_crosstab['llm'] = pred_col.replace('_pred', '')

    return df_crosstab


def get_confusion_metrics(df_confusion_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate confusion matrix metrics for each label
    """
    # Get diagonal elements (true positives)
    true_positives = np.diag(df_confusion_matrix)

    # Calculate total samples
    total_samples = df_confusion_matrix.sum().sum()

    # False positives and false negatives
    false_positives = df_confusion_matrix.sum(axis=1) - true_positives
    false_negatives = df_confusion_matrix.sum(axis=0) - true_positives

    # True negatives (sum of all correct predictions for other classes)
    true_negatives = total_samples - (true_positives + false_positives + false_negatives)

    # Calculate metrics
    accuracy = (true_positives + true_negatives) / total_samples
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    p_mcnemar = [mcnemar_test(fp, fn) for fp, fn in zip(false_positives, false_negatives)]

    df_results = pd.DataFrame({
        'tp': true_positives,
        'tn': true_negatives,
        'fp': false_positives,
        'fn': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'p_mcnemar': p_mcnemar
    })

    # Division by zero results in NaN values which are replaced by 0
    df_results.fillna(0, inplace=True)

    return df_results


def get_identification_confusion_metrics(df_confusion_matrices: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """
    Calculate confusion metrics for each index_name level.
    """
    df_agg = df_confusion_matrices.groupby([index_name, 'label'], observed=True).sum()

    # We care about the metrics for label 1 (fallacy) only
    confusion_metrics = {llm: get_confusion_metrics(cm.loc[llm]).loc[1] for llm, cm in
                         df_agg.groupby(index_name, observed=True)}
    df_confusion_metrics = pd.DataFrame(confusion_metrics).T

    for col in ['tp', 'tn', 'fp', 'fn']:
        df_confusion_metrics[col] = df_confusion_metrics[col].astype('UInt16')

    return df_confusion_metrics


def get_misclassifications(df_confusion_matrix: pd.DataFrame, n_misclassifications: int = 3) -> pd.DataFrame:
    """
    Return a DataFrame with accuracy and top N misclassifications per label.

    Parameters:
        df_confusion_matrix: Confusion matrix where rows are predicted labels and columns are actual labels
        n_misclassifications: Number of top misclassifications to include per label

    Returns:
        DataFrame with accuracy and top N misclassifications per label.
    """

    # Generate column names dynamically based on n_misclassifications
    result_columns = []
    for i in range(0, n_misclassifications):
        result_columns.extend([f'misclassification_{i + 1}', f'count_{i + 1}'])

    # Initialize the result DataFrame
    df_result = pd.DataFrame(index=df_confusion_matrix.columns, columns=result_columns)

    # Process each true label (column)
    for true_label in df_confusion_matrix.columns:
        # Get the column for this true label
        col = df_confusion_matrix[true_label]

        # Get misclassifications (exclude the correct prediction)
        misclassifications = col[col.index != true_label]

        # Sort misclassifications in descending order and get top n
        top_n_misclassifications = misclassifications.sort_values(ascending=False).head(n_misclassifications)

        # Fill misclassification information
        for i in range(0, n_misclassifications):
            count = top_n_misclassifications.iloc[i]
            label = top_n_misclassifications.index[i] if count > 0 else ''

            df_result.at[true_label, f'misclassification_{i + 1}'] = label
            df_result.at[true_label, f'count_{i + 1}'] = count

    df_result.sort_values(by='count_1', ascending=False, inplace=True)

    return df_result


def mcnemar_test(false_positives: int, false_negatives: int) -> float:
    """
    Test whether the false positives (FP) and false negatives (FN) are significantly different using McNemar's test.
    McNemar's test is the most appropriate choice because:
    - It's specifically designed for comparing paired nominal data in a confusion matrix
    - It's particularly suitable when you want to compare the off-diagonal elements of a 2x2 confusion matrix
    """
    # Create the contingency table for McNemar's test
    contingency_table = np.array([[0, false_positives],
                                  [false_negatives, 0]])

    return mcnemar(contingency_table).pvalue
