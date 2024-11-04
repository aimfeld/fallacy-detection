"""
This module contains functions for analyzing the fallacy experiments.
"""
import pandas as pd
import numpy as np
import re
from typing import Union
from .llms import LLM
from .experiment import RESPONSE_ERROR
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


def add_identification_scores(df_fallacies: pd.DataFrame, punish_missing: bool = True):
    response_cols = [col for col in df_fallacies.columns if col.endswith('_response')]
    for response_col in response_cols:
        llm_key = response_col.removesuffix('_response')
        pred_column = f"{llm_key}_pred"
        score_column = f"{llm_key}_score"
        df_fallacies[pred_column] = df_fallacies.apply(
            lambda row: _get_identification_prediction(row["label"], row[response_col]), axis=1
        )
        df_fallacies[pred_column] = pd.Categorical(df_fallacies[pred_column], categories=[1, 0])
        df_fallacies[score_column] = (df_fallacies['label'] == df_fallacies[pred_column]).astype('UInt8')

        if not punish_missing:
            # Set score to NA if prediction is missing so that it doesn't affect the accuracy
            df_fallacies.loc[df_fallacies[pred_column].isna(), score_column] = pd.NA


def _get_identification_prediction(label: int, response: str) -> Union[int, pd.NA]:
    # The model don't seem to respond in lowercase yes or no. But in chain-of-thought prompt
    # responses, the reasoning sometimes contains the word "no", when the final answer is "Yes".
    contains_yes = bool(re.search(r'\bYes\b', response))
    contains_no = bool(re.search(r'\bNo\b', response))

    if contains_yes == contains_no:
        return pd.NA  # Contains neither "Yes" nor "No", or contains both (e.g. "Yes. No. No. Yes. Yes."

    return 0 if contains_yes else 1  # Yes means reasoning step is correct, therefore 0, not a fallacy


def add_classification_scores(df_fallacies: pd.DataFrame, punish_missing: bool = True):
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
    if response == '' or response == RESPONSE_ERROR:
        return pd.NA

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
        df_info['llm_group'] = df_info.apply(lambda row: llms[row['llm']].group.value, axis=1)
        df_info['llm_group'] = pd.Categorical(df_info['llm_group'])
    if provider or add_all:
        df_info['llm_provider'] = df_info.apply(lambda row: llms[row['llm']].provider.value, axis=1)
        df_info['llm_provider'] = pd.Categorical(df_info['llm_provider'])

    if not has_llm_col:
        df_info.drop(columns='llm', inplace=True)

    return df_info


def get_confusion_matrices(df_fallacies: pd.DataFrame, actual_col: str) -> pd.DataFrame:
    """Returns a multi-index DataFrame with n*n confusion matrices for each LLM.

    Args:
        df_fallacies: DataFrame with fallacy predictions
        actual_col: Column name for the actual labels ('label' for identification, 'fallacy' for classification)

    Returns:
        Multi-index DataFrame with n*n confusion matrices for each LLM. Columns are actual labels, rows are predicted
        labels.
    """
    pred_cols = [col for col in df_fallacies.columns if col.endswith('_pred')]
    confusion_matrices: list[pd.DataFrame] = []

    for pred_col in pred_cols:
        # Fallacy identification (Yes/No): one confusion matrix per LLM and fallacy
        if actual_col == 'label':
            for fallacy in get_fallacy_list():
                df_confusion_matrix = get_crosstab(df_fallacies[df_fallacies['fallacy'] == fallacy], actual_col, pred_col)
                df_confusion_matrix['fallacy'] = fallacy

                confusion_matrices.append(df_confusion_matrix)

        # Fallacy classification (fallacy type): one confusion matrix per LLM
        elif actual_col == 'fallacy':
            df_confusion_matrix = get_crosstab(df_fallacies, actual_col, pred_col)

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


def get_crosstab(df_fallacies: pd.DataFrame, actual_col: str, pred_col: str) -> pd.DataFrame:
    """Returns a confusion matrix (cross-tabulation) for the given actual and predicted columns."""
    df_crosstab = pd.crosstab(df_fallacies[pred_col], df_fallacies[actual_col],
                              rownames=['predicted'], colnames=['actual'], dropna=False)

    # Drop the row containing invalid predictions to get an n*n confusion matrix
    df_crosstab = df_crosstab[df_crosstab.index.notna()]

    df_crosstab['llm'] = pred_col.replace('_pred', '')

    return df_crosstab


def get_identification_confusion_metrics(df_confusion_matrices: pd.DataFrame, groupby: list[str]) -> pd.DataFrame:
    df_metrics = df_confusion_matrices.groupby(groupby, observed=True).sum().unstack()
    if isinstance(df_metrics, pd.Series):
        df_metrics = df_metrics.to_frame().T
    df_metrics.columns = df_metrics.columns.to_flat_index()
    df_metrics.columns = ['TP', 'FN', 'FP', 'TN']

    for index, row in df_metrics.iterrows():
        accuracy, precision, recall, f1 = get_confusion_scores(row['TP'], row['TN'], row['FP'], row['FN'])
        mcnemar_p = mcnemar_test(row['FP'], row['FN'])

        df_metrics.loc[index, 'Accuracy'] = accuracy
        df_metrics.loc[index, 'Precision'] = precision
        df_metrics.loc[index, 'Recall'] = recall
        df_metrics.loc[index, 'F1'] = f1
        df_metrics.loc[index, 'McNemar-P'] = mcnemar_p

    return df_metrics


def get_confusion_scores(tp: int, tn: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    """Calculate accuracy, precision, recall, and F1-score from confusion matrix counts."""
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1


def get_misclassifications(df_confusion_matrix: pd.DataFrame, n_misclassifications: int = 3) -> pd.DataFrame:
    """Return a DataFrame with accuracy and top N misclassifications per label.

    Parameters:
        df_confusion_matrix: Confusion matrix where rows are predicted labels and columns are actual labels
        n_misclassifications: Number of top misclassifications to include per label

    Returns:
        DataFrame with accuracy and top N misclassifications per label.
    """

    # Generate column names dynamically based on n_misclassifications
    result_columns = ['accuracy']
    for i in range(1, n_misclassifications + 1):
        result_columns.extend([f'misclassification_{i}', f'count_{i}'])

    # Initialize the result DataFrame
    df_result = pd.DataFrame(index=df_confusion_matrix.columns, columns=result_columns)

    # Process each true label (column)
    for true_label in df_confusion_matrix.columns:
        # Get the column for this true label
        col = df_confusion_matrix[true_label]

        # Calculate accuracy
        total = col.sum()
        correct = col[true_label]
        accuracy = correct / total if total > 0 else 0

        # Get misclassifications (exclude the correct prediction)
        misclassifications = col[col.index != true_label]

        # Sort misclassifications in descending order and get top n
        top_n_misclassifications = misclassifications[misclassifications > 0].sort_values(ascending=False).head(n_misclassifications)

        # Fill the result row
        df_result.at[true_label, 'accuracy'] = accuracy

        # Fill misclassification information
        for i in range(1, n_misclassifications + 1):
            if i <= len(top_n_misclassifications):
                label = top_n_misclassifications.index[i - 1]
                count = top_n_misclassifications.iloc[i - 1]
            else:
                label = ''
                count = pd.NA

            df_result.at[true_label, f'misclassification_{i}'] = label
            df_result.at[true_label, f'count_{i}'] = count

    df_result.sort_values(by=['accuracy', 'count_1'], ascending=[True, False], inplace=True)

    # Convert accuracy to float and counts to int
    df_result['accuracy'] = df_result['accuracy'].astype(float)
    count_columns = [f'count_{i}' for i in range(1, n_misclassifications + 1)]
    df_result[count_columns] = df_result[count_columns].astype('UInt16')

    return df_result


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
