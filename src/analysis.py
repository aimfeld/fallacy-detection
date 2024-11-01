"""
This module contains functions for analyzing the fallacy experiments.
"""
import pandas as pd
import numpy as np
import re
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
        df_fallacies[pred_column] = pd.Categorical(df_fallacies[pred_column], categories=fallacies)
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
    df_type_accuracies = df_scores.groupby(['category', 'subcategory', 'fallacy'], observed=True).mean() * 100
    df_subcategory_accuracies = df_type_accuracies.groupby(['category', 'subcategory'], observed=True).mean()
    df_category_accuracies = df_subcategory_accuracies.groupby(['category'], observed=True).mean()
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
        df_info['llm_group'] = pd.Categorical(df_info['llm_group'])
    if provider or add_all:
        df_info['llm_provider'] = df_info.apply(lambda row: llms[row.name].provider.value, axis=1)
        df_info['llm_provider'] = pd.Categorical(df_info['llm_provider'])

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

def get_confusion_matrix(df_fallacies: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a multi-index DataFrame with confusion matrices for each LLM and fallacy.
    Columns are actual fallacies, rows are predicted fallacies.
    """
    pred_cols = [col for col in df_fallacies.columns if col.endswith('_pred')]
    llms = {llm.key: llm for llm in LLM}

    df_confusion_list = []
    for pred_col in pred_cols:
        llm_key = pred_col.replace('_pred', '')

        # Create confusion matrix for current LLM
        df_confusion = pd.crosstab(df_fallacies['gpt_4o_pred'], df_fallacies['fallacy'],
                                   rownames=['predicted'], colnames=['actual'], dropna=False)

        # Drop invalid predictions to get an n*n confusion matrix
        df_confusion = df_confusion[df_confusion.index.notna()]

        df_confusion['llm'] = llm_key
        df_confusion['llm_group'] = llms[llm_key].group.value

        df_confusion_list.append(df_confusion)

    # Create multi-index DataFrame
    df_result = pd.concat(df_confusion_list)
    df_result.index.name = 'fallacy'
    df_result = df_result.reset_index()
    add_taxonomy(df_result)
    df_result = df_result.set_index(['llm', 'llm_group', 'category', 'subcategory', 'fallacy'])

    return df_result

def sort_confusion_matrix(df_conf_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the confusion matrix by the sum of off-diagonal elements.
    """
    # Get off-diagonal sums using numpy
    off_diag_sums = df_conf_matrix.values.sum(axis=1) - np.diagonal(df_conf_matrix)
    sorted_idx = pd.Series(off_diag_sums, index=df_conf_matrix.index).sort_values(ascending=True).index
    return df_conf_matrix.loc[sorted_idx, sorted_idx]


def get_mispredictions(df_conf_matrix: pd.DataFrame, n_mispredictions: int = 3) -> pd.DataFrame:
    """
    Analyze a confusion matrix and return a DataFrame with accuracy and top N mispredictions per label.

    Parameters:
    df_conf_matrix (pd.DataFrame): Confusion matrix where rows are predictions and columns are true labels
    n_mispredictions (int): Number of top mispredictions to include per label

    Returns:
    pd.DataFrame: Analysis results with accuracy and top N mispredictions per label
    """

    # Generate column names dynamically based on n_mispredictions
    result_columns = ['accuracy']
    for i in range(1, n_mispredictions + 1):
        result_columns.extend([f'misprediction_{i}', f'count_{i}'])

    # Initialize the result DataFrame
    df_result = pd.DataFrame(index=df_conf_matrix.columns, columns=result_columns)

    # Process each true label (column)
    for true_label in df_conf_matrix.columns:
        # Get the column for this true label
        col = df_conf_matrix[true_label]

        # Calculate accuracy
        total = col.sum()
        correct = col[true_label]
        accuracy = correct / total if total > 0 else 0

        # Get mispredictions (exclude the correct prediction)
        mispredictions = col[col.index != true_label]

        # Sort mispredictions in descending order and get top n
        top_n_mispredictions = mispredictions[mispredictions > 0].sort_values(ascending=False).head(n_mispredictions)

        # Fill the result row
        df_result.at[true_label, 'accuracy'] = accuracy

        # Fill misprediction information
        for i in range(1, n_mispredictions + 1):
            if i <= len(top_n_mispredictions):
                label = top_n_mispredictions.index[i - 1]
                count = top_n_mispredictions.iloc[i - 1]
            else:
                label = ''
                count = pd.NA

            df_result.at[true_label, f'misprediction_{i}'] = label
            df_result.at[true_label, f'count_{i}'] = count

    df_result.sort_values(by=['accuracy', 'count_1'], ascending=[True, False], inplace=True)

    # Convert accuracy to float and counts to int
    df_result['accuracy'] = df_result['accuracy'].astype(float)
    count_columns = [f'count_{i}' for i in range(1, n_mispredictions + 1)]
    df_result[count_columns] = df_result[count_columns].astype('UInt16')

    return df_result


def add_confusion_scores(df_confusion: pd.DataFrame) -> pd.DataFrame:
    df = df_confusion.copy()
    for index, row in df.iterrows():
        accuracy, precision, recall, f1 = get_confusion_scores(row['TP'], row['TN'], row['FP'], row['FN'])
        mcnemar_p = mcnemar_test(row['FP'], row['FN'])
        df.loc[index, 'Accuracy'] = accuracy
        df.loc[index, 'Precision'] = precision
        df.loc[index, 'Recall'] = recall
        df.loc[index, 'F1'] = f1
        df.loc[index, 'McNemar-P'] = mcnemar_p

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
