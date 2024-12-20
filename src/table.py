"""
This module contains functions for displaying tables.
"""
from .analysis import add_llm_info
import pandas as pd

# Hong et al 2024 table labels
HONG_TABLE_LABELS = {
    'propositional': 'prop.',
    'quantificational': 'quant.',
    'syllogistic': 'syl.',
    'probabilistic': 'prob.',
    'formal': 'formal',
    'ambiguity': 'amb.',
    'inconsistency': 'incon.',
    'irrelevance': 'irrel.',
    'insufficiency': 'insuf.',
    'inappropriate presumption': 'inappr.',
    'informal': 'informal',
    'accuracy': 'accuracy',
}

CONFUSION_METRICS_LABELS = {
    'tp': 'TP',
    'tn': 'TN',
    'fp': 'FP',
    'fn': 'FN',
    'fp/fn': 'FP/FN',
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1',
    'p_mcnemar': 'P-McNemar',
}


def display_llm_table(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Display a table which has an llm-key index."""
    df_display = add_llm_info(df.copy(), label=True).set_index('llm_label', drop=True)

    return df_display.round(digits)


def get_llm_confusion_metrics_table(df_llm_confusion_metrics: pd.DataFrame) -> pd.DataFrame:
    df = df_llm_confusion_metrics.sort_values('accuracy', ascending=False)
    df = df[CONFUSION_METRICS_LABELS.keys()]
    df.columns = CONFUSION_METRICS_LABELS.values()
    df = display_llm_table(df)
    df.index.name = 'Model'

    return df


def get_hong_table(df_subcategory_accuracies: pd.DataFrame, df_category_accuracies: pd.DataFrame,
                   df_overall_accuracies: pd.DataFrame) -> pd.DataFrame:
    """Get the Hong et al. 2024 table."""
    df = df_subcategory_accuracies.droplevel(0, axis=1)
    df = df.join(df_category_accuracies).join(df_overall_accuracies)
    df = df[HONG_TABLE_LABELS.keys()]
    df.columns = HONG_TABLE_LABELS.values()
    df = display_llm_table(df.sort_values('accuracy', ascending=False), digits=1)
    df.index.name = 'Model'

    return df


def get_fallacy_search_table(df_metrics: pd.DataFrame, df_metrics_subset: pd.DataFrame) -> pd.DataFrame:
    """Get the fallacy search table with 3 levels and subset metrics."""
    df = df_metrics.join(df_metrics_subset, rsuffix='_subset').sort_values('f1_l2', ascending=False)
    col_labels = {
        'f1_l0': 'F1 Level 0',
        'f1_l1': 'F1 Level 1',
        'f1_l2': 'F1 Level 2',
        'f1_l0_subset': 'F1 Level 0 (Subset)',
        'f1_l1_subset': 'F1 Level 1 (Subset)',
        'f1_l2_subset': 'F1 Level 2 (Subset)',
    }

    df = df[col_labels.keys()]
    df.columns = col_labels.values()
    df = display_llm_table(df, digits=3)
    df.index.name = 'Model'

    return df
