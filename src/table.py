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


def get_hong_table(df_subcategory_accuracies: pd.DataFrame, df_category_accuracies: pd.DataFrame, df_overall_accuracies: pd.DataFrame) -> pd.DataFrame:
    """Get the Hong et al. 2024 table."""
    df = df_subcategory_accuracies.droplevel(0, axis=1)
    df = df.join(df_category_accuracies).join(df_overall_accuracies)
    df = df[HONG_TABLE_LABELS.keys()]
    df.columns = HONG_TABLE_LABELS.values()
    df = display_llm_table(df.sort_values('accuracy', ascending=False), digits=1)
    df.index.name = 'Model'

    return df