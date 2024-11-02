"""
This module contains functions for plotting data.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .analysis import get_confusion_scores, mcnemar_test

DPI = 600 # Quality of saved .png


def plot_accuracies(data: pd.DataFrame, figsize: tuple, title: str,
                    y='llm_label', y_label= '', hue: str = None,
                    legend_title: str = None, legend_loc: str = 'lower right', legend_anchor: tuple = (1, 0),
                    order: list[str] = None, hue_order: list[str] = None,
                    annotate: bool = False, xlim: tuple = (0, 100)):
    """
    Plot horizontal bar plot of accuracies
    """
    df = data.copy()
    if 'subcategory' in df.columns:
        df['subcategory'] = df['subcategory'].cat.remove_unused_categories() # Don't show unused subcategories in the plot

    _, ax = plt.subplots(figsize=figsize)
    # reset_index() prevents reindexing error when there are duplicate indices
    sns.barplot(x='accuracy', y=y, data=df.reset_index(), hue=hue, order=order, hue_order=hue_order, ax=ax)
    plt.title(title)
    plt.xlabel('Accuracy (%)')
    plt.ylabel(y_label)

    # Set x-axis limits, start from 50% (random guessing)
    ax.set_xlim(xlim)

    if annotate:
        for i in ax.containers:
            ax.bar_label(i, label_type='edge', color='#555555', fmt='%.1f%%', padding=5, fontsize=9)

    if hue and legend_title and legend_anchor:
        plt.legend(loc=legend_loc, bbox_to_anchor=legend_anchor, title=legend_title)

    save_plot(title)
    plt.show()


def plot_identification_confusion_matrix(metrics: pd.Series, title: str, figsize=(6, 5)):
    # Create custom annotation labels
    labels = np.array([
        [f'TP\n{int(metrics["TP"])}', f'FP\n{int(metrics["FP"])}'],
        [f'FN\n{int(metrics["FN"])}', f'TN\n{int(metrics["TN"])}']
    ])

    confusion_matrix = np.array([
        [metrics['TP'], metrics['FP']],
        [metrics['FN'], metrics['TN']]
    ])

    # Create figure and axes
    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(confusion_matrix,
                annot=labels,
                fmt='',
                cmap='YlGnBu',
                cbar=False,
                square=True,
                yticklabels=['fallacy', 'no fallacy'],
                xticklabels=['fallacy', 'no fallacy'],
                annot_kws={'size': 12}
                )

    # Add labels
    plt.title(title, pad=10)
    plt.xlabel('Actual', labelpad=10)
    plt.ylabel('Predicted', labelpad=10)

    # Add metrics text box
    accuracy, precision, recall, f1 = get_confusion_scores(metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN'])
    mcnemar_p = mcnemar_test(metrics['FP'], metrics['FN'])
    metrics_text = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}\nMcNemar-P: {mcnemar_p:.3f}"
    plt.text(2.1, 0.6, metrics_text, fontsize=10)

    plt.tight_layout()
    save_plot(title)
    plt.show()


def plot_classification_confusion_matrix(df_confusion, title: str, figsize=(10, 10)):
    # Create figure and axes
    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        df_confusion,
        annot=True,
        cmap='YlGnBu',
        cbar=False,
        square=True,
        yticklabels=df_confusion.columns,
        xticklabels=df_confusion.columns,
        annot_kws={'size': 12}
    )

    # Add labels
    plt.title(title, pad=10)
    plt.xlabel('Actual', labelpad=10)
    plt.ylabel('Predicted', labelpad=10)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    save_plot(title)
    plt.show()


def save_plot(title: str):
    plt.savefig(f'plot/{title}.svg', format='svg', dpi=DPI, bbox_inches='tight')
    plt.savefig(f'plot/{title}.png', dpi=DPI, bbox_inches='tight')
