"""
This module contains functions for plotting data.
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Horizontal accuracy bar plot
def plot_accuracies(data: pd.DataFrame, figsize: tuple, title: str,
                    y='llm_label', y_label= '',
                    hue: str = None, legend_title: str = None, legend_anchor: tuple = None,
                    annotate: bool = False, xlim: tuple = (0, 100)):
    _, ax = plt.subplots(figsize=figsize)
    sns.barplot(x='accuracy', y=y, data=data, hue=hue, ax=ax)
    plt.title(title)
    plt.xlabel('Accuracy (%)')
    plt.ylabel(y_label)

    # Set x-axis limits, start from 50% (random guessing)
    ax.set_xlim(xlim)

    if annotate:
        for i in ax.containers:
            ax.bar_label(i, label_type='center', color='white', fmt='%.1f%%', fontsize=9)

    if hue and legend_title and legend_anchor:
        plt.legend(loc='upper right', bbox_to_anchor=legend_anchor, title=legend_title)

    plt.show()