"""
This module contains the logic for fine-tuning.
"""

import pandas as pd
from enum import Enum
import json


class TuningSet(Enum):
    TRAIN = 'train'  # Training set
    VALIDATION = 'validation'  # Validation set to check progress during training
    TEST = 'test'  # Test set to evaluate the model

def tuning_train_test_split(df_fallacies: pd.DataFrame, groupby: list[str], n_train: int, n_validation: int):
    """
    Split the fallacies into train, validation, and test sets for fine-tuning
    """
    df_fallacies['tuning'] = TuningSet.TEST.value
    df_fallacies['cumcount'] = df_fallacies.groupby(groupby, observed=True).cumcount()
    df_fallacies.loc[df_fallacies['cumcount'] < n_train, 'tuning'] = TuningSet.TRAIN.value
    df_fallacies.loc[(df_fallacies['cumcount'] >= n_train) &
                     (df_fallacies['cumcount'] < n_train + n_validation), 'tuning'] = TuningSet.VALIDATION.value
    df_fallacies.drop(columns=['cumcount'], inplace=True)


def get_tuning_examples(df_fallacies: pd.DataFrame, prompt_template: str, system_prompt: str, tuning_set: TuningSet) -> list[dict]:
    """
    Get the OpenAI training data for fine-tuning
    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """
    df = df_fallacies[df_fallacies['tuning'] == tuning_set.value]
    examples: list[dict] = []
    for _, row in df.iterrows():
        prompt = prompt_template.replace('[step]', row['step'])
        example = {
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                },
                {
                    'role': 'assistant',
                    'content': row['fallacy']
                }
            ]
        }
        examples.append(example)

    return examples


def save_tuning_examples(examples: list[dict], filename: str):
    with open(filename, 'w') as f:
        f.writelines(json.dumps(example) + '\n' for example in examples)