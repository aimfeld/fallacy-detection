"""
This module contains the main experiment logic.
"""
import pandas as pd
from .llms import LLMs
from .utils import log
from .fallacies import create_fallacy_df, get_fallacy_list
from time import sleep
from enum import Enum
import json

# Constants
RESPONSE_ERROR = 'error'


class TuningSet(Enum):
    TRAIN = 'train'  # Training set
    VALIDATION = 'validation'  # Validation set to check progress during training
    TEST = 'test'  # Test set to evaluate the model


def get_fallacy_df(filename: str, only_incorrect: bool = False) -> pd.DataFrame:
    """
    Load the fallacy identification dataframe from a CSV file, or create a new one if the file doesn't exist.
    """
    try:
        df = pd.read_csv(filename)
        df = df.fillna('')

        log(f"Loaded existing fallacy dataframe from {filename}.")
    except FileNotFoundError:
        df = create_fallacy_df()

        log("Created new fallacy identification dataframe.")

    if only_incorrect:
        # Select only incorrect reasoning steps
        df = df[df['label'] == 1]

    return df


def save_fallacy_df(df_fallacies: pd.DataFrame, filename: str):
    df_fallacies.to_csv(filename, index=False)


def run_experiment(df_fallacies: pd.DataFrame, filename: str, prompt_template: str, llms: LLMs,
                   keep_existing_responses: bool = True, sleep_seconds: float = 0, log_responses: bool = False):
    """
    Run the experiment to get responses from the LLMs for each reasoning step.
    """
    for llm_name, llm in llms.items():
        response_column = f"{llm_name.value}_response"
        # Add a column to the dataframe for each LLM if it doesn't exist
        if response_column not in df_fallacies.columns:
            df_fallacies[response_column] = ''

        response_count = 0
        error_count = 0

        for index, row in df_fallacies.iterrows():
            # Continue if a valid response already exists
            if keep_existing_responses and row[response_column] != '' and row[response_column] != RESPONSE_ERROR:
                continue

            # Get the response from the LLM
            prompt = prompt_template.replace('[step]', row['step'])
            # log(f"Prompting LLM {llm_name.value}: {prompt}")

            try:
                response = llm.invoke(prompt)
                if log_responses:
                    log(f"Response from LLM {llm_name.value} (index={index}): {response}")

                # Huggingface endpoint returns a string, the other LLMs return a response object
                response_text = response if isinstance(response, str) else response.content
                df_fallacies.at[index, response_column] = response_text.replace("\n", " ").strip()

            except Exception as e:
                log(f"Error invoking LLM {llm_name.value}: {e}")

                df_fallacies.at[index, response_column] = RESPONSE_ERROR
                error_count += 1

            # Saving intermediate results
            response_count += 1
            if response_count % 10 == 0:
                save_fallacy_df(df_fallacies, filename)

            if response_count % 100 == 0:
                log(f"Processed {response_count} responses for LLM {llm_name.value} (index={index}).")

            if error_count > 10:
                log(f"Error count too high for LLM {llm_name.value}, skipping model.")
                break

            # Sleep between requests if specified
            if sleep_seconds > 0:
                sleep(sleep_seconds)


def get_classification_prompt_template() -> str:
    """
    Get the template for the classification prompt
    """
    # Newline characters will be preserved in the multi-line prompt
    prompt_template = """You are a logical fallacy classifier. Given an incorrect reasoning step, your task is to identify its type of fallacy.
Answer by choosing one of these fallacies:
[fallacies]
You should only answer the name of the fallacy.
What type of fallacy does the following reasoning step belong to?
[step]"""

    fallacies_list = get_fallacy_list()
    fallacies_string = "\n".join([f"({number + 1}) {fallacy}" for number, fallacy in enumerate(fallacies_list)])
    prompt_template = prompt_template.replace('[fallacies]', fallacies_string)

    return prompt_template


def tuning_train_test_split(df_fallacies: pd.DataFrame, n_train: int, n_validation: int):
    """
    Split the fallacies into train, validation, and test sets for fine-tuning
    """
    df_fallacies['tuning'] = TuningSet.TEST.value
    df_fallacies['cumcount'] = df_fallacies.groupby('fallacy').cumcount()
    df_fallacies.loc[df_fallacies['cumcount'] < n_train, 'tuning'] = TuningSet.TRAIN.value
    df_fallacies.loc[(df_fallacies['cumcount'] >= n_train) &
                     (df_fallacies['cumcount'] < n_train + n_validation), 'tuning'] = TuningSet.VALIDATION.value
    df_fallacies.drop(columns=['cumcount'], inplace=True)


def get_tuning_examples(df_fallacies: pd.DataFrame, tuning_set: TuningSet) -> list[dict]:
    """
    Get the OpenAI training data for fine-tuning
    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """
    df = df_fallacies[df_fallacies['tuning'] == tuning_set.value]
    prompt_template = get_classification_prompt_template()
    examples: list[dict] = []
    for _, row in df.iterrows():
        prompt = prompt_template.replace('[step]', row['step'])
        example = {
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a logical fallacy classifier. Given an incorrect reasoning step, your task is to identify its type of fallacy.'
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
