"""
This module contains the main experiment logic.
"""
import pandas as pd
from .llms import LLMs
from .utils import log
from .fallacies import create_fallacy_df
from time import sleep

# Constants
RESPONSE_ERROR = 'error'

def get_fallacy_df(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename)
        df = df.fillna('')

        log(f"Loaded existing fallacy dataframe from {filename}.")
    except FileNotFoundError:
        df = create_fallacy_df()

        log("Created new fallacy identification dataframe.")

    return df

def save_fallacy_df(df_fallacies: pd.DataFrame, filename: str):
    df_fallacies.to_csv(filename, index=False)


# Run experiment 1: fallacy identification with zero-shot prompt
def run_experiment(df_fallacies: pd.DataFrame, filename: str, prompt_template: str, llms: LLMs,
                   keep_existing_responses: bool = True, sleep_seconds: float = 0, log_responses: bool = False):
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

