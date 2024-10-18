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


# Run the fallacy identification experiment, preserving existing responses if desired.
def run_fallacy_identification_zero_shot(df_fallacies: pd.DataFrame, llms: LLMs, keep_existing_responses: bool = True,
                                         sleep_seconds: float = 0):
    for llm_name, llm in llms.items():
        response_column = f"{llm_name.value}_response"
        # Add a column to the dataframe for each LLM if it doesn't exist
        if response_column not in df_fallacies.columns:
            df_fallacies[response_column] = ''

        response_count = 0

        for index, row in df_fallacies.iterrows():
            # Continue if a valid response already exists
            if keep_existing_responses and row[response_column] != '' and row[response_column] != RESPONSE_ERROR:
                continue

            # Get the response from the LLM
            prompt = f"Is the following reasoning step correct? You can only answer \"Yes\" or \"No\".\n{row['step']}"
            # log(f"Prompting LLM {llm_name.value}: {prompt}")

            try:
                response = llm.invoke(prompt)
                # log(f"Response from LLM {llm_name.value}: {response}")

                # Truncate the response to 10 characters in case instructions are ignored
                response_text = response if isinstance(response, str) else response.content
                df_fallacies.at[index, response_column] = response_text.replace("\n", " ").strip()[0:10]

            except Exception as e:
                log(f"Error invoking LLM {llm_name.value}: {e}")

                df_fallacies.at[index, response_column] = RESPONSE_ERROR

            # Saving intermediate results
            response_count += 1
            if response_count % 10 == 0:
                save_fallacy_df(df_fallacies, 'data/fallacy_identification_zero_shot.csv')

            if response_count % 100 == 0:
                log(f"Processed {response_count} responses for LLM {llm_name.value} (index={index}).")

            # Sleep between requests if specified
            if sleep_seconds > 0:
                sleep(sleep_seconds)



