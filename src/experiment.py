import pandas as pd
from .llms import LLMs
from .utils import log

# Constants
RESPONSE_ERROR = 'error'


def get_fallacy_identification_df() -> pd.DataFrame:
    try:
        df = pd.read_csv('data/fallacy_identification.csv')
        df = df.fillna('')

        log("Loaded existing fallacy identification dataframe from CSV.")
    except FileNotFoundError:
        df = pd.read_json('fallacies/step_fallacy.test.jsonl', lines=True)
        df['step'] = df['step'].apply(_remove_square_brackets)

        log("Created new fallacy identification dataframe from JSONL.")

    return df


def save_fallacy_identification_df(df_fallacies: pd.DataFrame):
    df_fallacies.to_csv('data/fallacy_identification.csv', index=False)


# Run the fallacy identification experiment, preserving existing responses if desired.
def run_fallacy_identification(df_fallacies: pd.DataFrame, llms: LLMs, keep_existing_responses: bool = True):
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
                df_fallacies.at[index, response_column] = response.content.replace("\n", " ").strip()[0:10]

            except Exception as e:
                log(f"Error invoking LLM {llm_name.value}: {e}")

                df_fallacies.at[index, response_column] = RESPONSE_ERROR

            # Saving intermediate results
            response_count += 1
            if response_count % 10 == 0:
                save_fallacy_identification_df(df_fallacies)

            if response_count % 100 == 0:
                log(f"Processed {response_count} responses for LLM {llm_name.value} (index={index}).")


def _remove_square_brackets(string):
    return string.replace("[", "").replace("]", "")