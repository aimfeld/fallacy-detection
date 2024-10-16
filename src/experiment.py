import pandas as pd
from .llms import LLMs

RESPONSE_ERROR = 'error'

def remove_square_brackets(string):
    return string.replace("[", "").replace("]", "")


def get_fallacy_identification_df() -> pd.DataFrame:
    try:
        df = pd.read_csv('data/fallacy_identification.csv')
        df = df.fillna('')

        print("Loaded existing fallacy identification dataframe from CSV.")
    except FileNotFoundError:
        df = pd.read_json('fallacies/step_fallacy.test.jsonl', lines=True)
        df['step'] = df['step'].apply(remove_square_brackets)

        print("Created new fallacy identification dataframe from JSONL.")

    return df


def save_fallacy_identification_df(df_fallacies: pd.DataFrame):
    df_fallacies.to_csv('data/fallacy_identification.csv', index=False)

    print("Saved fallacy identification dataframe to CSV.")


# Run the fallacy identification experiment, preserving existing responses if desired.
def run_fallacy_identification(df_fallacies: pd.DataFrame, llms: LLMs, keep_existing_responses: bool = True):
    for llm_name, llm in llms.items():
        response_column = f"{llm_name}_response"
        # Add a column to the dataframe for each LLM if it doesn't exist
        if response_column not in df_fallacies.columns:
            df_fallacies[response_column] = ''

        for index, row in df_fallacies.iterrows():
            # Continue if a valid response already exists
            if keep_existing_responses and row[response_column] != '' and row[response_column] != RESPONSE_ERROR:
                continue

            # Get the response from the LLM
            prompt = f"Is the following reasoning step correct? You can only answer \"Yes\" or \"No\".\n{row['step']}"
            print(f"Prompting LLM {llm_name}: {prompt}")

            try:
                response = llm.invoke(prompt)
                # print(f"Response from LLM {llm_name}: {response}")

                df_fallacies.at[index, response_column] = response.content
            except Exception as e:
                print(f"Error invoking LLM {llm_name}: {e}")

                df_fallacies.at[index, response_column] = RESPONSE_ERROR
