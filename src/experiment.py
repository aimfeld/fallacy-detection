"""
This module contains the main experiment logic.
"""
import pandas as pd
from .llms import LLMs
from .utils import log
from .constants import RESPONSE_ERROR
from .fallacies import save_fallacy_df, get_fallacy_list
from .mafalda import save_mafalda_df, Fallacy
from time import sleep
from langchain_core.messages.ai import AIMessage

# Constants
TEXT_PLACEHOLDER = '[text]'
FALLACIES_PLACEHOLDER = '[fallacies]'


def run_experiment(df: pd.DataFrame, filename: str, prompt_template: str, llms: LLMs,
                   keep_existing_responses: bool = True, sleep_seconds: float = 0, log_responses: bool = False,
                   is_search: bool = False):
    """
    Run the experiment to get responses from the LLMs for each reasoning step.
    """
    for llm, chat_model in llms.items():
        response_column = f"{llm.key}_response"
        # Add a column to the dataframe for each LLM if it doesn't exist
        if response_column not in df.columns:
            df[response_column] = ''

        response_count = 0
        error_count = 0

        for index, row in df.iterrows():
            # Continue if a valid response already exists
            if keep_existing_responses and row[response_column] != '' and row[response_column] != RESPONSE_ERROR:
                continue

            try:
                if is_search:
                    response: AIMessage = chat_model.invoke({'system_prompt': prompt_template, 'input': row['text']})
                    df.at[index, response_column] = response # Validated pydantic FallacyResponse object
                else:
                    prompt = prompt_template.replace(TEXT_PLACEHOLDER, row['step'])
                    response: AIMessage = chat_model.invoke(prompt)

                    response_text = _filter_response_text(response.content)
                    df.at[index, response_column] = response_text

                if log_responses:
                    log(f"Response from LLM {llm.key} (index={index}): {response}")

            except Exception as e:
                log(f"Error invoking LLM {llm.key}: {e}")

                df.at[index, response_column] = RESPONSE_ERROR
                error_count += 1

            # Saving intermediate results
            response_count += 1
            if response_count % 10 == 0:
                save_mafalda_df(df, filename) if is_search else save_fallacy_df(df, filename)

            if response_count % 100 == 0:
                log(f"Processed {response_count} responses for LLM {llm.key} (index={index}).")

            if error_count > 30:
                log(f"Error count too high for LLM {llm.key}, skipping model.")
                break

            # Sleep between requests if specified
            if sleep_seconds > 0:
                sleep(sleep_seconds)


def get_identification_zero_shot_prompt_template() -> str:
    return f"""Is the following reasoning step correct? You can only answer "Yes" or "No".
{TEXT_PLACEHOLDER}"""


def get_identification_zero_shot_prompt_template_v2() -> str:
    """
    A slight variation of the zero-shot identification prompt, to see whether false positives can be reduced by wording.
    """
    return f"""Does the following reasoning step contain a logical fallacy? You can only answer "Yes" or "No".
{TEXT_PLACEHOLDER}"""


def get_identification_few_shot_prompt_template() -> str:
    return f"""Is the following reasoning step correct? You can only answer "Yes" or "No".
Since if it's raining then the streets are wet and it's raining now, therefore, the streets are wet.
Yes.
Since I found a shell on the beach and this shell was beautifully shaped and colored, therefore, all shells are beautifully shaped and colored.
No.
Since I am at home or I am in the city and I am at home, therefore, I am not in the city.
No.
Since heavy snowfall often leads to traffic jams and traffic jams cause delays, therefore, heavy snowfall can lead to delays.
Yes.
{TEXT_PLACEHOLDER}"""


def get_identification_cot_prompt_template() -> str:
    return f"""Is the following reasoning step correct?
Let's think step by step and then answer "Yes" or "No".
{TEXT_PLACEHOLDER}"""


def get_classification_prompt_template() -> str:
    # Newline characters will be preserved in the multi-line prompt
    prompt_template = f"""You are a logical fallacy classifier. Given an incorrect reasoning step, your task is to identify its type of fallacy.
Answer by choosing one of these fallacies:
{FALLACIES_PLACEHOLDER}
You should only answer the name of the fallacy.
What type of fallacy does the following reasoning step belong to?
{TEXT_PLACEHOLDER}"""

    fallacies_list = get_fallacy_list()
    fallacies_string = "\n".join([f"({number + 1}) {fallacy}" for number, fallacy in enumerate(fallacies_list)])
    prompt_template = prompt_template.replace(FALLACIES_PLACEHOLDER, fallacies_string)

    return prompt_template


def get_search_system_prompt() -> str:
    prompt = f"""You are an expert at detecting and analyzing logical fallacies. Your task is to detect and analyze logical fallacies in the provided text with high precision. 

Output Format:
Provide your analysis in JSON format with the following structure for each identified fallacy:
{{
  "fallacies": [
    {{
      "fallacy": "<fallacy_type>",
      "span": "<exact_text>",
      "reason": "<explanation>",
      "confidence": <0.0-1.0>
    }}
  ]
}}

Guidelines:
1. Fallacy Types: Only use fallacies from this approved list: {FALLACIES_PLACEHOLDER}
2. Text Spans:
   - Include the complete context needed to understand the fallacy, but not more than necessary.
   - Can overlap with other identified fallacies
   - Must be verbatim quotes from the original text
3. Reasoning:
   - Provide clear, specific explanations
   - Include both why it qualifies as a fallacy and how it violates logical reasoning
4. Confidence Score:
   - Rate your confidence in each identification from 0.0 to 1.0

Principles:
- Think step by step
- Be thorough but avoid over-identification
- Apply the principle of charity: interpret arguments in their strongest reasonable form
- Consider context and implicit assumptions
- Return an empty list if no clear logical fallacies are present
"""

    fallacies_string = ', '.join(e.value for e in Fallacy)
    prompt = prompt.replace(FALLACIES_PLACEHOLDER, fallacies_string)

    return prompt


def _filter_response_text(response_text: str) -> str:
    return response_text.replace("\n", " ").strip()

