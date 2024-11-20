"""
This module contains the logic for fallacy search, where an LLM is used to detect and analyze multiple logical fallacies
in a given text.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class FallacyEntry(BaseModel):
    """
    An identified fallacy, spanning one or more sentences.
    """
    fallacy: str = Field(description="The identified fallacy.")
    definition: str = Field(description="A one-sentence definition of the identified fallacy.")
    span: str = Field(
        description="The verbatim text span where the fallacy occurs, consisting of one or more contiguous sentences.")
    reason: str = Field(description="An explanation why the text span contains this fallacy.")
    defense: Optional[str] = Field(
        description="A counter-argument against the fallacy claim which explains how the argument could still be valid or reasonable.",
        default=None)
    confidence: float = Field(description="Confidence rating from 0.0 to 1.0.")


class FallacyResponse(BaseModel):
    """
    A response from the LLMs for a given input text.
    """
    fallacies: List[FallacyEntry] = Field(default_factory=list, title="The list of fallacies found in the text.")
    summary: str = Field(description="An overall summary of the logical reasoning quality.")
    rating: Optional[int] = Field(description="A rating of the overall logical reasoning quality from 1 to 10.")


def get_search_system_prompt() -> str:
    """
    This system prompt has been improved iteratively, based on the MAFALDA F1 score and partial inspection of the
    response quality by the author.
    """
    prompt = f"""You are an expert at detecting and analyzing logical fallacies. Your task is to detect and analyze logical fallacies in the provided text with high precision. 

Output Format:
Provide your analysis in JSON format with the following structure for each identified fallacy:
{{
  "fallacies": [
    {{
      "fallacy": "<fallacy_type>",
      "definition": "<fallacy_definition>",
      "span": "<text_span>",
      "reason": "<reason>",
      "defense": "<defense>",
      "confidence": <confidence>
    }}
  ],
  "summary": "<summary>",
  "rating": <rating>
}}

Guidelines:
1. Fallacy Types: Include any formal and informal logical fallacies
2. <fallacy_type>: The name of the identified formal or informal logical fallacy
3. <fallacy_definition>: A one-sentence definition of the identified fallacy
4. <text_span>:
   - Include the complete context needed to understand the fallacy, but keep the span as short as possible
   - Can overlap with other identified fallacies
   - Must be verbatim quotes from the original text
5. <reason>:
   - Provide clear, specific explanations
   - Include both why it qualifies as a fallacy and how it violates logical reasoning
6. <defense>:
   - Provide the strongest possible charitable interpretation under the assumption that the argument is valid or reasonable, and not a fallacy
   - Consider implicit premises that could validate the argument
7. <confidence>: Rate your confidence in each fallacy identification from 0.0 to 1.0, taking into account the reasoning and defense
8. <summary>: An overall summary of the logical reasoning quality. Take into account the identified fallacies.
9. <rating>: A rating of the overall logical reasoning quality. Use a scale from 1 to 10, where 1 is the lowest and 10 is the highest. If the provided text contains no arguments or reasoning (e.g. just factual statements or descriptions), the rating should be null.

Principles:
- Think step by step
- Be very thorough and include all potential fallacies in the provided text
- Adjust confidence scores downward in proportion to the strength and plausibility of the defense
- Consider context and implicit assumptions
- Return an empty list if no clear logical fallacies are present
"""

    return prompt


# noinspection PyArgumentList
def fallacy_search(text: str, model = 'gpt-4o-2024-08-06') -> FallacyResponse:
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=model,
        temperature=0.7,  # Higher temperature might generate more identified fallacies
        timeout=30.0,
        max_retries=2,
    )
    prompt = ChatPromptTemplate.from_messages([('system', '{system_prompt}'), ('user', '{input}')])

    # Models will generate validated structured outputs.
    pipe = prompt | llm.with_structured_output(FallacyResponse, method='json_schema')

    return pipe.invoke({'system_prompt': get_search_system_prompt(), 'input': text})


def pretty_print_fallacies(fallacy_response: FallacyResponse):
    print(json.dumps(fallacy_response.model_dump(mode='json')['fallacies'], indent=2, ensure_ascii=False))