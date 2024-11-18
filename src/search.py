"""
This module contains the logic for fallacy search, where an LLM is used to detect and analyze multiple logical fallacies
in a given text.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import regex
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class Fallacy(Enum):
    """
    MAFALDA fallacy types. 
    Fallacy search is restricted to these types so we can validate the results against the MAFALDA benchmark.
    But it works perfectly well without restriction.
    """
    APPEAL_TO_ANGER = "Appeal to Anger"
    APPEAL_TO_FEAR = "Appeal to Fear"
    APPEAL_TO_PITY = "Appeal to Pity"
    APPEAL_TO_POSITIVE_EMOTION = "Appeal to Positive Emotion"
    APPEAL_TO_RIDICULE = "Appeal to Ridicule"
    APPEAL_TO_WORSE_PROBLEMS = "Appeal to Worse Problems"
    CAUSAL_OVERSIMPLIFICATION = "Causal Oversimplification"
    CIRCULAR_REASONING = "Circular Reasoning"
    EQUIVOCATION = "Equivocation"
    FALLACY_OF_DIVISION = "Fallacy of Division"
    FALSE_ANALOGY = "False Analogy"
    FALSE_CAUSALITY = "False Causality"
    FALSE_DILEMMA = "False Dilemma"
    HASTY_GENERALIZATION = "Hasty Generalization"
    SLIPPERY_SLOPE = "Slippery Slope"
    STRAWMAN_FALLACY = "Strawman Fallacy"
    AD_HOMINEM = "Ad Hominem"
    AD_POPULUM = "Ad Populum"
    APPEAL_TO_AUTHORITY = "Appeal to Authority"
    APPEAL_TO_NATURE = "Appeal to Nature"
    APPEAL_TO_TRADITION = "Appeal to Tradition"
    GUILT_BY_ASSOCIATION = "Guilt by Association"
    TU_QUOQUE = "Tu Quoque"


class FallacyEntry(BaseModel):
    """
    A fallacy found in the MAFALDA dataset, spanning one or more sentences.
    """
    fallacy: Fallacy = Field(description="The identified fallacy.")
    span: str = Field(
        description="The verbatim text span where the fallacy occurs, consisting of one or more contiguous sentences.")
    reason: str = Field(description="An explanation why the text span contains this fallacy.")
    defense: Optional[str] = Field(
        description="A counter-argument against the fallacy claim which explains how the argument could still be valid or reasonable.",
        default=None)
    confidence: float = Field(description="Confidence rating from 0.0 to 1.0.")


class UnrestrictedFallacyEntry(FallacyEntry):
    fallacy: str = Field(description="The identified fallacy.")


class FallacyResponse(BaseModel):
    """
    A response from the LLMs for a given input text.
    """
    fallacies: List[FallacyEntry] = Field(default_factory=list, title="The list of fallacies found in the text.")


class UnrestrictedFallacyResponse(FallacyResponse):
    fallacies: List[UnrestrictedFallacyEntry] = Field(default_factory=list, title="The list of fallacies found in the text.")


def get_search_system_prompt(restrict_fallacies: bool = True) -> str:
    """
    This system prompt has been improved iteratively, based on the MAFALDA F1 score and partial inspection of the
    response quality by the author.

    Args:
        restrict_fallacies: Whether to restrict the identified fallacies to the MAFALDA fallacy types.
    """
    fallacy_types_placeholder = '[fallacy_types]'

    prompt = f"""You are an expert at detecting and analyzing logical fallacies. Your task is to detect and analyze logical fallacies in the provided text with high precision. 

Output Format:
Provide your analysis in JSON format with the following structure for each identified fallacy:
{{
  "fallacies": [
    {{
      "fallacy": "<fallacy_type>",
      "span": "<exact_text>",
      "reason": "<explanation>",
      "defense": "<counter_argument>",
      "confidence": <0.0-1.0>
    }}
  ]
}}

Guidelines:
1. Fallacy Types: {fallacy_types_placeholder}
2. Text Spans:
   - Include the complete context needed to understand the fallacy, but keep the span as short as possible
   - Can overlap with other identified fallacies
   - Must be verbatim quotes from the original text
3. Reasoning:
   - Provide clear, specific explanations
   - Include both why it qualifies as a fallacy and how it violates logical reasoning
4. Defense:
   - Provide the strongest possible charitable interpretation under the assumption that the argument is valid or reasonable, and not a fallacy
   - Consider implicit premises that could validate the argument
5. Confidence Score:
   - Rate your confidence in each fallacy identification from 0.0 to 1.0, taking into account the reasoning and defense

Principles:
- Think step by step
- Be very thorough and include all potential fallacies in the provided text
- Adjust confidence scores downward in proportion to the strength and plausibility of the defense
- Consider context and implicit assumptions
- Return an empty list if no clear logical fallacies are present
"""
    fallacies_string = ', '.join(e.value for e in Fallacy)
    fallacy_types = f'Only use fallacies from this approved list: {fallacies_string}' if restrict_fallacies \
        else 'Include any formal and informal logical fallacies'

    prompt = prompt.replace(fallacy_types_placeholder, fallacy_types)

    return prompt


def fallacy_search(text: str, model = 'gpt-4o-2024-08-06', restrict_fallacies: bool = True) -> FallacyResponse | UnrestrictedFallacyResponse:
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=model,
        temperature=0,  # Higher temperature might generate more identified fallacies
        timeout=30.0,
        max_retries=2,
    )
    prompt = ChatPromptTemplate.from_messages([('system', '{system_prompt}'), ('user', '{input}')])

    # Models will generate validated structured outputs.
    response_type = FallacyResponse if restrict_fallacies else UnrestrictedFallacyResponse
    pipe = prompt | llm.with_structured_output(response_type, method='json_schema')

    system_prompt = get_search_system_prompt(restrict_fallacies)

    return pipe.invoke({'system_prompt': system_prompt, 'input': text})


def fuzzy_match(span: str, text: str) -> tuple[int | None, int | None]:
    """
    Returns start and end indices of the first match of the span in the text, using fuzzy matching.
    """
    # Sometimes, the span is enclosed with '...some span bla bla...'
    span = span.removeprefix('...').removesuffix('...')

    # Sometimes, the model uses different quotation marks than the original text
    text = text.replace('"', "'")
    span = span.replace('"', "'")

    # Allow a few differences
    fuzzy_pattern = f'({regex.escape(span)}){{e<=5}}'
    if match := regex.search(fuzzy_pattern, text, regex.BESTMATCH):
        return match.start(), match.end()

    return None, None


def pretty_print_fallacies(fallacy_response: FallacyResponse):
    print(json.dumps(fallacy_response.model_dump(mode='json')['fallacies'], indent=2, ensure_ascii=False))