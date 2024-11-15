"""
This module functions for dealing with the MAFALDA dataset and evaluation by Helwe et al. (2024).
"""
import pandas as pd
import numpy as np
import json
import regex
from .utils import log
from .constants import RESPONSE_ERROR
from .mafalda_metrics.new_metrics import text_full_task_p_r_f1, AnnotatedText, GroundTruthSpan, PredictionSpan
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum



class Fallacy(Enum):
    """
    MAFALDA fallacy types.
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
    defense: Optional[str] = Field(description="A counter-argument against the fallacy claim which explains how the argument could still be valid or reasonable.", default=None)
    confidence: float = Field(description="Confidence rating from 0.0 to 1.0.")


class FallacyResponse(BaseModel):
    """
    A response from the LLMs for a given input text.
    """
    fallacies: List[FallacyEntry] = Field(default_factory=list, title="The list of fallacies found in the text.")

# Map fallacies to string labels
FALLACY_2_LABEL: dict[str, str] = {
    Fallacy.APPEAL_TO_ANGER.value: 'appeal to anger',
    Fallacy.APPEAL_TO_FEAR.value: 'appeal to fear',
    Fallacy.APPEAL_TO_PITY.value: 'appeal to pity',
    Fallacy.APPEAL_TO_POSITIVE_EMOTION.value: 'appeal to positive emotion',
    Fallacy.APPEAL_TO_RIDICULE.value: 'appeal to ridicule',
    Fallacy.APPEAL_TO_WORSE_PROBLEMS.value: 'appeal to worse problems',
    Fallacy.CAUSAL_OVERSIMPLIFICATION.value: 'causal oversimplification',
    Fallacy.CIRCULAR_REASONING.value: 'circular reasoning',
    Fallacy.EQUIVOCATION.value: 'equivocation',
    Fallacy.FALLACY_OF_DIVISION.value: 'fallacy of division',
    Fallacy.FALSE_ANALOGY.value: 'false analogy',
    Fallacy.FALSE_CAUSALITY.value: 'false causality',
    Fallacy.FALSE_DILEMMA.value: 'false dilemma',
    Fallacy.HASTY_GENERALIZATION.value: 'hasty generalization',
    Fallacy.SLIPPERY_SLOPE.value: 'slippery slope',
    Fallacy.STRAWMAN_FALLACY.value: 'straw man',
    Fallacy.AD_HOMINEM.value: 'ad hominem',
    Fallacy.AD_POPULUM.value: 'ad populum',
    Fallacy.APPEAL_TO_AUTHORITY.value: 'appeal to (false) authority',
    Fallacy.APPEAL_TO_NATURE.value: 'appeal to nature',
    Fallacy.APPEAL_TO_TRADITION.value: 'appeal to tradition',
    Fallacy.GUILT_BY_ASSOCIATION.value: 'guilt by association',
    Fallacy.TU_QUOQUE.value: 'tu quoque',
}

# Map string labels to numeric labels
LEVEL_2_NUMERIC = {
    "nothing": 0,
    "appeal to positive emotion": 1,
    "appeal to anger": 2,
    "appeal to fear": 3,
    "appeal to pity": 4,
    "appeal to ridicule": 5,
    "appeal to worse problems": 6,
    "causal oversimplification": 7,
    "circular reasoning": 8,
    "equivocation": 9,
    "false analogy": 10,
    "false causality": 11,
    "false dilemma": 12,
    "hasty generalization": 13,
    "slippery slope": 14,
    "straw man": 15,
    "fallacy of division": 16,
    "ad hominem": 17,
    "ad populum": 18,
    "appeal to (false) authority": 19,
    "appeal to nature": 20,
    "appeal to tradition": 21,
    "guilt by association": 22,
    "tu quoque": 23,
}


def create_mafalda_df() -> pd.DataFrame:
    df = pd.read_json('datasets/MAFALDA/gold_standard_dataset.jsonl', lines=True)

    return df


def get_mafalda_df(filename: str) -> pd.DataFrame:
    """
    Load the mafalda dataframe from a CSV file, or create a new one if the file doesn't exist.
    """
    try:
        df = pd.read_csv(filename)
        df = df.fillna('')

        log(f"Loaded existing mafalda dataframe from {filename}.")
    except FileNotFoundError:
        df = create_mafalda_df()

        log("Created new mafalda dataframe.")

    df['sentences_with_labels'] = df['sentences_with_labels'].apply(json.loads)

    # Parse json strings into validated FallacyResponse objects
    response_cols = [col for col in df.columns if col.endswith('_response')]
    for col in response_cols:
        df[col] = df[col].apply(lambda x: x if x in ['', RESPONSE_ERROR] else FallacyResponse.model_validate_json(x))

    # Parse strings with list of labels
    df['labels'] = df['labels'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)

    return df


def get_mafalda_fallacies_df() -> pd.DataFrame:
    df = pd.read_csv('datasets/MAFALDA/mafalda_fallacies.csv')
    df.set_index('fallacy', inplace=True)

    return df


def save_mafalda_df(df: pd.DataFrame, filename: str):
    """
    Save the mafalda dataframe to a CSV file.
    """
    df = df.copy()

    df['sentences_with_labels'] = df['sentences_with_labels'].apply(json.dumps)
    response_cols = [col for col in df.columns if col.endswith('_response')]
    for col in response_cols:
        df[col] = df[col].apply(lambda x: x if x in ['', RESPONSE_ERROR] else FallacyResponse.model_dump_json(x))

    df.to_csv(filename, index=False)


def get_llm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the mean precision, recall, and F1 score for each model.

    Args:
        df: DataFrame containing model responses and their corresponding precision, recall, and F1 scores.

    Returns:
        DataFrame with the mean precision, recall, and F1 scores for each model.
    """
    response_cols = [col for col in df.columns if col.endswith('_response')]
    llm_metrics: dict[str: pd.Series] = {}
    for response_col in response_cols:
        llm_key = response_col.removesuffix('_response')
        precision_col = f"{llm_key}_precision"
        recall_col = f"{llm_key}_recall"
        f1_col = f"{llm_key}_f1"
        metrics = df[[precision_col, recall_col, f1_col]].mean()

        confidence_ratings = []
        fallacy_counts = []
        fallacy_response: FallacyResponse
        for fallacy_response in df[response_col]:
            fallacy_counts.append(len(fallacy_response.fallacies))
            for entry in fallacy_response.fallacies:
                confidence_ratings.append(entry.confidence)

        metrics['fallacy_count'] = np.mean(fallacy_counts)
        metrics['confidence'] = np.mean(confidence_ratings)
        metrics.index = ['precision', 'recall', 'f1', 'fallacy_count', 'confidence']

        llm_metrics[llm_key] = metrics


    return pd.DataFrame(llm_metrics).T


def evaluate_responses(df: pd.DataFrame, confidence_threshold: float = 0.5):
    """
    Evaluates the precision, recall, and F1 score for responses in the dataframe based on provided confidence threshold.

    Args:
        df: DataFrame containing model responses
        confidence_threshold: Fallacies with a lower confidence rating will be ignored.
    """
    response_cols = [col for col in df.columns if col.endswith('_response')]
    for response_col in response_cols:
        llm_key = response_col.removesuffix('_response')
        precision_col = f"{llm_key}_precision"
        recall_col = f"{llm_key}_recall"
        f1_col = f"{llm_key}_f1"

        log(f'Evaluating responses for {llm_key} ...')
        for index, row in df.iterrows():
            precision, recall, f1 = _evaluate_response(row['text'], row['labels'], row[response_col].fallacies,
                                                       confidence_threshold)

            df.at[index, precision_col] = precision
            df.at[index, recall_col] = recall
            df.at[index, f1_col] = f1


def _evaluate_response(text: str, labels: list[list[int, int, str]], fallacy_entries: list[FallacyEntry],
                       confidence_threshold: float) \
        -> tuple[float, float, float]:
    gold_spans = []
    for label_span in labels:
        start, end, label = tuple(label_span)
        if label == 'to clean':
            continue
        span = GroundTruthSpan(text[start:end], {LEVEL_2_NUMERIC[label.lower()]}, [start, end])
        gold_spans.append(span)

    pred_spans = []
    for entry in fallacy_entries:
        if entry.confidence < confidence_threshold:
            continue
        start, end = _fuzzy_match(entry.span, text)
        if start is not None and end is not None:
            label = LEVEL_2_NUMERIC[FALLACY_2_LABEL[entry.fallacy.value]]
            span = PredictionSpan(entry.span, label, [start, end])
            pred_spans.append(span)
        else:
            print(f"Warning: failed to match span for fallacy {entry.fallacy}:\nspan: {entry.span}\ntext: {text}")

    return text_full_task_p_r_f1(AnnotatedText(pred_spans), AnnotatedText(gold_spans))


def _fuzzy_match(pattern: str, text: str) -> tuple[int | None, int | None]:
    # Sometimes, the span is enclosed with '...some span bla bla...'
    pattern = pattern.removeprefix('...').removesuffix('...')

    # Sometimes, the model uses different quotation marks than the original text
    text = text.replace('"', "'")
    pattern = pattern.replace('"', "'")

    # Allow a few differences
    fuzzy_pattern = f'({regex.escape(pattern)}){{e<=5}}'
    if match := regex.search(fuzzy_pattern, text, regex.BESTMATCH):
        return match.start(), match.end()

    return None, None
