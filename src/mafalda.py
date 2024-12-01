"""
This module functions for dealing with the MAFALDA dataset and evaluation by Helwe et al. (2024).
"""
import pandas as pd
import numpy as np
import json
import regex
from copy import deepcopy
from .utils import log
from .constants import RESPONSE_ERROR
from .mafalda_metrics.new_metrics import text_full_task_p_r_f1, AnnotatedText, PredictionSpan, GroundTruthSpan
from .mafalda_metrics.evaluate import build_ground_truth_spans, LEVEL_2_NUMERIC, LEVEL_2_TO_1
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


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

Span = tuple[int, int]

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


class FallacyResponse(BaseModel):
    """
    A response from the LLMs for a given input text.
    """
    fallacies: List[FallacyEntry] = Field(default_factory=list, title="The list of fallacies found in the text.")


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


def get_llm_metrics(df: pd.DataFrame, exclude_llms: list[str] = None) -> pd.DataFrame:
    """
    Calculates the mean precision, recall, and F1 score for each model.

    Args:
        df: DataFrame containing model responses and their corresponding precision, recall, and F1 scores.
        exclude_llms: List of LLM keys to ignore.
    Returns:
        DataFrame with the mean precision, recall, and F1 scores for each model.
    """
    response_cols = [col for col in df.columns if col.endswith('_response')]
    llm_metrics: dict[str: pd.Series] = {}
    mean_metrics = ['p_l2', 'r_l2', 'f1_l2', 'p_l1', 'r_l1', 'f1_l1', 'p_l0', 'r_l0', 'f1_l0']

    for response_col in response_cols:
        llm_key = response_col.removesuffix('_response')
        if exclude_llms is not None and llm_key in exclude_llms:
            continue

        mean_metric_cols = [f'{llm_key}_{metric}' for metric in mean_metrics]
        metrics = df[mean_metric_cols].mean()
        metrics.index = mean_metrics

        confidence_ratings = []
        fallacy_counts = []
        fallacy_response: FallacyResponse
        for fallacy_response in df[response_col]:
            fallacy_counts.append(len(fallacy_response.fallacies))
            for entry in fallacy_response.fallacies:
                confidence_ratings.append(entry.confidence)

        metrics['fallacies'] = np.mean(fallacy_counts)
        metrics['confidence'] = np.mean(confidence_ratings)
        metrics['mismatch_rate'] = np.sum(df[f'{llm_key}_mismatch_count']) / len(df)

        llm_metrics[llm_key] = metrics

    return pd.DataFrame(llm_metrics).T


def evaluate_responses(df: pd.DataFrame, confidence_threshold: float = 0.5,
                       add_uncovered_spans: bool = True):
    """
    Evaluates the precision, recall, and F1 score for responses in the dataframe based on provided confidence threshold.

    Args:
        df: DataFrame containing model responses
        confidence_threshold: Fallacies with a lower confidence rating will be ignored.
        add_uncovered_spans: Add None labels to gold standard and "nothing" labels to uncovered prediction spans.
                             If true, this replicates the calculations in Helwe et al. (2024).
                             However, perfomance metrics are then underestimated, see
                             https://github.com/ChadiHelwe/MAFALDA/issues/2
    """
    response_cols = [col for col in df.columns if col.endswith('_response')]
    for response_col in response_cols:
        llm_key = response_col.removesuffix('_response')

        log(f'Evaluating responses for {llm_key} ...')
        for index, row in df.iterrows():
            metrics = _evaluate_response(row['text'], row['labels'], row[response_col],
                                         confidence_threshold, add_uncovered_spans)
            for metric in metrics:
                df.at[index, f'{llm_key}_{metric}'] = metrics[metric]

        mismatch_count_col = f'{llm_key}_mismatch_count'
        df[mismatch_count_col] = df[mismatch_count_col].astype('UInt16')


def _evaluate_response(text: str, labels: list[list], fallacy_response: FallacyResponse,
                       confidence_threshold: float, add_uncovered_spans: bool) -> dict:
    """
    Use the metrics by Helwe et al. (2024) to evaluate a single FallacyResponse.
    """
    # Create gold standard annotations
    if add_uncovered_spans:
        # Add None label spans to gold standard annotations for uncovered spans
        gold_annotations_l2 = build_ground_truth_spans(text, labels)
    else:
        gold_spans = []
        for label_span in labels:
            start, end, label = tuple(label_span)
            if label == 'to clean':
                continue
            span = GroundTruthSpan(text[start:end], {LEVEL_2_NUMERIC[label.lower()]}, [start, end])
            gold_spans.append(span)
        gold_annotations_l2 = AnnotatedText(gold_spans)

    # Create prediction annotations
    # While Helwe et al. (2024) predict a label for each sentence with a separate prompt, we use a single prompt to
    # predict all fallacy spans and types in the given text using a single prompt.
    pred_spans: list[PredictionSpan] = []
    covered_spans: list[Span] = []
    mismatch_count = 0
    for entry in fallacy_response.fallacies:
        if entry.confidence < confidence_threshold:
            continue
        start, end = _fuzzy_match(entry.span, text)
        if start is not None and end is not None:
            label = LEVEL_2_NUMERIC[FALLACY_2_LABEL[entry.fallacy.value]]
            pred_spans.append(PredictionSpan(entry.span, label, [start, end]))
            covered_spans.append((start, end))
        else:
            mismatch_count += 1
            print(f"Warning: failed to match span for fallacy {entry.fallacy}:\nspan: {entry.span}\ntext: {text}")

    if add_uncovered_spans:
        # Add 'nothing' predictions for uncovered text spans
        for span in _get_uncovered_spans(covered_spans, len(text)):
            pred_spans.append(PredictionSpan(text[span[0]:span[1]], 0, [span[0], span[1]]))

    pred_annotations_l2 = AnnotatedText(pred_spans)

    pred_annotations_l1, gold_annotations_l1 = _get_level_annotations(pred_annotations_l2, gold_annotations_l2, level=1)
    pred_annotations_l0, gold_annotations_l0 = _get_level_annotations(pred_annotations_l2, gold_annotations_l2, level=0)
    
    p_l2, r_l2, f1_l2 = text_full_task_p_r_f1(pred_annotations_l2, gold_annotations_l2)
    p_l1, r_l1, f1_l1 = text_full_task_p_r_f1(pred_annotations_l1, gold_annotations_l1)
    p_l0, r_l0, f1_l0 = text_full_task_p_r_f1(pred_annotations_l0, gold_annotations_l0)
    
    return {
        'p_l2': p_l2, 'r_l2': r_l2, 'f1_l2': f1_l2,
        'p_l1': p_l1, 'r_l1': r_l1, 'f1_l1': f1_l1,
        'p_l0': p_l0, 'r_l0': r_l0, 'f1_l0': f1_l0,
        'mismatch_count': mismatch_count
    }

def _fuzzy_match(span: str, text: str) -> tuple[int | None, int | None]:
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

def _get_uncovered_spans(covered_spans: list[Span], text_length: int) -> list[Span]:
    """
    Returns spans of text that are not covered by given spans.
    """
    # Sort spans by start index to process them in order
    sorted_spans = sorted(covered_spans)

    # Initialize result list and current position
    uncovered_spans = []
    current_pos = 0

    # Process each span
    for start, end in sorted_spans:
        # If there's a gap before current span, add it to results
        if current_pos < start - 1:
            uncovered_spans.append((current_pos, start - 1))

        # Update current position to end of current span
        current_pos = max(current_pos, end + 1)

    # Add final span if there's remaining text
    if current_pos < text_length:
        uncovered_spans.append((current_pos, text_length))

    return uncovered_spans


def _get_level_annotations(pred_annotations_l2: AnnotatedText, gold_annotations_l2: AnnotatedText, level: int) \
        -> tuple[AnnotatedText, AnnotatedText]:
    """
    Returns annotations for the given level.
    """
    assert level in [0, 1], 'Invalid level'
    pred_annotations = deepcopy(pred_annotations_l2)
    gold_annotations = deepcopy(gold_annotations_l2)

    pred_span: PredictionSpan
    for pred_span in pred_annotations.spans:
        pred_span.label = _get_level_label(pred_span.label, level)
        
    gold_span: GroundTruthSpan
    for gold_span in gold_annotations.spans:
        gold_span.labels = {_get_level_label(label_l2, level) for label_l2 in gold_span.labels}


    return pred_annotations, gold_annotations

def _get_level_label(label_level_2: int | None, level: int | None) -> int | None:
    """
    Returns the label for the given level.
    """
    assert level in [0, 1], 'Invalid level'
    assert label_level_2 is None or 0 <= label_level_2 <= 24, 'Invalid label level 2'

    # None labels mark uncovered spans in gold standard annotations. It's unclear why None labels are converted to
    # None on level 1, but to 0 on level 0. However, we follow the implementation in evaluate.py by Helwe et al. (2024).
    # If no uncovered text spans are added, the inconsistency doesn't matter, since there are no None labels.
    if level == 1:
        return None if label_level_2 is None else LEVEL_2_TO_1[label_level_2]
    elif level == 0:
        return 1 if label_level_2 is not None and (0 < label_level_2 < 24) else 0

