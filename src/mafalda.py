"""
This module functions for dealing with the MAFALDA dataset by Helwe et al. (2024).
"""
import pandas as pd
import json
from .utils import log
from .constants import RESPONSE_ERROR
from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class FallacyType(Enum):
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
    ABUSIVE_AD_HOMINEM = "Abusive Ad Hominem"
    AD_POPULUM = "Ad Populum"
    APPEAL_TO_AUTHORITY = "Appeal to Authority"
    APPEAL_TO_NATURE = "Appeal to Nature"
    APPEAL_TO_TRADITION = "Appeal to Tradition"
    GUILT_BY_ASSOCIATION = "Guilt by Association"
    TU_QUOQUE = "Tu Quoque"


class Fallacy(BaseModel):
    """
    A fallacy found in the MAFALDA dataset, spanning one or more sentences.
    """
    fallacy: FallacyType = Field(description="The identified fallacy.")
    span: str = Field(
        description="The verbatim text span where the fallacy occurs, consisting of one or more contiguous sentences.")
    reason: str = Field(description="An explanation why the text span contains this fallacy.")


class FallacyResponse(BaseModel):
    """
    A response from the LLMs for a given input text.
    """
    fallacies: List[Fallacy] = Field(default_factory=list, title="The list of fallacies found in the text.")


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

    response_cols = [col for col in df.columns if col.endswith('_response')]
    for col in response_cols:
        df[col] = df[col].apply(lambda x: x if x in ['', RESPONSE_ERROR] else FallacyResponse.model_validate_json(x))

    return df


def get_mafalda_fallacies_df() -> pd.DataFrame:
    return pd.read_csv('datasets/MAFALDA/mafalda_fallacies.csv')


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