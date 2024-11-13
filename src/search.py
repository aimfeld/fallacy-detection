"""
This module contains models and functions for fallacy search.
"""

from pydantic import BaseModel, Field
from typing import List


class Fallacy(BaseModel):
    fallacy: str = Field(description="The identified fallacy.")
    span: str = Field(
        description="The verbatim text span where the fallacy occurs, consisting of one or more contiguous sentences.")
    reason: str = Field(description="An explanation why the text span contains this fallacy.")


class FallacyResponse(BaseModel):
    fallacies: List[Fallacy] = Field(default_factory=list, title="The list of fallacies found in the text.")