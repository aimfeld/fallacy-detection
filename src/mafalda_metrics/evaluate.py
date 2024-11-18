"""
Code copied from https://github.com/ChadiHelwe/MAFALDA/blob/0df434477b914a20f55c0592ba05a53fe924c65b/src/evaluate.py
to ensure exact methodology for calculating performance metrics. Unused code parts have been removed.
"""
from collections import OrderedDict
from typing import Any
from .new_metrics import AnnotatedText, GroundTruthSpan


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

LEVEL_2_TO_1 = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 3,
    18: 3,
    19: 3,
    20: 3,
    21: 3,
    22: 3,
    23: 3,
    24: 4,
}


def build_ground_truth_spans(text: str, labels: list[list[Any]]):
    dict_labels = OrderedDict()
    for label in labels:
        if "to clean" in label[2].lower():
            continue
        if (label[0], label[1]) not in dict_labels:
            dict_labels[(label[0], label[1])] = set([LEVEL_2_NUMERIC[label[2].lower()]])
        else:
            dict_labels[(label[0], label[1])].add(LEVEL_2_NUMERIC[label[2].lower()])

    current = 0
    end = len(text)
    uncovered_ranges = []

    # Find and store ranges of text that are not covered
    for idx in dict_labels:
        # If there is a gap before the current labeled span, add it as uncovered
        if current < idx[0]:
            uncovered_ranges.append((current, idx[0] - 1))

        # Update the current index to the end of the labeled span
        current = max(current, idx[1] + 1)

    # If there is any remaining text after the last label, add it as uncovered
    if current < end:
        uncovered_ranges.append((current, end))

    # If there were no labels at all, the entire text is uncovered
    if len(dict_labels) == 0:
        uncovered_ranges.append((0, end))

    # Add uncovered ranges to the dictionary with a None labe
    for i in uncovered_ranges:
        dict_labels[i] = set([None])

    # Construct the list of GroundTruthSpan objects
    ground_truth_spans = []

    for idx in dict_labels:
        # Create a GroundTruthSpan for each labeled and uncovered span
        ground_truth_spans.append(
            GroundTruthSpan(text[idx[0] : idx[1]], dict_labels[idx], [idx[0], idx[1]])
        )

    return AnnotatedText(ground_truth_spans)
