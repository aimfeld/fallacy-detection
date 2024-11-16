"""
Code copied from https://github.com/ChadiHelwe/MAFALDA/blob/0df434477b914a20f55c0592ba05a53fe924c65b/src/evaluate.py
to ensure exact methodology for calculating performance metrics. Unused code parts have been removed.
"""
from collections import OrderedDict
from typing import Any
from .new_metrics import AnnotatedText, GroundTruthSpan


KEWORDS_LEVEL_1_NUMERIC = {
    "emotion": 1,
    "logic": 2,
    "credibility": 3,
}

LEVEL_2_TO_LEVEL_1 = {
    "nothing": 0,
    "appeal to positive emotion": 1,
    "appeal to anger": 1,
    "appeal to fear": 1,
    "appeal to pity": 1,
    "appeal to ridicule": 1,
    "appeal to worse problems": 1,
    "causal oversimplification": 2,
    "circular reasoning": 2,
    "equivocation": 2,
    "false analogy": 2,
    "false causality": 2,
    "false dilemma": 2,
    "hasty generalization": 2,
    "slippery slope": 2,
    "straw man": 2,
    "fallacy of division": 2,
    "ad hominem": 3,
    "ad populum": 3,
    "appeal to (false) authority": 3,
    "appeal to nature": 3,
    "appeal to tradition": 3,
    "guilt by association": 3,
    "tu quoque": 3,
}


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


NUMERIC_TO_LEVEL_2 = {
    0: "nothing",
    1: "appeal to positive emotion",
    2: "appeal to anger",
    3: "appeal to fear",
    4: "appeal to pity",
    5: "appeal to ridicule",
    6: "appeal to worse problems",
    7: "causal oversimplification",
    8: "circular reasoning",
    9: "equivocation",
    10: "false analogy",
    11: "false causality",
    12: "false dilemma",
    13: "hasty generalization",
    14: "slippery slope",
    15: "straw man",
    16: "fallacy of division",
    17: "ad hominem",
    18: "ad populum",
    19: "appeal to (false) authority",
    20: "appeal to nature",
    21: "appeal to tradition",
    22: "guilt by association",
    23: "tu quoque",
    24: "unknown",
}

KEYWORDS_LEVEL_2_NUMERIC = {
    "emotion": 1,
    "anger": 2,
    "fear": 3,
    "pity": 4,
    "ridicule": 5,
    "worse": 6,
    "problems": 6,
    "oversimplification": 7,
    "circular": 8,
    "equivocation": 9,
    "analogy": 10,
    "causality": 11,
    "dilemma": 12,
    "generalization": 13,
    "slippery": 14,
    "slope": 14,
    "straw": 15,
    "division": 16,
    "hominem": 17,
    "populum": 18,
    "authority": 19,
    "nature": 20,
    "tradition": 21,
    "association": 22,
    "quoque": 23,
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


FALLACIES_LEVEL_2_TO_LEVEL_1 = {
    "nothing": "nothing",
    "appeal to positive emotion": "emotion",
    "appeal to anger": "emotion",
    "appeal to fear": "emotion",
    "appeal to pity": "emotion",
    "appeal to ridicule": "emotion",
    "appeal to worse problems": "emotion",
    "causal oversimplification": "logic",
    "circular reasoning": "logic",
    "equivocation": "logic",
    "false analogy": "logic",
    "false causality": "logic",
    "false dilemma": "logic",
    "hasty generalization": "logic",
    "slippery slope": "logic",
    "straw man": "logic",
    "fallacy of division": "logic",
    "ad hominem": "credibility",
    "ad populum": "credibility",
    "appeal to (false) authority": "credibility",
    "appeal to nature": "credibility",
    "appeal to tradition": "credibility",
    "guilt by association": "credibility",
    "tu quoque": "credibility",
    "unknown": "unknown",
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
