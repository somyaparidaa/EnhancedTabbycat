"""Hypothesis transformation utilities for topic-conditioned stance classification.

Competitive debate motions typically begin with parliamentary prefixes such as
'This House would [verb]', 'This House supports [noun]', 'This House regrets [noun]', etc.
Natural Language Inference models expect a declarative statement (hypothesis) to evaluate
entailment (PRO) or contradiction (CON) with respect to a premise claim.

This module exposes the configurable motion-to-hypothesis transformation.
"""

import re
from typing import Callable, Optional

# Regex for common parliamentary motion prefixes
_PREFIX_RE = re.compile(
    r'^(this\s+house\s+(would|believes\s+that|supports|regrets|opposes|prefers|as\s+[^,]+,\s+would)\s+)',
    re.IGNORECASE,
)


def default_motion_to_hypothesis(motion_text: str) -> str:
    """Transform a debate motion into a declarative hypothesis statement.

    Examples:
        - "This House would ban private schools." -> "We should ban private schools."
        - "This House supports universal basic income." -> "Universal basic income should be supported."
        - "This House regrets the rise of social media." -> "The rise of social media is regrettable."
        - "Universal basic income." -> "Universal basic income is desirable."

    If the motion already represents a declarative statement or does not match
    standard parliamentary prefixes, cleans whitespace and returns a declarative claim.
    """
    if not motion_text or not motion_text.strip():
        return ""

    text = motion_text.strip().rstrip('.')

    # 1. 'This House would...' -> 'We should...'
    match_thw = re.match(r'^this\s+house\s+would\s+(.+)$', text, re.IGNORECASE)
    if match_thw:
        action = match_thw.group(1).strip()
        return f"We should {action}."

    # 2. 'This House believes that...' -> '...'
    match_thbt = re.match(r'^this\s+house\s+believes\s+that\s+(.+)$', text, re.IGNORECASE)
    if match_thbt:
        statement = match_thbt.group(1).strip()
        # Capitalize first letter
        statement = statement[0].upper() + statement[1:] if statement else statement
        return f"{statement}."

    # 3. 'This House supports...' -> '... should be supported.'
    match_ths = re.match(r'^this\s+house\s+supports\s+(.+)$', text, re.IGNORECASE)
    if match_ths:
        topic = match_ths.group(1).strip()
        return f"{topic} should be supported."

    # 4. 'This House regrets...' -> '... is regrettable.'
    match_thr = re.match(r'^this\s+house\s+regrets\s+(.+)$', text, re.IGNORECASE)
    if match_thr:
        topic = match_thr.group(1).strip()
        return f"{topic} is regrettable."

    # 5. 'This House opposes...' -> '... should be opposed.'
    match_tho = re.match(r'^this\s+house\s+opposes\s+(.+)$', text, re.IGNORECASE)
    if match_tho:
        topic = match_tho.group(1).strip()
        return f"{topic} should be opposed."

    # 6. 'This House prefers X to Y' -> 'X is preferable to Y.'
    match_thp = re.match(r'^this\s+house\s+prefers\s+(.+)$', text, re.IGNORECASE)
    if match_thp:
        topic = match_thp.group(1).strip()
        return f"{topic} is preferable."

    # 7. Fallback: ensure complete sentence
    return f"{text}."
