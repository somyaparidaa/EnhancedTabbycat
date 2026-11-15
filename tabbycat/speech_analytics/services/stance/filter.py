"""Heuristic procedural sentence filter for debate transcripts.

NOTE / LIMITATION:
This is a rule-based heuristic filter designed to filter out common debate procedural
utterances (salutations, floor management, POI interjections, time queries, roadmap signposting).
It is NOT a trained claim extractor or argument mining model.
It does NOT claim that all filtered sentences are non-claims, nor that all passed
sentences are substantive claims.
"""

import re
from typing import Tuple

# Common debate salutations and house formalities
_SALUTATION_PATTERNS = [
    r'^(good\s+(morning|afternoon|evening)|hello|hi|welcome)\b',
    r'^(madam|mr\.?|mister)\s+(speaker|president|chair|deputy)\b',
    r'^(members|ladies\s+and\s+gentlemen)\s+of\s+the\s+house\b',
    r'^(thank\s+you|thanks)(,\s*(madam|mr\.?|chair))?\b',
]

# POI offers, rejections, acceptances
_POI_PATTERNS = [
    r'^(point\s+of\s+(information|order|clarification|personal\s+privilege))\b',
    r'^(on\s+that\s+point|on\s+this\s+point)\b',
    r'^(yes,?\s*(please|go\s+ahead|opening|closing)|no\s+thank\s+you|rejected|sit\s+down|not\s+now)\b',
    r'^(i\s+will\s+take\s+(a\s+point|your\s+point|closing|opening))\b',
]

# Pacing, time checks, floor yielding, procedural closures
_PROCEDURAL_PATTERNS = [
    r'^(can\s+you\s+hear\s+me|am\s+i\s+audible|is\s+my\s+mic\s+working)\b',
    r'^(how\s+much\s+time\s+do\s+i\s+have|what\s+is\s+my\s+time|time\s+check)\b',
    r'^(starting\s+my\s+timer\s+in|timer\s+starts\s+in|time\s+starts\s+now)\b',
    r'^(with\s+that,?\s*i\s+(yield|conclude|beg\s+to\s+propose|beg\s+to\s+oppose|proudly\s+propose|proudly\s+oppose))\b',
    r'^(i\s+yield\s+my\s+remaining\s+time|that\s+concludes\s+my\s+speech|thank\s+you\s+very\s+much)\b',
]

# Roadmap / signposting without substantive reasoning
_SIGNPOST_PATTERNS = [
    r'^(in\s+this\s+speech,?\s*i\s+will\s+(have|present|cover)\s+(two|three|four|\d+)\s+(points|arguments|contentions|clashes))\b',
    r'^(first,?\s*rebuttal;?\s*second,?\s*substantive)\b',
    r'^(moving\s+on\s+to\s+my\s+(first|second|third|next|final)\s+(point|argument|contention|clash))\b',
    r'^(now\s+to\s+my\s+(first|second|third|next|final)\s+(point|argument))\b',
]

_ALL_HEURISTIC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (_SALUTATION_PATTERNS + _POI_PATTERNS + _PROCEDURAL_PATTERNS + _SIGNPOST_PATTERNS)
]


def is_procedural_sentence(text: str, min_word_count: int = 4) -> Tuple[bool, str]:
    """Evaluate whether a sentence should be skipped as procedural or non-claim.

    Args:
        text: Sentence text string.
        min_word_count: Sentences shorter than this are skipped as fragments/noise.

    Returns:
        Tuple of (is_procedural: bool, reason: str)
    """
    clean_text = text.strip()
    if not clean_text:
        return True, "EMPTY_SENTENCE"

    for pattern in _ALL_HEURISTIC_PATTERNS:
        if pattern.search(clean_text):
            return True, "MATCHED_PROCEDURAL_HEURISTIC"

    words = re.findall(r'\b\w+\b', clean_text)
    if len(words) < min_word_count:
        return True, "TOO_SHORT_FRAGMENT"

    return False, "ELIGIBLE_CANDIDATE_CLAIM"
