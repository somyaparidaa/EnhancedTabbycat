"""Base interface and data contracts for topic-conditioned stance classification."""

import abc
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hypothesis import default_motion_to_hypothesis


@dataclass(frozen=True)
class StancePrediction:
    """Represents a single topic-conditioned stance classification result.

    Preserves verified training conventions:
    - Label mapping:
        0 = contradiction -> CON
        1 = entailment -> PRO
        2 = neutral -> UNCERTAIN
    - Preserves all three softmax probabilities.
    - Exposes uncertainty status without discarding raw probabilities.
    """
    sentence_id: str
    claim_text: str
    motion: str
    motion_hypothesis: str
    predicted_stance: str             # "PRO", "CON", "UNCERTAIN"
    contradiction_probability: float  # P(CON)
    entailment_probability: float     # P(PRO)
    neutral_probability: float        # P(UNCERTAIN)
    uncertainty_status: str           # "CONFIDENT", "LOW_MARGIN", "BELOW_THRESHOLD", "DOMINANT_NEUTRAL"
    model_version: str
    evidence_reference: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable dictionary."""
        return asdict(self)


class BaseStanceClassifier(abc.ABC):
    """Abstract interface for debate topic-conditioned stance classification."""

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        margin_threshold: float = 0.15,
        hypothesis_transform: Optional[Callable[[str], str]] = None,
    ):
        """
        Args:
            confidence_threshold: Minimum probability for PRO or CON to be considered confident.
            margin_threshold: Minimum margin between top-1 and runner-up probability.
            hypothesis_transform: Optional function converting motion to hypothesis string.
        """
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.hypothesis_transform = hypothesis_transform or default_motion_to_hypothesis

    @property
    @abc.abstractmethod
    def model_version(self) -> str:
        """Identifier/version tag of the underlying model."""
        pass

    @abc.abstractmethod
    def predict_one(
        self,
        motion: str,
        claim_text: str,
        sentence_id: str = "s_0001",
        evidence_reference: Optional[Dict[str, Any]] = None,
    ) -> StancePrediction:
        """Classify stance of a single claim with respect to a motion.

        Strict input order constraint:
        Premise = claim_text
        Hypothesis = to_hypothesis(motion)
        """
        pass

    @abc.abstractmethod
    def predict_batch(
        self,
        motion: str,
        items: List[Dict[str, Any]],
    ) -> List[StancePrediction]:
        """Classify a batch of claims with respect to a motion.

        Each item in `items` is a dict with:
        - 'claim_text' or 'text': str
        - 'sentence_id' or 'id': str
        - 'evidence_reference' (optional): dict
        """
        pass

    def evaluate_uncertainty(
        self, p_con: float, p_pro: float, p_neutral: float
    ) -> Tuple[str, str]:
        """Determine stance label and uncertainty status from 3-way softmax probabilities.

        Returns:
            Tuple of (predicted_stance, uncertainty_status)
        """
        # Top-1 label check among all 3 classes
        probs = [
            ("CON", p_con),
            ("PRO", p_pro),
            ("UNCERTAIN", p_neutral),
        ]
        sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)
        top_label, top_prob = sorted_probs[0]
        second_label, second_prob = sorted_probs[1]

        # 1. If neutral is the highest probability class
        if top_label == "UNCERTAIN":
            return "UNCERTAIN", "DOMINANT_NEUTRAL"

        # 2. Check margin between top class and second class
        margin = top_prob - second_prob
        if margin < self.margin_threshold:
            return top_label, "LOW_MARGIN"

        # 3. Check absolute confidence threshold
        if top_prob < self.confidence_threshold:
            return top_label, "BELOW_THRESHOLD"

        return top_label, "CONFIDENT"
