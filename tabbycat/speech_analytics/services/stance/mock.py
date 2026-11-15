"""Mock stance classifier for fast, deterministic unit testing without PyTorch dependencies."""

import re
from typing import Any, Callable, Dict, List, Optional

from .base import BaseStanceClassifier, StancePrediction


class MockStanceClassifier(BaseStanceClassifier):
    """Deterministic mock stance classifier for tests."""

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        margin_threshold: float = 0.15,
        hypothesis_transform: Optional[Callable[[str], str]] = None,
        model_version: str = "mock-deberta-stance-v1",
    ):
        super().__init__(
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            hypothesis_transform=hypothesis_transform,
        )
        self._version = model_version

    @property
    def model_version(self) -> str:
        return self._version

    def predict_one(
        self,
        motion: str,
        claim_text: str,
        sentence_id: str = "s_0001",
        evidence_reference: Optional[Dict[str, Any]] = None,
    ) -> StancePrediction:
        """Heuristic mock prediction based on keyword presence."""
        hypothesis = self.hypothesis_transform(motion)
        clean_text = claim_text.lower()

        # Deterministic keyword simulation
        if any(w in clean_text for w in ["support", "propose", "benefits", "effective", "accelerate", "vital"]):
            p_con = 0.05
            p_pro = 0.88
            p_neu = 0.07
        elif any(w in clean_text for w in ["oppose", "harm", "dangerous", "fail", "kill", "disaster", "flawed"]):
            p_con = 0.85
            p_pro = 0.08
            p_neu = 0.07
        elif any(w in clean_text for w in ["neutral", "unclear", "maybe", "both"]):
            p_con = 0.20
            p_pro = 0.25
            p_neu = 0.55
        elif any(w in clean_text for w in ["close", "tied"]):
            p_con = 0.48
            p_pro = 0.46
            p_neu = 0.06
        else:
            # Default balanced distribution
            p_con = 0.33
            p_pro = 0.34
            p_neu = 0.33

        stance, status = self.evaluate_uncertainty(p_con, p_pro, p_neu)

        return StancePrediction(
            sentence_id=sentence_id,
            claim_text=claim_text,
            motion=motion,
            motion_hypothesis=hypothesis,
            predicted_stance=stance,
            contradiction_probability=round(p_con, 4),
            entailment_probability=round(p_pro, 4),
            neutral_probability=round(p_neu, 4),
            uncertainty_status=status,
            model_version=self.model_version,
            evidence_reference=evidence_reference,
        )

    def predict_batch(
        self,
        motion: str,
        items: List[Dict[str, Any]],
    ) -> List[StancePrediction]:
        results: List[StancePrediction] = []
        for it in items:
            c_text = it.get('claim_text') or it.get('text', '')
            s_id = it.get('sentence_id') or it.get('id', 's_0001')
            ev = it.get('evidence_reference') or it.get('evidence')
            results.append(self.predict_one(motion, c_text, sentence_id=s_id, evidence_reference=ev))
        return results
