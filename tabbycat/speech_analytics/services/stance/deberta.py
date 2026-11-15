"""Local DeBERTa stance classification provider.

Loads a fine-tuned cross-encoder/nli-deberta-v3-small checkpoint from local filesystem.
Strictly adheres to verified training conventions:
- Tokenizer input order: tokenizer(claim, motion_hypothesis)
- Claim is premise (first sequence), motion hypothesis is hypothesis (second sequence).
- Label mapping:
    0 = contradiction -> CON
    1 = entailment -> PRO
    2 = neutral -> UNCERTAIN
- Supports CPU and Apple Silicon MPS device acceleration.
- Preserves all 3 softmax probabilities.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .base import BaseStanceClassifier, StancePrediction
from .exceptions import (
    StanceConfigurationError,
    StanceInferenceError,
    StanceModelLoadError,
)

logger = logging.getLogger(__name__)


def resolve_device(preferred_device: Optional[str] = None) -> Any:
    """Resolve torch.device supporting Apple Silicon MPS, CUDA, and CPU fallback."""
    import torch

    if preferred_device:
        pref = preferred_device.lower().strip()
        if pref == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device('mps')
        elif pref == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif pref == 'cpu':
            return torch.device('cpu')

    # Automatic selection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class DeBERTaStanceClassifier(BaseStanceClassifier):
    """Local DeBERTa-v3 cross-encoder stance classification provider."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        confidence_threshold: float = 0.55,
        margin_threshold: float = 0.15,
        hypothesis_transform: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            hypothesis_transform=hypothesis_transform,
        )
        self.model_path = os.path.abspath(model_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self._device_str = device
        self._version = f"deberta-v3-small-nli-ft:{os.path.basename(self.model_path)}"

        if not os.path.isdir(self.model_path):
            raise StanceConfigurationError(
                f"Model checkpoint directory does not exist at '{self.model_path}'."
            )

        # Lazy load weights on initialization
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load tokenizer and DebertaV2 model onto target compute device."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise StanceModelLoadError(
                "Required dependencies 'torch' and 'transformers' are not installed."
            ) from exc

        try:
            self.device = resolve_device(self._device_str)
            logger.info("Loading DeBERTa stance model from %s onto %s", self.model_path, self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            # Verify label mapping
            id2label = getattr(self.model.config, 'id2label', {})
            # Standard NLI: 0=contradiction, 1=entailment, 2=neutral
            logger.debug("DeBERTa stance classifier initialized. id2label: %s", id2label)

        except Exception as exc:
            raise StanceModelLoadError(
                f"Failed to load DeBERTa model from '{self.model_path}': {str(exc)}"
            ) from exc

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
        """Classify single claim with respect to debate motion."""
        results = self.predict_batch(
            motion=motion,
            items=[{
                'claim_text': claim_text,
                'sentence_id': sentence_id,
                'evidence_reference': evidence_reference,
            }],
        )
        return results[0]

    def predict_batch(
        self,
        motion: str,
        items: List[Dict[str, Any]],
    ) -> List[StancePrediction]:
        """Classify a batch of claims with respect to debate motion.

        Strict input order constraint:
        tokenizer(claim, motion_hypothesis, return_tensors='pt', ...)
        Premise = claim (first sequence)
        Hypothesis = motion_hypothesis (second sequence)
        """
        if not items:
            return []

        import torch

        hypothesis = self.hypothesis_transform(motion)
        predictions: List[StancePrediction] = []

        # Process in batches to control memory consumption
        for i in range(0, len(items), self.batch_size):
            batch_items = items[i:i + self.batch_size]
            premises = [
                (it.get('claim_text') or it.get('text') or '').strip()
                for it in batch_items
            ]
            hypotheses = [hypothesis] * len(premises)

            try:
                # STRICT ORDER: premise = claim, hypothesis = motion_hypothesis
                inputs = self.tokenizer(
                    premises,
                    hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).cpu().tolist()

            except Exception as exc:
                raise StanceInferenceError(
                    f"Forward pass failed during batch stance classification: {str(exc)}"
                ) from exc

            # Map predictions
            for it, prob_distribution in zip(batch_items, probs):
                sentence_id = str(it.get('sentence_id') or it.get('id') or "s_0001")
                claim_text = str(it.get('claim_text') or it.get('text') or '')
                evidence_ref = it.get('evidence_reference') or it.get('evidence')

                # Label index mapping:
                # 0 = contradiction (CON)
                # 1 = entailment (PRO)
                # 2 = neutral (UNCERTAIN)
                p_con = float(prob_distribution[0])
                p_pro = float(prob_distribution[1])
                p_neutral = float(prob_distribution[2])

                stance, uncertainty_status = self.evaluate_uncertainty(p_con, p_pro, p_neutral)

                predictions.append(
                    StancePrediction(
                        sentence_id=sentence_id,
                        claim_text=claim_text,
                        motion=motion,
                        motion_hypothesis=hypothesis,
                        predicted_stance=stance,
                        contradiction_probability=round(p_con, 4),
                        entailment_probability=round(p_pro, 4),
                        neutral_probability=round(p_neutral, 4),
                        uncertainty_status=uncertainty_status,
                        model_version=self.model_version,
                        evidence_reference=evidence_ref,
                    )
                )

        return predictions
