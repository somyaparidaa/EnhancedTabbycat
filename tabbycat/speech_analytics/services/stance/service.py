"""High-level stance classification service orchestrating Stage 2B debate transcripts with DeBERTa.

Integrates:
1. DebateAnalysisInput (Stage 2B contract)
2. Procedural sentence filtering (heuristic)
3. DeBERTa topic-conditioned cross-encoder (premise=claim, hypothesis=motion_hypothesis)
4. EvidenceReference preservation with verbatim quotes and timestamps
5. Three-way probability preservation (CON, PRO, UNCERTAIN)
6. Robust handling of missing motions, unresolved speakers, and timing estimates
"""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseStanceClassifier, StancePrediction
from .exceptions import StanceConfigurationError
from .factory import get_stance_classifier
from .filter import is_procedural_sentence

logger = logging.getLogger(__name__)

STANCE_ANALYSIS_SCHEMA_VERSION = "1.0.0"


def classify_transcript_stance(
    analysis_bundle: Dict[str, Any],
    classifier: Optional[BaseStanceClassifier] = None,
    custom_motion: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify topic-conditioned stance for all eligible sentences in a Stage 2B analysis bundle.

    Args:
        analysis_bundle: The DebateAnalysisInput dictionary (from transcript.get_analysis_input() or build_debate_analysis_input()).
        classifier: Optional StanceClassifier instance (defaults to get_stance_classifier()).
        custom_motion: Optional motion string overriding the bundle's motion.

    Returns:
        Structured dictionary adhering to stance_classification_v1 schema containing:
        - schema_version
        - motion
        - model_version
        - predictions (list of sentence-level stance prediction dictionaries)
        - stats
        - warnings
    """
    if not isinstance(analysis_bundle, dict):
        raise ValueError(f"analysis_bundle must be a dictionary, got {type(analysis_bundle).__name__}")

    warnings: List[str] = []

    # 1. Resolve Motion
    motion_text = custom_motion
    if not motion_text:
        debate_ctx = analysis_bundle.get('debate_context') or {}
        motion_obj = debate_ctx.get('motion') or {}
        primary_motion = motion_obj.get('primary_motion') or {}
        motion_text = (primary_motion.get('text') or '').strip()

    if not motion_text:
        warnings.append("No primary motion text found in debate context; stance inference cannot be performed.")
        return {
            'schema_version': STANCE_ANALYSIS_SCHEMA_VERSION,
            'motion': "",
            'model_version': "none",
            'predictions': [],
            'stats': {
                'total_sentences': len((analysis_bundle.get('transcript') or {}).get('sentences', [])),
                'total_evaluated': 0,
                'total_pro': 0,
                'total_con': 0,
                'total_uncertain': 0,
                'total_skipped_procedural': 0,
            },
            'warnings': warnings,
        }

    # 2. Resolve Classifier
    clf = classifier or get_stance_classifier()
    model_version = clf.model_version

    # 3. Extract sentences and metadata
    transcript_data = analysis_bundle.get('transcript') or {}
    sentences = transcript_data.get('sentences', [])

    # Check speaker mapping context
    speaker_mapping_ctx = (analysis_bundle.get('debate_context') or {}).get('speaker_mapping') or {}
    mapping_status = speaker_mapping_ctx.get('status', 'UNKNOWN')
    mapping_dict = speaker_mapping_ctx.get('mapping') or {}

    if mapping_status in ('UNRESOLVED_MULTI_SPEAKER', 'DEBATE_ONLY_UNASSIGNED'):
        warnings.append(
            f"Speaker mapping is {mapping_status}; sentence-level stances are evaluated without individual debater attribution."
        )

    # 4. Filter Procedural Sentences vs Candidate Claims
    candidate_items: List[Dict[str, Any]] = []
    sentence_records: List[Dict[str, Any]] = []

    for s in sentences:
        s_id = s.get('id') or s.get('sentence_id') or 's_0000'
        s_text = (s.get('text') or '').strip()
        speaker_label = s.get('speaker') or s.get('speaker_identifier') or "Unassigned"
        start_time = s.get('start_time')
        end_time = s.get('end_time')
        is_approx = s.get('is_approximate', False)

        # Build standard EvidenceReference preserving exact quote, timing, speaker
        speaker_meta = mapping_dict.get(speaker_label, {})
        speech_role = speaker_meta.get('role')

        evidence_ref = {
            'source': "target_speech",
            'speaker_identifier': speaker_label,
            'speech_role': speech_role,
            'quote': s_text,
            'start_time': start_time,
            'end_time': end_time,
            'is_approximate': is_approx,
            'observation_type': "direct_observation",
        }

        # Procedural filter
        is_proc, reason = is_procedural_sentence(s_text)

        rec = {
            'sentence_id': s_id,
            'text': s_text,
            'speaker': speaker_label,
            'is_procedural': is_proc,
            'skip_reason': reason,
            'evidence_reference': evidence_ref,
        }
        sentence_records.append(rec)

        if not is_proc:
            candidate_items.append({
                'sentence_id': s_id,
                'claim_text': s_text,
                'evidence_reference': evidence_ref,
            })

    # 5. Execute Batch Inference for Candidate Claims
    predictions_map: Dict[str, StancePrediction] = {}
    if candidate_items:
        batch_predictions = clf.predict_batch(motion=motion_text, items=candidate_items)
        for pred in batch_predictions:
            predictions_map[pred.sentence_id] = pred

    # 6. Format Consolidated Predictions List
    all_predictions: List[Dict[str, Any]] = []
    total_pro = 0
    total_con = 0
    total_uncertain = 0
    total_skipped = 0

    for rec in sentence_records:
        s_id = rec['sentence_id']
        s_text = rec['text']
        ev_ref = rec['evidence_reference']

        if rec['is_procedural']:
            total_skipped += 1
            all_predictions.append({
                'sentence_id': s_id,
                'sentence_text': s_text,
                'motion': motion_text,
                'motion_hypothesis': clf.hypothesis_transform(motion_text),
                'predicted_stance': "NOT_APPLICABLE",
                'contradiction_probability': None,
                'entailment_probability': None,
                'neutral_probability': None,
                'uncertainty_status': "SKIPPED_PROCEDURAL",
                'filter_reason': rec['skip_reason'],
                'model_version': model_version,
                'evidence_reference': ev_ref,
            })
        else:
            pred = predictions_map.get(s_id)
            if pred:
                if pred.predicted_stance == "PRO":
                    total_pro += 1
                elif pred.predicted_stance == "CON":
                    total_con += 1
                else:
                    total_uncertain += 1

                pred_dict = pred.to_dict()
                pred_dict['sentence_text'] = s_text
                pred_dict['filter_reason'] = "ELIGIBLE_CANDIDATE_CLAIM"
                all_predictions.append(pred_dict)

    total_evaluated = len(candidate_items)

    return {
        'schema_version': STANCE_ANALYSIS_SCHEMA_VERSION,
        'motion': motion_text,
        'model_version': model_version,
        'predictions': all_predictions,
        'stats': {
            'total_sentences': len(sentences),
            'total_evaluated': total_evaluated,
            'total_pro': total_pro,
            'total_con': total_con,
            'total_uncertain': total_uncertain,
            'total_skipped_procedural': total_skipped,
        },
        'warnings': warnings,
    }
