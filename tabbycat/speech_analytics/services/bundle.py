"""Debate analysis input bundle builder and validator.

Combines preprocessed transcript sentences, paragraphs, and structured debate context
into a versioned, serializable, and reproducible data contract ready for downstream
NLP analysis (e.g. Hugging Face models, LLMs, or heuristic classifiers).
"""

import datetime
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from .context_builder import DebateContextBuilder
from .preprocessor import TranscriptPreprocessor

logger = logging.getLogger(__name__)

CURRENT_INPUT_CONTRACT_VERSION = "1.0.0"


class DebateAnalysisInputBundle:
    """Builds and validates the complete input contract for debate speech analysis."""

    def __init__(
        self,
        contract_version: str = CURRENT_INPUT_CONTRACT_VERSION,
        paragraph_pause_threshold: float = 2.5,
    ):
        self.contract_version = contract_version
        self.preprocessor = TranscriptPreprocessor(paragraph_pause_threshold=paragraph_pause_threshold)
        self.context_builder = DebateContextBuilder()

    def build(
        self,
        transcript: Any,
        recording: Optional[Any] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct the full analysis input bundle from a SpeechTranscript model or dict.

        Args:
            transcript: SpeechTranscript model instance or dictionary with text/metadata.
            recording: Optional SpeechRecording model instance (if transcript is a dict or unlinked).
            custom_metadata: Optional supplementary metadata to include.

        Returns:
            A deterministic, JSON-serializable dictionary adhering to the version 1.0.0 input contract.
        """
        # Resolve recording and debate objects
        rec_obj = recording or getattr(transcript, 'recording', None)
        debate_obj = getattr(rec_obj, 'debate', None) if rec_obj else None
        target_speaker_id = getattr(rec_obj, 'speaker_id', None) if rec_obj else None

        # 1. Preprocess transcript into sentences and paragraphs
        processed_transcript = self.preprocessor.preprocess(transcript)

        # Diarized speakers and speaker metrics from preprocessed transcript
        diarized_speakers = sorted(list(set(
            s['speaker'] for s in processed_transcript['sentences']
            if s.get('speaker') and s['speaker'] != "Unassigned"
        )))

        speaker_stats: Dict[str, Dict[str, Any]] = {}
        for s in processed_transcript['sentences']:
            spk = s.get('speaker')
            if not spk or spk == "Unassigned":
                continue
            if spk not in speaker_stats:
                speaker_stats[spk] = {'word_count': 0, 'duration': 0.0}
            speaker_stats[spk]['word_count'] += s.get('word_count', 0)
            if s.get('start_time') is not None and s.get('end_time') is not None:
                dur = max(0.0, s['end_time'] - s['start_time'])
                speaker_stats[spk]['duration'] += dur

        # 2. Extract Tabbycat debate context
        if debate_obj:
            debate_context = self.context_builder.build_context(
                debate=debate_obj,
                target_speaker_id=target_speaker_id,
                diarized_speakers=diarized_speakers,
                speaker_stats=speaker_stats,
            )
        else:
            debate_context = {
                'tournament': None,
                'round': None,
                'debate': None,
                'motion': None,
                'teams': [],
                'speeches': [],
                'speaker_mapping': {
                    'status': 'NO_DEBATE_CONTEXT',
                    'target_speaker': None,
                    'diarized_speakers': diarized_speakers,
                    'unresolved_speakers': diarized_speakers,
                    'mapping': {},
                },
            }

        # 3. Assess context completeness
        warnings: List[str] = list(processed_transcript.get('warnings', []))

        context_completeness = "isolated_speech"
        if not debate_obj:
            context_completeness = "isolated_speech"
            warnings.append("No debate or tournament context linked to transcript.")
        elif not debate_context.get('motion') or not debate_context['motion'].get('primary_motion'):
            warnings.append("No motion text found for this debate round.")
        elif debate_context['speaker_mapping']['status'] == "UNRESOLVED_MULTI_SPEAKER":
            warnings.append("Multi-speaker debate recording with unassigned individual speaker roles.")
        elif debate_context['speaker_mapping']['status'] == "RESOLVED_TARGET_WITH_POIS":
            warnings.append("Multiple diarized voices detected; primary target speaker separated from POIs/interjections.")

        # 4. Construct metadata
        recording_id = getattr(rec_obj, 'id', None)
        transcript_id = getattr(transcript, 'id', None)

        # Generate deterministic fingerprint of input content
        content_hash = self._generate_fingerprint(
            full_text=processed_transcript['full_text'],
            sentences=processed_transcript['sentences'],
            debate_context=debate_context,
        )

        bundle: Dict[str, Any] = {
            'contract_version': self.contract_version,
            'fingerprint': content_hash,
            'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'provenance': {
                'transcript_id': transcript_id,
                'recording_id': recording_id,
                'audio_duration': processed_transcript.get('duration'),
                'provider': (transcript.metadata or {}).get('provider') if hasattr(transcript, 'metadata') and isinstance(transcript.metadata, dict) else 'unknown',
                'language_code': processed_transcript['language_code'],
                'overall_confidence': processed_transcript['overall_confidence'],
                'context_completeness': context_completeness,
            },
            'debate_context': debate_context,
            'transcript': {
                'full_text': processed_transcript['full_text'],
                'stats': processed_transcript['stats'],
                'sentences': processed_transcript['sentences'],
                'paragraphs': processed_transcript['paragraphs'],
            },
            'warnings': warnings,
        }

        if custom_metadata:
            bundle['custom_metadata'] = custom_metadata

        return bundle

    def _generate_fingerprint(
        self, full_text: str, sentences: List[Dict[str, Any]], debate_context: Dict[str, Any]
    ) -> str:
        """Create a deterministic SHA-256 fingerprint from the structured inputs.

        Uses canonical JSON serialization with sorted keys and stable separators
        to guarantee reproducibility regardless of dictionary key ordering.
        Materially includes:
        - Full transcript text
        - Normalized sentences (id, speaker, start_time, end_time, text, is_approximate)
        - Debate ID and confirmed ballot status
        - Tournament format name and format configurations
        - Primary motion text and info slide
        - Team IDs and side assignments
        - Speaker mapping status, target speaker, and resolved speech roles
        Excludes volatile values like generated_at timestamps.
        """
        canonical_payload = {
            'full_text': full_text,
            'sentences': [
                {
                    'id': s['id'],
                    'speaker': s['speaker'],
                    'start_time': s['start_time'],
                    'end_time': s['end_time'],
                    'text': s['text'],
                    'is_approximate': s.get('is_approximate', False),
                }
                for s in sentences
            ],
            'tournament': {
                'format_name': (debate_context.get('tournament') or {}).get('format_name'),
                'teams_in_debate': (debate_context.get('tournament') or {}).get('teams_in_debate'),
                'substantive_speakers': (debate_context.get('tournament') or {}).get('substantive_speakers'),
                'reply_scores_enabled': (debate_context.get('tournament') or {}).get('reply_scores_enabled'),
            },
            'debate': {
                'id': (debate_context.get('debate') or {}).get('id'),
                'confirmed_ballot_available': (debate_context.get('debate') or {}).get('confirmed_ballot_available'),
            },
            'motion': {
                'text': ((debate_context.get('motion') or {}).get('primary_motion') or {}).get('text'),
                'info_slide': ((debate_context.get('motion') or {}).get('primary_motion') or {}).get('info_slide'),
            },
            'teams': sorted([
                {
                    'team_id': tm.get('team_id'),
                    'side_code': tm.get('side_code'),
                }
                for tm in debate_context.get('teams', [])
            ], key=lambda x: str(x.get('team_id'))),
            'speaker_mapping': {
                'status': (debate_context.get('speaker_mapping') or {}).get('status'),
                'target_speaker_id': ((debate_context.get('speaker_mapping') or {}).get('target_speaker') or {}).get('id'),
            },
            'speeches': sorted([
                {
                    'position': sp.get('position'),
                    'side_code': sp.get('side_code'),
                    'role_abbr': sp.get('role_abbr'),
                    'speaker_id': sp.get('speaker_id'),
                }
                for sp in debate_context.get('speeches', [])
            ], key=lambda x: (str(x.get('side_code')), x.get('position') or 0)),
        }

        canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def build_debate_analysis_input(
    transcript: Any,
    recording: Optional[Any] = None,
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Helper functional interface to generate the analysis input bundle."""
    builder = DebateAnalysisInputBundle()
    return builder.build(
        transcript=transcript,
        recording=recording,
        custom_metadata=custom_metadata,
    )
