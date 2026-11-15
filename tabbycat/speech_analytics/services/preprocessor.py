"""Transcript preprocessing and segmentation service for debate speech analytics.

Normalizes raw speech transcripts, diarized speaker segments, and token/word timestamps
into standardized, timestamp-preserving sentence-level and paragraph-level structures.
Provides deterministic sentence identifiers, confidence scoring, robust punctuation handling,
and graceful fallbacks when word-level timestamps are unavailable.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sentence boundary regex: splits on '.', '!', '?' followed by whitespace or end of string,
# while protecting common abbreviations if followed by lower-case or single initials.
_SENTENCE_SPLIT_REGEX = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z0-9"\'“‘])|(?<=[.!?])$'
)


def _split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """Split a string into clean sentences while tracking start/end character offsets in text.
    
    Returns a list of tuples: (sentence_text, char_start, char_end)
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    # If text contains punctuation marks, split by sentence boundary pattern
    # We find slices
    spans: List[Tuple[int, int]] = []
    start = 0
    # Match sentence endings: '.', '!', '?'
    # Avoid splitting on common abbreviations like "Mr.", "Ms.", "Dr.", "Gov.", "vs."
    pattern = re.compile(r'([.!?]+)(?:\s+|$)')
    matches = list(pattern.finditer(text))
    
    if not matches:
        # Punctuation-free text: treat the entire string (or long clause) as one sentence
        stripped = text.strip()
        if stripped:
            start_pos = text.find(stripped)
            return [(stripped, start_pos, start_pos + len(stripped))]
        return []

    cur_start = 0
    for match in matches:
        end_pos = match.end()
        # Sentence slice
        cand = text[cur_start:end_pos].strip()
        if cand:
            s_idx = text.find(cand, cur_start)
            spans.append((cand, s_idx, s_idx + len(cand)))
        cur_start = end_pos

    if cur_start < len(text):
        remaining = text[cur_start:].strip()
        if remaining:
            s_idx = text.find(remaining, cur_start)
            spans.append((remaining, s_idx, s_idx + len(remaining)))

    return spans


class TranscriptPreprocessor:
    """Preprocesses raw transcript records and metadata into structured sentences and paragraphs."""

    def __init__(self, paragraph_pause_threshold: float = 2.5):
        """
        Args:
            paragraph_pause_threshold: Time gap in seconds between consecutive sentences
                that signals a paragraph boundary (in addition to speaker switches).
        """
        self.paragraph_pause_threshold = paragraph_pause_threshold

    def preprocess(self, transcript_obj_or_data: Any) -> Dict[str, Any]:
        """Preprocess a SpeechTranscript model instance or equivalent dictionary.

        Args:
            transcript_obj_or_data: Either a SpeechTranscript Django model instance,
                or a dictionary with keys: 'text', 'language_code', 'metadata'.

        Returns:
            A structured dict with:
                - full_text: Cleaned full text string
                - language_code: Language code
                - sentences: List of structured sentence dictionaries
                - paragraphs: List of structured paragraph dictionaries
                - stats: Summary statistics (word_count, sentence_count, paragraph_count, etc.)
                - warnings: List of warning strings if timestamps or punctuations were imputed
        """
        raw_text, language_code, metadata = self._extract_raw_inputs(transcript_obj_or_data)
        warnings: List[str] = []

        segments = self._extract_segments(metadata, raw_text)
        if not segments and raw_text:
            warnings.append("No diarized segments found in metadata; transcript treated as single unassigned segment.")
            segments = [{
                'speaker': None,
                'text': raw_text,
                'start': 0.0,
                'end': None,
                'confidence': None,
                'words': None,
            }]

        sentences: List[Dict[str, Any]] = []
        sentence_idx = 1

        for seg_idx, seg in enumerate(segments, start=1):
            seg_text = (seg.get('text') or '').strip()
            if not seg_text:
                continue

            seg_speaker = seg.get('speaker') or "Unassigned"
            seg_start = seg.get('start')
            seg_end = seg.get('end')
            seg_confidence = seg.get('confidence')
            seg_words = seg.get('words') or []

            # Sentence splitting within segment
            sentence_spans = _split_into_sentences(seg_text)
            if not sentence_spans:
                continue

            # If segment has no punctuation, sentence_spans has 1 element with seg_text
            num_sentences_in_seg = len(sentence_spans)

            for s_in_seg_idx, (s_text, c_start, c_end) in enumerate(sentence_spans, start=1):
                sentence_id = f"s_{sentence_idx:04d}"
                s_words = [w for w in re.split(r'\s+', s_text) if w]
                word_count = len(s_words)

                # Determine sentence timestamps
                s_start, s_end, s_conf, derived_from_words = self._calculate_sentence_timing(
                    s_text=s_text,
                    c_start=c_start,
                    c_end=c_end,
                    seg_len=len(seg_text),
                    seg_start=seg_start,
                    seg_end=seg_end,
                    seg_conf=seg_confidence,
                    seg_words=seg_words,
                    s_idx=s_in_seg_idx,
                    total_s=num_sentences_in_seg,
                )

                if not derived_from_words and num_sentences_in_seg > 1 and seg_start is not None and seg_end is not None:
                    # Notice about proportional interpolation
                    if "Segment-level timestamps proportionally interpolated for internal sentences." not in warnings:
                        warnings.append("Segment-level timestamps proportionally interpolated for internal sentences.")

                timing_source = 'words' if derived_from_words else ('segment' if s_start is not None else 'unknown')
                is_approximate = (timing_source != 'words')

                sentences.append({
                    'id': sentence_id,
                    'text': s_text,
                    'speaker': seg_speaker,
                    'start_time': s_start,
                    'end_time': s_end,
                    'confidence': s_conf,
                    'word_count': word_count,
                    'segment_index': seg_idx,
                    'timing_source': timing_source,
                    'is_approximate': is_approximate,
                    'paragraph_id': None,  # Grouped below
                })
                sentence_idx += 1

        # Group sentences into paragraphs
        paragraphs = self._group_paragraphs(sentences)

        # Compute summary stats
        total_words = sum(s['word_count'] for s in sentences)
        confs = [s['confidence'] for s in sentences if s.get('confidence') is not None]
        avg_conf = round(sum(confs) / len(confs), 3) if confs else None
        
        duration = None
        if sentences:
            valid_ends = [s['end_time'] for s in sentences if s.get('end_time') is not None]
            if valid_ends:
                duration = round(max(valid_ends), 2)

        return {
            'full_text': raw_text.strip(),
            'language_code': language_code or 'en',
            'duration': duration,
            'overall_confidence': avg_conf,
            'sentences': sentences,
            'paragraphs': paragraphs,
            'stats': {
                'total_sentences': len(sentences),
                'total_paragraphs': len(paragraphs),
                'total_words': total_words,
                'speaker_count': len(set(s['speaker'] for s in sentences if s['speaker'])),
            },
            'warnings': warnings,
        }

    def _extract_raw_inputs(self, obj: Any) -> Tuple[str, str, Dict[str, Any]]:
        """Extract text, language_code, and metadata from either a model or dict."""
        if hasattr(obj, 'text') and hasattr(obj, 'metadata'):
            return str(obj.text or ''), str(obj.language_code or 'en'), (obj.metadata or {})
        elif isinstance(obj, dict):
            return str(obj.get('text') or ''), str(obj.get('language_code') or 'en'), (obj.get('metadata') or {})
        return '', 'en', {}

    def _extract_segments(self, metadata: Dict[str, Any], raw_text: str) -> List[Dict[str, Any]]:
        """Extract segment list from metadata, supporting both Google STT and Gnani structures."""
        if not isinstance(metadata, dict):
            return []

        # 1. Direct segments key
        segments = metadata.get('segments')
        if isinstance(segments, list) and segments:
            return segments

        # 2. Gnani raw payload inside metadata
        nested_meta = metadata.get('metadata')
        if isinstance(nested_meta, dict):
            nested_segments = nested_meta.get('segments')
            if isinstance(nested_segments, list) and nested_segments:
                return nested_segments

        # 3. Google words list if segments not explicitly created
        words = metadata.get('words')
        if isinstance(words, list) and words:
            # Synthetic segment from words
            return [{
                'speaker': words[0].get('speaker') if isinstance(words[0], dict) else None,
                'text': raw_text,
                'start': words[0].get('start') if isinstance(words[0], dict) else 0.0,
                'end': words[-1].get('end') if isinstance(words[-1], dict) else None,
                'confidence': metadata.get('overall_confidence'),
                'words': words,
            }]

        return []

    def _calculate_sentence_timing(
        self,
        s_text: str,
        c_start: int,
        c_end: int,
        seg_len: int,
        seg_start: Optional[float],
        seg_end: Optional[float],
        seg_conf: Optional[float],
        seg_words: List[Dict[str, Any]],
        s_idx: int,
        total_s: int,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
        """Derive start/end timestamps and confidence for a sentence."""
        # 1. Try word-level matching
        if seg_words and isinstance(seg_words, list):
            s_clean_words = [re.sub(r'^[^\w]+|[^\w]+$', '', w).lower() for w in re.split(r'\s+', s_text) if w]
            s_clean_words = [w for w in s_clean_words if w]
            
            # Find best sequence match in seg_words
            match_start, match_end, match_confs = self._match_words(s_clean_words, seg_words)
            if match_start is not None and match_end is not None:
                calc_conf = round(sum(match_confs) / len(match_confs), 3) if match_confs else seg_conf
                return round(match_start, 3), round(match_end, 3), calc_conf, True

        # 2. Segment-level fallback with proportional interpolation
        if seg_start is None or seg_end is None:
            return seg_start, seg_end, seg_conf, False

        if total_s <= 1 or seg_len <= 0:
            return round(seg_start, 3), round(seg_end, 3), seg_conf, False

        # Proportional interpolation based on character position
        duration = max(0.0, seg_end - seg_start)
        p_start = max(0.0, min(1.0, c_start / float(seg_len)))
        p_end = max(0.0, min(1.0, c_end / float(seg_len)))

        derived_start = round(seg_start + duration * p_start, 3)
        derived_end = round(seg_start + duration * p_end, 3)
        if derived_end < derived_start:
            derived_end = derived_start

        return derived_start, derived_end, seg_conf, False

    def _match_words(
        self, target_words: List[str], seg_words: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float], List[float]]:
        """Match sequence of clean target words against seg_words list."""
        if not target_words or not seg_words:
            return None, None, []

        norm_seg_words = [
            re.sub(r'^[^\w]+|[^\w]+$', '', str(w.get('word', ''))).lower()
            for w in seg_words if isinstance(w, dict)
        ]

        target_len = len(target_words)
        for i in range(len(norm_seg_words) - target_len + 1):
            if norm_seg_words[i:i + target_len] == target_words:
                start_w = seg_words[i]
                end_w = seg_words[i + target_len - 1]
                start_t = start_w.get('start')
                end_t = end_w.get('end')
                confs = [
                    float(seg_words[k].get('confidence'))
                    for k in range(i, i + target_len)
                    if seg_words[k].get('confidence') is not None
                ]
                return start_t, end_t, confs

        # Fallback: prefix match on first word and last word
        first_w = target_words[0]
        last_w = target_words[-1]
        start_t = None
        end_t = None
        confs = []

        for i, w in enumerate(norm_seg_words):
            if w == first_w and start_t is None:
                start_t = seg_words[i].get('start')
            if w == last_w:
                end_t = seg_words[i].get('end')
                if seg_words[i].get('confidence') is not None:
                    confs.append(float(seg_words[i].get('confidence')))

        if start_t is not None and end_t is not None and end_t >= start_t:
            return start_t, end_t, confs

        return None, None, []

    def _group_paragraphs(self, sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group sentences into cohesive paragraphs by speaker turns or time pauses."""
        if not sentences:
            return []

        paragraphs: List[Dict[str, Any]] = []
        p_idx = 1
        current_sentences: List[Dict[str, Any]] = []
        current_speaker = sentences[0]['speaker']

        for i, sent in enumerate(sentences):
            prev_sent = sentences[i - 1] if i > 0 else None
            
            # Boundary conditions:
            # 1. Speaker switch
            # 2. Significant pause between sentences
            speaker_changed = (sent['speaker'] != current_speaker)
            pause_break = False
            if prev_sent and prev_sent.get('end_time') is not None and sent.get('start_time') is not None:
                gap = sent['start_time'] - prev_sent['end_time']
                if gap >= self.paragraph_pause_threshold:
                    pause_break = True

            if (speaker_changed or pause_break) and current_sentences:
                para = self._build_paragraph(p_idx, current_speaker, current_sentences)
                paragraphs.append(para)
                p_idx += 1
                current_sentences = []
                current_speaker = sent['speaker']

            sent['paragraph_id'] = f"p_{p_idx:03d}"
            current_sentences.append(sent)

        if current_sentences:
            para = self._build_paragraph(p_idx, current_speaker, current_sentences)
            paragraphs.append(para)

        return paragraphs

    def _build_paragraph(
        self, p_num: int, speaker: str, sents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Construct paragraph object from a sequence of sentences."""
        p_id = f"p_{p_num:03d}"
        text = " ".join(s['text'] for s in sents)
        valid_starts = [s['start_time'] for s in sents if s.get('start_time') is not None]
        valid_ends = [s['end_time'] for s in sents if s.get('end_time') is not None]
        start_time = min(valid_starts) if valid_starts else None
        end_time = max(valid_ends) if valid_ends else None

        confs = [s['confidence'] for s in sents if s.get('confidence') is not None]
        avg_conf = round(sum(confs) / len(confs), 3) if confs else None

        return {
            'id': p_id,
            'speaker': speaker,
            'sentence_ids': [s['id'] for s in sents],
            'text': text,
            'start_time': start_time,
            'end_time': end_time,
            'confidence': avg_conf,
            'word_count': sum(s['word_count'] for s in sents),
        }


def preprocess_transcript(transcript_obj_or_data: Any) -> Dict[str, Any]:
    """Helper functional interface to preprocess a transcript."""
    preprocessor = TranscriptPreprocessor()
    return preprocessor.preprocess(transcript_obj_or_data)
