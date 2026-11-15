"""Google Cloud Speech-to-Text provider service.
Supports synchronous recognition (short files <10MB / <=60s) and is architected
for long-running recognition operations using Google Cloud Speech-to-Text v1.
Uses Application Default Credentials (ADC) without hardcoded secrets.
"""

import logging
import mimetypes
import os
import shutil
import struct
import subprocess
import tempfile
import uuid
import wave
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

from django.conf import settings
from google.api_core import exceptions as google_exceptions
from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError
from google.cloud import speech_v1

from .base import (
    BaseSpeechToTextClient,
    STTAPIResponseError,
    STTAuthenticationError,
    STTConfigurationError,
    STTConnectionError,
    STTError,
    STTMalformedResponseError,
    STTRateLimitError,
    STTTimeoutError,
)

logger = logging.getLogger(__name__)


def detect_audio_format(content: bytes, filename: str = "") -> Dict[str, Any]:
    """Inspect raw audio bytes and filename to determine format, sample rate, and channels.
    
    Does NOT assume extension accurately represents the format.
    Returns a dict with:
        'format': 'wav' | 'flac' | 'mp3' | 'ogg' | 'm4a' | 'unknown'
        'sample_rate': int or None
        'channels': int or None
        'sample_width': int or None
    """
    info: Dict[str, Any] = {
        'format': 'unknown',
        'sample_rate': None,
        'channels': None,
        'sample_width': None,
    }

    if not content:
        return info

    # 1. Check WAV / RIFF header
    if len(content) >= 12 and content[:4] == b'RIFF' and content[8:12] == b'WAVE':
        info['format'] = 'wav'
        # Parse fmt chunk if present
        idx = 12
        while idx + 8 <= len(content):
            chunk_id = content[idx:idx+4]
            chunk_size = struct.unpack('<I', content[idx+4:idx+8])[0]
            if chunk_id == b'fmt ' and chunk_size >= 16:
                fmt_data = content[idx+8:idx+8+chunk_size]
                audio_format, channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack('<HHIIHH', fmt_data[:16])
                info['channels'] = channels
                info['sample_rate'] = sample_rate
                info['sample_width'] = bits_per_sample // 8
                info['audio_format'] = audio_format  # 1 = PCM
                break
            idx += 8 + chunk_size
        return info

    # 2. Check FLAC header (starts with b'fLaC')
    if len(content) >= 4 and content[:4] == b'fLaC':
        info['format'] = 'flac'
        return info

    # 3. Check OGG header (starts with b'OggS')
    if len(content) >= 4 and content[:4] == b'OggS':
        info['format'] = 'ogg'
        return info

    # 4. Check MP3 header (ID3 tag or sync word 0xFFE0 / 0xFFF0)
    if len(content) >= 3 and content[:3] == b'ID3':
        info['format'] = 'mp3'
        return info
    if len(content) >= 2 and content[0] == 0xFF and (content[1] & 0xE0) == 0xE0:
        info['format'] = 'mp3'
        return info

    # 5. Check MP4 / M4A / AAC container (ftyp atom at byte 4)
    if len(content) >= 8 and content[4:8] == b'ftyp':
        info['format'] = 'm4a'
        return info

    # Fallback to extension if magic bytes could not identify
    ext = os.path.splitext(filename)[1].lower() if filename else ''
    if ext in ('.m4a', '.mp4', '.aac'):
        info['format'] = 'm4a'
    elif ext == '.wav':
        info['format'] = 'wav'
    elif ext == '.mp3':
        info['format'] = 'mp3'
    elif ext == '.flac':
        info['format'] = 'flac'
    elif ext == '.ogg':
        info['format'] = 'ogg'

    return info


def convert_audio_to_wav(
    source_content: bytes,
    source_filename: str = "audio.m4a",
    target_sample_rate: int = 16000,
) -> Tuple[bytes, str]:
    """Convert audio content (such as AAC/M4A) to standard 16-bit linear PCM WAV.

    Uses `ffmpeg` if available on the system PATH.
    If `ffmpeg` is not found, attempts `afconvert` (native on macOS).
    Never touches or modifies the original recording or database files.
    All conversions operate in isolated temporary files and cleans them up.
    Returns:
        Tuple of (converted_wav_bytes, converted_filename)
    Raises:
        STTConfigurationError if no conversion tool is installed or conversion fails.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    afconvert_bin = shutil.which("afconvert")

    if not ffmpeg_bin and not afconvert_bin:
        raise STTConfigurationError(
            "Audio conversion required for audio format (e.g. M4A/AAC), but neither 'ffmpeg' nor 'afconvert' "
            "is installed or accessible in PATH."
        )

    # Prepare temp files
    source_ext = os.path.splitext(source_filename)[1] or ".m4a"
    temp_dir = tempfile.mkdtemp(prefix="tabbycat_stt_conv_")
    src_temp_path = os.path.join(temp_dir, f"input{source_ext}")
    out_temp_path = os.path.join(temp_dir, "converted_16k.wav")

    try:
        with open(src_temp_path, "wb") as f:
            f.write(source_content)

        if ffmpeg_bin:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", src_temp_path,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", str(target_sample_rate),
                "-ac", "1",
                out_temp_path,
            ]
        else:
            # macOS native afconvert
            cmd = [
                afconvert_bin,
                "-f", "WAVE",
                "-d", f"LEI16@{target_sample_rate}",
                "-c", "1",
                src_temp_path,
                out_temp_path,
            ]

        logger.info("Executing audio conversion: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )

        if result.returncode != 0:
            err_msg = result.stderr.decode("utf-8", errors="replace").strip()
            logger.error("Audio conversion failed (code %d): %s", result.returncode, err_msg)
            raise STTConfigurationError(f"Failed to transcode audio to WAV: {err_msg}")

        if not os.path.exists(out_temp_path) or os.path.getsize(out_temp_path) == 0:
            raise STTConfigurationError("Audio conversion produced an empty or missing WAV file.")

        with open(out_temp_path, "rb") as f:
            converted_bytes = f.read()

        base_name = os.path.splitext(os.path.basename(source_filename))[0]
        out_filename = f"{base_name}_converted.wav"
        return converted_bytes, out_filename

    except subprocess.TimeoutExpired as exc:
        logger.error("Audio conversion process timed out: %s", exc)
        raise STTTimeoutError("Audio conversion timed out while converting to WAV.") from exc
    except STTError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during audio conversion: %s", exc)
        raise STTConfigurationError(f"Unexpected error during audio transcoding: {exc}") from exc
    finally:
        # Strictly ensure temporary directory and conversion files are removed
        shutil.rmtree(temp_dir, ignore_errors=True)


class GoogleSpeechClient(BaseSpeechToTextClient):
    """Google Cloud Speech-to-Text client implementing BaseSpeechToTextClient.

    Employs Application Default Credentials (ADC).
    Supports:
      1. Synchronous recognition for local audio files.
      2. Storage of completed transcripts in memory/local operation state.
      3. Normalization of word offsets and speaker diarization tags.
      4. Auto-detection of audio formats and safe transcoding of unsupported formats (e.g. M4A/AAC to LINEAR16 WAV).
    """

    DEFAULT_LANGUAGE_CODE = "en-US"
    DEFAULT_MODEL = "latest_long"

    # In-memory store for completed synchronous operations or tracked local operations
    _local_operations: Dict[str, Dict[str, Any]] = {}

    def __init__(self, client: Optional[speech_v1.SpeechClient] = None):
        if client:
            self._client = client
        else:
            try:
                self._client = speech_v1.SpeechClient()
            except (DefaultCredentialsError, GoogleAuthError) as exc:
                logger.error("Google Cloud authentication failed: %s", exc)
                raise STTAuthenticationError(
                    f"Google Cloud Speech authentication failed using ADC: {exc}"
                ) from exc
            except Exception as exc:
                logger.error("Failed to initialize Google Speech client: %s", exc)
                raise STTConfigurationError(
                    f"Failed to initialize Google Speech client: {exc}"
                ) from exc

    def _map_google_exception(self, exc: Exception) -> STTError:
        """Map google.api_core exceptions to provider-neutral STT errors."""
        if isinstance(exc, (DefaultCredentialsError, GoogleAuthError, google_exceptions.Unauthenticated)):
            return STTAuthenticationError(f"Google authentication failed: {exc}")
        if isinstance(exc, google_exceptions.PermissionDenied):
            return STTAuthenticationError(f"Google Speech-to-Text API permission denied: {exc}")
        if isinstance(exc, google_exceptions.ResourceExhausted):
            return STTRateLimitError(f"Google Cloud quota/rate limit exceeded: {exc}", status_code=429)
        if isinstance(exc, google_exceptions.DeadlineExceeded):
            return STTTimeoutError(f"Google Speech-to-Text request timed out: {exc}")
        if isinstance(exc, (google_exceptions.ServiceUnavailable, google_exceptions.BadGateway)):
            return STTConnectionError(f"Google Speech-to-Text service unavailable: {exc}")
        if isinstance(exc, google_exceptions.GoogleAPICallError):
            return STTAPIResponseError(f"Google Speech-to-Text API error ({exc.code}): {exc.message}")
        return STTError(f"Unexpected Google Speech error: {exc}")

    def _read_audio_content_and_filename(
        self, audio_file: Union[BinaryIO, Tuple[str, BinaryIO, str]]
    ) -> Tuple[bytes, str]:
        """Read bytes and determine filename from audio_file input."""
        if isinstance(audio_file, tuple):
            filename, file_stream, _ = audio_file
            if hasattr(file_stream, 'seek') and callable(file_stream.seek):
                file_stream.seek(0)
            content = file_stream.read()
        else:
            if hasattr(audio_file, 'seek') and callable(audio_file.seek):
                audio_file.seek(0)
            filename = getattr(audio_file, 'name', 'audio.wav')
            content = audio_file.read()

        if isinstance(content, str):
            content = content.encode('utf-8')
        return content, filename

    def _build_recognition_config(
        self,
        audio_format_info: Dict[str, Any],
        language_code: str,
        enable_diarization: bool = True,
        min_speakers: int = 2,
        max_speakers: int = 6,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> speech_v1.RecognitionConfig:
        """Construct Google RecognitionConfig dynamically matched to actual audio properties."""
        fmt = audio_format_info.get('format', 'unknown')
        sample_rate = audio_format_info.get('sample_rate')
        channels = audio_format_info.get('channels')

        config_kwargs: Dict[str, Any] = {
            'language_code': language_code or getattr(settings, 'GOOGLE_SPEECH_LANGUAGE_CODE', self.DEFAULT_LANGUAGE_CODE),
            'enable_word_time_offsets': True,
        }

        # Resolve model: prefer explicit kwarg, then settings, then default
        selected_model = model if model is not None else getattr(settings, 'GOOGLE_SPEECH_MODEL', self.DEFAULT_MODEL)
        # Avoid passing invalid model values (such as empty string or 'prisma-v2.5' from other providers)
        if selected_model and selected_model.strip():
            config_kwargs['model'] = selected_model.strip()

        # Set encoding and sample rate based on verified format
        if fmt == 'wav':
            config_kwargs['encoding'] = speech_v1.RecognitionConfig.AudioEncoding.LINEAR16
            if sample_rate:
                config_kwargs['sample_rate_hertz'] = sample_rate
            if channels and channels > 1:
                config_kwargs['audio_channel_count'] = channels
        elif fmt == 'flac':
            config_kwargs['encoding'] = speech_v1.RecognitionConfig.AudioEncoding.FLAC
            if sample_rate:
                config_kwargs['sample_rate_hertz'] = sample_rate
        elif fmt == 'mp3':
            config_kwargs['encoding'] = speech_v1.RecognitionConfig.AudioEncoding.MP3
            if sample_rate:
                config_kwargs['sample_rate_hertz'] = sample_rate
        elif fmt == 'ogg':
            config_kwargs['encoding'] = speech_v1.RecognitionConfig.AudioEncoding.OGG_OPUS
            if sample_rate:
                config_kwargs['sample_rate_hertz'] = sample_rate
        else:
            # Leave encoding unspecified or allow Google to auto-detect if header permits
            config_kwargs['encoding'] = speech_v1.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED

        if enable_diarization:
            config_kwargs['diarization_config'] = speech_v1.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=min_speakers,
                max_speaker_count=max_speakers,
            )

        config_kwargs.update(kwargs)
        return speech_v1.RecognitionConfig(**config_kwargs)

    def create_batch_job(
        self,
        audio_file: Union[BinaryIO, Tuple[str, BinaryIO, str]],
        language_code: str = "en-US",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Submit audio to Google Speech-to-Text.

        1. Inspects audio content bytes for true format (does not blindly trust extension).
        2. If format is unsupported by Google Speech v1 (e.g. M4A/AAC), transcodes to a temporary 16kHz LINEAR16 WAV.
        3. Never modifies or deletes the original user audio file.
        4. Executes synchronous recognize() and returns normalized transcript payload.
        """
        try:
            content, filename = self._read_audio_content_and_filename(audio_file)
        except Exception as exc:
            raise STTMalformedResponseError(f"Failed to read audio file input: {exc}") from exc

        # Detect true format
        format_info = detect_audio_format(content, filename=filename)
        logger.info("Detected audio format for %s: %s", filename, format_info)

        # Transcode unsupported formats (such as M4A / AAC container or unknown non-WAV/MP3/FLAC/OGG)
        if format_info['format'] in ('m4a', 'unknown'):
            logger.info("Transcoding %s (%s) to LINEAR16 WAV for Google STT compatibility", filename, format_info['format'])
            content, filename = convert_audio_to_wav(content, source_filename=filename, target_sample_rate=16000)
            format_info = detect_audio_format(content, filename=filename)

        config = self._build_recognition_config(
            audio_format_info=format_info,
            language_code=language_code,
            **kwargs,
        )
        audio = speech_v1.RecognitionAudio(content=content)

        job_id = f"google_sync_{uuid.uuid4().hex[:16]}"
        logger.info("Executing Google Speech synchronous recognition for job %s (%s)", job_id, filename)

        try:
            response = self._client.recognize(config=config, audio=audio)
        except Exception as exc:
            mapped_exc = self._map_google_exception(exc)
            logger.warning("Google Speech recognize failed for job %s: %s", job_id, mapped_exc.message)
            raise mapped_exc from exc

        # Normalize the response immediately
        normalized = self.normalize_google_response(response, language_code=config.language_code)

        # Store in local operations map for status check and retrieval
        self._local_operations[job_id] = {
            'job_id': job_id,
            'status': 'COMPLETED',
            'transcript_payload': normalized,
            'error': None,
        }

        return {
            'job_id': job_id,
            'status': 'COMPLETED',
            'raw_summary': {'results_count': len(response.results)},
            'transcript_payload': normalized,
        }

    def start_batch_job(self, job_id: str) -> Dict[str, Any]:
        """For synchronous recognition, job is already processed upon creation."""
        op = self._local_operations.get(job_id)
        if op:
            return {'job_id': job_id, 'status': op['status']}
        return {'job_id': job_id, 'status': 'PROCESSING'}

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Poll the status of the Google transcription job."""
        if job_id in self._local_operations:
            op = self._local_operations[job_id]
            return {
                'job_id': job_id,
                'status': op['status'],
                'progress': 100 if op['status'] == 'COMPLETED' else 0,
                'error': op.get('error'),
            }

        # If job_id represents an external long-running operation in future:
        if job_id.startswith("projects/") or "operations/" in job_id:
            try:
                operation = self._client.transport.operations_client.get_operation(job_id)
                if operation.done:
                    if operation.error and operation.error.message:
                        return {'job_id': job_id, 'status': 'FAILED', 'error': operation.error.message}
                    return {'job_id': job_id, 'status': 'COMPLETED'}
                return {'job_id': job_id, 'status': 'IN_PROGRESS'}
            except Exception as exc:
                raise self._map_google_exception(exc) from exc

        return {'job_id': job_id, 'status': 'UNKNOWN', 'error': 'Unknown job ID.'}

    def fetch_completed_transcript(self, job_id: str) -> Dict[str, Any]:
        """Retrieve the normalized transcript for the job."""
        op = self._local_operations.get(job_id)
        if op and op.get('transcript_payload'):
            return op['transcript_payload']

        raise STTMalformedResponseError(f"No completed transcript found for Google job '{job_id}'.")

    def normalize_google_response(
        self, response: speech_v1.RecognizeResponse, language_code: str = "en-US"
    ) -> Dict[str, Any]:
        """Normalize Google RecognizeResponse into Tabbycat's SpeechTranscript structure.

        Extracts:
          - Full concatenated transcript text.
          - Overall confidence average.
          - Word-level timestamps and speaker diarization tags.
          - Formatted diarized segments compatible with Tabbycat's UI templates.
        """
        full_text_parts: List[str] = []
        confidences: List[float] = []
        all_words: List[Dict[str, Any]] = []

        for result in response.results:
            if not result.alternatives:
                continue
            alt = result.alternatives[0]
            if alt.transcript:
                full_text_parts.append(alt.transcript.strip())
            if alt.confidence > 0:
                confidences.append(alt.confidence)

            # Process word timestamps and speaker tags
            for word_info in alt.words:
                start_sec = (
                    word_info.start_time.total_seconds()
                    if hasattr(word_info, 'start_time') and word_info.start_time
                    else 0.0
                )
                end_sec = (
                    word_info.end_time.total_seconds()
                    if hasattr(word_info, 'end_time') and word_info.end_time
                    else start_sec
                )
                speaker_tag = getattr(word_info, 'speaker_tag', 1) or 1
                word_conf = getattr(word_info, 'confidence', alt.confidence) or alt.confidence

                all_words.append({
                    'word': word_info.word,
                    'start': round(start_sec, 3),
                    'end': round(end_sec, 3),
                    'speaker': f"Speaker {speaker_tag}",
                    'confidence': round(word_conf, 3),
                })

        # Calculate overall metrics
        full_text = " ".join(full_text_parts).strip()
        overall_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
        duration = all_words[-1]['end'] if all_words else None

        # Group words into speaker segments
        segments = self._group_words_into_segments(all_words)

        return {
            'text': full_text,
            'language_code': language_code,
            'duration': duration,
            'overall_confidence': overall_confidence,
            'segments': segments,
            'metadata': {
                'provider': 'google',
                'word_count': len(all_words),
                'total_results': len(response.results),
            },
        }

    def _group_words_into_segments(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group sequential words with the same speaker into cohesive speech segments."""
        if not words:
            return []

        segments: List[Dict[str, Any]] = []
        current_speaker = words[0]['speaker']
        current_words: List[str] = [words[0]['word']]
        start_time = words[0]['start']
        end_time = words[0]['end']
        segment_confidences = [words[0]['confidence']]

        for w in words[1:]:
            if w['speaker'] == current_speaker:
                current_words.append(w['word'])
                end_time = w['end']
                segment_confidences.append(w['confidence'])
            else:
                segments.append({
                    'speaker': current_speaker,
                    'text': " ".join(current_words),
                    'start': start_time,
                    'end': end_time,
                    'confidence': round(sum(segment_confidences) / len(segment_confidences), 3),
                })
                current_speaker = w['speaker']
                current_words = [w['word']]
                start_time = w['start']
                end_time = w['end']
                segment_confidences = [w['confidence']]

        if current_words:
            segments.append({
                'speaker': current_speaker,
                'text': " ".join(current_words),
                'start': start_time,
                'end': end_time,
                'confidence': round(sum(segment_confidences) / len(segment_confidences), 3),
            })

        return segments
