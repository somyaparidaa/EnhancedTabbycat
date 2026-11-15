"""Service layer for speech analytics and external speech-to-text integrations."""

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
from .bundle import DebateAnalysisInputBundle, build_debate_analysis_input
from .context_builder import DebateContextBuilder, build_debate_context
from .factory import get_stt_client
from .gnani import GnaniPrismaBatchClient
from .google_speech import GoogleSpeechClient
from .preprocessor import TranscriptPreprocessor, preprocess_transcript
from .stance import (
    BaseStanceClassifier,
    DeBERTaStanceClassifier,
    MockStanceClassifier,
    StanceConfigurationError,
    StanceError,
    StanceInferenceError,
    StanceModelLoadError,
    StancePrediction,
    classify_transcript_stance,
    get_stance_classifier,
    is_procedural_sentence,
)

__all__ = [
    'BaseSpeechToTextClient',
    'STTError',
    'STTConfigurationError',
    'STTAuthenticationError',
    'STTRateLimitError',
    'STTAPIResponseError',
    'STTConnectionError',
    'STTTimeoutError',
    'STTMalformedResponseError',
    'get_stt_client',
    'GnaniPrismaBatchClient',
    'GoogleSpeechClient',
    'TranscriptPreprocessor',
    'preprocess_transcript',
    'DebateContextBuilder',
    'build_debate_context',
    'DebateAnalysisInputBundle',
    'build_debate_analysis_input',
    'BaseStanceClassifier',
    'StancePrediction',
    'DeBERTaStanceClassifier',
    'MockStanceClassifier',
    'get_stance_classifier',
    'classify_transcript_stance',
    'is_procedural_sentence',
    'StanceError',
    'StanceConfigurationError',
    'StanceModelLoadError',
    'StanceInferenceError',
]
