"""Unit tests for Stage 2B transcript stance classification service."""

from django.test import SimpleTestCase

from speech_analytics.contract import validate_stance_classification
from speech_analytics.services.stance.filter import is_procedural_sentence
from speech_analytics.services.stance.mock import MockStanceClassifier
from speech_analytics.services.stance.service import classify_transcript_stance


class StanceProceduralFilterTests(SimpleTestCase):
    """Tests for the rule-based procedural sentence filter."""

    def test_salutations_filtered(self):
        cases = [
            "Good morning Madam Speaker.",
            "Madam President, members of the house.",
            "Thank you Mr. Speaker.",
            "Welcome ladies and gentlemen.",
        ]
        for c in cases:
            is_proc, reason = is_procedural_sentence(c)
            self.assertTrue(is_proc, f"Expected procedural for: {c}")
            self.assertEqual(reason, "MATCHED_PROCEDURAL_HEURISTIC")

    def test_poi_interactions_filtered(self):
        cases = [
            "Point of information!",
            "Point of clarification Madam Speaker.",
            "On that point, opening opposition?",
            "Yes, please go ahead.",
            "No thank you, sit down.",
            "I will take your point now.",
        ]
        for c in cases:
            is_proc, reason = is_procedural_sentence(c)
            self.assertTrue(is_proc, f"Expected procedural for: {c}")
            self.assertEqual(reason, "MATCHED_PROCEDURAL_HEURISTIC")

    def test_time_and_pacing_filtered(self):
        cases = [
            "Am I audible to the house?",
            "Starting my timer in 3, 2, 1.",
            "With that, I beg to propose the motion.",
            "I yield my remaining time to the chair.",
        ]
        for c in cases:
            is_proc, reason = is_procedural_sentence(c)
            self.assertTrue(is_proc, f"Expected procedural for: {c}")

    def test_short_fragments_filtered(self):
        cases = ["Yes.", "Wait.", "Okay.", "One second."]
        for c in cases:
            is_proc, reason = is_procedural_sentence(c)
            self.assertTrue(is_proc, f"Expected fragment for: {c}")
            self.assertEqual(reason, "TOO_SHORT_FRAGMENT")

    def test_substantive_arguments_passed(self):
        cases = [
            "Carbon taxes will accelerate our transition away from fossil fuels.",
            "Universal healthcare significantly reduces long-term hospitalization costs.",
            "Subsidies distort market competition and hurt small business innovation.",
            "Banning private schools will drastically improve public education funding.",
        ]
        for c in cases:
            is_proc, reason = is_procedural_sentence(c)
            self.assertFalse(is_proc, f"Expected substantive for: {c}")
            self.assertEqual(reason, "ELIGIBLE_CANDIDATE_CLAIM")


class TranscriptStanceServiceTests(SimpleTestCase):
    """Tests for classify_transcript_stance orchestrator."""

    def setUp(self):
        self.mock_classifier = MockStanceClassifier()
        self.synthetic_bundle = {
            'contract_version': '1.0.0',
            'fingerprint': 'abc123hash',
            'generated_at': '2026-11-15T10:00:00Z',
            'provenance': {
                'transcript_id': 101,
                'recording_id': 55,
                'language_code': 'en-US',
                'context_completeness': 'isolated_speech',
            },
            'debate_context': {
                'tournament': {'name': 'Australs 2026', 'format_name': 'Australs'},
                'round': {'seq': 1, 'name': 'Round 1'},
                'debate': {'id': 20, 'matchup': 'Team A vs Team B', 'confirmed_ballot_available': True},
                'motion': {
                    'primary_motion': {
                        'text': 'This House would ban targeted advertising.',
                        'info_slide': 'Targeted ads track user habits across web services.',
                    }
                },
                'teams': [
                    {'team_id': 1, 'team_name': 'Team A', 'side_code': 'AFF'},
                    {'team_id': 2, 'team_name': 'Team B', 'side_code': 'NEG'},
                ],
                'speeches': [
                    {'position': 1, 'side_code': 'AFF', 'role_name': 'Affirmative Speaker 1', 'speaker_id': 11},
                ],
                'speaker_mapping': {
                    'status': 'RESOLVED_SINGLE_SPEAKER',
                    'target_speaker': {'id': 11, 'name': 'Alice', 'speech_role': 'Affirmative Speaker 1'},
                    'diarized_speakers': ['Speaker 1'],
                    'unresolved_speakers': [],
                    'mapping': {
                        'Speaker 1': {'speaker_id': 11, 'speaker_name': 'Alice', 'role': 'Affirmative Speaker 1'},
                    }
                },
            },
            'transcript': {
                'full_text': (
                    'Good morning Madam Speaker. '
                    'First, targeted ads violate digital privacy and exploit user addictions. '
                    'Point of information! '
                    'No thank you. '
                    'Furthermore, surveillance capitalism harms market competition.'
                ),
                'stats': {'total_sentences': 5, 'total_words': 29, 'total_paragraphs': 2, 'speaker_count': 1},
                'sentences': [
                    {
                        'id': 's_0001',
                        'text': 'Good morning Madam Speaker.',
                        'speaker': 'Speaker 1',
                        'start_time': 0.0,
                        'end_time': 2.0,
                        'word_count': 4,
                        'timing_source': 'words',
                        'is_approximate': False,
                        'paragraph_id': 'p_001',
                    },
                    {
                        'id': 's_0002',
                        'text': 'First, targeted ads violate digital privacy and exploit user addictions.',
                        'speaker': 'Speaker 1',
                        'start_time': 2.5,
                        'end_time': 6.8,
                        'word_count': 10,
                        'timing_source': 'words',
                        'is_approximate': False,
                        'paragraph_id': 'p_001',
                    },
                    {
                        'id': 's_0003',
                        'text': 'Point of information!',
                        'speaker': 'Speaker 1',
                        'start_time': 7.0,
                        'end_time': 8.2,
                        'word_count': 3,
                        'timing_source': 'segment',
                        'is_approximate': True,
                        'paragraph_id': 'p_002',
                    },
                    {
                        'id': 's_0004',
                        'text': 'No thank you.',
                        'speaker': 'Speaker 1',
                        'start_time': 8.5,
                        'end_time': 9.5,
                        'word_count': 3,
                        'timing_source': 'segment',
                        'is_approximate': True,
                        'paragraph_id': 'p_002',
                    },
                    {
                        'id': 's_0005',
                        'text': 'Furthermore, surveillance capitalism harms market competition.',
                        'speaker': 'Speaker 1',
                        'start_time': 10.0,
                        'end_time': 14.5,
                        'word_count': 6,
                        'timing_source': 'words',
                        'is_approximate': False,
                        'paragraph_id': 'p_002',
                    },
                ],
                'paragraphs': [],
            },
            'warnings': [],
        }

    def test_classify_transcript_stance_end_to_end(self):
        result = classify_transcript_stance(self.synthetic_bundle, classifier=self.mock_classifier)

        # Basic metadata assertions
        self.assertEqual(result['schema_version'], "1.0.0")
        self.assertEqual(result['motion'], "This House would ban targeted advertising.")
        self.assertEqual(len(result['predictions']), 5)

        # Procedural sentences skipped
        s1 = next(p for p in result['predictions'] if p['sentence_id'] == 's_0001')
        self.assertEqual(s1['predicted_stance'], "NOT_APPLICABLE")
        self.assertEqual(s1['uncertainty_status'], "SKIPPED_PROCEDURAL")
        self.assertIsNone(s1['contradiction_probability'])
        self.assertIsNone(s1['entailment_probability'])

        s3 = next(p for p in result['predictions'] if p['sentence_id'] == 's_0003')
        self.assertEqual(s3['predicted_stance'], "NOT_APPLICABLE")
        self.assertEqual(s3['uncertainty_status'], "SKIPPED_PROCEDURAL")

        # Substantive sentence evaluated
        s2 = next(p for p in result['predictions'] if p['sentence_id'] == 's_0002')
        self.assertIn(s2['predicted_stance'], ["PRO", "CON", "UNCERTAIN"])
        self.assertIsNotNone(s2['contradiction_probability'])
        self.assertIsNotNone(s2['entailment_probability'])
        self.assertIsNotNone(s2['neutral_probability'])

        # Check evidence reference preservation
        ev = s2['evidence_reference']
        self.assertEqual(ev['quote'], "First, targeted ads violate digital privacy and exploit user addictions.")
        self.assertEqual(ev['start_time'], 2.5)
        self.assertEqual(ev['end_time'], 6.8)
        self.assertFalse(ev['is_approximate'])
        self.assertEqual(ev['speaker_identifier'], "Speaker 1")
        self.assertEqual(ev['speech_role'], "Affirmative Speaker 1")

        # Check stats
        stats = result['stats']
        self.assertEqual(stats['total_sentences'], 5)
        self.assertEqual(stats['total_skipped_procedural'], 3)  # s1, s3, s4
        self.assertEqual(stats['total_evaluated'], 2)           # s2, s5

        # Validate against JSON schema
        is_valid, errors = validate_stance_classification(result)
        self.assertTrue(is_valid, f"Schema validation errors: {errors}")

    def test_missing_motion_handled_gracefully(self):
        bundle_no_motion = dict(self.synthetic_bundle)
        bundle_no_motion['debate_context'] = {'motion': None}

        result = classify_transcript_stance(bundle_no_motion, classifier=self.mock_classifier)
        self.assertEqual(result['motion'], "")
        self.assertEqual(len(result['predictions']), 0)
        self.assertEqual(result['stats']['total_evaluated'], 0)
        self.assertIn("No primary motion text found", result['warnings'][0])

        is_valid, errors = validate_stance_classification(result)
        self.assertTrue(is_valid, f"Schema errors on missing motion: {errors}")

    def test_unresolved_speaker_mapping_warning(self):
        bundle_unresolved = dict(self.synthetic_bundle)
        bundle_unresolved['debate_context'] = dict(self.synthetic_bundle['debate_context'])
        bundle_unresolved['debate_context']['speaker_mapping'] = {
            'status': 'UNRESOLVED_MULTI_SPEAKER',
            'diarized_speakers': ['Speaker 1', 'Speaker 2'],
            'unresolved_speakers': ['Speaker 1', 'Speaker 2'],
            'mapping': {},
        }

        result = classify_transcript_stance(bundle_unresolved, classifier=self.mock_classifier)
        self.assertGreater(len(result['warnings']), 0)
        self.assertIn("UNRESOLVED_MULTI_SPEAKER", result['warnings'][0])
        # Inference should still run without crashing
        self.assertEqual(result['stats']['total_evaluated'], 2)
