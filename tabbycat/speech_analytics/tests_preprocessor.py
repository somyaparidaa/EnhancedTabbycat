"""Unit tests for transcript preprocessing and segmentation service."""

from django.test import SimpleTestCase

from speech_analytics.services.preprocessor import TranscriptPreprocessor, preprocess_transcript


class TranscriptPreprocessorTests(SimpleTestCase):
    """Test suite for TranscriptPreprocessor."""

    def setUp(self):
        self.preprocessor = TranscriptPreprocessor(paragraph_pause_threshold=2.0)

    def test_empty_transcript(self):
        result = self.preprocessor.preprocess({
            'text': '',
            'language_code': 'en',
            'metadata': {},
        })
        self.assertEqual(result['full_text'], '')
        self.assertEqual(result['sentences'], [])
        self.assertEqual(result['paragraphs'], [])
        self.assertEqual(result['stats']['total_sentences'], 0)
        self.assertEqual(result['stats']['total_paragraphs'], 0)

    def test_single_speaker_without_diarization_metadata(self):
        raw_text = "Good morning Madam Speaker. Today we are proud to propose the motion."
        result = self.preprocessor.preprocess({
            'text': raw_text,
            'language_code': 'en-US',
            'metadata': {},
        })
        self.assertEqual(result['full_text'], raw_text)
        self.assertEqual(len(result['sentences']), 2)
        self.assertEqual(result['sentences'][0]['id'], "s_0001")
        self.assertEqual(result['sentences'][0]['text'], "Good morning Madam Speaker.")
        self.assertEqual(result['sentences'][1]['id'], "s_0002")
        self.assertEqual(result['sentences'][1]['text'], "Today we are proud to propose the motion.")
        self.assertEqual(result['sentences'][0]['speaker'], "Unassigned")
        self.assertIn("No diarized segments found in metadata", result['warnings'][0])
        self.assertEqual(len(result['paragraphs']), 1)

    def test_word_level_timestamps_exact_matching(self):
        metadata = {
            'segments': [
                {
                    'speaker': 'Speaker 1',
                    'text': 'We propose that artificial intelligence should be regulated. This will safeguard consumers.',
                    'start': 0.0,
                    'end': 6.0,
                    'confidence': 0.95,
                    'words': [
                        {'word': 'We', 'start': 0.0, 'end': 0.4, 'confidence': 0.9},
                        {'word': 'propose', 'start': 0.4, 'end': 0.9, 'confidence': 0.95},
                        {'word': 'that', 'start': 0.9, 'end': 1.1, 'confidence': 0.98},
                        {'word': 'artificial', 'start': 1.1, 'end': 1.6, 'confidence': 0.94},
                        {'word': 'intelligence', 'start': 1.6, 'end': 2.3, 'confidence': 0.96},
                        {'word': 'should', 'start': 2.3, 'end': 2.6, 'confidence': 0.97},
                        {'word': 'be', 'start': 2.6, 'end': 2.8, 'confidence': 0.99},
                        {'word': 'regulated.', 'start': 2.8, 'end': 3.5, 'confidence': 0.95},
                        {'word': 'This', 'start': 3.7, 'end': 4.0, 'confidence': 0.92},
                        {'word': 'will', 'start': 4.0, 'end': 4.3, 'confidence': 0.94},
                        {'word': 'safeguard', 'start': 4.3, 'end': 5.1, 'confidence': 0.93},
                        {'word': 'consumers.', 'start': 5.1, 'end': 6.0, 'confidence': 0.96},
                    ],
                }
            ]
        }
        result = self.preprocessor.preprocess({
            'text': metadata['segments'][0]['text'],
            'language_code': 'en-US',
            'metadata': metadata,
        })
        sentences = result['sentences']
        self.assertEqual(len(sentences), 2)
        
        # Sentence 1
        self.assertEqual(sentences[0]['text'], 'We propose that artificial intelligence should be regulated.')
        self.assertEqual(sentences[0]['start_time'], 0.0)
        self.assertEqual(sentences[0]['end_time'], 3.5)
        self.assertEqual(sentences[0]['timing_source'], 'words')
        self.assertFalse(sentences[0]['is_approximate'])
        self.assertEqual(sentences[0]['speaker'], 'Speaker 1')

        # Sentence 2
        self.assertEqual(sentences[1]['text'], 'This will safeguard consumers.')
        self.assertEqual(sentences[1]['start_time'], 3.7)
        self.assertEqual(sentences[1]['end_time'], 6.0)
        self.assertEqual(sentences[1]['timing_source'], 'words')
        self.assertFalse(sentences[1]['is_approximate'])

    def test_segment_level_timestamp_proportional_fallback(self):
        metadata = {
            'segments': [
                {
                    'speaker': 'Speaker 1',
                    'text': 'First argument. Second argument.',
                    'start': 10.0,
                    'end': 20.0,
                    'confidence': 0.88,
                    'words': None,  # No word-level timestamps
                }
            ]
        }
        result = self.preprocessor.preprocess({
            'text': metadata['segments'][0]['text'],
            'language_code': 'en-US',
            'metadata': metadata,
        })
        sentences = result['sentences']
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]['timing_source'], 'segment')
        self.assertTrue(sentences[0]['is_approximate'])
        self.assertEqual(sentences[1]['timing_source'], 'segment')
        self.assertTrue(sentences[1]['is_approximate'])
        self.assertAlmostEqual(sentences[0]['start_time'], 10.0, places=1)
        self.assertLess(sentences[0]['end_time'], sentences[1]['end_time'])
        self.assertAlmostEqual(sentences[1]['end_time'], 20.0, places=1)
        self.assertIn("Segment-level timestamps proportionally interpolated", result['warnings'][0])

    def test_punctuation_free_transcript(self):
        text = "we believe that social media algorithms polarize public discourse and lead to harms"
        metadata = {
            'segments': [
                {
                    'speaker': 'Speaker 1',
                    'text': text,
                    'start': 1.0,
                    'end': 7.5,
                    'confidence': 0.9,
                }
            ]
        }
        result = self.preprocessor.preprocess({
            'text': text,
            'language_code': 'en',
            'metadata': metadata,
        })
        self.assertEqual(len(result['sentences']), 1)
        self.assertEqual(result['sentences'][0]['text'], text)
        self.assertEqual(result['sentences'][0]['start_time'], 1.0)
        self.assertEqual(result['sentences'][0]['end_time'], 7.5)

    def test_paragraph_grouping_by_speaker_and_pause(self):
        metadata = {
            'segments': [
                {
                    'speaker': 'Speaker 1',
                    'text': 'Opening point. Continuing thought.',
                    'start': 0.0,
                    'end': 5.0,
                    'confidence': 0.9,
                    'words': [
                        {'word': 'Opening', 'start': 0.0, 'end': 1.0, 'confidence': 0.9},
                        {'word': 'point.', 'start': 1.0, 'end': 2.0, 'confidence': 0.9},
                        {'word': 'Continuing', 'start': 4.5, 'end': 4.8, 'confidence': 0.9},  # 2.5s gap (> 2.0 threshold)
                        {'word': 'thought.', 'start': 4.8, 'end': 5.0, 'confidence': 0.9},
                    ],
                },
                {
                    'speaker': 'Speaker 2',
                    'text': 'Point of information Madam Speaker!',
                    'start': 5.5,
                    'end': 7.0,
                    'confidence': 0.92,
                    'words': [
                        {'word': 'Point', 'start': 5.5, 'end': 5.8, 'confidence': 0.9},
                        {'word': 'of', 'start': 5.8, 'end': 6.0, 'confidence': 0.9},
                        {'word': 'information', 'start': 6.0, 'end': 6.5, 'confidence': 0.9},
                        {'word': 'Madam', 'start': 6.5, 'end': 6.7, 'confidence': 0.9},
                        {'word': 'Speaker!', 'start': 6.7, 'end': 7.0, 'confidence': 0.9},
                    ],
                }
            ]
        }
        result = self.preprocessor.preprocess({
            'text': "Opening point. Continuing thought. Point of information Madam Speaker!",
            'language_code': 'en',
            'metadata': metadata,
        })
        paragraphs = result['paragraphs']
        # Due to 2.5s gap between sent 1 and sent 2, and speaker switch between sent 2 and sent 3:
        # Paragraph 1: sent 1 (Speaker 1)
        # Paragraph 2: sent 2 (Speaker 1)
        # Paragraph 3: sent 3 (Speaker 2)
        self.assertEqual(len(paragraphs), 3)
        self.assertEqual(paragraphs[0]['speaker'], 'Speaker 1')
        self.assertEqual(paragraphs[1]['speaker'], 'Speaker 1')
        self.assertEqual(paragraphs[2]['speaker'], 'Speaker 2')
        self.assertEqual(paragraphs[2]['text'], 'Point of information Madam Speaker!')

    def test_unknown_timing_source_flagged_approximate(self):
        result = self.preprocessor.preprocess({
            'text': 'A debate sentence with unknown timestamps.',
            'language_code': 'en',
            'metadata': {
                'segments': [
                    {
                        'speaker': 'Speaker 1',
                        'text': 'A debate sentence with unknown timestamps.',
                        'start': None,
                        'end': None,
                    }
                ]
            },
        })
        sentences = result['sentences']
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0]['timing_source'], 'unknown')
        self.assertTrue(sentences[0]['is_approximate'])
        self.assertIsNone(sentences[0]['start_time'])
        self.assertIsNone(sentences[0]['end_time'])

    def test_empty_segments_filtered(self):
        metadata = {
            'segments': [
                {'speaker': 'Speaker 1', 'text': '', 'start': 0.0, 'end': 1.0},
                {'speaker': 'Speaker 1', 'text': '   ', 'start': 1.0, 'end': 2.0},
                {'speaker': 'Speaker 2', 'text': 'Actual debate speech.', 'start': 2.0, 'end': 4.0},
            ]
        }
        result = self.preprocessor.preprocess({
            'text': 'Actual debate speech.',
            'language_code': 'en',
            'metadata': metadata,
        })
        self.assertEqual(len(result['sentences']), 1)
        self.assertEqual(result['sentences'][0]['speaker'], 'Speaker 2')
        self.assertEqual(result['sentences'][0]['text'], 'Actual debate speech.')
