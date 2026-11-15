import copy
import json
import os
from django.test import SimpleTestCase

from speech_analytics.contract import (
    DebateAnalysisValidationError,
    ensure_valid_debate_analysis,
    load_debate_analysis_schema,
    validate_debate_analysis,
)

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), 'schemas')
EXAMPLES_DIR = os.path.join(SCHEMA_DIR, 'examples')


class DebateAnalysisContractTests(SimpleTestCase):
    """Unit tests for the Stage 2.1 structured AI debate analysis contract and validator."""

    def setUp(self):
        with open(os.path.join(EXAMPLES_DIR, 'valid_bp_pm_analysis.json'), 'r') as f:
            self.valid_bp_payload = json.load(f)

        with open(os.path.join(EXAMPLES_DIR, 'valid_isolated_speech_analysis.json'), 'r') as f:
            self.valid_isolated_payload = json.load(f)

    def test_schema_loads_successfully(self):
        schema = load_debate_analysis_schema("1.0.0")
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema.get("title"), "DebateSpeechAnalysisV1")
        self.assertIn("required", schema)

    def test_schema_unsupported_version_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_debate_analysis_schema("99.0.0")

    def test_valid_bp_pm_example_passes_validation(self):
        is_valid, errors = validate_debate_analysis(self.valid_bp_payload)
        self.assertTrue(is_valid, f"Expected valid payload but got errors: {errors}")
        self.assertEqual(errors, [])

    def test_valid_isolated_speech_example_passes_validation(self):
        is_valid, errors = validate_debate_analysis(self.valid_isolated_payload)
        self.assertTrue(is_valid, f"Expected valid payload but got errors: {errors}")
        self.assertEqual(errors, [])

    def test_ensure_valid_debate_analysis_returns_payload_on_success(self):
        result = ensure_valid_debate_analysis(self.valid_bp_payload)
        self.assertEqual(result, self.valid_bp_payload)

    def test_non_dict_payload_fails(self):
        is_valid, errors = validate_debate_analysis("not a dict")
        self.assertFalse(is_valid)
        self.assertTrue(any("must be a dictionary" in e for e in errors))

    # --- Missing Required Sections ---

    def test_missing_required_top_level_section_fails(self):
        required_sections = [
            "schema_version",
            "metadata",
            "argument_analysis",
            "rebuttal_analysis",
            "clash_analysis",
            "role_fulfilment",
            "structure_analysis",
            "feedback",
            "limitations",
        ]
        for section in required_sections:
            payload = copy.deepcopy(self.valid_bp_payload)
            del payload[section]
            is_valid, errors = validate_debate_analysis(payload)
            self.assertFalse(is_valid, f"Expected invalidation when missing '{section}'")
            self.assertTrue(any(section in e for e in errors))

    def test_missing_metadata_required_field_fails(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        del payload['metadata']['target_speaker_name']
        is_valid, errors = validate_debate_analysis(payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("target_speaker_name" in e for e in errors))

    def test_unsupported_context_completeness_fails(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        payload['metadata']['context_completeness'] = 'total_guess'
        is_valid, errors = validate_debate_analysis(payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("context_completeness" in e for e in errors))

    # --- Schema Rejection of Arbitrary Rankings / Scores ---

    def test_arbitrary_scores_or_ranks_rejected_by_additional_properties(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        payload['overall_score'] = 95
        payload['ranking'] = 1
        is_valid, errors = validate_debate_analysis(payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("additional" in e.lower() or "overall_score" in e for e in errors))

    # --- Rebuttal and Argument Validation ---

    def test_unsupported_engagement_type_fails(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        payload['rebuttal_analysis'][0]['engagement_type'] = 'vibe_check'
        is_valid, errors = validate_debate_analysis(payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("engagement_type" in e for e in errors))

    def test_unsupported_rebuttal_effectiveness_fails(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        payload['rebuttal_analysis'][0]['effectiveness'] = 'godlike'
        is_valid, errors = validate_debate_analysis(payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("effectiveness" in e for e in errors))

    def test_unsupported_observation_type_fails(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        payload['argument_analysis'][0]['evidence'][0]['observation_type'] = 'hallucinated'
        is_valid, errors = validate_debate_analysis(payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("observation_type" in e for e in errors))

    # --- Evidence Integrity Validation ---

    def test_evidence_with_inverted_timestamps_fails_integrity_check(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        payload['argument_analysis'][0]['evidence'][0]['start_time'] = 120.0
        payload['argument_analysis'][0]['evidence'][0]['end_time'] = 45.0  # start > end
        is_valid, errors = validate_debate_analysis(payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("start_time" in e and "end_time" in e for e in errors))

    def test_evidence_with_blank_quote_fails_integrity_check(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        payload['argument_analysis'][0]['evidence'][0]['quote'] = "   "
        is_valid, errors = validate_debate_analysis(payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("quote" in e for e in errors))

    def test_evidence_from_prior_speech_supported_and_valid(self):
        # Already in valid_bp_payload rebuttal_analysis
        rebuttal_ev = self.valid_bp_payload['rebuttal_analysis'][0]['evidence']
        sources = [e['source'] for e in rebuttal_ev]
        self.assertIn('prior_speech', sources)
        self.assertIn('target_speech', sources)

    def test_missing_limitations_fields_fails(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        del payload['limitations']['missing_debate_context']
        is_valid, errors = validate_debate_analysis(payload)
        self.assertFalse(is_valid)
        self.assertTrue(any("missing_debate_context" in e for e in errors))

    def test_ensure_valid_raises_custom_exception(self):
        payload = copy.deepcopy(self.valid_bp_payload)
        payload['schema_version'] = '2.0.0'
        with self.assertRaises(DebateAnalysisValidationError) as ctx:
            ensure_valid_debate_analysis(payload)
        self.assertIn("failed validation", str(ctx.exception))
