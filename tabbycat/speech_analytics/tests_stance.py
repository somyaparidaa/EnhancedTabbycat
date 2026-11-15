"""Unit tests for topic-conditioned stance classification services and providers."""

import os
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from speech_analytics.services.stance.base import StancePrediction
from speech_analytics.services.stance.exceptions import (
    StanceConfigurationError,
    StanceInferenceError,
    StanceModelLoadError,
)
from speech_analytics.services.stance.factory import (
    get_stance_classifier,
    reset_stance_classifier_cache,
)
from speech_analytics.services.stance.hypothesis import default_motion_to_hypothesis
from speech_analytics.services.stance.mock import MockStanceClassifier


class StanceHypothesisTransformTests(SimpleTestCase):
    """Tests for debate motion to declarative hypothesis conversion."""

    def test_thw_prefix(self):
        motion = "This House would ban private schools."
        hyp = default_motion_to_hypothesis(motion)
        self.assertEqual(hyp, "We should ban private schools.")

    def test_thbt_prefix(self):
        motion = "This House believes that artificial intelligence is dangerous."
        hyp = default_motion_to_hypothesis(motion)
        self.assertEqual(hyp, "Artificial intelligence is dangerous.")

    def test_ths_prefix(self):
        motion = "This House supports universal basic income."
        hyp = default_motion_to_hypothesis(motion)
        self.assertEqual(hyp, "universal basic income should be supported.")

    def test_thr_prefix(self):
        motion = "This House regrets the rise of cryptocurrency."
        hyp = default_motion_to_hypothesis(motion)
        self.assertEqual(hyp, "the rise of cryptocurrency is regrettable.")

    def test_thp_prefix(self):
        motion = "This House prefers nuclear energy to fossil fuels."
        hyp = default_motion_to_hypothesis(motion)
        self.assertEqual(hyp, "nuclear energy to fossil fuels is preferable.")

    def test_plain_motion(self):
        motion = "Carbon taxation"
        hyp = default_motion_to_hypothesis(motion)
        self.assertEqual(hyp, "Carbon taxation.")


class MockStanceClassifierTests(SimpleTestCase):
    """Tests for the deterministic MockStanceClassifier provider."""

    def setUp(self):
        self.classifier = MockStanceClassifier(confidence_threshold=0.55, margin_threshold=0.15)

    def test_predict_one_pro(self):
        pred = self.classifier.predict_one(
            motion="This House would subsidize solar panels.",
            claim_text="Solar panels accelerate our transition to clean energy.",
            sentence_id="s_001",
        )
        self.assertEqual(pred.predicted_stance, "PRO")
        self.assertEqual(pred.uncertainty_status, "CONFIDENT")
        self.assertGreater(pred.entailment_probability, pred.contradiction_probability)
        self.assertEqual(pred.sentence_id, "s_001")
        self.assertIn("solar panels", pred.motion_hypothesis.lower())

    def test_predict_one_con(self):
        pred = self.classifier.predict_one(
            motion="This House would ban nuclear power.",
            claim_text="Nuclear power bans are dangerous and fail energy grids.",
            sentence_id="s_002",
        )
        self.assertEqual(pred.predicted_stance, "CON")
        self.assertEqual(pred.uncertainty_status, "CONFIDENT")
        self.assertGreater(pred.contradiction_probability, pred.entailment_probability)

    def test_predict_one_uncertain_dominant_neutral(self):
        pred = self.classifier.predict_one(
            motion="This House would ban homework.",
            claim_text="It is unclear whether students prefer homework or not.",
            sentence_id="s_003",
        )
        self.assertEqual(pred.predicted_stance, "UNCERTAIN")
        self.assertEqual(pred.uncertainty_status, "DOMINANT_NEUTRAL")
        self.assertGreater(pred.neutral_probability, 0.50)

    def test_predict_one_uncertain_low_margin(self):
        pred = self.classifier.predict_one(
            motion="This House would adopt the euro.",
            claim_text="The economic debate is very close and tied.",
            sentence_id="s_004",
        )
        # In mock, "close" gives p_con=0.48, p_pro=0.46 (margin 0.02 < 0.15 threshold)
        self.assertEqual(pred.uncertainty_status, "LOW_MARGIN")

    def test_predict_batch(self):
        items = [
            {'sentence_id': 's_001', 'claim_text': 'We support this proposal.'},
            {'sentence_id': 's_002', 'claim_text': 'We oppose this disastrous policy.'},
        ]
        results = self.classifier.predict_batch("This House would reform tax law.", items)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].predicted_stance, "PRO")
        self.assertEqual(results[1].predicted_stance, "CON")

    def test_to_dict_preserves_all_probabilities(self):
        pred = self.classifier.predict_one(
            motion="This House would ban crypto.",
            claim_text="Crypto accelerates financial fraud.",
            sentence_id="s_010",
        )
        d = pred.to_dict()
        self.assertIn("contradiction_probability", d)
        self.assertIn("entailment_probability", d)
        self.assertIn("neutral_probability", d)
        self.assertIn("motion_hypothesis", d)
        self.assertIn("uncertainty_status", d)
        self.assertIn("model_version", d)


class StanceFactoryTests(SimpleTestCase):
    """Tests for stance classifier factory and singleton caching."""

    def setUp(self):
        reset_stance_classifier_cache()

    def tearDown(self):
        reset_stance_classifier_cache()

    @override_settings(STANCE_CLASSIFIER_BACKEND='mock')
    def test_get_stance_classifier_mock(self):
        c1 = get_stance_classifier()
        self.assertIsInstance(c1, MockStanceClassifier)
        c2 = get_stance_classifier()
        # Verify singleton caching
        self.assertIs(c1, c2)

    @override_settings(STANCE_CLASSIFIER_BACKEND='mock')
    def test_force_reload_resets_cache(self):
        c1 = get_stance_classifier()
        c2 = get_stance_classifier(force_reload=True)
        self.assertIsNot(c1, c2)

    def test_unsupported_backend_raises_error(self):
        with self.assertRaises(StanceConfigurationError):
            get_stance_classifier(backend='non_existent_provider')

    def test_missing_model_path_raises_error(self):
        with self.assertRaises(StanceConfigurationError):
            get_stance_classifier(backend='deberta', model_path='/path/does/not/exist/at/all')


class DeBERTaLocalCheckpointTests(SimpleTestCase):
    """Integration test using the locally available checkpoint at /Users/somya/Developer/stance_ft."""

    CHECKPOINT_PATH = "/Users/somya/Developer/stance_ft"

    def setUp(self):
        if not os.path.exists(self.CHECKPOINT_PATH):
            self.skipTest(f"Local checkpoint not found at {self.CHECKPOINT_PATH}")

    def test_live_checkpoint_inference_cpu(self):
        from speech_analytics.services.stance.deberta import DeBERTaStanceClassifier

        classifier = DeBERTaStanceClassifier(
            model_path=self.CHECKPOINT_PATH,
            device="cpu",
            confidence_threshold=0.55,
            margin_threshold=0.15,
        )

        motion = "This House would subsidize renewable energy."
        claim = "Subsidies make solar and wind infrastructure scalable and cost-effective."

        pred = classifier.predict_one(
            motion=motion,
            claim_text=claim,
            sentence_id="s_100",
        )

        # Check prediction structure and probabilities
        self.assertIn(pred.predicted_stance, ["PRO", "CON", "UNCERTAIN"])
        self.assertGreaterEqual(pred.contradiction_probability, 0.0)
        self.assertGreaterEqual(pred.entailment_probability, 0.0)
        self.assertGreaterEqual(pred.neutral_probability, 0.0)

        # Sum of softmax probabilities should be approx 1.0
        prob_sum = (
            pred.contradiction_probability
            + pred.entailment_probability
            + pred.neutral_probability
        )
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

        # Verify hypothesis was constructed properly
        self.assertEqual(pred.motion_hypothesis, "We should subsidize renewable energy.")

    def test_live_checkpoint_batch_inference(self):
        from speech_analytics.services.stance.deberta import DeBERTaStanceClassifier

        classifier = DeBERTaStanceClassifier(
            model_path=self.CHECKPOINT_PATH,
            device="cpu",
        )

        motion = "This House would ban single-use plastics."
        items = [
            {'sentence_id': 's_01', 'claim_text': 'Banning plastics eliminates tons of landfill waste.'},
            {'sentence_id': 's_02', 'claim_text': 'Plastic bans increase costs for struggling small businesses.'},
        ]

        batch_preds = classifier.predict_batch(motion, items)
        self.assertEqual(len(batch_preds), 2)
        self.assertEqual(batch_preds[0].sentence_id, 's_01')
        self.assertEqual(batch_preds[1].sentence_id, 's_02')
