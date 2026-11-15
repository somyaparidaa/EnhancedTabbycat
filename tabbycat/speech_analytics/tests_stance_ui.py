"""Unit and integration tests for Stance Analysis UI and Views."""

import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from draw.models import Debate, DebateTeam
from motions.models import Motion
from participants.models import Institution, Speaker, Team
from speech_analytics.models import SpeechRecording, SpeechTranscript
from speech_analytics.services.stance.exceptions import StanceModelLoadError
from tournaments.models import Round, Tournament

User = get_user_model()


@override_settings(
    STANCE_CLASSIFIER_BACKEND="mock",
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
class StanceAnalysisViewTests(TestCase):
    def setUp(self):
        super().setUp()
        self.tournament = Tournament.objects.create(name="Stance Cup", slug="stance-cup")
        self.round = Round.objects.create(tournament=self.tournament, seq=1, name="Round 1", abbreviation="R1")
        self.motion = Motion.objects.create(
            tournament=self.tournament,
            text="This House would ban the development of autonomous weapons systems.",
            reference="Motion 1",
        )
        self.motion.rounds.add(self.round)
        self.institution = Institution.objects.create(code="INS", name="Debate Institution")

        self.team1 = Team.objects.create(tournament=self.tournament, institution=self.institution, reference="Gov Team")
        self.team2 = Team.objects.create(tournament=self.tournament, institution=self.institution, reference="Opp Team")
        self.speaker1 = Speaker.objects.create(team=self.team1, name="Alice")
        self.speaker2 = Speaker.objects.create(team=self.team2, name="Bob")

        self.debate = Debate.objects.create(round=self.round)
        DebateTeam.objects.create(debate=self.debate, team=self.team1, side=0)
        DebateTeam.objects.create(debate=self.debate, team=self.team2, side=1)

        audio_file = SimpleUploadedFile("speech.wav", b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00", content_type="audio/wav")
        self.recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=audio_file,
            status=SpeechRecording.Status.COMPLETED,
        )
        self.transcript = SpeechTranscript.objects.create(
            recording=self.recording,
            text="Thank you Madam Speaker. Autonomous weapons undermine ethical accountability and must be banned.",
            language_code="en",
            metadata={
                "provider": "google",
                "segments": [
                    {
                        "speaker": "Speaker 1",
                        "text": "Thank you Madam Speaker.",
                        "start": 0.0,
                        "end": 2.0,
                        "confidence": 0.95,
                    },
                    {
                        "speaker": "Speaker 1",
                        "text": "Autonomous weapons undermine ethical accountability and must be banned.",
                        "start": 2.0,
                        "end": 8.0,
                        "confidence": 0.98,
                    },
                ],
            },
        )

        # Another tournament for permission/isolation testing
        self.other_tournament = Tournament.objects.create(name="Other Cup", slug="other-cup")
        self.other_round = Round.objects.create(tournament=self.other_tournament, seq=1, name="R1", abbreviation="R1")
        self.other_debate = Debate.objects.create(round=self.other_round)
        other_audio = SimpleUploadedFile("other.wav", b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00", content_type="audio/wav")
        self.other_recording = SpeechRecording.objects.create(
            debate=self.other_debate,
            audio_file=other_audio,
            status=SpeechRecording.Status.COMPLETED,
        )

        self.admin_user = User.objects.create_superuser(username="admin", password="password", email="admin@test.com")
        self.client.force_login(self.admin_user)

    def test_stance_analysis_url_isolation(self):
        """Cross-tournament recording access returns 404."""
        url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.other_tournament.slug,
            "recording_id": self.recording.id,
        })
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 404)

    def test_unauthenticated_access_denied(self):
        """Unauthenticated requests are redirected or rejected."""
        self.client.logout()
        url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        resp = self.client.post(url)
        self.assertIn(resp.status_code, [302, 403])

    def test_run_stance_analysis_json_success(self):
        """POST with JSON Accept header runs analysis, stores metadata, and returns JSON payload."""
        url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        resp = self.client.post(url, HTTP_ACCEPT="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("stance_analysis", data)

        stance_res = data["stance_analysis"]
        self.assertEqual(stance_res["motion"], self.motion.text)
        self.assertEqual(stance_res["stats"]["total_sentences"], 2)
        self.assertEqual(stance_res["stats"]["total_skipped_procedural"], 1)
        self.assertEqual(stance_res["stats"]["total_evaluated"], 1)

        # Verify persisted in database
        self.transcript.refresh_from_db()
        self.assertIn("stance_analysis", self.transcript.metadata)
        self.assertEqual(self.transcript.metadata["stance_analysis"]["motion"], self.motion.text)

    def test_run_stance_analysis_html_redirect_and_template_display(self):
        """POST with HTML redirect sets flash message and populates transcript detail page."""
        detail_url = reverse("speech-transcript-detail", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        stance_url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })

        # Run analysis via HTML POST
        resp = self.client.post(stance_url, HTTP_REFERER=detail_url, HTTP_ACCEPT="text/html", follow=True)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Stance analysis completed successfully!")

        # Verify UI template elements
        self.assertContains(resp, "Topic Stance Analysis (DeBERTa NLI)")
        self.assertContains(resp, "Experimental NLP Analysis Notice")
        self.assertContains(resp, "Autonomous weapons undermine ethical accountability and must be banned.")
        self.assertContains(resp, "PRO")
        self.assertContains(resp, "Thank you Madam Speaker.")
        self.assertContains(resp, "MATCHED_PROCEDURAL_HEURISTIC")
        self.assertContains(resp, "Re-run Stance Analysis")

    def test_missing_transcript_error_handling(self):
        """Recording without transcript returns 400."""
        audio = SimpleUploadedFile("empty.wav", b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00", content_type="audio/wav")
        recording_no_transcript = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker2,
            audio_file=audio,
            status=SpeechRecording.Status.COMPLETED,
        )
        url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": recording_no_transcript.id,
        })
        resp = self.client.post(url, HTTP_ACCEPT="application/json")
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.json()["success"])

    def test_model_loading_failure_gracefully_handled(self):
        """If model checkpoint fails to load, view returns 503 with informative error."""
        url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        with patch("speech_analytics.views.get_stance_classifier", side_effect=StanceModelLoadError("Weights corrupt")):
            resp = self.client.post(url, HTTP_ACCEPT="application/json")
            self.assertEqual(resp.status_code, 503)
            data = resp.json()
            self.assertFalse(data["success"])
            self.assertIn("Weights corrupt", data["error"])

    def test_custom_motion_override_respected(self):
        """User can submit custom motion override if debate has no motion assigned."""
        custom_motion = "This House would legalize all drugs."
        url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        resp = self.client.post(url, data={"custom_motion": custom_motion}, HTTP_ACCEPT="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["stance_analysis"]["motion"], custom_motion)
        self.assertEqual(data["stance_analysis"]["motion_source"], "custom_override")

        # Verify custom override does NOT pollute or modify official tournament motions
        self.assertFalse(Motion.objects.filter(text=custom_motion).exists())

    def test_confirmed_ballot_motion_precedence(self):
        """Confirmed ballot motion takes precedence over round-assigned motions."""
        from motions.models import RoundMotion
        ballot_motion = Motion.objects.create(
            tournament=self.tournament,
            text="This House would ban private schooling.",
            reference="Private Schooling",
        )
        RoundMotion.objects.create(round=self.round, motion=ballot_motion, seq=2)

        from results.models import BallotSubmission
        # Create confirmed ballot with ballot_motion
        ballot = BallotSubmission.objects.create(
            debate=self.debate,
            confirmed=True,
            motion=ballot_motion,
        )

        url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        resp = self.client.post(url, HTTP_ACCEPT="application/json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["stance_analysis"]["motion"], ballot_motion.text)
        self.assertEqual(data["stance_analysis"]["motion_source"], "official_debate_context")

    def test_cached_analysis_invalidated_or_stale_when_motion_changes(self):
        """Detail view detects when cached analysis motion differs from current official motion."""
        # 1. Run analysis with initial motion
        stance_url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        detail_url = reverse("speech-transcript-detail", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        self.client.post(stance_url, HTTP_REFERER=detail_url, HTTP_ACCEPT="text/html")

        # 2. Detail page shows official debate motion
        resp = self.client.get(detail_url)
        self.assertContains(resp, "Official Debate Motion")
        self.assertNotContains(resp, "Motion Changed")

        # 3. Change the round motion text in database
        self.motion.text = "This House would mandate voting in national elections."
        self.motion.save()

        # 4. Detail page now flags analysis as stale/changed
        resp = self.client.get(detail_url)
        self.assertContains(resp, "Motion Changed")
        self.assertContains(resp, "This House would mandate voting in national elections.")

    def test_missing_motion_requires_motion_or_override(self):
        """When debate has no motion, UI shows prompt and stance view returns warning."""
        # Remove motion from round
        self.motion.rounds.clear()

        detail_url = reverse("speech-transcript-detail", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        resp = self.client.get(detail_url)
        self.assertContains(resp, "No Motion Assigned to this Debate")
        self.assertContains(resp, "Supply Analysis Motion Override")

        # POST without custom_motion produces empty prediction and warning
        stance_url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        post_resp = self.client.post(stance_url, HTTP_ACCEPT="application/json")
        self.assertEqual(post_resp.status_code, 200)
        data = post_resp.json()
        self.assertEqual(data["stance_analysis"]["motion"], "")
        self.assertIn("No primary motion text found in debate context", data["stance_analysis"]["warnings"][0])

    def test_evidence_reference_and_timestamps_reach_ui(self):
        """Timestamps and approximate flags render in UI table."""
        stance_url = reverse("speech-transcript-stance", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        detail_url = reverse("speech-transcript-detail", kwargs={
            "tournament_slug": self.tournament.slug,
            "recording_id": self.recording.id,
        })
        self.client.post(stance_url, HTTP_REFERER=detail_url, HTTP_ACCEPT="text/html")

        resp = self.client.get(detail_url)
        self.assertEqual(resp.status_code, 200)
        # Check start & end timestamps render
        self.assertContains(resp, "2.0s -")
        self.assertContains(resp, "8.0s")
        # Check approx indicator
        self.assertContains(resp, "Approx")
