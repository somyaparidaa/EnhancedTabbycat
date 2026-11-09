import io
import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from draw.models import Debate, DebateTeam
from participants.models import Institution, Speaker, Team
from speech_analytics.models import SpeechRecording, SpeechTranscript
from speech_analytics.services.exceptions import GnaniAPIResponseError, GnaniAuthenticationError
from tournaments.models import Round, Tournament

User = get_user_model()


@override_settings(
    GNANI_API_KEY="test-mock-api-key",
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
class SpeechRecordingWorkflowTests(TestCase):
    def setUp(self):
        super().setUp()
        self.tournament = Tournament.objects.create(name="STT Cup", slug="stt-cup")
        self.round = Round.objects.create(tournament=self.tournament, seq=1, name="Round 1", abbreviation="R1")
        self.institution = Institution.objects.create(code="INS", name="Debate Institution")

        self.team1 = Team.objects.create(tournament=self.tournament, institution=self.institution, reference="Team 1")
        self.team2 = Team.objects.create(tournament=self.tournament, institution=self.institution, reference="Team 2")
        self.speaker1 = Speaker.objects.create(team=self.team1, name="Alice")
        self.speaker2 = Speaker.objects.create(team=self.team2, name="Bob")

        self.debate = Debate.objects.create(round=self.round)
        DebateTeam.objects.create(debate=self.debate, team=self.team1, side=0)
        DebateTeam.objects.create(debate=self.debate, team=self.team2, side=1)

        # Another tournament for authorization/scoping tests
        self.other_tournament = Tournament.objects.create(name="Other Cup", slug="other-cup")
        self.other_round = Round.objects.create(tournament=self.other_tournament, seq=1, name="R1", abbreviation="R1")
        self.other_debate = Debate.objects.create(round=self.other_round)

        # Superuser for admin tests
        self.admin_user = User.objects.create_superuser(username="admin", password="password", email="admin@test.com")
        self.client.force_login(self.admin_user)

    def _create_audio_file(self, name="speech.wav", content=b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00", content_type="audio/wav"):
        return SimpleUploadedFile(name, content, content_type=content_type)

    # 1. Successful upload & submission workflow
    # 1. Upload and transcription submission via JSON request
    @patch('speech_analytics.views.GnaniPrismaBatchClient')
    def test_upload_and_transcription_submission_success(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_client.create_batch_job.return_value = {'job_id': 'gnani_job_999', 'status': 'CREATED'}
        mock_client.start_batch_job.return_value = {'job_id': 'gnani_job_999', 'status': 'STARTING'}

        upload_url = reverse('speech-recording-upload', kwargs={'tournament_slug': self.tournament.slug})
        audio_file = self._create_audio_file()

        response = self.client.post(
            upload_url,
            {
                'debate': self.debate.id,
                'speaker': self.speaker1.id,
                'audio_file': audio_file,
            },
            HTTP_ACCEPT='application/json',
        )

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertNotIn('job_id', data)
        self.assertEqual(data['status'], SpeechRecording.Status.PROCESSING)

        recording = SpeechRecording.objects.get(pk=data['recording_id'])
        self.assertEqual(recording.job_id, 'gnani_job_999')
        self.assertEqual(recording.status, SpeechRecording.Status.PROCESSING)
        self.assertEqual(recording.speaker, self.speaker1)
        mock_client.create_batch_job.assert_called_once()
        mock_client.start_batch_job.assert_called_once_with('gnani_job_999')

    # 2. Upload with job start failure retains job_id and sets FAILED
    @patch('speech_analytics.views.GnaniPrismaBatchClient')
    def test_upload_job_start_failure_retains_job_id(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_client.create_batch_job.return_value = {'job_id': 'gnani_job_start_fail', 'status': 'CREATED'}
        mock_client.start_batch_job.side_effect = GnaniAPIResponseError("Start failed", status_code=500)

        upload_url = reverse('speech-recording-upload', kwargs={'tournament_slug': self.tournament.slug})
        audio_file = self._create_audio_file()

        response = self.client.post(
            upload_url,
            {
                'debate': self.debate.id,
                'speaker': self.speaker1.id,
                'audio_file': audio_file,
            },
            HTTP_ACCEPT='application/json',
        )

        self.assertEqual(response.status_code, 502)
        recording = SpeechRecording.objects.get(debate=self.debate, speaker=self.speaker1)
        # job_id must be retained so it can be investigated/recovered without duplicate creation
        self.assertEqual(recording.job_id, 'gnani_job_start_fail')
        self.assertEqual(recording.status, SpeechRecording.Status.FAILED)
        self.assertIn("Start failed", recording.error_message)

    # 3. Audio file validation: rejects invalid extension and non-audio magic bytes
    def test_audio_file_validation_rejections(self):
        upload_url = reverse('speech-recording-upload', kwargs={'tournament_slug': self.tournament.slug})

        # Disallowed extension (.txt)
        bad_ext_file = SimpleUploadedFile("speech.txt", b"plain text content", content_type="text/plain")
        response = self.client.post(
            upload_url,
            {
                'debate': self.debate.id,
                'speaker': self.speaker1.id,
                'audio_file': bad_ext_file,
            },
            HTTP_ACCEPT='application/json',
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('audio_file', response.json()['errors'])

        # Allowed extension but invalid binary signature (fake wav)
        fake_wav_file = SimpleUploadedFile("speech.wav", b"NOT_A_REAL_AUDIO_FILE", content_type="audio/wav")
        response = self.client.post(
            upload_url,
            {
                'debate': self.debate.id,
                'speaker': self.speaker1.id,
                'audio_file': fake_wav_file,
            },
            HTTP_ACCEPT='application/json',
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('audio_file', response.json()['errors'])

    # 4. Status Check Endpoint: In-Progress polling
    @patch('speech_analytics.views.GnaniPrismaBatchClient')
    def test_status_check_in_progress(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_client.get_job_status.return_value = {'job_id': 'job_123', 'status': 'IN_PROGRESS', 'progress': 50}

        recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_audio_file(),
            job_id='job_123',
            status=SpeechRecording.Status.PROCESSING,
        )

        status_url = reverse('speech-recording-status', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording.id,
        })
        response = self.client.get(status_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], SpeechRecording.Status.PROCESSING)
        self.assertNotIn('job_id', data)
        self.assertFalse(data['transcript_available'])
        self.assertFalse(SpeechTranscript.objects.filter(recording=recording).exists())

    # 5. Status Check Endpoint: Completed job fetches transcript and sets COMPLETED
    @patch('speech_analytics.views.GnaniPrismaBatchClient')
    def test_status_check_completed_saves_transcript(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_client.get_job_status.return_value = {'job_id': 'job_123', 'status': 'COMPLETED'}
        mock_client.fetch_completed_transcript.return_value = {
            'text': "Debate speech text from Prisma.",
            'language_code': 'en-IN',
            'metadata': {'confidence': 0.99},
        }

        recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_audio_file(),
            job_id='job_123',
            status=SpeechRecording.Status.PROCESSING,
        )

        status_url = reverse('speech-recording-status', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording.id,
        })
        response = self.client.get(status_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], SpeechRecording.Status.COMPLETED)
        self.assertTrue(data['transcript_available'])

        recording.refresh_from_db()
        self.assertEqual(recording.status, SpeechRecording.Status.COMPLETED)
        transcript = recording.transcript
        self.assertEqual(transcript.text, "Debate speech text from Prisma.")
        self.assertEqual(transcript.language_code, "en-IN")

    # 6. Status Check: Repeated check on completed recording is idempotent (zero API calls)
    @patch('speech_analytics.views.GnaniPrismaBatchClient')
    def test_repeated_status_check_on_completed_recording(self, MockClientClass):
        mock_client = MockClientClass.return_value

        recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_audio_file(),
            job_id='job_already_done',
            status=SpeechRecording.Status.COMPLETED,
        )
        SpeechTranscript.objects.create(
            recording=recording,
            text="Already saved transcript.",
            language_code="en",
        )

        status_url = reverse('speech-recording-status', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording.id,
        })
        response = self.client.get(status_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], SpeechRecording.Status.COMPLETED)
        self.assertTrue(data['transcript_available'])
        # Should not contact Gnani API at all
        mock_client.get_job_status.assert_not_called()
        mock_client.fetch_completed_transcript.assert_not_called()

    # 7. Status Check: Failed job updates status and error message
    @patch('speech_analytics.views.GnaniPrismaBatchClient')
    def test_status_check_failed_job(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_client.get_job_status.return_value = {
            'job_id': 'job_fail',
            'status': 'FAILED',
            'error': 'Unsupported audio format detected during batch processing',
        }

        recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_audio_file(),
            job_id='job_fail',
            status=SpeechRecording.Status.PROCESSING,
        )

        status_url = reverse('speech-recording-status', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording.id,
        })
        response = self.client.get(status_url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], SpeechRecording.Status.FAILED)
        self.assertIn("Unsupported audio format", data['error'])

        recording.refresh_from_db()
        self.assertEqual(recording.status, SpeechRecording.Status.FAILED)

    # 8. Cross-tournament access restriction
    def test_cross_tournament_access_rejected(self):
        # Recording belongs to self.tournament
        recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_audio_file(),
        )

        # Attempt to access recording under other_tournament URL
        wrong_url = reverse('speech-recording-status', kwargs={
            'tournament_slug': self.other_tournament.slug,
            'recording_id': recording.id,
        })
        response = self.client.get(wrong_url)
        self.assertEqual(response.status_code, 404)

    # 9. Unauthenticated access denied
    def test_unauthenticated_access_denied(self):
        self.client.logout()
        upload_url = reverse('speech-recording-upload', kwargs={'tournament_slug': self.tournament.slug})
        response = self.client.get(upload_url)
        # Tabbycat AdministratorMixin redirects unauthenticated users to login
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    # 10. Dashboard loads for authorized administrator
    def test_dashboard_loads_for_administrator(self):
        dashboard_url = reverse('speech-analytics-dashboard', kwargs={'tournament_slug': self.tournament.slug})
        response = self.client.get(dashboard_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'speech_analytics/recording_list.html')
        self.assertIn('recordings', response.context)

    # 11. Dashboard shows only recordings for the current tournament and renders all 4 statuses
    def test_dashboard_filters_by_tournament_and_renders_statuses(self):
        # 4 recordings with different statuses for current tournament
        rec_pending = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_audio_file("pending.wav"),
            status=SpeechRecording.Status.PENDING,
        )
        rec_proc = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker2,
            audio_file=self._create_audio_file("proc.wav"),
            status=SpeechRecording.Status.PROCESSING,
        )
        rec_comp = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_audio_file("comp.wav"),
            status=SpeechRecording.Status.COMPLETED,
        )
        SpeechTranscript.objects.create(
            recording=rec_comp,
            text="Completed debate speech.",
            language_code="en",
        )
        rec_failed = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker2,
            audio_file=self._create_audio_file("failed.wav"),
            status=SpeechRecording.Status.FAILED,
            error_message="Provider timeout",
        )

        # Recording for another tournament
        rec_other = SpeechRecording.objects.create(
            debate=self.other_debate,
            audio_file=self._create_audio_file("other.wav"),
            status=SpeechRecording.Status.COMPLETED,
        )

        dashboard_url = reverse('speech-analytics-dashboard', kwargs={'tournament_slug': self.tournament.slug})
        response = self.client.get(dashboard_url)
        self.assertEqual(response.status_code, 200)

        rendered_recordings = list(response.context['recordings'])
        self.assertEqual(len(rendered_recordings), 4)
        self.assertIn(rec_pending, rendered_recordings)
        self.assertIn(rec_proc, rendered_recordings)
        self.assertIn(rec_comp, rendered_recordings)
        self.assertIn(rec_failed, rendered_recordings)
        self.assertNotIn(rec_other, rendered_recordings)

        # HTML content check for badges and links
        content = response.content.decode('utf-8')
        self.assertIn("Pending", content)
        self.assertIn("Processing", content)
        self.assertIn("Completed", content)
        self.assertIn("Failed", content)
        self.assertIn("View Transcript", content)

    # 12. Dashboard empty state renders when no recordings exist
    def test_dashboard_empty_state(self):
        dashboard_url = reverse('speech-analytics-dashboard', kwargs={'tournament_slug': self.tournament.slug})
        response = self.client.get(dashboard_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No speech recordings uploaded yet")
        self.assertContains(response, "Upload First Recording")

    # 13. Unauthenticated access to dashboard denied
    def test_dashboard_unauthenticated_denied(self):
        self.client.logout()
        dashboard_url = reverse('speech-analytics-dashboard', kwargs={'tournament_slug': self.tournament.slug})
        response = self.client.get(dashboard_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

    # 14. Authenticated non-admin access to dashboard denied
    def test_dashboard_non_admin_authenticated_denied(self):
        non_admin = User.objects.create_user(username="regular_user", password="password")
        self.client.force_login(non_admin)
        dashboard_url = reverse('speech-analytics-dashboard', kwargs={'tournament_slug': self.tournament.slug})
        response = self.client.get(dashboard_url)
        self.assertEqual(response.status_code, 403)

    # 15. GET upload page renders successfully for tournament admin
    def test_upload_page_get_renders_successfully(self):
        upload_url = reverse('speech-recording-upload', kwargs={'tournament_slug': self.tournament.slug})
        response = self.client.get(upload_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'speech_analytics/recording_upload.html')

        # Form fields present
        self.assertIn('form', response.context)
        form = response.context['form']
        self.assertIn('debate', form.fields)
        self.assertIn('speaker', form.fields)
        self.assertIn('audio_file', form.fields)

        # Content assertions
        content = response.content.decode('utf-8')
        self.assertIn('name="csrfmiddlewaretoken"', content)
        self.assertIn('enctype="multipart/form-data"', content)
        self.assertIn('Supported Formats', content)
        self.assertIn('Maximum File Size', content)
        self.assertIn('Cancel and return to dashboard', content)

    # 16. Unauthenticated and non-admin access to upload page denied
    def test_upload_page_unauthorized_denied(self):
        upload_url = reverse('speech-recording-upload', kwargs={'tournament_slug': self.tournament.slug})

        # Unauthenticated -> redirect to login
        self.client.logout()
        response = self.client.get(upload_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

        # Authenticated non-admin -> 403
        non_admin = User.objects.create_user(username="upload_regular_user", password="password")
        self.client.force_login(non_admin)
        response = self.client.get(upload_url)
        self.assertEqual(response.status_code, 403)

    # 17. Successful HTML form submission redirects to dashboard with success message
    @patch('speech_analytics.views.GnaniPrismaBatchClient')
    def test_upload_form_post_success_redirects_to_dashboard(self, MockClientClass):
        mock_client = MockClientClass.return_value
        mock_client.create_batch_job.return_value = {'job_id': 'gnani_job_html_123', 'status': 'CREATED'}
        mock_client.start_batch_job.return_value = {'job_id': 'gnani_job_html_123', 'status': 'STARTING'}

        upload_url = reverse('speech-recording-upload', kwargs={'tournament_slug': self.tournament.slug})
        dashboard_url = reverse('speech-analytics-dashboard', kwargs={'tournament_slug': self.tournament.slug})
        audio_file = self._create_audio_file("html_upload.wav")

        response = self.client.post(
            upload_url,
            {
                'debate': self.debate.id,
                'speaker': self.speaker1.id,
                'audio_file': audio_file,
            },
            follow=True,
        )

        self.assertRedirects(response, dashboard_url)
        # Verify success message displayed
        messages_list = list(response.context['messages'])
        self.assertTrue(any("uploaded successfully" in str(m) for m in messages_list))

        # Verify job was recorded and started
        recording = SpeechRecording.objects.get(debate=self.debate, speaker=self.speaker1)
        self.assertEqual(recording.job_id, 'gnani_job_html_123')
        self.assertEqual(recording.status, SpeechRecording.Status.PROCESSING)
        # Verify job_id is NOT in the rendered HTML
        self.assertNotIn('gnani_job_html_123', response.content.decode('utf-8'))
        mock_client.create_batch_job.assert_called_once()
        mock_client.start_batch_job.assert_called_once_with('gnani_job_html_123')

    # 18. Invalid HTML form submission renders errors and preserves form context
    def test_upload_form_post_invalid_renders_errors(self):
        upload_url = reverse('speech-recording-upload', kwargs={'tournament_slug': self.tournament.slug})
        bad_file = SimpleUploadedFile("speech.txt", b"plain text content", content_type="text/plain")

        response = self.client.post(
            upload_url,
            {
                'debate': self.debate.id,
                'speaker': self.speaker1.id,
                'audio_file': bad_file,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'speech_analytics/recording_upload.html')
        self.assertIn('form', response.context)
        form = response.context['form']
        self.assertTrue(form.errors)
        self.assertIn('audio_file', form.errors)
        # Context preserved
        self.assertEqual(form.data.get('speaker'), str(self.speaker1.id))
        self.assertContains(response, "Please correct the errors indicated below.")

    # 19. Transcript Detail: Completed recording displays full transcript and diarization
    def test_transcript_detail_completed_displays_transcript(self):
        recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_audio_file("comp_detail.wav"),
            status=SpeechRecording.Status.COMPLETED,
        )
        SpeechTranscript.objects.create(
            recording=recording,
            text="This is the winning speech transcript.",
            language_code="en-IN",
            metadata={
                "model": "prisma-v2.5",
                "diarization": [
                    {"speaker": "Speaker 1", "start": 0.0, "end": 4.5, "text": "This is the winning", "confidence": 0.95},
                    {"speaker": "Speaker 1", "start": 4.6, "end": 6.8, "text": "speech transcript.", "confidence": 0.98},
                ],
            },
        )

        detail_url = reverse('speech-transcript-detail', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording.id,
        })
        response = self.client.get(detail_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'speech_analytics/transcript_detail.html')
        self.assertContains(response, "This is the winning speech transcript.")
        self.assertContains(response, "Speaker Diarization & Segments")
        self.assertContains(response, "Speaker 1")
        self.assertContains(response, "en-IN")
        self.assertContains(response, "Alice")
        self.assertContains(response, "Audio Playback")

    # 20. Transcript Detail: Pending and Processing states display gracefully without crash
    def test_transcript_detail_pending_and_processing_states(self):
        recording_p = SpeechRecording.objects.create(
            debate=self.debate,
            audio_file=self._create_audio_file("pending_detail.wav"),
            status=SpeechRecording.Status.PENDING,
        )
        detail_url = reverse('speech-transcript-detail', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording_p.id,
        })
        response = self.client.get(detail_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Pending Transcription")
        self.assertContains(response, "No Transcript Available")

        recording_p.status = SpeechRecording.Status.PROCESSING
        recording_p.save()
        response2 = self.client.get(detail_url)
        self.assertEqual(response2.status_code, 200)
        self.assertContains(response2, "Processing Audio")

    # 21. Transcript Detail: Failed recording displays safe error information
    def test_transcript_detail_failed_state(self):
        recording_f = SpeechRecording.objects.create(
            debate=self.debate,
            audio_file=self._create_audio_file("failed_detail.wav"),
            status=SpeechRecording.Status.FAILED,
            error_message="Audio sample rate not supported by engine",
        )
        detail_url = reverse('speech-transcript-detail', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording_f.id,
        })
        response = self.client.get(detail_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Transcription Failed")
        self.assertContains(response, "Audio sample rate not supported by engine")

    # 22. Transcript Detail: Cross-tournament access returns 404
    def test_transcript_detail_cross_tournament_404(self):
        recording_other = SpeechRecording.objects.create(
            debate=self.other_debate,
            audio_file=self._create_audio_file("other_rec.wav"),
            status=SpeechRecording.Status.COMPLETED,
        )
        detail_url = reverse('speech-transcript-detail', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording_other.id,
        })
        response = self.client.get(detail_url)
        self.assertEqual(response.status_code, 404)

    # 23. Transcript Detail: Unauthorized access denied
    def test_transcript_detail_unauthorized_denied(self):
        recording = SpeechRecording.objects.create(
            debate=self.debate,
            audio_file=self._create_audio_file("auth_rec.wav"),
            status=SpeechRecording.Status.COMPLETED,
        )
        detail_url = reverse('speech-transcript-detail', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording.id,
        })

        # Unauthenticated -> redirect to login
        self.client.logout()
        response = self.client.get(detail_url)
        self.assertEqual(response.status_code, 302)
        self.assertIn('/accounts/login/', response.url)

        # Authenticated non-admin -> 403
        non_admin = User.objects.create_user(username="detail_non_admin", password="password")
        self.client.force_login(non_admin)
        response_non_admin = self.client.get(detail_url)
        self.assertEqual(response_non_admin.status_code, 403)

    # 24. Audio Streaming Endpoint: Requires admin permission, respects scoping, streams audio
    def test_audio_stream_view(self):
        recording = SpeechRecording.objects.create(
            debate=self.debate,
            audio_file=self._create_audio_file("stream_test.wav"),
            status=SpeechRecording.Status.COMPLETED,
        )
        audio_url = reverse('speech-recording-audio', kwargs={
            'tournament_slug': self.tournament.slug,
            'recording_id': recording.id,
        })

        # Admin user gets file stream
        response = self.client.get(audio_url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('audio', response['Content-Type'])
        # Consume streaming content
        content = b"".join(response.streaming_content)
        self.assertGreater(len(content), 0)

        # Cross-tournament access returns 404
        cross_url = reverse('speech-recording-audio', kwargs={
            'tournament_slug': self.other_tournament.slug,
            'recording_id': recording.id,
        })
        response_cross = self.client.get(cross_url)
        self.assertEqual(response_cross.status_code, 404)

        # Unauthenticated -> redirect to login
        self.client.logout()
        response_anon = self.client.get(audio_url)
        self.assertEqual(response_anon.status_code, 302)

        # Authenticated non-admin -> 403
        non_admin = User.objects.create_user(username="audio_stream_non_admin", password="password")
        self.client.force_login(non_admin)
        response_non_admin = self.client.get(audio_url)
        self.assertEqual(response_non_admin.status_code, 403)
