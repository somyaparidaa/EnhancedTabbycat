import io
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from draw.models import Debate, DebateTeam
from participants.models import Institution, Speaker, Team
from speech_analytics.models import SpeechRecording, SpeechTranscript, speech_recording_upload_path
from tournaments.models import Round, Tournament


class SpeechRecordingModelTests(TestCase):
    def setUp(self):
        super().setUp()
        self.tournament = Tournament.objects.create(name="STT Championship", slug="stt-champ")
        self.round = Round.objects.create(tournament=self.tournament, seq=1, name="Round 1", abbreviation="R1")
        self.institution = Institution.objects.create(code="INS", name="Institute of Tech")

        self.team1 = Team.objects.create(tournament=self.tournament, institution=self.institution, reference="Team 1")
        self.team2 = Team.objects.create(tournament=self.tournament, institution=self.institution, reference="Team 2")
        self.team3 = Team.objects.create(tournament=self.tournament, institution=self.institution, reference="Team 3")

        self.speaker1 = Speaker.objects.create(team=self.team1, name="Alice")
        self.speaker2 = Speaker.objects.create(team=self.team2, name="Bob")
        self.outsider_speaker = Speaker.objects.create(team=self.team3, name="Charlie")

        # Debate with team1 and team2
        self.debate = Debate.objects.create(round=self.round)
        DebateTeam.objects.create(debate=self.debate, team=self.team1, side=0)
        DebateTeam.objects.create(debate=self.debate, team=self.team2, side=1)

    def _create_dummy_audio(self, name="speech.wav"):
        return SimpleUploadedFile(name, b"dummy audio content", content_type="audio/wav")

    def test_valid_speaker_assignment(self):
        """A speaker whose team is competing in the debate can be successfully assigned."""
        recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_dummy_audio("alice.wav"),
        )
        self.assertIsNotNone(recording.pk)
        self.assertEqual(recording.speaker, self.speaker1)
        self.assertEqual(recording.status, SpeechRecording.Status.PENDING)

    def test_speaker_belonging_to_different_debate_raises_validation_error(self):
        """A speaker whose team is not competing in the debate is rejected."""
        with self.assertRaises(ValidationError) as ctx:
            SpeechRecording.objects.create(
                debate=self.debate,
                speaker=self.outsider_speaker,
                audio_file=self._create_dummy_audio("outsider.wav"),
            )
        self.assertIn('speaker', ctx.exception.message_dict)

    def test_recording_without_associated_speaker(self):
        """A recording can be created without specifying an individual speaker."""
        recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=None,
            audio_file=self._create_dummy_audio("debate_overview.wav"),
        )
        self.assertIsNotNone(recording.pk)
        self.assertIsNone(recording.speaker)

    def test_multiple_recordings_for_same_speaker_and_debate(self):
        """Multiple recordings for the same speaker in the same debate are allowed."""
        rec1 = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_dummy_audio("substantive.wav"),
        )
        rec2 = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_dummy_audio("reply.wav"),
        )
        self.assertNotEqual(rec1.pk, rec2.pk)
        self.assertEqual(SpeechRecording.objects.filter(debate=self.debate, speaker=self.speaker1).count(), 2)

    def test_unique_upload_filenames(self):
        """Uploading two files with the same initial filename generates distinct paths."""
        rec1 = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_dummy_audio("test.wav"),
        )
        rec2 = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker2,
            audio_file=self._create_dummy_audio("test.wav"),
        )
        self.assertNotEqual(rec1.audio_file.name, rec2.audio_file.name)
        self.assertIn("speech_recordings/stt-champ/round_1/debate_", rec1.audio_file.name)
        self.assertIn("speech_recordings/stt-champ/round_1/debate_", rec2.audio_file.name)
        self.assertTrue(rec1.audio_file.name.endswith(".wav"))

    def test_invalid_debate_relationship_in_upload_path(self):
        """upload_path raises explicit ValidationError when debate hierarchy is invalid."""
        unsaved_recording = SpeechRecording(debate=None)
        with self.assertRaises(ValidationError):
            speech_recording_upload_path(unsaved_recording, "test.wav")

    def test_speech_transcript_one_to_one(self):
        """SpeechTranscript maintains 1:1 relationship with SpeechRecording."""
        recording = SpeechRecording.objects.create(
            debate=self.debate,
            speaker=self.speaker1,
            audio_file=self._create_dummy_audio("alice.wav"),
        )
        transcript = SpeechTranscript.objects.create(
            recording=recording,
            text="Members of the house...",
            language_code="en",
            metadata={"confidence": 0.98},
        )
        self.assertEqual(recording.transcript, transcript)

        # Attempting a second transcript for the same recording raises ValidationError/IntegrityError
        with self.assertRaises(Exception):
            SpeechTranscript.objects.create(
                recording=recording,
                text="Duplicate transcript...",
            )
