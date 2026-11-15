"""Unit tests for DebateContextBuilder and DebateAnalysisInputBundle."""

from django.test import TestCase

from draw.models import Debate, DebateSide, DebateTeam
from motions.models import Motion, RoundMotion
from participants.models import Institution, Speaker, Team
from results.models import BallotSubmission, SpeakerScore
from speech_analytics.contract import validate_debate_analysis_input
from speech_analytics.models import SpeechRecording, SpeechTranscript
from speech_analytics.services.bundle import build_debate_analysis_input
from speech_analytics.services.context_builder import DebateContextBuilder
from tournaments.models import Round, Tournament


class DebateContextBuilderTests(TestCase):
    """Test suite for extracting Tabbycat debate domain models into context dictionaries."""

    def setUp(self):
        # 1. Create BP tournament
        self.tournament_bp = Tournament.objects.create(
            name="World Universities Debating Championship",
            slug="wudc-test",
        )
        self.tournament_bp.preferences['debate_rules__teams_in_debate'] = 4
        self.tournament_bp.preferences['debate_rules__substantive_speakers'] = 2
        self.tournament_bp.preferences['debate_rules__reply_scores_enabled'] = False

        self.round_bp = Round.objects.create(
            tournament=self.tournament_bp,
            seq=1,
            name="Round 1",
            stage=Round.Stage.PRELIMINARY,
            draw_type=Round.DrawType.RANDOM,
        )

        self.motion_bp = Motion.objects.create(
            tournament=self.tournament_bp,
            text="This House would ban targeted advertising.",
            reference="Ban Ads",
            info_slide="Targeted advertising tracks browsing habits across sites.",
        )
        RoundMotion.objects.create(round=self.round_bp, motion=self.motion_bp, seq=1)

        self.debate_bp = Debate.objects.create(round=self.round_bp)

        # Teams and speakers for BP
        inst = Institution.objects.create(name="Oxford", code="OX")
        self.teams_bp = []
        self.speakers_bp = []
        for side_idx, side_enum in enumerate([DebateSide.OG, DebateSide.OO, DebateSide.CG, DebateSide.CO]):
            team = Team.objects.create(
                tournament=self.tournament_bp,
                short_name=f"Team {side_enum}",
                reference=f"T{side_idx}",
                institution=inst,
            )
            DebateTeam.objects.create(debate=self.debate_bp, team=team, side=side_enum)
            spk1 = Speaker.objects.create(name=f"Speaker {side_enum}.1", team=team)
            spk2 = Speaker.objects.create(name=f"Speaker {side_enum}.2", team=team)
            self.teams_bp.append(team)
            self.speakers_bp.extend([spk1, spk2])

        # Confirmed ballot for BP
        self.ballot_bp = BallotSubmission.objects.create(
            debate=self.debate_bp,
            confirmed=True,
            motion=self.motion_bp,
        )
        for dt in self.debate_bp.debateteam_set.all():
            team_spks = dt.team.speaker_set.all().order_by('id')
            SpeakerScore.objects.create(
                ballot_submission=self.ballot_bp,
                debate_team=dt,
                speaker=team_spks[0],
                position=1,
                score=75.0,
            )
            SpeakerScore.objects.create(
                ballot_submission=self.ballot_bp,
                debate_team=dt,
                speaker=team_spks[1],
                position=2,
                score=76.0,
            )

        # 2. Create 2-team (Australs) tournament
        self.tournament_australs = Tournament.objects.create(
            name="Australs Championship",
            slug="australs-test",
        )
        self.tournament_australs.preferences['debate_rules__teams_in_debate'] = 2
        self.tournament_australs.preferences['debate_rules__substantive_speakers'] = 3
        self.tournament_australs.preferences['debate_rules__reply_scores_enabled'] = True

        self.round_australs = Round.objects.create(
            tournament=self.tournament_australs,
            seq=2,
            name="Round 2",
            stage=Round.Stage.PRELIMINARY,
            draw_type=Round.DrawType.RANDOM,
        )
        self.motion_australs = Motion.objects.create(
            tournament=self.tournament_australs,
            text="This House supports universal basic income.",
            reference="UBI",
        )
        RoundMotion.objects.create(round=self.round_australs, motion=self.motion_australs, seq=1)
        self.debate_australs = Debate.objects.create(round=self.round_australs)

        team_aff = Team.objects.create(tournament=self.tournament_australs, short_name="Sydney Aff")
        team_neg = Team.objects.create(tournament=self.tournament_australs, short_name="Melbourne Neg")
        DebateTeam.objects.create(debate=self.debate_australs, team=team_aff, side=DebateSide.AFF)
        DebateTeam.objects.create(debate=self.debate_australs, team=team_neg, side=DebateSide.NEG)

    def test_bp_context_extraction_with_confirmed_ballot(self):
        builder = DebateContextBuilder()
        pm_speaker = self.speakers_bp[0]  # Speaker OG.1 (Prime Minister)

        context = builder.build_context(
            debate=self.debate_bp,
            target_speaker_id=pm_speaker.id,
            diarized_speakers=["Speaker 1"],
        )

        # Tournament rules
        self.assertEqual(context['tournament']['teams_in_debate'], 4)
        self.assertEqual(context['tournament']['format_name'], "British Parliamentary (BP)")

        # Motion & Info Slide
        self.assertEqual(context['motion']['primary_motion']['text'], "This House would ban targeted advertising.")
        self.assertEqual(context['motion']['primary_motion']['info_slide'], "Targeted advertising tracks browsing habits across sites.")

        # Teams
        self.assertEqual(len(context['teams']), 4)
        og_team = next(t for t in context['teams'] if t['side_code'] == 'OG')
        self.assertEqual(og_team['institution'], "Oxford")

        # Speeches & Roles
        self.assertEqual(len(context['speeches']), 8)
        pm_speech = next(s for s in context['speeches'] if s['speaker_id'] == pm_speaker.id)
        self.assertEqual(pm_speech['role_abbr'], "PM")
        self.assertEqual(pm_speech['role_name'], "Prime Minister")

        # Speaker mapping resolution
        mapping = context['speaker_mapping']
        self.assertEqual(mapping['status'], "RESOLVED_SINGLE_SPEAKER")
        self.assertEqual(mapping['target_speaker']['id'], pm_speaker.id)
        self.assertEqual(mapping['target_speaker']['speech_role'], "Prime Minister")
        self.assertEqual(mapping['unresolved_speakers'], [])

    def test_australs_two_team_context_extraction(self):
        builder = DebateContextBuilder()
        context = builder.build_context(
            debate=self.debate_australs,
            target_speaker_id=None,
            diarized_speakers=["Speaker 1", "Speaker 2"],
        )

        self.assertEqual(context['tournament']['teams_in_debate'], 2)
        self.assertEqual(context['tournament']['format_name'], "Australs / 3v3 with Reply")
        self.assertEqual(len(context['teams']), 2)
        self.assertEqual(context['teams'][0]['side_code'], "AFF")
        self.assertEqual(context['teams'][1]['side_code'], "NEG")

        # No ballot submitted yet: generates expected speech positions
        self.assertEqual(len(context['speeches']), 8)  # 3 substantive + 1 reply each
        aff1 = next(s for s in context['speeches'] if s['role_abbr'] == 'Aff1')
        self.assertIn("Speaker 1", aff1['role_name'])

        # Speaker mapping for unassigned debate-level recording
        self.assertEqual(context['speaker_mapping']['status'], "UNRESOLVED_MULTI_SPEAKER")
        self.assertEqual(len(context['speaker_mapping']['unresolved_speakers']), 2)

    def test_speaker_mapping_with_pois_and_interjections(self):
        builder = DebateContextBuilder()
        pm_speaker = self.speakers_bp[0]

        context = builder.build_context(
            debate=self.debate_bp,
            target_speaker_id=pm_speaker.id,
            diarized_speakers=["Speaker 1", "Speaker 2", "Speaker 3"],
            speaker_stats={
                "Speaker 1": {"word_count": 800, "duration": 360.0},
                "Speaker 2": {"word_count": 25, "duration": 12.0},
                "Speaker 3": {"word_count": 15, "duration": 8.0},
            },
        )
        mapping = context['speaker_mapping']
        self.assertEqual(mapping['status'], "RESOLVED_TARGET_WITH_POIS")
        self.assertEqual(mapping['target_speaker']['id'], pm_speaker.id)
        self.assertEqual(mapping['unresolved_speakers'], ["Speaker 2", "Speaker 3"])
        self.assertEqual(mapping['mapping']['Speaker 2']['type'], "unresolved_poi")
        self.assertEqual(mapping['resolution_metadata']['method'], "dominant_heuristic_word_or_duration")

    def test_speaker_mapping_chair_speaks_first_briefly(self):
        """Test case where Chair speaks first briefly (Speaker 1) but target debater is dominant (Speaker 2)."""
        builder = DebateContextBuilder()
        pm_speaker = self.speakers_bp[0]

        context = builder.build_context(
            debate=self.debate_bp,
            target_speaker_id=pm_speaker.id,
            diarized_speakers=["Speaker 1", "Speaker 2"],
            speaker_stats={
                "Speaker 1": {"word_count": 10, "duration": 4.0},   # Chair: "Order, order."
                "Speaker 2": {"word_count": 950, "duration": 410.0}, # Target debater: 7-minute speech
            },
        )
        mapping = context['speaker_mapping']
        self.assertEqual(mapping['status'], "RESOLVED_TARGET_WITH_POIS")
        self.assertEqual(mapping['resolution_metadata']['dominant_label'], "Speaker 2")
        self.assertEqual(mapping['mapping']['Speaker 2']['speaker_id'], pm_speaker.id)
        self.assertEqual(mapping['mapping']['Speaker 2']['type'], "target_speaker")
        self.assertEqual(mapping['unresolved_speakers'], ["Speaker 1"])
        self.assertEqual(mapping['mapping']['Speaker 1']['type'], "unresolved_poi")

    def test_speaker_mapping_ambiguous_equal_duration(self):
        """Test case where two diarized speakers have similar speaking times (e.g. cross-examination or co-speech)."""
        builder = DebateContextBuilder()
        pm_speaker = self.speakers_bp[0]

        context = builder.build_context(
            debate=self.debate_bp,
            target_speaker_id=pm_speaker.id,
            diarized_speakers=["Speaker 1", "Speaker 2"],
            speaker_stats={
                "Speaker 1": {"word_count": 450, "duration": 200.0},
                "Speaker 2": {"word_count": 430, "duration": 195.0},
            },
        )
        mapping = context['speaker_mapping']
        # Must preserve unresolved mapping rather than guessing
        self.assertEqual(mapping['status'], "UNRESOLVED_MULTI_SPEAKER")
        self.assertEqual(mapping['resolution_metadata']['method'], "ambiguous_evidence_preserved_unresolved")
        self.assertEqual(len(mapping['unresolved_speakers']), 2)
        self.assertIsNone(mapping['mapping']['Speaker 1']['speaker_id'])
        self.assertIsNone(mapping['mapping']['Speaker 2']['speaker_id'])

    def test_speaker_mapping_no_usable_metrics(self):
        """Test case where no word count or duration stats exist for multi-speaker diarization."""
        builder = DebateContextBuilder()
        pm_speaker = self.speakers_bp[0]

        context = builder.build_context(
            debate=self.debate_bp,
            target_speaker_id=pm_speaker.id,
            diarized_speakers=["Speaker 1", "Speaker 2"],
            speaker_stats={},  # empty
        )
        mapping = context['speaker_mapping']
        # Must preserve unresolved mapping without guessing
        self.assertEqual(mapping['status'], "UNRESOLVED_MULTI_SPEAKER")
        self.assertEqual(mapping['resolution_metadata']['method'], "ambiguous_evidence_preserved_unresolved")
        self.assertEqual(len(mapping['unresolved_speakers']), 2)


class DebateAnalysisInputBundleTests(TestCase):
    """Test suite for versioned DebateAnalysisInput bundle generation and validation."""

    def setUp(self):
        self.tournament = Tournament.objects.create(name="WUDC 2026", slug="wudc-2026")
        self.tournament.preferences['debate_rules__teams_in_debate'] = 4
        self.round_obj = Round.objects.create(tournament=self.tournament, seq=1, name="R1")
        self.motion = Motion.objects.create(
            tournament=self.tournament,
            text="This House would abolish patent protection for pharmaceuticals.",
            reference="Pharma Patents",
        )
        RoundMotion.objects.create(round=self.round_obj, motion=self.motion, seq=1)
        self.debate = Debate.objects.create(round=self.round_obj)
        self.team = Team.objects.create(tournament=self.tournament, short_name="Cambridge")
        DebateTeam.objects.create(debate=self.debate, team=self.team, side=DebateSide.OG)
        self.speaker = Speaker.objects.create(name="Alice", team=self.team)

        # Recording
        self.recording = SpeechRecording(
            debate=self.debate,
            speaker=self.speaker,
            audio_file="speech_recordings/wudc/dummy.wav",
            status=SpeechRecording.Status.COMPLETED,
        )
        self.recording.save()

        # Transcript
        self.transcript = SpeechTranscript.objects.create(
            recording=self.recording,
            text="First, monopolies kill innovation. Second, lives are saved by generic drugs.",
            language_code="en-US",
            metadata={
                'provider': 'google',
                'overall_confidence': 0.96,
                'segments': [
                    {
                        'speaker': 'Speaker 1',
                        'text': 'First, monopolies kill innovation. Second, lives are saved by generic drugs.',
                        'start': 0.0,
                        'end': 8.5,
                        'confidence': 0.96,
                        'words': [
                            {'word': 'First,', 'start': 0.0, 'end': 0.5, 'confidence': 0.95},
                            {'word': 'monopolies', 'start': 0.5, 'end': 1.2, 'confidence': 0.96},
                            {'word': 'kill', 'start': 1.2, 'end': 1.5, 'confidence': 0.98},
                            {'word': 'innovation.', 'start': 1.5, 'end': 2.3, 'confidence': 0.97},
                            {'word': 'Second,', 'start': 3.0, 'end': 3.5, 'confidence': 0.94},
                            {'word': 'lives', 'start': 3.5, 'end': 4.0, 'confidence': 0.96},
                            {'word': 'are', 'start': 4.0, 'end': 4.2, 'confidence': 0.99},
                            {'word': 'saved', 'start': 4.2, 'end': 4.7, 'confidence': 0.97},
                            {'word': 'by', 'start': 4.7, 'end': 4.9, 'confidence': 0.98},
                            {'word': 'generic', 'start': 4.9, 'end': 5.5, 'confidence': 0.95},
                            {'word': 'drugs.', 'start': 5.5, 'end': 6.2, 'confidence': 0.96},
                        ]
                    }
                ]
            }
        )

    def test_bundle_structure_and_schema_validation(self):
        bundle = self.transcript.get_analysis_input()

        self.assertEqual(bundle['contract_version'], "1.0.0")
        self.assertIsNotNone(bundle['fingerprint'])
        self.assertEqual(bundle['provenance']['transcript_id'], self.transcript.id)
        self.assertEqual(bundle['provenance']['recording_id'], self.recording.id)
        self.assertEqual(bundle['provenance']['language_code'], "en-US")

        # Debate context assertions
        self.assertEqual(bundle['debate_context']['motion']['primary_motion']['text'], self.motion.text)
        self.assertEqual(bundle['debate_context']['speaker_mapping']['status'], "RESOLVED_SINGLE_SPEAKER")
        self.assertEqual(bundle['debate_context']['speaker_mapping']['target_speaker']['name'], "Alice")

        # Transcript preprocessing assertions
        sentences = bundle['transcript']['sentences']
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0]['text'], "First, monopolies kill innovation.")
        self.assertEqual(sentences[0]['start_time'], 0.0)
        self.assertEqual(sentences[0]['end_time'], 2.3)
        self.assertEqual(sentences[0]['timing_source'], "words")
        self.assertEqual(sentences[1]['text'], "Second, lives are saved by generic drugs.")
        self.assertEqual(sentences[1]['start_time'], 3.0)
        self.assertEqual(sentences[1]['end_time'], 6.2)

        # Validate against JSON schema
        is_valid, errors = validate_debate_analysis_input(bundle)
        self.assertTrue(is_valid, f"Validation errors: {errors}")

    def test_bundle_determinism(self):
        bundle1 = build_debate_analysis_input(self.transcript)
        bundle2 = build_debate_analysis_input(self.transcript)

        self.assertEqual(bundle1['fingerprint'], bundle2['fingerprint'])
        self.assertEqual(bundle1['transcript']['sentences'], bundle2['transcript']['sentences'])
        self.assertEqual(bundle1['transcript']['paragraphs'], bundle2['transcript']['paragraphs'])

    def test_fingerprint_sensitivity_to_motion(self):
        bundle_orig = build_debate_analysis_input(self.transcript)
        
        # Modify motion
        self.motion.text = "This House would substantially regulate pharmaceutical pricing."
        self.motion.save()
        bundle_modified = build_debate_analysis_input(self.transcript)

        self.assertNotEqual(bundle_orig['fingerprint'], bundle_modified['fingerprint'])

    def test_fingerprint_sensitivity_to_team_sides(self):
        bundle_orig = build_debate_analysis_input(self.transcript)

        # Change team side from OG to OO
        dt = self.debate.debateteam_set.first()
        dt.side = DebateSide.OO
        dt.save()
        bundle_modified = build_debate_analysis_input(self.transcript)

        self.assertNotEqual(bundle_orig['fingerprint'], bundle_modified['fingerprint'])

    def test_fingerprint_sensitivity_to_speaker_mapping(self):
        bundle_orig = build_debate_analysis_input(self.transcript)

        # Unlink speaker from recording
        self.recording.speaker = None
        self.recording.save()
        bundle_modified = build_debate_analysis_input(self.transcript)

        self.assertNotEqual(bundle_orig['fingerprint'], bundle_modified['fingerprint'])
