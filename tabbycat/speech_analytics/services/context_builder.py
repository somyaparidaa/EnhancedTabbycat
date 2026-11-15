"""Debate context extraction service for competitive debate speech analytics.

Extracts structured tournament, round, debate, motion, side, team, and speaker data
from native Tabbycat models. Supports all tournament formats:
- British Parliamentary (BP, 4 teams: Opening Government, Opening Opposition, Closing Government, Closing Opposition)
- Two-team formats (Australs, WSDC, American Parliamentary, etc.: Affirmative/Proposition, Negative/Opposition)
- Multi-motion vetoes and info slides
- Speaker rosters and confirmed ballot speech positions
- Speaker mapping resolution (target speaker vs diarization labels)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from django.core.exceptions import ObjectDoesNotExist
from django.utils.translation import gettext as _

from draw.models import Debate, DebateSide
from motions.models import RoundMotion
from results.models import SpeakerScore
from results.utils import _BP_POSITION_NAMES, side_and_position_names
from tournaments.models import Tournament
from tournaments.utils import get_side_name

logger = logging.getLogger(__name__)

# Standard BP speaker role abbreviations mapped to full titles
BP_ROLE_LABELS = {
    'PM': 'Prime Minister',
    'DPM': 'Deputy Prime Minister',
    'LO': 'Leader of the Opposition',
    'DLO': 'Deputy Leader of the Opposition',
    'MG': 'Member for the Government',
    'GW': 'Government Whip',
    'MO': 'Member for the Opposition',
    'OW': 'Opposition Whip',
}

# Standard side name codes for BP
BP_SIDE_CODES = {
    DebateSide.OG: 'OG',
    DebateSide.OO: 'OO',
    DebateSide.CG: 'CG',
    DebateSide.CO: 'CO',
}

TWO_TEAM_SIDE_CODES = {
    DebateSide.AFF: 'AFF',
    DebateSide.NEG: 'NEG',
}


class DebateContextBuilder:
    """Extracts and formats comprehensive competitive debate context for NLP analysis."""

    def build_context(
        self,
        debate: Debate,
        target_speaker_id: Optional[int] = None,
        diarized_speakers: Optional[List[str]] = None,
        speaker_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build structured debate context dictionary.

        Args:
            debate: The Tabbycat Debate instance.
            target_speaker_id: Optional Speaker PK linked to the recording.
            diarized_speakers: Optional list of speaker labels identified in the transcript (e.g. ['Speaker 1', 'Speaker 2']).
            speaker_stats: Optional dict mapping speaker label to metrics like {'word_count': int, 'duration': float}.

        Returns:
            Structured context dictionary.
        """
        round_obj = debate.round
        tournament = round_obj.tournament

        # 1. Tournament metadata & rules
        tournament_info = self._extract_tournament_info(tournament)

        # 2. Round metadata
        round_info = self._extract_round_info(round_obj)

        # 3. Motion & Info slide
        motion_info = self._extract_motion_info(round_obj, debate)

        # 4. Sides & Teams
        teams_info = self._extract_teams_and_sides(tournament, debate)

        # 5. Speeches & Speaker scores (from confirmed or available ballot)
        speeches_info, confirmed_ballot_found = self._extract_speeches(tournament, debate)

        # 6. Speaker mapping resolution
        speaker_mapping = self._resolve_speaker_mapping(
            debate=debate,
            target_speaker_id=target_speaker_id,
            diarized_speakers=diarized_speakers or [],
            teams_info=teams_info,
            speeches_info=speeches_info,
            speaker_stats=speaker_stats or {},
        )

        return {
            'tournament': tournament_info,
            'round': round_info,
            'debate': {
                'id': debate.id,
                'matchup': debate.matchup if hasattr(debate, 'matchup') else str(debate),
                'importance': getattr(debate, 'importance', None),
                'bracket': getattr(debate, 'bracket', None),
                'confirmed_ballot_available': confirmed_ballot_found,
            },
            'motion': motion_info,
            'teams': teams_info,
            'speeches': speeches_info,
            'speaker_mapping': speaker_mapping,
        }

    def _extract_tournament_info(self, tournament: Tournament) -> Dict[str, Any]:
        """Extract tournament rules and formatting settings."""
        teams_in_debate = tournament.pref('teams_in_debate')
        substantive_speakers = tournament.pref('substantive_speakers')
        reply_scores_enabled = tournament.pref('reply_scores_enabled')
        side_names_pref = tournament.pref('side_names')

        format_name = "British Parliamentary (BP)" if teams_in_debate == 4 else f"{teams_in_debate}-Team Format"
        if teams_in_debate == 2:
            if substantive_speakers == 3 and reply_scores_enabled:
                format_name = "Australs / 3v3 with Reply"
            elif substantive_speakers == 3 and not reply_scores_enabled:
                format_name = "Australian Easters / 3v3"
            elif substantive_speakers == 2:
                format_name = "2v2 Format"

        return {
            'id': tournament.id,
            'name': tournament.name,
            'slug': tournament.slug,
            'format_name': format_name,
            'teams_in_debate': teams_in_debate,
            'substantive_speakers': substantive_speakers,
            'reply_scores_enabled': reply_scores_enabled,
            'side_names_style': side_names_pref,
        }

    def _extract_round_info(self, round_obj: Any) -> Dict[str, Any]:
        """Extract round stage, sequence, and elimination info."""
        return {
            'id': round_obj.id,
            'seq': round_obj.seq,
            'name': round_obj.name,
            'abbreviation': round_obj.abbreviation,
            'stage': round_obj.stage,
            'stage_display': round_obj.get_stage_display() if hasattr(round_obj, 'get_stage_display') else str(round_obj.stage),
            'is_break_round': round_obj.is_break_round if hasattr(round_obj, 'is_break_round') else False,
            'draw_type': round_obj.draw_type,
        }

    def _extract_motion_info(self, round_obj: Any, debate: Debate) -> Dict[str, Any]:
        """Extract motion, info slide, and any veto/alternative motions."""
        # Check if debate's confirmed ballot specifies a motion
        confirmed_ballot = getattr(debate, 'confirmed_ballot', None)
        chosen_motion = None
        if confirmed_ballot and getattr(confirmed_ballot, 'motion', None):
            chosen_motion = confirmed_ballot.motion

        # Fetch round motions
        round_motions = list(
            RoundMotion.objects.filter(round=round_obj)
            .select_related('motion')
            .order_by('seq')
        )

        motions_list = []
        for rm in round_motions:
            m = rm.motion
            m_dict = {
                'id': m.id,
                'reference': m.reference,
                'text': m.text,
                'info_slide': m.info_slide or "",
                'seq': rm.seq,
                'is_chosen': (chosen_motion and chosen_motion.id == m.id) or (len(round_motions) == 1),
            }
            motions_list.append(m_dict)

        primary_motion = None
        if chosen_motion:
            primary_motion = {
                'id': chosen_motion.id,
                'reference': chosen_motion.reference,
                'text': chosen_motion.text,
                'info_slide': chosen_motion.info_slide or "",
            }
        elif motions_list:
            primary_motion = {
                'id': motions_list[0]['id'],
                'reference': motions_list[0]['reference'],
                'text': motions_list[0]['text'],
                'info_slide': motions_list[0]['info_slide'],
            }

        return {
            'primary_motion': primary_motion,
            'round_motions': motions_list,
            'has_multiple_motions': len(round_motions) > 1,
        }

    def _extract_teams_and_sides(self, tournament: Tournament, debate: Debate) -> List[Dict[str, Any]]:
        """Extract all teams in the debate with their sides, codes, and institutional info."""
        teams_data: List[Dict[str, Any]] = []

        debate_teams = list(
            debate.debateteam_set.select_related('team', 'team__institution').order_by('side')
        )

        for dt in debate_teams:
            team = dt.team
            side_code = self._get_side_code(tournament, dt.side)
            side_name = get_side_name(tournament, dt.side, 'full')
            side_abbr = get_side_name(tournament, dt.side, 'abbr')

            # Extract rostered speakers for this team
            speakers = [
                {
                    'id': spk.id,
                    'name': spk.name,
                    'code_name': getattr(spk, 'code_name', '') or '',
                }
                for spk in team.speaker_set.all().order_by('id')
            ]

            teams_data.append({
                'debate_team_id': dt.id,
                'team_id': team.id,
                'team_name': team.short_name,
                'code_name': team.code_name or '',
                'institution': team.institution.name if team.institution else None,
                'side_raw': dt.side,
                'side_code': side_code,
                'side_name': side_name,
                'side_abbr': side_abbr,
                'rostered_speakers': speakers,
            })

        return teams_data

    def _get_side_code(self, tournament: Tournament, side_int: int) -> str:
        """Resolve standard side code (e.g. OG, OO, CG, CO, AFF, NEG)."""
        teams_in_debate = tournament.pref('teams_in_debate')
        if teams_in_debate == 4:
            return BP_SIDE_CODES.get(side_int, f"Side_{side_int}")
        elif teams_in_debate == 2:
            return TWO_TEAM_SIDE_CODES.get(side_int, f"Side_{side_int}")
        return f"Side_{side_int}"

    def _extract_speeches(
        self, tournament: Tournament, debate: Debate
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Extract speeches, roles, and assigned speakers from ballot submission."""
        ballot = getattr(debate, 'confirmed_ballot', None)
        ballot_found = bool(ballot)

        # Fallback to any latest ballot if confirmed not found
        if not ballot:
            ballot = debate.ballotsubmission_set.filter(discarded=False).order_by('-version').first()

        speeches: List[Dict[str, Any]] = []

        if ballot:
            scores = (
                SpeakerScore.objects.filter(ballot_submission=ballot)
                .select_related('speaker', 'debate_team', 'debate_team__team')
                .order_by('position', 'debate_team__side')
            )

            for score in scores:
                side_code = self._get_side_code(tournament, score.debate_team.side)
                role_abbr, role_name = self._resolve_speech_role(tournament, score.debate_team.side, score.position)

                speeches.append({
                    'position': score.position,
                    'side_code': side_code,
                    'role_abbr': role_abbr,
                    'role_name': role_name,
                    'speaker_id': score.speaker.id if score.speaker else None,
                    'speaker_name': score.speaker.name if score.speaker else _("Unassigned"),
                    'team_id': score.debate_team.team.id if score.debate_team and score.debate_team.team else None,
                    'team_name': score.debate_team.team.short_name if score.debate_team and score.debate_team.team else "",
                    'is_ghost': getattr(score, 'ghost', False),
                })
        else:
            # Construct standard expected speech roles if no ballot has been submitted yet
            speeches = self._generate_expected_speech_roles(tournament, debate)

        return speeches, ballot_found

    def _resolve_speech_role(
        self, tournament: Tournament, side_int: int, position: int
    ) -> Tuple[str, str]:
        """Determine standard role abbreviation and full descriptive title."""
        teams_in_debate = tournament.pref('teams_in_debate')

        if teams_in_debate == 4:
            # BP roles: side 0 (OG): pos 1=PM, pos 2=DPM
            # side 1 (OO): pos 1=LO, pos 2=DLO
            # side 2 (CG): pos 1=MG, pos 2=GW
            # side 3 (CO): pos 1=MO, pos 2=OW
            if 0 <= side_int < len(_BP_POSITION_NAMES) and 1 <= position <= 2:
                abbr = str(_BP_POSITION_NAMES[side_int][position - 1])
                full_name = BP_ROLE_LABELS.get(abbr, abbr)
                return abbr, full_name

        # 2-team roles
        side_abbr = get_side_name(tournament, side_int, 'abbr')
        if position == tournament.reply_position:
            return f"{side_abbr}R", f"{get_side_name(tournament, side_int, 'full')} Reply"

        return f"{side_abbr}{position}", f"{get_side_name(tournament, side_int, 'full')} Speaker {position}"

    def _generate_expected_speech_roles(
        self, tournament: Tournament, debate: Debate
    ) -> List[Dict[str, Any]]:
        """Generate expected speech slots when ballot results are not yet entered."""
        expected: List[Dict[str, Any]] = []
        debate_teams = list(debate.debateteam_set.select_related('team').order_by('side'))

        for dt in debate_teams:
            side_code = self._get_side_code(tournament, dt.side)
            for pos in tournament.positions:
                role_abbr, role_name = self._resolve_speech_role(tournament, dt.side, pos)
                expected.append({
                    'position': pos,
                    'side_code': side_code,
                    'role_abbr': role_abbr,
                    'role_name': role_name,
                    'speaker_id': None,
                    'speaker_name': _("Pending Ballot Submission"),
                    'team_id': dt.team.id,
                    'team_name': dt.team.short_name,
                    'is_ghost': False,
                })

        return expected

    def _resolve_speaker_mapping(
        self,
        debate: Debate,
        target_speaker_id: Optional[int],
        diarized_speakers: List[str],
        teams_info: List[Dict[str, Any]],
        speeches_info: List[Dict[str, Any]],
        speaker_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Resolve speaker mapping status between target speaker and transcript diarization labels.

        Classification:
        - RESOLVED_SINGLE_SPEAKER: Recording is linked to a target speaker, and only 1 diarized label exists.
        - RESOLVED_TARGET_WITH_POIS: Recording is linked to target speaker, and a dominant primary speaker label
          is identifiably distinguishable from secondary POIs/floor interjections.
        - UNRESOLVED_MULTI_SPEAKER: Recording is debate-level or ambiguous multi-speaker without clear dominance,
          preserving unresolved mapping without guessing.
        - DEBATE_ONLY_UNASSIGNED: Recording is debate-level with no individual speaker assigned.
        """
        unresolved_speakers: List[str] = []
        mapping_details: Dict[str, Any] = {}
        speaker_stats = speaker_stats or {}

        target_speech = None
        target_team = None
        target_speaker_obj = None

        if target_speaker_id:
            # Find target speaker in speeches
            for sp in speeches_info:
                if sp.get('speaker_id') == target_speaker_id:
                    target_speech = sp
                    break

            # Find target speaker in team rosters
            for tm in teams_info:
                for spk in tm['rostered_speakers']:
                    if spk['id'] == target_speaker_id:
                        target_team = tm
                        target_speaker_obj = spk
                        break
                if target_team:
                    break

        num_diarized = len(diarized_speakers)
        resolution_metadata: Dict[str, Any] = {
            'method': 'unassigned',
            'dominant_label': None,
            'evidence': {},
        }

        if target_speaker_id and num_diarized <= 1:
            status = "RESOLVED_SINGLE_SPEAKER"
            resolution_metadata['method'] = "single_speaker_direct"
            if diarized_speakers:
                label = diarized_speakers[0]
                resolution_metadata['dominant_label'] = label
                mapping_details[label] = {
                    'speaker_id': target_speaker_id,
                    'speaker_name': target_speaker_obj['name'] if target_speaker_obj else f"Speaker #{target_speaker_id}",
                    'role': target_speech['role_name'] if target_speech else "Speaker",
                    'type': "target_speaker",
                    'evidence': speaker_stats.get(label, {}),
                }

        elif target_speaker_id and num_diarized > 1:
            # Multi-diarization: evaluate dominant speaker using aggregate word count and duration
            dominant_candidate, is_ambiguous, candidate_evidence = self._find_dominant_speaker(
                diarized_speakers, speaker_stats
            )
            resolution_metadata['evidence'] = candidate_evidence
            resolution_metadata['dominant_label'] = dominant_candidate

            if dominant_candidate and not is_ambiguous:
                status = "RESOLVED_TARGET_WITH_POIS"
                resolution_metadata['method'] = "dominant_heuristic_word_or_duration"
                
                # Primary dominant speaker mapped to target speaker
                mapping_details[dominant_candidate] = {
                    'speaker_id': target_speaker_id,
                    'speaker_name': target_speaker_obj['name'] if target_speaker_obj else f"Speaker #{target_speaker_id}",
                    'role': target_speech['role_name'] if target_speech else "Speaker",
                    'type': "target_speaker",
                    'evidence': candidate_evidence.get(dominant_candidate, {}),
                }

                # Secondary labels remain unresolved POIs/interjections
                for other_label in diarized_speakers:
                    if other_label == dominant_candidate:
                        continue
                    unresolved_speakers.append(other_label)
                    mapping_details[other_label] = {
                        'speaker_id': None,
                        'speaker_name': _("Unresolved (POI / Interjection)"),
                        'role': "POI / Floor Interjection",
                        'type': "unresolved_poi",
                        'evidence': candidate_evidence.get(other_label, {}),
                    }
            else:
                # Ambiguous or insufficient evidence: preserve unresolved mapping without guessing
                status = "UNRESOLVED_MULTI_SPEAKER"
                resolution_metadata['method'] = "ambiguous_evidence_preserved_unresolved"
                unresolved_speakers.extend(diarized_speakers)
                for label in diarized_speakers:
                    mapping_details[label] = {
                        'speaker_id': None,
                        'speaker_name': _("Unassigned Diarized Speaker (Ambiguous)"),
                        'role': "Unknown Speech Role",
                        'type': "unassigned",
                        'evidence': candidate_evidence.get(label, {}),
                    }

        elif not target_speaker_id and num_diarized > 0:
            status = "UNRESOLVED_MULTI_SPEAKER"
            resolution_metadata['method'] = "debate_only_unassigned"
            unresolved_speakers.extend(diarized_speakers)
            for label in diarized_speakers:
                mapping_details[label] = {
                    'speaker_id': None,
                    'speaker_name': _("Unassigned Diarized Speaker"),
                    'role': "Unknown Speech Role",
                    'type': "unassigned",
                    'evidence': speaker_stats.get(label, {}),
                }
        else:
            status = "DEBATE_ONLY_UNASSIGNED"
            resolution_metadata['method'] = "no_speakers"

        return {
            'status': status,
            'target_speaker': {
                'id': target_speaker_id,
                'name': target_speaker_obj['name'] if target_speaker_obj else None,
                'team': target_team['team_name'] if target_team else None,
                'side_code': target_team['side_code'] if target_team else (target_speech['side_code'] if target_speech else None),
                'speech_role': target_speech['role_name'] if target_speech else None,
                'speech_position': target_speech['position'] if target_speech else None,
            } if target_speaker_id else None,
            'diarized_speakers': diarized_speakers,
            'unresolved_speakers': unresolved_speakers,
            'mapping': mapping_details,
            'resolution_metadata': resolution_metadata,
        }

    def _find_dominant_speaker(
        self, diarized_speakers: List[str], speaker_stats: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """Identify candidate primary speaker by aggregate word count or speech duration.

        Returns:
            Tuple of (dominant_label, is_ambiguous, evidence_dict)
        """
        evidence: Dict[str, Any] = {}
        for spk in diarized_speakers:
            st = speaker_stats.get(spk, {})
            words = st.get('word_count', 0) or 0
            dur = st.get('duration', 0.0) or 0.0
            evidence[spk] = {
                'word_count': words,
                'duration': round(dur, 2) if dur else 0.0,
            }

        # Check if usable evidence exists
        total_words = sum(e['word_count'] for e in evidence.values())
        total_duration = sum(e['duration'] for e in evidence.values())

        if total_words == 0 and total_duration <= 0.0:
            # No usable metrics; cannot safely determine dominant speaker
            return None, True, evidence

        # Score by word count first if available, otherwise duration
        if total_words > 0:
            sorted_by_words = sorted(
                diarized_speakers,
                key=lambda s: evidence[s]['word_count'],
                reverse=True,
            )
            top1 = sorted_by_words[0]
            top1_words = evidence[top1]['word_count']
            top2_words = evidence[sorted_by_words[1]]['word_count'] if len(sorted_by_words) > 1 else 0

            # Dominance criterion: top speaker must have at least 60% of total words
            # and at least 1.5x the runner-up words
            share = top1_words / float(total_words)
            if share >= 0.60 and (top2_words == 0 or top1_words >= 1.5 * top2_words):
                return top1, False, evidence
            return top1, True, evidence  # Ambiguous close contest

        if total_duration > 0.0:
            sorted_by_dur = sorted(
                diarized_speakers,
                key=lambda s: evidence[s]['duration'],
                reverse=True,
            )
            top1 = sorted_by_dur[0]
            top1_dur = evidence[top1]['duration']
            top2_dur = evidence[sorted_by_dur[1]]['duration'] if len(sorted_by_dur) > 1 else 0.0

            share = top1_dur / float(total_duration)
            if share >= 0.60 and (top2_dur == 0.0 or top1_dur >= 1.5 * top2_dur):
                return top1, False, evidence
            return top1, True, evidence

        return None, True, evidence



def build_debate_context(
    debate: Debate,
    target_speaker_id: Optional[int] = None,
    diarized_speakers: Optional[List[str]] = None,
    speaker_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Helper functional interface to extract debate context."""
    builder = DebateContextBuilder()
    return builder.build_context(
        debate=debate,
        target_speaker_id=target_speaker_id,
        diarized_speakers=diarized_speakers,
        speaker_stats=speaker_stats,
    )
