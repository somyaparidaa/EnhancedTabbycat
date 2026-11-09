import os
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils.translation import gettext_lazy as _

from draw.models import Debate
from participants.models import Speaker
from .models import SpeechRecording

# Audio magic bytes / signatures for binary content validation
AUDIO_MAGIC_SIGNATURES = (
    b'RIFF',       # WAV (starts with RIFF....WAVE)
    b'ID3',        # MP3 with ID3v2 tag
    b'\xff\xfb',   # MP3 frame sync (MPEG-1 Layer 3)
    b'\xff\xf3',   # MP3 frame sync (MPEG-2 Layer 3)
    b'\xff\xf2',   # MP3 frame sync (MPEG-2.5 Layer 3)
    b'OggS',       # Ogg Vorbis/Opus
    b'fLaC',       # FLAC
    b'\xff\xf1',   # AAC (ADTS frame)
    b'\xff\xf9',   # AAC (ADTS frame)
)


class SpeechRecordingUploadForm(forms.ModelForm):
    """Form for uploading debate audio recordings and validating audio file integrity."""

    class Meta:
        model = SpeechRecording
        fields = ('debate', 'speaker', 'audio_file')

    def __init__(self, tournament, *args, **kwargs):
        self.tournament = tournament
        super().__init__(*args, **kwargs)

        # Restrict debates and speakers strictly to the current tournament
        self.fields['debate'].queryset = Debate.objects.filter(
            round__tournament=self.tournament
        ).select_related('round', 'venue')

        self.fields['speaker'].queryset = Speaker.objects.filter(
            team__tournament=self.tournament
        ).select_related('team')
        self.fields['speaker'].required = False

    def clean_audio_file(self):
        audio_file = self.cleaned_data.get('audio_file')
        if not audio_file:
            return audio_file

        # 1. Enforce maximum file size
        max_size = getattr(settings, 'SPEECH_ANALYTICS_MAX_UPLOAD_SIZE', 100 * 1024 * 1024)
        if audio_file.size > max_size:
            max_mb = max_size // (1024 * 1024)
            raise ValidationError(
                _("The uploaded file exceeds the %(max_mb)s MB size limit.") % {'max_mb': max_mb}
            )

        # 2. Enforce allowed extension
        allowed_extensions = getattr(
            settings,
            'SPEECH_ANALYTICS_ALLOWED_EXTENSIONS',
            ('.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'),
        )
        file_ext = os.path.splitext(audio_file.name)[1].lower()
        allowed_str = ', '.join(allowed_extensions)
        if file_ext not in allowed_extensions:
            raise ValidationError(
                _("Unsupported file format %(ext)s. Allowed formats: %(allowed)s") % {
                    'ext': file_ext,
                    'allowed': allowed_str,
                }
            )

        # 3. MIME and binary magic byte validation
        # Validate content_type header if provided
        content_type = getattr(audio_file, 'content_type', '')
        if content_type and not (content_type.startswith('audio/') or content_type == 'application/octet-stream'):
            raise ValidationError(_("Uploaded file content type '%(ct)s' is not a valid audio format.") % {'ct': content_type})

        # Inspect initial header bytes
        pos = audio_file.tell() if hasattr(audio_file, 'tell') else 0
        try:
            audio_file.seek(0)
            header = audio_file.read(16)
        finally:
            if hasattr(audio_file, 'seek'):
                audio_file.seek(pos)

        # Check against audio signatures (or MP4/M4A ftyp box)
        is_magic_matched = any(header.startswith(sig) for sig in AUDIO_MAGIC_SIGNATURES)
        is_m4a = (len(header) >= 12 and header[4:8] == b'ftyp')

        if not (is_magic_matched or is_m4a):
            raise ValidationError(_("File contents do not match a valid audio signature."))

        return audio_file

    def clean(self):
        cleaned_data = super().clean()
        debate = cleaned_data.get('debate')
        speaker = cleaned_data.get('speaker')

        if debate and debate.round.tournament_id != self.tournament.id:
            raise ValidationError({'debate': _("Selected debate does not belong to this tournament.")})

        if speaker and speaker.team.tournament_id != self.tournament.id:
            raise ValidationError({'speaker': _("Selected speaker does not belong to this tournament.")})

        if debate and speaker:
            eligible_team_ids = set(debate.debateteam_set.values_list('team_id', flat=True))
            if speaker.team_id not in eligible_team_ids:
                raise ValidationError({
                    'speaker': _(
                        "The selected speaker (%(speaker)s) belongs to %(team)s, "
                        "which is not competing in %(debate)s."
                    ) % {
                        'speaker': speaker.name,
                        'team': speaker.team.short_name,
                        'debate': str(debate),
                    }
                })

        return cleaned_data
