import os
import uuid

from django.core.exceptions import ObjectDoesNotExist, ValidationError
from django.db import models
from django.utils.translation import gettext_lazy as _


def speech_recording_upload_path(instance, filename):
    """Generate upload path for debate speech audio recordings.
    Organized by tournament slug, round sequence, and debate ID.
    Uses UUID-prefixed filenames to guarantee collision resistance.
    """
    try:
        debate = instance.debate
        round_obj = debate.round
        tournament = round_obj.tournament
        tournament_slug = tournament.slug
        round_seq = round_obj.seq
        debate_id = debate.id
    except (ObjectDoesNotExist, AttributeError) as exc:
        raise ValidationError(
            _("Cannot generate upload path: SpeechRecording must be associated with a valid Debate, Round, and Tournament.")
        ) from exc

    if not tournament_slug or round_seq is None or not debate_id:
        raise ValidationError(
            _("Cannot generate upload path: Associated Debate, Round, or Tournament lacks required identifier.")
        )

    root, ext = os.path.splitext(filename)
    unique_filename = f"{uuid.uuid4().hex}{ext.lower()}"
    return f"speech_recordings/{tournament_slug}/round_{round_seq}/debate_{debate_id}/{unique_filename}"


class SpeechRecording(models.Model):
    """Represents an uploaded audio recording of a speech or debate round."""

    class Status(models.TextChoices):
        PENDING = 'P', _("Pending")
        PROCESSING = 'R', _("Processing")
        COMPLETED = 'C', _("Completed")
        FAILED = 'F', _("Failed")

    debate = models.ForeignKey(
        'draw.Debate',
        models.CASCADE,
        related_name='speech_recordings',
        verbose_name=_("debate"),
    )
    speaker = models.ForeignKey(
        'participants.Speaker',
        models.SET_NULL,
        blank=True,
        null=True,
        related_name='speech_recordings',
        verbose_name=_("speaker"),
        help_text=_("Optional: specific speaker associated with this speech recording."),
    )
    audio_file = models.FileField(
        upload_to=speech_recording_upload_path,
        verbose_name=_("audio file"),
        help_text=_("Debate speech audio recording."),
    )
    status = models.CharField(
        max_length=1,
        choices=Status.choices,
        default=Status.PENDING,
        verbose_name=_("status"),
    )
    job_id = models.CharField(
        max_length=128,
        blank=True,
        null=True,
        db_index=True,
        verbose_name=_("Gnani job ID"),
        help_text=_("Asynchronous Batch STT job ID from Gnani Prisma."),
    )
    error_message = models.TextField(
        blank=True,
        verbose_name=_("error message"),
        help_text=_("Details of transcription failure, if any."),
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_("created at"),
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_("updated at"),
    )

    class Meta:
        ordering = ['-created_at']
        verbose_name = _("speech recording")
        verbose_name_plural = _("speech recordings")

    def __str__(self):
        speaker_name = self.speaker.name if self.speaker else _("Unassigned Speaker")
        debate_str = str(self.debate) if self.debate_id else _("Unknown Debate")
        return f"SpeechRecording #{self.pk} - {speaker_name} ({debate_str}) [{self.get_status_display()}]"

    def clean(self):
        super().clean()
        if not self.debate_id:
            raise ValidationError({'debate': _("SpeechRecording must be associated with a valid Debate.")})

        if self.speaker:
            try:
                eligible_team_ids = set(
                    self.debate.debateteam_set.values_list('team_id', flat=True)
                )
            except ObjectDoesNotExist as exc:
                raise ValidationError({'debate': _("Associated debate could not be found.")}) from exc

            try:
                speaker_team_id = self.speaker.team_id
            except ObjectDoesNotExist as exc:
                raise ValidationError({'speaker': _("Associated speaker could not be found.")}) from exc

            if speaker_team_id not in eligible_team_ids:
                team_name = getattr(self.speaker.team, 'short_name', _("unknown team"))
                raise ValidationError({
                    'speaker': _(
                        "The selected speaker (%(speaker)s) belongs to %(team)s, "
                        "which is not competing in %(debate)s."
                    ) % {
                        'speaker': self.speaker.name,
                        'team': team_name,
                        'debate': str(self.debate),
                    }
                })

    def save(self, *args, **kwargs):
        # Enforce validation in all standard programmatic creation/update paths
        # without triggering recursive saves.
        self.full_clean()
        super().save(*args, **kwargs)


class SpeechTranscript(models.Model):
    """Represents the transcript and response metadata generated from a SpeechRecording."""

    recording = models.OneToOneField(
        SpeechRecording,
        models.CASCADE,
        related_name='transcript',
        verbose_name=_("speech recording"),
    )
    text = models.TextField(
        blank=True,
        verbose_name=_("transcribed text"),
    )
    language_code = models.CharField(
        max_length=15,
        default='en',
        verbose_name=_("language code"),
        help_text=_("Language code returned or requested for transcription (e.g., 'en', 'hi')."),
    )
    metadata = models.JSONField(
        blank=True,
        null=True,
        verbose_name=_("transcription metadata"),
        help_text=_("Sanitized response payload and token/timing metadata from Gnani Prisma API."),
    )
    created_at = models.DateTimeField(
        auto_now_add=True,
        verbose_name=_("created at"),
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        verbose_name=_("updated at"),
    )

    class Meta:
        ordering = ['-created_at']
        verbose_name = _("speech transcript")
        verbose_name_plural = _("speech transcripts")

    def __str__(self):
        return f"SpeechTranscript for Recording #{self.recording_id} ({self.language_code})"
