import logging
from django.db import transaction
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404
from django.utils.translation import gettext as _
from django.views.generic import View
from django.views.generic.edit import FormView

from tournaments.mixins import TournamentMixin
from utils.mixins import AdministratorMixin

from .forms import SpeechRecordingUploadForm
from .models import SpeechRecording, SpeechTranscript
from .services.exceptions import GnaniSTTError
from .services.gnani import GnaniPrismaBatchClient

logger = logging.getLogger(__name__)


class SpeechRecordingUploadView(AdministratorMixin, TournamentMixin, FormView):
    """View to handle authenticated debate audio uploads and submit batch transcription jobs."""

    form_class = SpeechRecordingUploadForm
    page_title = _("Upload Speech Recording")

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['tournament'] = self.tournament
        return kwargs

    def form_valid(self, form):
        recording = form.save(commit=False)
        recording.status = SpeechRecording.Status.PENDING
        recording.save()

        # Submit recording through Gnani Prisma Batch STT client
        try:
            client = GnaniPrismaBatchClient()
            job_info = client.create_batch_job(
                audio_file=recording.audio_file.file,
                language_code="en-IN",
                model="prisma-v2.5",
            )
            job_id = job_info.get('job_id')
            if not job_id:
                raise GnaniSTTError("No job_id returned by Gnani API on job creation.")

            # Save job_id immediately after successful creation
            recording.job_id = str(job_id)
            recording.error_message = ""
            recording.save(update_fields=['job_id', 'error_message', 'updated_at'])

            # Trigger job execution
            client.start_batch_job(recording.job_id)
            recording.status = SpeechRecording.Status.PROCESSING
            recording.save(update_fields=['status', 'updated_at'])

        except GnaniSTTError as exc:
            logger.warning("Gnani STT submission failed for recording #%s: %s", recording.pk, exc.message)
            recording.status = SpeechRecording.Status.FAILED
            recording.error_message = exc.message
            recording.save(update_fields=['status', 'error_message', 'updated_at'])
            return JsonResponse({
                'success': False,
                'recording_id': recording.id,
                'status': recording.status,
                'error': recording.error_message,
            }, status=502)
        except Exception as exc:
            logger.exception("Unexpected error during speech recording submission for #%s", recording.pk)
            recording.status = SpeechRecording.Status.FAILED
            recording.error_message = _("An internal error occurred while submitting audio for transcription.")
            recording.save(update_fields=['status', 'error_message', 'updated_at'])
            return JsonResponse({
                'success': False,
                'recording_id': recording.id,
                'status': recording.status,
                'error': recording.error_message,
            }, status=500)

        # Do not expose internal provider job_id to client
        return JsonResponse({
            'success': True,
            'recording_id': recording.id,
            'status': recording.status,
            'error': None,
        }, status=201)

    def form_invalid(self, form):
        return JsonResponse({
            'success': False,
            'errors': form.errors,
        }, status=400)


class SpeechRecordingStatusCheckView(AdministratorMixin, TournamentMixin, View):
    """Endpoint to check the transcription status of an existing recording and persist transcripts upon completion."""

    def get(self, request, *args, **kwargs):
        recording_id = self.kwargs.get('recording_id')

        # Retrieve recording strictly scoped to the current tournament
        recording = get_object_or_404(
            SpeechRecording.objects.select_related('debate__round__tournament'),
            pk=recording_id,
            debate__round__tournament=self.tournament,
        )

        # 1. If already completed, return immediately without touching external API
        if recording.status == SpeechRecording.Status.COMPLETED:
            has_transcript = hasattr(recording, 'transcript')
            return JsonResponse({
                'recording_id': recording.id,
                'status': recording.status,
                'transcript_available': has_transcript,
                'error': None,
            })

        # 2. If no job_id exists (e.g. initial submission failed)
        if not recording.job_id:
            return JsonResponse({
                'recording_id': recording.id,
                'status': recording.status,
                'transcript_available': False,
                'error': recording.error_message or _("No active transcription job associated with this recording."),
            })

        # 3. Poll status from Gnani Batch STT
        client = GnaniPrismaBatchClient()
        try:
            status_data = client.get_job_status(recording.job_id)
            job_status = (status_data.get('status') or '').upper()

            if job_status == 'COMPLETED':
                # Step A: Fetch completed transcript OUTSIDE the database transaction
                transcript_payload = client.fetch_completed_transcript(recording.job_id)

                # Step B: Open a short atomic transaction and row lock to persist safely
                with transaction.atomic():
                    locked_rec = SpeechRecording.objects.select_for_update().get(pk=recording.pk)

                    if locked_rec.status != SpeechRecording.Status.COMPLETED:
                        SpeechTranscript.objects.update_or_create(
                            recording=locked_rec,
                            defaults={
                                'text': transcript_payload.get('text', ''),
                                'language_code': transcript_payload.get('language_code', 'en'),
                                'metadata': transcript_payload.get('metadata'),
                            },
                        )
                        locked_rec.status = SpeechRecording.Status.COMPLETED
                        locked_rec.error_message = ""
                        locked_rec.save(update_fields=['status', 'error_message', 'updated_at'])
                        recording = locked_rec

            elif job_status == 'FAILED':
                err_msg = status_data.get('error') or _("Transcription job failed during processing.")
                recording.status = SpeechRecording.Status.FAILED
                recording.error_message = str(err_msg)
                recording.save(update_fields=['status', 'error_message', 'updated_at'])

            else:
                # STARTING, QUEUED, IN_PROGRESS
                if recording.status != SpeechRecording.Status.PROCESSING:
                    recording.status = SpeechRecording.Status.PROCESSING
                    recording.save(update_fields=['status', 'updated_at'])

        except GnaniSTTError as exc:
            logger.warning("Status check failed for recording #%s: %s", recording.pk, exc.message)
            return JsonResponse({
                'recording_id': recording.id,
                'status': recording.status,
                'transcript_available': hasattr(recording, 'transcript'),
                'error': exc.message,
            }, status=502)
        except Exception as exc:
            logger.exception("Unexpected error during status check for recording #%s", recording.pk)
            return JsonResponse({
                'recording_id': recording.id,
                'status': recording.status,
                'transcript_available': hasattr(recording, 'transcript'),
                'error': _("Failed to communicate with transcription service."),
            }, status=500)

        # Do not expose internal provider job_id to client
        return JsonResponse({
            'recording_id': recording.id,
            'status': recording.status,
            'transcript_available': hasattr(recording, 'transcript'),
            'error': recording.error_message or None,
        })
