import json
import logging
from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.http import FileResponse, Http404, HttpResponseRedirect, JsonResponse
from django.shortcuts import get_object_or_404
from django.utils.translation import gettext as _, gettext_lazy as _lazy
from django.views.generic import DetailView, ListView, View
from django.views.generic.edit import FormView

from tournaments.mixins import TournamentMixin
from utils.misc import reverse_tournament
from utils.mixins import AdministratorMixin

from .forms import SpeechRecordingUploadForm
from .models import SpeechRecording, SpeechTranscript
from .services.exceptions import GnaniSTTError
from .services.gnani import GnaniPrismaBatchClient

logger = logging.getLogger(__name__)


class SpeechAnalyticsDashboardView(AdministratorMixin, TournamentMixin, ListView):
    """Dashboard view listing debate audio recordings and their transcription status."""

    model = SpeechRecording
    template_name = 'speech_analytics/recording_list.html'
    context_object_name = 'recordings'
    page_title = _lazy("Speech Analytics")
    page_emoji = "🎙️"

    def get_queryset(self):
        return SpeechRecording.objects.filter(
            debate__round__tournament=self.tournament
        ).select_related(
            'debate__round__tournament',
            'debate__venue',
            'speaker__team',
        ).prefetch_related(
            'transcript',
            'debate__debateteam_set__team',
        ).order_by('-created_at')


class SpeechRecordingUploadView(AdministratorMixin, TournamentMixin, FormView):
    """View to handle authenticated debate audio uploads and submit batch transcription jobs."""

    form_class = SpeechRecordingUploadForm
    template_name = 'speech_analytics/recording_upload.html'
    page_title = _lazy("Upload Speech Recording")
    page_emoji = "🎙️"

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['tournament'] = self.tournament
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        max_size = getattr(settings, 'SPEECH_ANALYTICS_MAX_UPLOAD_SIZE', 100 * 1024 * 1024)
        context['max_upload_size_mb'] = max_size // (1024 * 1024)
        allowed_exts = getattr(
            settings,
            'SPEECH_ANALYTICS_ALLOWED_EXTENSIONS',
            ('.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'),
        )
        context['allowed_extensions_display'] = ', '.join(allowed_exts)
        return context

    def get_success_url(self):
        return reverse_tournament('speech-analytics-dashboard', self.tournament)

    def _is_json_request(self):
        accept_header = self.request.headers.get('accept', '')
        return ('application/json' in accept_header) or (self.request.content_type == 'application/json')

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
            if self._is_json_request():
                return JsonResponse({
                    'success': False,
                    'recording_id': recording.id,
                    'status': recording.status,
                    'error': recording.error_message,
                }, status=502)
            messages.error(
                self.request,
                _("Recording uploaded, but initiating transcription failed: %(error)s") % {'error': recording.error_message},
            )
            return self.render_to_response(self.get_context_data(form=form))
        except Exception as exc:
            logger.exception("Unexpected error during speech recording submission for #%s", recording.pk)
            recording.status = SpeechRecording.Status.FAILED
            recording.error_message = _("An internal error occurred while submitting audio for transcription.")
            recording.save(update_fields=['status', 'error_message', 'updated_at'])
            if self._is_json_request():
                return JsonResponse({
                    'success': False,
                    'recording_id': recording.id,
                    'status': recording.status,
                    'error': recording.error_message,
                }, status=500)
            messages.error(self.request, recording.error_message)
            return self.render_to_response(self.get_context_data(form=form))

        # Successful submission
        if self._is_json_request():
            # Do not expose internal provider job_id to client
            return JsonResponse({
                'success': True,
                'recording_id': recording.id,
                'status': recording.status,
                'error': None,
            }, status=201)

        messages.success(
            self.request,
            _("Audio recording uploaded successfully! Transcription has started."),
        )
        return HttpResponseRedirect(self.get_success_url())

    def form_invalid(self, form):
        if self._is_json_request():
            return JsonResponse({
                'success': False,
                'errors': form.errors,
            }, status=400)
        messages.error(
            self.request,
            _("Please correct the errors indicated below."),
        )
        return self.render_to_response(self.get_context_data(form=form))


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


class SpeechTranscriptDetailView(AdministratorMixin, TournamentMixin, DetailView):
    """View to display detailed transcript, diarized segments, and status for a recording."""

    model = SpeechRecording
    template_name = 'speech_analytics/transcript_detail.html'
    context_object_name = 'recording'
    pk_url_kwarg = 'recording_id'
    page_title = _lazy("Speech Transcript")
    page_emoji = "📝"

    def get_queryset(self):
        return SpeechRecording.objects.filter(
            debate__round__tournament=self.tournament,
        ).select_related(
            'debate__round__tournament',
            'debate__venue',
            'speaker__team',
        ).prefetch_related(
            'transcript',
            'debate__debateteam_set__team',
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        recording = self.object
        transcript = getattr(recording, 'transcript', None)
        context['transcript'] = transcript

        diarized_segments = []
        transcript_metadata_json = ""
        if transcript and transcript.metadata:
            meta = transcript.metadata
            try:
                transcript_metadata_json = json.dumps(meta, indent=2, ensure_ascii=False)
            except Exception:
                transcript_metadata_json = str(meta)

            if isinstance(meta, dict):
                # Check for standard diarization structures
                segments = meta.get('diarization') or meta.get('segments') or meta.get('words')
                if isinstance(segments, list):
                    for item in segments:
                        if isinstance(item, dict) and ('speaker' in item or 'text' in item):
                            diarized_segments.append(item)

        context['diarized_segments'] = diarized_segments
        context['transcript_metadata_json'] = transcript_metadata_json
        return context


class SpeechRecordingAudioStreamView(AdministratorMixin, TournamentMixin, View):
    """Protected audio streaming endpoint for tournament administrators."""

    def get(self, request, *args, **kwargs):
        recording_id = self.kwargs.get('recording_id')
        recording = get_object_or_404(
            SpeechRecording.objects.select_related('debate__round__tournament'),
            pk=recording_id,
            debate__round__tournament=self.tournament,
        )

        if not recording.audio_file:
            raise Http404(_("No audio file found for this recording."))

        try:
            file_handle = recording.audio_file.open('rb')
        except (OSError, IOError) as exc:
            raise Http404(_("Audio file could not be opened.")) from exc

        return FileResponse(file_handle, as_attachment=False)
