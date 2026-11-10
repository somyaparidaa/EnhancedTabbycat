# Speech Analytics Module (STT-1)

The **Speech Analytics** module in EnhancedTabbycat integrates debate audio recording management with automated speech-to-text (STT) transcription powered by **Gnani AI Prisma v2.5 / v3.0 Batch API**.

---

## 1. Application Architecture

The module is structured as an isolated Django application located at `tabbycat/speech_analytics/`:

- **Models (`speech_analytics/models.py`)**:
  - `SpeechRecording`: Stores debate audio file references, status (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`), Gnani `job_id`, associated `Debate`, optional `Speaker`, and failure error messages. File paths are collision-resistant and partitioned by tournament, round, and debate ID (`speech_recordings/<tournament_slug>/round_<seq>/debate_<id>/<uuid>.<ext>`).
  - `SpeechTranscript`: A one-to-one relationship with `SpeechRecording`. Stores full transcribed text, detected/requested language code, and structured metadata (diarized segments, confidence, audio duration, sanitized provider payloads).
- **Service Client (`speech_analytics/services/gnani.py`)**:
  - `GnaniPrismaBatchClient`: Server-side client handling batch job creation (`POST /stt/v3/batch/jobs`), job execution (`POST /stt/v3/batch/jobs/{id}/start`), status polling (`GET /stt/v3/batch/jobs/{id}`), file URL lookup (`GET /stt/v3/batch/jobs/{id}/files`), and secure transcript downloading with SSRF and credential leak protections.
  - Automatic metadata sanitization (`sanitize_metadata`) strips auth headers, tokens, and secret keys while preserving legitimate domain fields (such as keyterms, confidence, timestamps, and segments).
- **Views & Workflow (`speech_analytics/views.py`)**:
  - `SpeechAnalyticsDashboardView`: Tournament-scoped dashboard displaying all recordings for the tournament, their status badges, and actions.
  - `SpeechRecordingUploadView`: Validates audio files (size, extension, MIME/magic signatures) and creates Gnani batch jobs upon upload.
  - `SpeechRecordingStatusCheckView`: Dual-protocol endpoint (HTML form and JSON) that polls Gnani job status. Employs row locking (`select_for_update()`) and atomic transactions *after* fetching transcripts outside the transaction to preserve concurrency and idempotency.
  - `SpeechRecordingRetryView`: Safely retries `FAILED` recordings via POST, ensuring only failed recordings can create a new Gnani job while preserving the original audio.
  - `SpeechTranscriptDetailView`: Displays transcript text, diarization segments, audio player, and metadata.
  - `SpeechRecordingAudioStreamView`: Protected binary streaming endpoint for debate audio files, restricting access strictly to authorized tournament administrators.
- **Access Control**:
  - All routes inherit from `AdministratorMixin` and `TournamentMixin`, enforcing staff/superuser credentials and strict tournament slug scoping.

---

## 2. Recording & Transcription Workflow

```
Administrator Upload
       │
       ▼
[SpeechRecordingUploadView]
       │
       ├─► Validates file (size, format, magic bytes, debate/speaker tournament match)
       ├─► Saves SpeechRecording (Status: PENDING)
       ├─► Calls GnaniPrismaBatchClient.create_batch_job()
       ├─► Persists returned job_id
       └─► Calls GnaniPrismaBatchClient.start_batch_job() (Status: PROCESSING)
       │
       ▼
Administrator Poll / Check Status
       │
       ▼
[SpeechRecordingStatusCheckView]
       │
       ├─► Polls GnaniPrismaBatchClient.get_job_status(job_id)
       │
       ├─► If IN_PROGRESS: Remains PROCESSING
       ├─► If FAILED: Records error_message, sets status to FAILED
       └─► If COMPLETED:
             ├─► Fetches transcript payload outside DB transaction
             ├─► Opens transaction.atomic() + select_for_update() row lock
             ├─► Creates/updates SpeechTranscript
             └─► Sets status to COMPLETED
```

If a recording enters `FAILED` status, administrators can trigger a retry via `SpeechRecordingRetryView`:
- Validates that recording status is currently `FAILED` (rejects `PROCESSING`, `PENDING`, or `COMPLETED`).
- Retries with the original audio file, generating and recording a new provider `job_id`.

---

## 3. Configuration & Environment Variables

The module is configured via environment variables in `tabbycat/settings/core.py`. No credentials are hardcoded.

| Variable | Description | Default |
|---|---|---|
| `GNANI_API_KEY` | Private API key for Gnani AI Prisma Batch STT | `""` (Required in production) |
| `GNANI_API_BASE_URL` | Base URL for Gnani API | `https://api.vachana.ai` |
| `GNANI_CONNECT_TIMEOUT` | Connect timeout in seconds | `10` |
| `GNANI_READ_TIMEOUT` | Read timeout in seconds | `60` |
| `SPEECH_MAX_UPLOAD_SIZE`| Maximum allowed audio file upload size in bytes | `104857600` (100 MB) |

### Local Development Setup

To configure for local testing, set the variables in your local environment or `.env` / `local.py`:

```bash
export GNANI_API_KEY="your-gnani-api-key"
export GNANI_API_BASE_URL="https://api.vachana.ai"
```

---

## 4. Supported Audio Formats and Limits

- **Allowed Extensions**: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.aac`.
- **Validation**:
  - File extension matching.
  - MIME type header checking.
  - Initial binary magic bytes / signature inspection (`RIFF`, `ID3`, MPEG sync frames, `OggS`, `fLaC`, `ftyp`).
- **Maximum File Size**: 100 MB default (configurable via `SPEECH_MAX_UPLOAD_SIZE`).

---

## 5. Security & Isolation Measures

- **No Public `job_id` Exposure**: Internal provider job identifiers are stripped from user-facing JSON and HTML responses.
- **CSRF & Safe Methods**: State-altering operations (`check-status`, `retry`, `upload`) strictly require `POST` with CSRF tokens.
- **Open-Redirect Protection**: All redirects (including `HTTP_REFERER` redirects) are validated with `url_has_allowed_host_and_scheme` to guarantee they only redirect within the same host and scheme.
- **SSRF Prevention**: Transcript downloads require HTTPS. If redirected to cloud storage (e.g. S3 buckets), authorization headers are stripped to prevent credential leakage.
- **Row-Level Concurrency Locking**: `select_for_update()` prevents race conditions or duplicate transcripts from concurrent requests.

---

## 6. Running Tests

Run the speech analytics test suite using Pipenv:

```bash
env -u DEBUG pipenv run python manage.py test speech_analytics
```

Run Django system and migration checks:

```bash
env -u DEBUG pipenv run python manage.py check
env -u DEBUG pipenv run python manage.py makemigrations speech_analytics --check --dry-run
```

*Note: All unit and integration tests mock external Gnani API calls. Tests run without network access and consume zero API credits.*

---

## 7. Known Limitations & Deferred Features

1. **Live Provider Verification**: Automated tests run against mock fixtures. Live Gnani Prisma API integration requires a valid provisioned API key and live billing credits.
2. **Diarization Speaker Mapping**: Diarized speaker segments (`Speaker 1`, `Speaker 2`) are parsed and rendered with timestamps and confidence scores when provided by the STT engine. However, the system does not attempt heuristic or automatic assignment of diarized labels to tournament speakers, preventing inaccurate debate speaker attributions.
3. **No Deletion Operation**: In alignment with Tabbycat's auditing model, recordings cannot be permanently deleted from the web UI.
