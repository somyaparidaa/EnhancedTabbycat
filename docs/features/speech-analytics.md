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

---

## 8. Stage 2: Long Audio & Google Cloud Speech Integration

Stage 2 introduces production-ready long audio transcription support via Google Cloud Speech-to-Text v1 and Google Cloud Storage (GCS).

### 8.1 Dual Synchronous & Long-Running Asynchronous Processing

- **Short Audio (<= 60s and < 10MB)**: Uses synchronous `client.recognize()` directly on in-memory audio bytes, completing immediately during upload or retry without touching Cloud Storage.
- **Long Audio (> 60s or >= 10MB)**: Uses Google Cloud Storage and `client.long_running_recognize()`.
  1. The audio file is automatically transcoded to 16kHz mono `LINEAR16` WAV if uploaded in an unsupported container (e.g., M4A/AAC).
  2. The transcoded audio is uploaded to a collision-resistant GCS path (`gs://<bucket_name>/speech_recordings/<uuid>/<filename>`).
  3. A long-running recognition operation is started using Google Speech-to-Text API v1.
  4. The operation ID (`projects/.../locations/global/operations/...`) is saved as the recording's `job_id`, and status remains `PROCESSING` without blocking the web request.
  5. Status is polled non-blockingly through the existing "Check Status" view.
  6. Upon completion, the full transcript, word timestamps, and diarized speaker segments are normalized and saved to `SpeechTranscript`.
  7. The temporary GCS audio object is deleted only after the transcript is successfully persisted (configurable via `GOOGLE_CLOUD_STORAGE_CLEANUP_ON_SUCCESS`).

### 8.2 Google Cloud Configuration & IAM Setup

1. **Required Environment Variables**:
   - `SPEECH_TRANSCRIPTION_PROVIDER="google"`: Activates Google Cloud Speech-to-Text.
   - `GOOGLE_CLOUD_STORAGE_BUCKET_NAME="<your-gcs-bucket-name>"`: Dedicated bucket for temporary long-audio transcription objects.
   - `GOOGLE_SPEECH_LANGUAGE_CODE="en-US"` (default `en-US`): Target speech language code.
   - `GOOGLE_SPEECH_MODEL="latest_long"` (default `latest_long`): Google Speech v1 model optimized for debate speeches.
   - `GOOGLE_CLOUD_STORAGE_CLEANUP_ON_SUCCESS="true"` (default `true`): Deletes temporary GCS audio objects after successful transcript persistence.

2. **IAM Permissions**:
   - The user/service account running Tabbycat (via ADC or service account) requires:
     - `roles/storage.objectAdmin` or `roles/storage.objectUser` on the configured bucket.
     - `roles/speech.client` on the Google Cloud project.
   - Ensure the Google Cloud Speech service agent (`service-<project-number>@gcp-sa-speech.iam.gserviceaccount.com`) has read access to the bucket (`Storage Object Viewer`).

3. **Testing Full Debate Audio**:
   To test a full debate recording (e.g. 7-minute speech or 60-minute debate round):
   ```bash
   export SPEECH_TRANSCRIPTION_PROVIDER="google"
   export GOOGLE_CLOUD_STORAGE_BUCKET_NAME="my-debate-audio-bucket"
   export GOOGLE_SPEECH_MODEL="latest_long"
   ```
   Upload the audio via the tournament dashboard. The request will return immediately with status `Processing`. Click **Check Status** once processing finishes to view the complete transcript with diarized segments.

---

## 9. Stage 2.1: AI Debate Analysis Contract (v1.0.0)

Stage 2 introduces structured, objective AI-powered competitive debate analysis. The analysis contract is decoupled from any specific generative model (such as Gnani Evon) and formalizes competitive debate evaluation dimensions.

### 8.1 Output Structure & Dimensions

The contract (`tabbycat/speech_analytics/schemas/debate_analysis_v1.json`) defines seven core substantive dimensions validated by `speech_analytics.contract`:

1. **`metadata`**: Target speaker, debate format (e.g. `BP`, `Australs`), side, speech role, motion text, info slide, and `context_completeness` (`full_debate`, `partial_round`, or `isolated_speech`).
2. **`argument_analysis`**: Constructive claims, underlying reasoning/mechanisms, analytical strengths, vulnerabilities, and unsupported claims or logical gaps.
3. **`rebuttal_analysis`**: Targets opposing arguments, classifying engagement type (`direct_rebuttal`, `counter_framing`, `mere_mention`, `mitigation`, `turning_the_argument`), logical effectiveness, and missed rebuttal opportunities. Crucially distinguishes actual refutation from mere superficial mention.
4. **`clash_analysis`**: Central disagreements in the debate, speaker positioning, comparative weighing metrics (scale, reversibility, likelihood), and unaddressed dimensions.
5. **`role_fulfilment`**: Expected normative responsibilities for the speech position and format, fulfilled duties, and missing or insufficiently addressed responsibilities.
6. **`structure_analysis`**: Organisation and progression, logical coherence, signposting/transitions, and time management pacing.
7. **`feedback`**: Actionable constructive feedback divided into strengths, areas for improvement, and actionable suggestions.
8. **`limitations`**: Explicit accounting of missing debate context (e.g. unrecorded prior speeches or POIs), inaudible audio / transcript gaps, uncertain interpretations, and unevaluated dimensions (e.g. delivery, body language).

### 8.2 Evidence & Provenance Handling

Every observation across arguments, rebuttals, clashes, role fulfilment, and structure must ground itself in verifiable evidence via `EvidenceReference`:
- **Verbatim Excerpt**: `quote` referencing the source transcript text.
- **Provenance Source**: `source` indicating whether the excerpt originates from `target_speech`, `prior_speech`, `subsequent_speech`, or `motion_context`. This enables evaluating rebuttals against actual speeches earlier in the debate without fabricating opposition claims.
- **Timestamps**: Optional audio start/end offsets (`start_time`, `end_time`) with strict ordering validation (`start_time <= end_time`).
- **Observation Type**: Explicit classification as `direct_observation` (cited fact/quote) versus `model_interpretation` (AI analytical inference).

### 8.3 Context Completeness & Future Analytics

Competitive debate analysis requires full round context to evaluate refutations, extensions, and clash weighing fairly. The contract represents missing information explicitly (`context_completeness: "isolated_speech"` vs `"full_debate"`):
- The model must never construe the absence of an unprovided speech as proof that a speaker failed to respond to it.
- The contract strictly forbids arbitrary numerical overall scores, speaker points, or debate rankings, preserving debate adjudication integrity while providing actionable qualitative analytics for speaker development.



---

## 10. Stage 2B: Transcript Intelligence & Debate Context Contract (v1.0.0)

Stage 2B bridges the raw Speech-to-Text layer (`SpeechTranscript`) and downstream NLP models by converting unsegmented transcripts into timestamp-preserving, debate-aware data structures.

### 10.1 Preprocessing & Sentence Segmentation (`TranscriptPreprocessor`)

- **Sentence-Level Splitting**: Segments transcript text into clean grammatical sentences (`s_0001`, `s_0002`, ...) while preserving verbatim character offsets.
- **Word-Level Timestamp Precision**: When word-level timestamps (`words` array) exist in STT metadata, each sentence derives its exact `start_time` and `end_time` by matching sentence boundaries to word token offsets.
- **Segment Fallback & Interpolation**: When only segment-level timestamps are available, start and end bounds are proportionally interpolated according to character positions, and an explicit warning is recorded (`Segment-level timestamps proportionally interpolated for internal sentences.`).
- **Paragraph Grouping**: Sentences are grouped into paragraphs (`p_001`, `p_002`, ...) upon speaker switches or significant acoustic pauses ($\ge 2.5$s default threshold).
- **Graceful Fault Tolerance**: Handles empty segments, noisy transcripts, punctuation-free transcripts, and missing confidence scores without failing.

### 10.2 Native Tabbycat Context Extraction (`DebateContextBuilder`)

Extracts rich domain context without duplicating Tabbycat models or creating redundant database tables:
- **Tournament & Rules**: Format name, team counts (`teams_in_debate` = 2 vs 4), speaker sequence (`substantive_speakers`, `reply_scores_enabled`), and side naming configurations (`aff-neg`, `gov-opp`, `prop-opp`).
- **Round & Stage**: Preliminary vs elimination round, round sequence number, and draw type.
- **Motion & Info Slide**: Primary debate motion, motion text, reference, info slide, and any vetoed or alternative round motions (`RoundMotion`).
- **Teams & Sides**: Roster of competing teams with side allocations (`OG`, `OO`, `CG`, `CO` for British Parliamentary; `AFF`, `NEG` for 2-team formats), code names, and institution affiliations.
- **Speeches & Roles**: Speech positions mapped from confirmed `BallotSubmission` results (e.g. `PM`, `DPM`, `LO`, `DLO`, `MG`, `GW`, `MO`, `OW` in BP, or `Aff1`, `Neg1`, `AffR`, `NegR` in 2-team formats). If no ballot has been submitted yet, expected speech slots are synthesized from tournament rules.
- **Speaker Mapping & POI Disambiguation**:
  - `RESOLVED_SINGLE_SPEAKER`: Target speaker specified and aligns with the diarization label.
  - `RESOLVED_TARGET_WITH_POIS`: Target speaker identified as primary speaker, while secondary diarized labels are isolated as points of information (POIs) or floor interjections without hallucinating speaker identities.
  - `UNRESOLVED_MULTI_SPEAKER`: Multi-speaker recording without individual speaker linkage.
  - `DEBATE_ONLY_UNASSIGNED`: Recording linked to the debate with no individual speaker assigned.

### 10.3 Analysis Input Contract (`DebateAnalysisInputBundle`)

Generated programmatically via `transcript.get_analysis_input()` or `build_debate_analysis_input(transcript)`:

- **JSON Schema**: Validated against `tabbycat/speech_analytics/schemas/debate_analysis_input_v1.json`.
- **Reproducibility Fingerprint**: Every bundle generates a deterministic SHA-256 fingerprint over full text, sentence timestamps, and debate motion IDs.
- **UI Inspection**: Tournament administrators can view and inspect the structured analysis input bundle directly in the **Transcript Detail** view under *View Structured Debate Analysis Input (NLP Context Bundle)*.

#### Illustrative Debate Analysis Input Payload

```json
{
  "contract_version": "1.0.0",
  "fingerprint": "7c4840d21054b5dfd4fcf7f33be16a5061483bf9872be9c3f4e2f9dcfd0eefaa",
  "generated_at": "2026-11-15T06:05:00.000000+00:00",
  "provenance": {
    "transcript_id": 14,
    "recording_id": 12,
    "audio_duration": 420.5,
    "provider": "google",
    "language_code": "en-US",
    "overall_confidence": 0.96,
    "context_completeness": "isolated_speech"
  },
  "debate_context": {
    "tournament": {
      "id": 1,
      "name": "World Universities Debating Championship 2026",
      "slug": "wudc-2026",
      "format_name": "British Parliamentary (BP)",
      "teams_in_debate": 4,
      "substantive_speakers": 2,
      "reply_scores_enabled": false,
      "side_names_style": "gov-opp"
    },
    "round": {
      "id": 1,
      "seq": 1,
      "name": "Round 1",
      "abbreviation": "R1",
      "stage": "P",
      "stage_display": "Preliminary",
      "is_break_round": false,
      "draw_type": "R"
    },
    "debate": {
      "id": 5,
      "matchup": "Oxford A vs Harvard B vs Cambridge C vs Sydney D",
      "importance": 0,
      "bracket": 0.0,
      "confirmed_ballot_available": true
    },
    "motion": {
      "primary_motion": {
        "id": 3,
        "reference": "Pharma Patents",
        "text": "This House would abolish patent protection for pharmaceuticals.",
        "info_slide": "Pharmaceutical patents grant temporary monopolies to developers."
      },
      "round_motions": [
        {
          "id": 3,
          "reference": "Pharma Patents",
          "text": "This House would abolish patent protection for pharmaceuticals.",
          "info_slide": "Pharmaceutical patents grant temporary monopolies to developers.",
          "seq": 1,
          "is_chosen": true
        }
      ],
      "has_multiple_motions": false
    },
    "teams": [
      {
        "debate_team_id": 10,
        "team_id": 101,
        "team_name": "Oxford A",
        "code_name": "Falcon",
        "institution": "University of Oxford",
        "side_raw": 0,
        "side_code": "OG",
        "side_name": "Opening Government",
        "side_abbr": "OG",
        "rostered_speakers": [
          { "id": 201, "name": "Alice Smith", "code_name": "AS01" },
          { "id": 202, "name": "Bob Jones", "code_name": "BJ02" }
        ]
      }
    ],
    "speeches": [
      {
        "position": 1,
        "side_code": "OG",
        "role_abbr": "PM",
        "role_name": "Prime Minister",
        "speaker_id": 201,
        "speaker_name": "Alice Smith",
        "team_id": 101,
        "team_name": "Oxford A",
        "is_ghost": false
      }
    ],
    "speaker_mapping": {
      "status": "RESOLVED_TARGET_WITH_POIS",
      "target_speaker": {
        "id": 201,
        "name": "Alice Smith",
        "team": "Oxford A",
        "side_code": "OG",
        "speech_role": "Prime Minister",
        "speech_position": 1
      },
      "diarized_speakers": ["Speaker 1", "Speaker 2"],
      "unresolved_speakers": ["Speaker 2"],
      "mapping": {
        "Speaker 1": {
          "speaker_id": 201,
          "speaker_name": "Alice Smith",
          "role": "Prime Minister",
          "type": "target_speaker"
        },
        "Speaker 2": {
          "speaker_id": null,
          "speaker_name": "Unresolved (POI / Interjection)",
          "role": "POI / Floor Interjection",
          "type": "unresolved_poi"
        }
      }
    }
  },
  "transcript": {
    "full_text": "We are proud to propose the motion. First, monopolies create artificial scarcity. Point of information! Rejected, thank you.",
    "stats": {
      "total_sentences": 4,
      "total_paragraphs": 3,
      "total_words": 18,
      "speaker_count": 2
    },
    "sentences": [
      {
        "id": "s_0001",
        "text": "We are proud to propose the motion.",
        "speaker": "Speaker 1",
        "start_time": 0.0,
        "end_time": 2.4,
        "confidence": 0.98,
        "word_count": 7,
        "segment_index": 1,
        "timing_source": "words",
        "paragraph_id": "p_001"
      },
      {
        "id": "s_0002",
        "text": "First, monopolies create artificial scarcity.",
        "speaker": "Speaker 1",
        "start_time": 2.5,
        "end_time": 5.1,
        "confidence": 0.95,
        "word_count": 5,
        "segment_index": 1,
        "timing_source": "words",
        "paragraph_id": "p_001"
      },
      {
        "id": "s_0003",
        "text": "Point of information!",
        "speaker": "Speaker 2",
        "start_time": 5.8,
        "end_time": 6.9,
        "confidence": 0.92,
        "word_count": 3,
        "segment_index": 2,
        "timing_source": "words",
        "paragraph_id": "p_002"
      },
      {
        "id": "s_0004",
        "text": "Rejected, thank you.",
        "speaker": "Speaker 1",
        "start_time": 7.2,
        "end_time": 8.5,
        "confidence": 0.97,
        "word_count": 3,
        "segment_index": 3,
        "timing_source": "words",
        "paragraph_id": "p_003"
      }
    ],
    "paragraphs": [
      {
        "id": "p_001",
        "speaker": "Speaker 1",
        "sentence_ids": ["s_0001", "s_0002"],
        "text": "We are proud to propose the motion. First, monopolies create artificial scarcity.",
        "start_time": 0.0,
        "end_time": 5.1,
        "confidence": 0.965,
        "word_count": 12
      },
      {
        "id": "p_002",
        "speaker": "Speaker 2",
        "sentence_ids": ["s_0003"],
        "text": "Point of information!",
        "start_time": 5.8,
        "end_time": 6.9,
        "confidence": 0.92,
        "word_count": 3
      },
      {
        "id": "p_003",
        "speaker": "Speaker 1",
        "sentence_ids": ["s_0004"],
        "text": "Rejected, thank you.",
        "start_time": 7.2,
        "end_time": 8.5,
        "confidence": 0.97,
        "word_count": 3
      }
    ]
  },
  "warnings": [
    "Multiple diarized voices detected; primary target speaker separated from POIs/interjections."
  ]
}
```
