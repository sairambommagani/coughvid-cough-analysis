# 03 Software requirements specification (SRS)

IEC 62304 5.2. Each requirement has an id, is verifiable, and is traced to design, tests and risk controls in 08.
"Shall" = mandatory. RC-xx = risk control from 05.

## 1. Functional requirements

| Id | Requirement | Source |
|---|---|---|
| REQ-FUN-01 | The system shall accept one cough recording per request through `POST /analyze` (multipart field `file`, at most 20 MB). | intended use |
| REQ-FUN-02 | The system shall return a cough check: `is_cough` (true / false) and the calibrated cough probability. | intended use |
| REQ-FUN-03 | When the cough check is positive and the quality gate passes, the system shall return the cough type `wet`, `dry` or `indeterminate`, the wet probability, and the three acoustic features that moved the score most. | intended use |
| REQ-FUN-04 | The system shall return the segments where coughs were found, the number of coughs and the SNR. | usability |
| REQ-FUN-05 | The analysis of a 10 s recording shall finish within 30 s on the reference CPU (4 cores, no GPU). Measured on the developer laptop with PANNs: 12.4 s for the first request after start-up (model warm-up), about 0.5 s after that. | performance |
| REQ-FUN-06 | The web interface shall let a user record (10 s maximum) or upload a recording and show the result. | usability |

## 2. Input requirements

| Id | Requirement | Risk control |
|---|---|---|
| REQ-INP-01 | The system shall detect the audio format from the file content (magic numbers), not from the file name or Content-Type. | RC-08 |
| REQ-INP-02 | The system shall accept WebM, Ogg, MP4, M4A, 3GP, WAV, MP3, AAC, FLAC, AMR, AIFF and Matroska audio, and list them at `GET /formats`. | - |
| REQ-INP-03 | The system shall reject: non-audio content with 415, an empty file with 400, more than 20 MB with 413, audio that cannot be decoded with 422; each with a message saying what to do. | RC-08 |
| REQ-INP-04 | The system shall convert every accepted input to 16 kHz mono float PCM, peak-normalised, before any model. | RC-08 |
| REQ-INP-05 | Decoding shall be stopped after 30 s (malformed or hostile files). | RC-09 |
| REQ-INP-06 | The web recorder shall record Opus (WebM/Opus, else Ogg/Opus) where the browser supports it, else MP4/AAC, and name the file after the format actually recorded. | RC-08 |

## 3. Intended-use and labelling requirements

| Id | Requirement | Risk control |
|---|---|---|
| REQ-INT-01 | In production mode the system shall not compute, return or display COVID-19 or respiratory-symptom outputs. | RC-03 |
| REQ-INT-02 | Production shall be the default mode; research mode only with `COUGH_MODE=research`. | RC-03 |
| REQ-INT-03 | In production the test-set browser (`/examples*`) shall be disabled. | RC-03 |
| REQ-INT-04 | In production `GET /model-info` shall report the performance of the intended-use outputs only. | RC-03 |
| REQ-INT-05 | Cough type shall be reported only if a cough was detected, at least 2 coughs were found and SNR >= 10 dB; otherwise "not assessed" with the reason and a request to record again. | RC-01 |
| REQ-INT-06 | A wet probability from 0.40 to 0.60 shall be reported as "indeterminate". | RC-02 |
| REQ-LAB-01 | Every response and the web interface shall show the intended use and the regulatory status "Not a medical device ... not licensed by CDSCO ...". | RC-04 |
| REQ-LAB-02 | `GET /intended-use` shall return the intended use, users, contraindications, safety class intent and the allowed outputs. | RC-04 |

## 4. Audit requirements

| Id | Requirement | Risk control |
|---|---|---|
| REQ-AUD-01 | Every request to `/analyze` shall produce exactly one audit record with: request id (UUID4), UTC timestamp, user id, organisation id, model version, SHA-256 of the model card, SHA-256 of the input bytes, detected input format, HTTP status and the complete output JSON. | RC-10 |
| REQ-AUD-02 | In production a request without valid `X-User-Id` and `X-Org-Id` (opaque ids: letters, digits, `. _ : -`, no `@`) shall be refused with 401 and audited with the ids left empty. | RC-10 |
| REQ-AUD-03 | Audit records shall be hash-chained (each record holds the SHA-256 of the previous one) and `python -m cough.audit --verify` shall detect any changed, removed or reordered record. | RC-10 |
| REQ-AUD-04 | The audit log shall be append-only (opened only in append mode, flushed and fsync'ed per record) and the chain shall continue across restarts and daily files. | RC-10 |
| REQ-AUD-05 | If the audit record cannot be written, the system shall return 503 and no result. | RC-10 |
| REQ-AUD-06 | Rejected requests and erasures shall be audited as well. | RC-10 |
| REQ-AUD-07 | Files of finished days shall be sealable read-only; in deployment the audit folder shall be on write-once storage (REQ-ENV-03). | RC-10 |
| REQ-AUD-08 | The response shall carry the request id (body and `X-Request-ID` header) so a clinician can quote it. | RC-10 |

## 5. Personal data (PHI) requirements

| Id | Requirement | Risk control |
|---|---|---|
| REQ-PHI-01 | By default the system shall store, per request, only: input hash, acoustic features, output, model version, user and organisation ids. Audio shall not be stored. | RC-11 |
| REQ-PHI-02 | The system shall refuse (400) any form field or query parameter other than the audio file, so location or personal metadata cannot reach inference. | RC-11 |
| REQ-PHI-03 | Audio shall be stored only if `COUGH_AUDIO_RETENTION_DAYS` > 0, for that many days, and be deleted by `python -m cough.phi --purge`. | RC-11 |
| REQ-PHI-04 | At start-up the system shall refuse to serve a model whose inputs include latitude, longitude or other forbidden metadata (fail closed). | RC-11 |
| REQ-PHI-05 | `DELETE /subjects/{user_id}` shall delete that person's stored features and audio; a person may erase only their own data unless the caller is an admin account. | RC-11 |
| REQ-PHI-06 | The original file name shall not be stored or logged (it may contain a name). | RC-11 |

## 6. Model version requirements

| Id | Requirement | Risk control |
|---|---|---|
| REQ-VER-01 | Every training run shall create a new version `coughvid_v<MAJOR.MINOR.PATCH>_fusion_<YYYYMMDD>` with a model card holding version, parent, date, git commit, uncommitted-changes flag, training data SHA-256, split sizes, metrics file, test ROC-AUC, SOUP versions and SHA-256 of every model file. | RC-12 |
| REQ-VER-02 | At start-up the system shall verify every model file against the model card. | RC-12 |
| REQ-VER-03 | In production the system shall refuse (503) to serve a model version that fails verification or has no model card. | RC-12 |
| REQ-VER-04 | Model files shall never be overwritten: training refuses a folder that already holds a model; registered files are read-only; a version cannot be registered twice. | RC-12 |
| REQ-VER-05 | An earlier version shall be servable on demand (`COUGH_MODEL_VERSION` or `python -m cough.versioning --use`). | RC-07 |
| REQ-VER-06 | Every response and audit record shall name the model version that produced it. | RC-12 |

## 7. Requirements on the environment (deployment)

| Id | Requirement |
|---|---|
| REQ-ENV-01 | An authenticating gateway shall set `X-User-Id` and `X-Org-Id` from the logged-in account; clients must not be able to set them. |
| REQ-ENV-02 | All traffic shall use TLS 1.2 or later. |
| REQ-ENV-03 | The audit folder shall be replicated to write-once (WORM) storage hosted in India (MeitY-empanelled cloud), retained for at least 1 year (DPDP Rules 2025) and at least 180 days of ICT logs in India (CERT-In Directions 2022). |
| REQ-ENV-04 | Stored features and any retained audio shall be encrypted at rest; access limited to named roles. |
| REQ-ENV-05 | `python -m cough.phi --purge` shall run daily; `python -m cough.audit --verify` daily with an alert on failure. |
| REQ-ENV-06 | Monitoring (11 section 4) shall track request volume, rejection rates, cough-check rate, cough-type distribution and SNR distribution per model version. |
