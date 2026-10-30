# 04 Software architecture and design

IEC 62304 5.3 (architecture), 5.4 (detailed design for the risk-control units).

## 1. Context

```
 clinician / patient (browser, phone)
        |  HTTPS (TLS)
        v
 authenticating gateway  -- sets X-User-Id, X-Org-Id from the login (REQ-ENV-01)
        |
        v
 COUGHVID service (this software)
   cough/api.py ------------------------------------------------------------+
     | 1 identity check (caller)          -> 401 if missing                  |
     | 2 metadata refusal (form/query)    -> 400                             |
     | 3 format check   formats.py        -> 400/413/415                     |
     | 4 decode         audio.py + ffmpeg -> 422 (30 s limit)                |
     | 5 features       features.py                                         |
     | 6 models         LightGBM, PANNs+LR, CNN (only allowed tasks)        |
     | 7 output rules   samd.py: quality gate, indeterminate band, labels   |
     | 8 store          phi.py: hash + features (audio only if retention)  |
     | 9 audit          audit.py: hash-chained record, fsync               |-> audit/ (WORM in deployment)
     +-> response (request id, model version, labelling)                  |
   versioning.py: loads LATEST version, verifies file hashes at start-up   |
        |                                                                  |
        v                                                                  v
 artifacts/<version>/ (read-only)                         data_store/ (features.jsonl, audio/ if retention)
```

## 2. Software items

| Item | Files | Responsibility | Class |
|---|---|---|---|
| SI-1 API and orchestration | `cough/api.py` | endpoints, request flow above, error mapping, mode handling | B |
| SI-2 Intended-use controls | `cough/samd.py` | mode, allowed outputs, quality gate, indeterminate band, labelling, forbidden inputs | B (Class C detail) |
| SI-3 Input format check | `cough/formats.py` | content-based format detection, supported list, rejection reasons | B (Class C detail) |
| SI-4 Audio processing | `cough/audio.py`, `cough/features.py` | ffmpeg decoding with time limit, normalisation, cough segmentation, SNR, handcrafted features, log-mel | B |
| SI-5 Models | `cough/cnn.py`, `cough/panns.py`, `cough/train.py` (fusion) | inference of the three models, fusion, calibration | B |
| SI-6 Audit trail | `cough/audit.py` | append-only hash-chained log, verify, seal, find | B (Class C detail) |
| SI-7 PHI store | `cough/phi.py` | feature store, opt-in audio retention, purge, erasure | B (Class C detail) |
| SI-8 Model versions | `cough/versioning.py`, `cough/train.py` | semantic versions, model cards, registry, integrity, read-only, rollback | B (Class C detail) |
| SI-9 Web interface | `cough/static/index.html` | recorder (Opus / MP4), upload, result display, labelling | B |
| SI-10 Training and evaluation (offline) | `cough/prepare.py`, `embed.py`, `train.py`, `evaluate.py`, `data.py` | produce and evaluate model versions; not part of the running service | B |

SOUP used by each item: 06.

## 3. Interfaces

| Interface | Direction | Content |
|---|---|---|
| `POST /analyze` | in | multipart `file`; headers `X-User-Id`, `X-Org-Id`; nothing else accepted |
| `POST /analyze` | out | JSON: request_id, mode, intended_use, regulatory_status, input (format, codec, converted_to), recording, cough_check, cough_type, model_version, disclaimer, data_retention; header `X-Request-ID` |
| `GET /intended-use`, `/formats`, `/health`, `/model-info` | out | labelling, formats, status and integrity, performance of allowed tasks |
| `DELETE /subjects/{user_id}` | in/out | erasure of one person's stored data |
| ffmpeg subprocess | out/in | file path in, f32le PCM out; 30 s timeout |
| Model folder | in | `config.json`, `models.joblib`, `cnn.pt`, `metrics.json`, `test_predictions.parquet`, `version.json` |
| Audit folder | out | `audit-YYYYMMDD.jsonl`, one JSON record per line |
| Data store | out | `features.jsonl`; `audio/<org>/<request_id>.<ext>` only with retention |

## 4. Segregation and SOUP isolation (IEC 62304 5.3.5)

- ffmpeg, the parser of untrusted input, runs in a separate process with a time limit; its crash or hang cannot
  corrupt the service state, and it only receives files that passed the format check and the size limit.
- The research outputs are separated by mode, not by code branches in the client: in production the disease
  models are not evaluated at all (`Engine.scores` skips tasks outside `samd.allowed_tasks()`).
- Model files are read-only and hash-verified, so a SOUP or operator error cannot silently change the served model.

## 5. Detailed design of the risk-control units

### 5.1 Output gating (SI-2, RC-01..RC-04)
```
analyse_file(path):
  p = scores(path, tasks = allowed_tasks())        # production: ("cough", "wet") only
  cough_check = is_cough if p_cough >= 0.5 and segments found
  if not cough_check: return message "record again"
  if "wet" in p:
     ok, reason = quality_gate(n_coughs, snr, is_cough)    # >= 2 coughs and SNR >= 10 dB
     if ok:  label = "indeterminate" if 0.40 <= p_wet <= 0.60 else ("wet" if p_wet > 0.60 else "dry")
     else:   label = "not assessed", reason
  research mode only: COVID-19 / symptom scores, labelled research_use_only
  every response: intended_use, regulatory_status, disclaimer, model_version
```

### 5.2 Format check (SI-3, RC-08)
First 4 KB inspected: `RIFF....WAVE` -> wav (codec from the fmt chunk), `ID3` or MPEG layer III frame header -> mp3,
`OggS` -> ogg (OpusHead / vorbis / FLAC), EBML `1A 45 DF A3` -> webm (doctype "webm") or mkv (codec A_OPUS /
A_VORBIS / A_AAC), `ftyp` at offset 4 -> m4a / 3gp / mp4 by brand, `fLaC` -> flac, `#!AMR` -> amr, `FORM....AIFF`
-> aiff, ADTS sync -> aac. Known non-audio signatures (PNG, JPEG, GIF, PDF, ZIP, JSON, HTML, plain text) give a
named reason. The temporary file gets the extension of the detected format, so a renamed file decodes correctly.

### 5.3 Audit record (SI-6, RC-10)
```
rec = {seq, request_id, timestamp(UTC ms), event, user_id, org_id, status, mode, model_version,
       model_card_sha256, input_sha256, input_format, output, prev_hash}
rec.hash = SHA-256(prev_hash + canonical_json(rec))     # canonical: sorted keys, no spaces
append line to audit-YYYYMMDD.jsonl; flush; fsync       # under a lock; seq and prev kept in memory, re-read at start
verify(): recompute every hash from GENESIS = 64 zeros; first mismatch reported with file, line, seq and reason
```
Response is returned only after the record is on disk; on failure 503 (REQ-AUD-05). Invalid ids are written as
null, never verbatim (an e-mail address must not enter an immutable log).

### 5.4 Model integrity (SI-8, RC-12)
`register()` writes `version.json` with SHA-256 of every model file, then sets all files read-only. At start-up
`Engine` calls `verify_integrity()`; in production any missing or different file, or a missing card, stops the
model from being served (HTTP 503 on every request) instead of serving an unknown model. `train.run()` refuses a
folder that already contains `models.joblib` or `version.json`.

### 5.5 PHI store (SI-7, RC-11)
Feature record per request (JSON line): request_id, user_id, org_id, input_sha256, model_version, stored_at,
features (numbers only). Erasure rewrites `features.jsonl` without that person's lines (atomic replace) and
deletes their retained audio. Purge deletes audio whose age exceeds its retention days.

## 6. Data flow of personal data

| Data | Where it goes | Kept |
|---|---|---|
| Audio bytes | temp file (deleted in `finally`), ffmpeg pipe | not kept (default); N days with retention |
| Input SHA-256 | audit record, feature record | audit: >= 1 year (WORM); features: until erasure |
| Acoustic features | feature record | until erasure |
| User / org id | audit record, feature record | as above |
| File name | used only to compare with the detected format | never stored |
| Location, demographics | refused at the API (400); models cannot use them (fail closed) | never received |
