# 09 Audit trail and personal-data (PHI) protection design

Requirements REQ-AUD-*, REQ-PHI-*, REQ-ENV-* (03); risk controls RC-10, RC-11 (05). India is the target market:
the DPDP Act 2023 and DPDP Rules 2025 apply to the personal data; in this document "PHI" means any health-related
personal data (the term comes from HIPAA, the obligations from the DPDP Act).

## 1. Audit trail

### 1.1 What is recorded
One record per request to `/analyze` (event `inference` or `rejected`) and per erasure (event `erasure`):

| Field | Meaning |
|---|---|
| seq | position in the chain (1, 2, 3 ...) |
| request_id | UUID4, also returned to the caller in the body and the `X-Request-ID` header |
| timestamp | UTC, ISO 8601 with milliseconds |
| event | inference / rejected / erasure |
| user_id, org_id | opaque account ids from the gateway (null if missing or invalid; an e-mail is never written) |
| status | HTTP status returned |
| mode | production / research |
| model_version | e.g. `coughvid_v1.0.0_fusion_20261021` |
| model_card_sha256 | SHA-256 of the served version's `version.json` (which holds the hashes of every model file) |
| input_sha256 | SHA-256 of the uploaded bytes (proves which input produced the output, without storing it) |
| input_format | detected format, e.g. webm |
| output | the complete JSON returned to the caller |
| prev_hash, hash | hash chain (below) |

No audio, no features, no file name, no location in the audit trail.

### 1.2 Append-only and tamper evidence
- `hash = SHA-256(prev_hash + canonical JSON of the record without hash)`; the first record chains from 64 zeros.
  Any edit, deletion, insertion or reordering breaks every later hash: `python -m cough.audit --verify` reports the
  first bad record (file, line, seq, reason). Run daily; alert on failure (REQ-ENV-05).
- Files are opened only in append mode; each record is flushed and fsync'ed before the response leaves; a lock
  serialises writers; a restarted process continues the sequence from the last record.
- No output without its record: if the write fails, the API answers 503 (REQ-AUD-05).
- `python -m cough.audit --seal` makes the files of finished days read-only.

### 1.3 WORM storage in deployment (REQ-ENV-03)
A local file system can be changed by an administrator, so the hash chain gives tamper *evidence*, not tamper
*proofing*. In deployment the audit folder is shipped every hour to object storage with a write-once retention lock
(compliance mode; for example S3 Object Lock or Azure immutable blob storage) in an Indian region of a
MeitY-empanelled cloud, retention at least 1 year (DPDP Rules 2025 keep logs for at least one year) and the ICT
logs at least 180 days in India (CERT-In Directions, 28 April 2022). The last hash of each day is also recorded in
a separate system (the daily verification report), so even a full replacement of the log would be detected.

### 1.4 Using the audit trail
`python -m cough.audit --find <request_id>` returns the record of a response a clinician questions: which model
version, which input (hash), which output, who and when. Problem reports (02 section 8) quote request ids.

## 2. Personal data: what is processed and stored

| Data | Personal data? | Default | With audio retention on |
|---|---|---|---|
| Cough audio | yes (health data, voice biometric) | processed in memory / temp file, deleted at the end of the request | stored N days under `audio/<org>/<request_id>.<ext>`, then purged |
| Input SHA-256 | pseudonymous | audit + feature record | same |
| Acoustic features (numbers) | pseudonymous health data | feature record `features.jsonl` | same |
| Output | health data | audit + response | same |
| User id, org id | yes (identifiers) | audit + feature record | same |
| File name | possibly (may contain a name) | never stored | never stored |
| Latitude / longitude, age, gender, symptoms, dates | yes | refused at the API (400) | refused |

### 2.1 Data minimisation
The COUGHVID training data contained latitude and longitude, age and gender. None of them is a model input
(`samd.check_model_inputs` refuses a model that would use them, so a future model cannot add them silently), and
the API refuses any field other than the audio file (REQ-PHI-02). Location was used only offline, to group
recordings for a leak-free train / test split.

### 2.2 Retention (documented)

| Data | Retention | Deletion |
|---|---|---|
| Audio | 0 days by default; with `COUGH_AUDIO_RETENTION_DAYS=N`, N days (set per organisation by contract, for clinician review) | `python -m cough.phi --purge`, daily |
| Feature records | until the person's erasure request or the end of the contract with the organisation; reviewed yearly | `DELETE /subjects/{user_id}`; contract end: organisation's folder deleted |
| Audit records | at least 1 year, then as required by the contract and law | expiry of the WORM retention; never before |

### 2.3 Right to erasure (DPDP Act s. 12)
`DELETE /subjects/{user_id}` (by the person, or an admin account on their behalf) deletes their feature records and
any retained audio, and records the erasure in the audit trail. The audit records themselves are kept: they hold no
audio and no features, and they are required evidence of what the software did; this exception, its legal basis
and the retention period are stated in the privacy notice (10). If the law or contract requires, the user id in the
audit trail can be pseudonymised per organisation at the gateway, so erasure of the mapping makes the audit records
anonymous.

### 2.4 Security safeguards (DPDP Rules 2025, Rule 6)
- Encryption in transit (TLS 1.2+) and at rest for the data store and audio (REQ-ENV-02, 04).
- Access control: only the service account writes the data store; named roles read it; audit storage is
  write-once.
- Activity logs: the audit trail, plus gateway access logs.
- Backups of the data store (encrypted); the audit trail is backed up by its WORM replication.
- Processor contracts carry these terms (10).
- Breach response: section 3 of 10 (Data Protection Board and affected persons; full report within 72 hours;
  CERT-In within 6 hours).
