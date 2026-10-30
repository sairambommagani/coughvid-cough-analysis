# 07 Verification test protocol and report

IEC 62304 5.5 (unit verification), 5.6 (integration testing), 5.7 (system testing). All tests are automated with
pytest and run on synthetic recordings in the real file formats (no patient data).

## 1. Procedure

1. Environment: the release environment; record its SOUP versions with `python scripts/soup_versions.py`.
2. Run `python -m pytest -q` in the repository root. ffmpeg and PyTorch must be installed (tests that need them are
   skipped otherwise: a skip is a failure for a release).
3. Pass criterion for the release: every test passes, 0 skipped, 0 failed.
4. Record the result in section 4 (date, git commit, number passed, environment) and keep the console output.

## 2. Test cases

Level: U = unit, I = integration, S = system (production API on a trained, registered model).

| Test case | Level | What is checked (expected result) | Requirement | Automated test |
|---|---|---|---|---|
| TC-FMT-01 | U | each of 10 formats (WebM/Opus, Ogg/Opus, Ogg/Vorbis, MP4/AAC, M4A/AAC, AAC, MP3, WAV 16-bit, WAV 8 kHz, FLAC) is detected from its bytes with the right container and codec | REQ-INP-01, 02 | `test_formats.py::test_detect`, `test_matrix_covers_every_generator_format` |
| TC-FMT-02 | I | each format decodes to the same model input (16 kHz mono, peak 1.0, length within 0.15 s of the WAV, same cough count +-1) | REQ-INP-04 | `test_decode_to_model_input` |
| TC-FMT-03 | U | an MP3 renamed `.webm` is detected as MP3, flagged `extension_matches = false`, and decodes | REQ-INP-01 | `test_wrong_extension_is_detected_and_still_decodes` |
| TC-FMT-04 | U | PNG, PDF, ZIP, text, JSON and random bytes are rejected with a named reason | REQ-INP-03 | `test_not_audio_is_rejected` |
| TC-FMT-05 | U | empty and 2-byte inputs are rejected | REQ-INP-03 | `test_empty_and_tiny` |
| TC-FMT-06 | I | a truncated WebM is detected as WebM but fails to decode | REQ-INP-03 | `test_truncated_file_fails_to_decode` |
| TC-FMT-07 | S | `POST /analyze` returns 200 for every format, with the right name, a wrong name and no name (30 cases); input format reported; model version present | REQ-FUN-01, REQ-INP-01..04, REQ-VER-06 | `test_api_accepts_every_format` |
| TC-FMT-08 | S | non-audio -> 415, empty -> 400, > 20 MB -> 413, truncated -> 422; `GET /formats` lists the formats | REQ-INP-02, 03 | `test_api_rejects_non_audio`, `test_api_empty_and_too_large`, `test_api_truncated_is_422`, `test_api_lists_formats` |
| TC-INP-05 | U | decoding stops with an error when it exceeds the time limit | REQ-INP-05 | `test_tc_inp_05_decoder_time_limit` |
| TC-INT-01 | S | production response on a trained model has no `covid_19` / `respiratory_symptoms`; research mode has them | REQ-INT-01 | `test_cough.py::test_pipeline_and_api` |
| TC-INT-02 | U | default mode is production; only `cough`, `wet` allowed; invalid mode refused | REQ-INT-02 | `test_samd.py::test_tc_int_02_default_mode_is_production` |
| TC-INT-03 | S | `/examples` returns 404 in production | REQ-INT-03 | `test_pipeline_and_api` |
| TC-INT-04 | S | `/model-info` in production reports only `cough` and `wet` | REQ-INT-04 | `test_pipeline_and_api` |
| TC-INT-05 | U, S | labelling: intended use and "Not a medical device" in every response | REQ-LAB-01, 02 | `test_tc_int_05_labelling`, `test_pipeline_and_api` |
| TC-RSK-01 | U | quality gate: passes 3 coughs / 25 dB; refuses 1 cough, 6 dB, no cough | REQ-INT-05 | `test_tc_rsk_01_quality_gate` |
| TC-RSK-02 | U | 0.10 dry, 0.40 / 0.50 / 0.60 indeterminate, 0.85 wet | REQ-INT-06 | `test_tc_rsk_02_indeterminate_band` |
| TC-AUD-01 | S | the audit record of a real inference has request id, user, org, model version, model card hash, input SHA-256 equal to the uploaded bytes, and output equal to the response; `X-Request-ID` equals the body | REQ-AUD-01, 08, REQ-VER-06 | `test_pipeline_and_api` |
| TC-AUD-02 | U, S | ids with `@`, spaces or empty are invalid; production without ids -> 401 | REQ-AUD-02 | `test_tc_aud_02_id_rules`, `test_pipeline_and_api` |
| TC-AUD-03 | U, S | changing an output, deleting a record: verify fails at the right record; restoring the file: verify passes; real log verifies | REQ-AUD-03 | `test_tc_aud_03_hash_chain_detects_tampering`, `test_pipeline_and_api` |
| TC-AUD-04 | U | a second logger on the same folder continues the sequence and the chain | REQ-AUD-04 | `test_tc_aud_04_append_only_and_chain_survives_restart` |
| TC-AUD-05 | U | finished days are sealed read-only | REQ-AUD-07 | `test_tc_aud_05_seal_makes_old_days_read_only` |
| TC-AUD-06 | S | a rejected upload (415) has its own audit record with status, input hash and user | REQ-AUD-06 | `test_tc_aud_06_rejected_requests_are_audited` |
| TC-AUD-07 | S | if the audit write fails the API returns 503 and no result | REQ-AUD-05 | `test_tc_aud_07_no_result_without_audit_record` |
| TC-PHI-01 | S | the stored feature record has no latitude / longitude and no file name | REQ-PHI-01, 06 | `test_pipeline_and_api` |
| TC-PHI-02 | S | an upload with a `latitude` form field is refused with 400 | REQ-PHI-02 | `test_pipeline_and_api` |
| TC-PHI-03 | U, S | audio is not stored by default | REQ-PHI-01, 03 | `test_tc_phi_03_audio_not_stored_by_default`, `test_pipeline_and_api` |
| TC-PHI-04 | U | a model using latitude, longitude, lat_*, age or gender is refused; the pipeline's own feature list passes | REQ-PHI-04 | `test_tc_phi_04_forbidden_model_inputs`, `test_tc_phi_04_trained_feature_list_is_audio_only` |
| TC-PHI-05 | U, S | erasure deletes exactly that person's features and audio; others stay; via API | REQ-PHI-05 | `test_tc_phi_05_erasure`, `test_pipeline_and_api` |
| TC-PHI-06 | U | retained audio survives until its retention period and is purged after it | REQ-PHI-03 | `test_tc_phi_06_retention_and_purge` |
| TC-VER-01 | U | the model card has the semantic version, name pattern, training data SHA-256, metrics file, file hashes and SOUP versions | REQ-VER-01 | `test_tc_ver_01_model_card_has_version_data_hash_metrics`, `test_semver_and_names` |
| TC-VER-02 | S | `/health` reports integrity "verified" for a registered version | REQ-VER-02 | `test_pipeline_and_api` |
| TC-VER-03 | U, S | a changed model file fails verification; production returns 503 for it; files are read-only; no double registration | REQ-VER-03, 04 | `test_tc_ver_03_integrity_and_no_overwrite`, `test_pipeline_and_api` |
| TC-VER-04 | U | training refuses a folder that already holds a model | REQ-VER-04 | `test_tc_ver_04_train_refuses_to_overwrite` |
| TC-VER-05 | U | versions 1.0.0 -> 1.1.0 -> 1.1.1 -> 2.0.0, LATEST follows, `COUGH_MODEL_VERSION=1.1.0` rolls back, duplicate refused | REQ-VER-05 | `test_registry_latest_and_rollback` |
| TC-VER-06 | U | an un-adopted flat model becomes 1.0.0 before a new version; a crashed run does not block the next | REQ-VER-01, 04 | `test_crashed_run_and_unadopted_legacy` |
| TC-VER-07 | U | a model card without file hashes (registered before integrity checks) gets them once, recorded as added later; a second attempt is refused | REQ-VER-02, 04 | `test_tc_ver_07_migration_adds_hashes_once` |
| TC-DOC-01 | U | the regulatory documents exist and the intended use in 01 equals the code | REQ-LAB-01 | `test_tc_doc_01_regulatory_documents_present` |
| TC-REG-01 | I | existing functional tests: segmentation, expert labels, metrics, full pipeline from raw data to served prediction | REQ-FUN-02..04 | `test_cough.py` (all) |

Not automated (checked by review at release, recorded in 11): REQ-FUN-05 response time (measured in the browser
check), REQ-FUN-06 and REQ-INP-06 recorder behaviour in Chrome / Edge / Firefox / Safari, REQ-ENV-* (deployment).

## 3. Model performance verification (system level)

The test-set metrics of the served version are in its `metrics.json` (hash in the model card) and summarised in the
model card (`test_roc_auc`). Acceptance for a new version: cough check ROC-AUC >= 0.97 and wet / dry ROC-AUC not
lower than the parent version's lower 95% confidence bound, on the same held-out split (11 section 2).

Version 1.0.0 (`coughvid_v1.0.0_fusion_20261021`): cough check 0.988 (0.98-0.99); wet / dry 0.674 (0.59-0.75).

## 4. Test report

| Date | Git commit | Environment | Result | By |
|---|---|---|---|---|
| 2026-10-28 | 34a2e4c | Windows, Python 3.11 (developer laptop) | 76 passed (before the SaMD changes) | Sairam |
| 2026-10-29 | release commit (SaMD conversion, on top of 34a2e4c) | Windows, Python 3.11 (developer laptop; SOUP versions in 06a) | 97 passed, 0 skipped, 0 failed | Sairam |
| 2026-10-29 | same | browser check, production mode (Chrome) | banner and intended use shown; WebM/Opus recorded; no disease outputs; request id shown; audit verify ok (2 records); model integrity verified | Sairam |
