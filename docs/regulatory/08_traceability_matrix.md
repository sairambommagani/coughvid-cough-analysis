# 08 Traceability matrix

Hazard -> risk control -> requirement -> design (code) -> verification. IEC 62304 5.1.1, 5.7.4, 7.3.3.

| Hazard (05) | Risk control | Requirement (03) | Design / code (04) | Verification (07) |
|---|---|---|---|---|
| H-01 mis-triage | RC-03 no disease outputs in production | REQ-INT-01, 02, 03, 04 | `samd.allowed_tasks`, `Engine.scores`, `_research_only`, `model_info` | TC-INT-01..04 |
| H-01, H-11 | RC-04 labelling | REQ-LAB-01, 02 | `samd.labelling`, `/intended-use`, UI header and result footer | TC-INT-05, TC-DOC-01 |
| H-02 SNR shortcut | RC-01 quality gate | REQ-INT-05 | `samd.quality_gate`, `Engine.analyse_file` | TC-RSK-01 |
| H-02, H-05 | RC-02 indeterminate band | REQ-INT-06 | `samd.cough_type_label` | TC-RSK-02 |
| H-03 demographic failure | RC-05 no demographic / location input | REQ-PHI-02, 04 | `api.analyze` extra-field refusal, `samd.check_model_inputs` | TC-PHI-02, TC-PHI-04 |
| H-03 | RC-03 | REQ-INT-01 | as above | TC-INT-01 |
| H-04 time drift | RC-06 monitoring | REQ-ENV-06 | 11 section 4 | release review |
| H-04 | RC-07 change control, rollback | REQ-VER-05 | `versioning.resolve_artifacts`, `--use` | TC-VER-05 |
| H-04, H-07 | RC-12 version traceability and integrity | REQ-VER-01..04, 06 | `versioning.register`, `verify_integrity`, `lock_files`, `Engine.__init__`, `train.run` guard | TC-VER-01..04, 06, 07, TC-AUD-01 |
| H-06 non-cough accepted | RC-01 | REQ-INT-05 | as above | TC-RSK-01 |
| H-08 malformed input | RC-08 format check | REQ-INP-01..04, 06 | `formats.sniff`, `api.check_upload`, `audio.load_audio`, recorder `pickFormat` | TC-FMT-01..08 |
| H-08 | RC-09 decoder isolation and limit | REQ-INP-05 | `audio.load_audio` timeout | TC-INP-05 |
| H-09 audit loss | RC-10 audit trail | REQ-AUD-01..08 | `audit.AuditLog`, `verify`, `seal`, `api._audited` | TC-AUD-01..07 |
| H-10 data breach | RC-11 PHI minimisation | REQ-PHI-01, 03, 05, 06 | `phi.store_features`, `store_audio`, `erase_subject`, `purge_expired`; temp file deleted | TC-PHI-01, 03, 05, 06 |
| H-10 | deployment controls | REQ-ENV-01..05 | 09, 10 | release review |

Coverage: every requirement in 03 appears above or in 07 section 2 (functional requirements REQ-FUN-* are
verified by TC-REG-01 and TC-FMT-07); every risk control has at least one verification; no test exists without a
requirement.
