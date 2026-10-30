# Regulatory documentation - COUGHVID cough analysis software

| Field | Value |
|---|---|
| Software | COUGHVID cough analysis (API + web interface) |
| Software version | 1.1.0 (API), model `coughvid_v1.0.0_fusion_20261021` |
| Intended IEC 62304 software safety class | B (rationale in 02, section 2) |
| Target market | India (decided by the lead, 2026-10-29) |
| Regulator / device class | CDSCO, Medical Devices Rules 2017; CDSCO class A (low risk) for the intended use, per the CDSCO Guidance Document on Medical Device Software (CDSCO/MD/GD/MDSW/01/2026) |
| Regulatory status | Not a medical device. Investigational software built to SaMD practice; not licensed by CDSCO |
| Standards followed (intent, not certified) | IS/ISO/IEC 62304:2006+A1:2015 (software life cycle), ISO 14971:2019 (risk management), IEC 82304-1 (health software product), IEC 81001-5-1 (health software security), IMDRF SaMD N12 (categorisation); ISO 13485 QMS not yet in place |
| Privacy framework | Digital Personal Data Protection Act 2023 + DPDP Rules 2025 (India); CERT-In Directions 2022 (incident reporting, logs). HIPAA BAA kept as a template in case of a US deployment |
| Owner | Sairam (developer); reviewer: lead engineer |
| Status | Draft for review, 2026-10-29 |

## Documents

| No. | Document | IEC 62304 / ISO 14971 reference |
|---|---|---|
| 01 | Intended use statement and labelling | IEC 82304-1 7.2, IMDRF N12 |
| 02 | Software development plan, safety classification, maintenance, configuration and problem resolution | 62304: 4.3, 5.1, 6, 8, 9 |
| 03 | Software requirements specification (SRS) | 62304: 5.2 |
| 04 | Software architecture and design | 62304: 5.3, 5.4 |
| 05 | Risk management file | ISO 14971: 4-10; 62304: 7 |
| 06 | SOUP list (and 06a generated versions) | 62304: 5.3.3, 5.3.4, 7.1.2, 7.1.3, 8.1.2 |
| 07 | Verification test protocol and report | 62304: 5.5, 5.6, 5.7 |
| 08 | Traceability matrix (requirement - design - test - risk control) | 62304: 5.1.1, 5.7.4, 7.3.3 |
| 09 | Audit trail and PHI / data protection design | 62304: 5.2.2 (security); DPDP Rules 2025 Rule 6 (safeguards, logs); IEC 81001-5-1 |
| 10 | Legal and privacy (India first): processor agreement / BAA, DPIA, consent, right to erasure, breach reporting, labelling | DPDP Act 2023 s. 6, 8, 9, 10, 12; DPDP Rules 2025; HIPAA 164.504(e) |
| 11 | Release, model change control and post-market monitoring | 62304: 5.8, 6, 8 |

## Where each item of the lead's request is answered

| Request | Document | Code |
|---|---|---|
| 1. Intended use statement | 01 | `cough/samd.py` (INTENDED_USE, output gating) |
| 2. Risk file (mis-triage, SNR shortcut, age 50+ AUC 0.53, time drift) | 05 | risk controls RC-01..RC-12 |
| 3. Requirements, design, test protocol, SOUP list | 03, 04, 07, 06 | `tests/` |
| 4. Versioned models: semantic version + training data hash + metrics; no silent overwrite | 11, 04 | `cough/versioning.py`, `cough/train.py` |
| 5. Audit: request id, user/org id, model version, input hash, output, timestamp; append-only | 09 | `cough/audit.py`, `cough/api.py` |
| 6. PHI: audio only with documented retention; hash + features by default; no lat/lon in inference | 09 | `cough/phi.py`, `cough/samd.py` |
| 7. Legal: BAA, DPIA, consent, right to deletion; "not a medical device" in UI | 10, 01 | `DELETE /subjects/{id}`, UI banner |

## Document control

All documents live in git next to the code; a change to a document is a commit and is reviewed like code. The
version of this documentation set is the git commit of the release (recorded in every model card).
