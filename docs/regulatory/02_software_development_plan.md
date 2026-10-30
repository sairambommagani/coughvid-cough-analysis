# 02 Software development plan

IEC 62304 clauses 4.3 (safety classification), 5.1 (development planning), 6 (maintenance), 8 (configuration
management), 9 (problem resolution).

## 1. Scope

The software system is the COUGHVID cough analysis service: the FastAPI service (`cough/api.py`), the web
interface (`cough/static/index.html`), the inference pipeline (decoding, segmentation, features, models), the
training pipeline that produces model versions, and the controls described in documents 01-11. Out of scope: the
hosting platform, the authenticating gateway that sets the user and organisation ids, and the client devices.
Their requirements on this software are stated in 03 (REQ-ENV-*).

## 2. Software safety classification (IEC 62304 4.3)

| Class | Definition (IEC 62304) |
|---|---|
| A | No injury or damage to health is possible |
| B | Non-serious injury is possible |
| C | Death or serious injury is possible |

**Decision: Class B** (confirmed by the lead engineer on 2026-10-29).

Reasoning. The hazardous situations a software failure can contribute to (risk file 05):

1. A wrong cough type (e.g. "dry" for a wet, productive cough) is read by the clinician. The clinician may give
   less weight to a chest infection, and treatment may be delayed: at worst a **non-serious injury** (a delay
   that the clinician's own examination normally catches).
2. A wrong cough check ("no clear cough") makes the user record again: inconvenience, no injury.
3. Disease outputs (COVID-19, respiratory symptoms) could cause a **serious injury** through false reassurance
   (missed infection, onward transmission). They are therefore removed from the production software (RC-03), not
   just labelled. Without that removal the classification would have to be C.

Serious injury is not credible for the production outputs because (a) the output only informs a clinician, who
examines the patient (external risk control, IEC 62304 4.3 a, as amended in 2015), (b) the condition is
non-serious (routine consultation), and (c) the software does not drive or trigger treatment. Death or serious
injury is therefore not a credible result of a software failure: Class B.

Not to be confused with the CDSCO device class (MDR-2017): CDSCO class A (low risk), from the CDSCO software
guidance Table 2 (non-serious condition x inform clinical management). IEC 62304 classifies the harm from a
software failure, CDSCO classifies the device; the two scales are independent.

Software items and their class. All items are Class B. The items that implement risk controls (output gating
`samd.py`, quality gate, labelling, format check `formats.py`, audit `audit.py`, integrity check `versioning.py`)
are developed with the Class C level of detailed design (04 section 5) as extra margin.

## 3. Life cycle model and deliverables

Incremental development; each increment goes through the activities below before release. Required for Class B by
IEC 62304 and where each is answered:

| IEC 62304 activity | Class B required | Deliverable |
|---|---|---|
| 5.1 Development planning | yes | this document |
| 5.2 Requirements analysis | yes | 03 SRS |
| 5.3 Architectural design (incl. SOUP 5.3.3, 5.3.4, segregation 5.3.5) | yes | 04, 06 |
| 5.4 Detailed design | for Class C; done for risk-control units | 04 section 5 |
| 5.5 Unit implementation and verification | yes | code review + unit tests (07) |
| 5.6 Integration and integration testing | yes | `tests/test_cough.py`, `tests/test_formats.py` (07) |
| 5.7 System testing | yes | 07 system tests, test report |
| 5.8 Release | yes | 11 release checklist |
| 6 Maintenance | yes | section 6 below, 11 |
| 7 Risk management (incl. 7.1.2 SOUP hazards) | yes | 05, 06 |
| 8 Configuration management | yes | section 7 below |
| 9 Problem resolution | yes | section 8 below |

## 4. Standards, methods and tools

- Language Python 3.11; style: readable functions with docstrings that state intent and risk controls.
- Version control: git, repository `coughvid-cough-analysis` (private). Every change is a commit; releases are tags.
- Tests: pytest; all tests must pass before a release (07). Tests run on synthetic data in the real file formats,
  so they need no patient data.
- Training data: COUGHVID v3 (Zenodo 7024894, CC BY 4.0), used under its licence; no other patient data is used
  for development.
- Tools are SOUP or development tools: listed in 06.

## 5. Verification approach (unit, integration, system)

Unit acceptance criteria: code reviewed by a second engineer; unit tests for every risk-control function; no
reduction of the measured model performance without a recorded decision. Integration: the full pipeline from
synthetic raw data to a served prediction (`test_pipeline_and_api`). System: the production API against the
requirements (07, TC-*). Regression: the whole suite on every change.

## 6. Maintenance (IEC 62304 clause 6)

- Feedback and problem reports from users and monitoring are handled as in section 8.
- SOUP is monitored for new versions and published vulnerabilities (06 section 3) before each release and at
  least monthly; ffmpeg advisories are checked at every release because ffmpeg parses untrusted input.
- A model retraining is a change to the software: it follows 11 (model change control) and gets a new semantic
  version. The served version is recorded in every audit record, so every output can be traced to its model.

## 7. Configuration management (IEC 62304 clause 8)

Configuration items: source code, documents (this folder), tests, `requirements.txt` and `requirements.lock.txt`
(exact SOUP versions, generated by `scripts/soup_versions.py`), model versions (`artifacts/<version>/` with model
card `version.json`, `artifacts/registry.json`).

- Identification: code by git commit and release tag; models by semantic version
  `coughvid_v<MAJOR.MINOR.PATCH>_fusion_<YYYYMMDD>` (11); SOUP by name and exact version (06a).
- Change control: every change through a reviewed commit; model changes through 11.
- Status accounting: `python -m cough.versioning --list` shows every model version with its parent, git commit,
  notes and test ROC-AUC; git log shows code changes.
- Model files are never overwritten: training refuses an existing model folder, registered versions are read-only,
  and the API refuses to serve a version whose file hashes differ from its model card.

## 8. Problem resolution (IEC 62304 clause 9)

Every problem (bug, wrong output reported by a clinician, monitoring alert, SOUP vulnerability) gets a problem
report (GitHub issue) with: date, reporter, request id(s) from the audit trail if an output is involved, software
and model version, description, severity (safety relevant yes / no), investigation, root cause, change made,
verification, and closure. A safety-relevant problem triggers a review of the risk file (05) and, if needed, a
rollback to the previous model version (`python -m cough.versioning --use <version>`). Trends are reviewed monthly.

## 9. Roles

| Role | Person |
|---|---|
| Developer, document author | Sairam |
| Reviewer and approver (code, risk file, release) | Lead engineer (Chandan) |
| Clinical input for risk evaluation and labelling | to be assigned (a clinician) before any clinical use |
