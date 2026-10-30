# 05 Risk management file

ISO 14971:2019 (risk management for medical devices) and IEC 62304 clause 7 (software contribution to hazardous
situations). All numbers come from the held-out test set of COUGHVID v3 (README section 3) unless stated.

## 1. Scope and plan

- Product: COUGHVID cough analysis software, production mode, intended use in 01.
- Life-cycle phases covered: development, release, use, maintenance (retraining), decommissioning.
- Method: preliminary hazard analysis per hazard below -> estimate risk (severity x probability) -> control ->
  verify the control (test id) -> residual risk -> overall residual risk and benefit-risk.
- Review: at every release, after every safety-relevant problem report, and when monitoring shows drift (11).

## 2. Risk acceptability

Severity (S): S1 negligible (inconvenience) - S2 minor (temporary, no intervention) - S3 serious (needs medical
intervention, e.g. delayed treatment of an infection) - S4 critical (permanent impairment, life-threatening) - S5
catastrophic (death).

Probability of harm (P): P1 improbable (< 1 in 100,000 uses) - P2 remote (< 1 in 10,000) - P3 occasional (< 1 in
1,000) - P4 probable (< 1 in 100) - P5 frequent (>= 1 in 100).

| | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| S5 | R | U | U | U | U |
| S4 | R | R | U | U | U |
| S3 | A | R | R | U | U |
| S2 | A | A | R | R | U |
| S1 | A | A | A | A | R |

A = acceptable; R = reduce as far as possible, acceptable only with benefit-risk justification; U = unacceptable.

Probabilities include the clinician's review (the external control in 02 section 2): P is the probability that
the software output contributes to harm, not the model's error rate. The model error rates are stated next to
each hazard so the estimate can be checked.

## 3. Hazard analysis

### H-01 Mis-triage from a disease output (COVID-19 / respiratory symptoms)
- Hazardous situation: a user or clinician reads a low COVID-19 score as "not infected" (false reassurance), or a
  high score as "infected" (unnecessary isolation, missed other diagnosis).
- Evidence: COVID-19 ROC-AUC 0.600 (95% CI 0.54-0.66); at the 90%-specificity threshold the model caught only 20%
  of COVID-19 cases (80% missed); four self-reported answers alone do better (0.670). Respiratory symptoms 0.627.
- Initial risk: S3 x P5 = **U**.
- Controls: RC-03 (disease outputs not computed or returned in production; research mode only, labelled),
  RC-04 (labelling: "does not diagnose COVID-19"), intended use excludes diagnosis and screening (01).
- Verification: TC-INT-01, TC-INT-02, TC-INT-03, TC-INT-04, TC-INT-05.
- Residual: no disease output reaches a user in production; residual misuse = research mode deployed by mistake:
  S3 x P1 = **A** (mode defaults to production; research mode is shown on every response and in the UI).

### H-02 SNR shortcut: the model learns the recording, not the cough
- Hazardous situation: cough type (or a disease score) follows the microphone and room noise rather than the
  cough; a noisy recording of a dry cough is labelled wet (or the reverse), misleading the clinician.
- Evidence: COVID-19 recordings were noisier (median SNR 19.9 dB vs 23.1 dB); SNR is the most important
  handcrafted feature; SNR alone reaches ROC-AUC 0.563; without SNR / loudness / length features the handcrafted
  model drops from 0.609 to 0.594.
- Initial risk: S3 x P3 = R.
- Controls: RC-01 quality gate (cough type only with SNR >= 10 dB and >= 2 coughs; otherwise "not assessed, record
  again"), RC-02 indeterminate band, RC-03 (the disease models, where the shortcut was measured, are removed),
  recording instructions in the UI (quiet room, 20 cm), echo cancellation / noise suppression / auto gain turned
  off in the recorder so the device does not alter the cough. Monitoring of the SNR distribution per version (RC-06).
- Verification: TC-RSK-01, TC-RSK-02, TC-INT-01.
- Residual: S3 x P2 = **R**, accepted (benefit-risk in section 5). Open action A-02: measure cough-type performance
  per SNR band and per device before clinical use.

### H-03 Demographic failure (age 50+ COVID-19 ROC-AUC 0.53)
- Hazardous situation: the model is much weaker for a group of patients, so its output is wrong more often for
  them without anyone knowing.
- Evidence: COVID-19 fusion ROC-AUC by age: <30 0.561, 30-50 0.654, 50+ 0.527 (257 recordings, 17 COVID-19,
  near random); by gender female 0.590, male 0.620. Subgroup results for cough type were not computed (58 wet test
  coughs are too few to split).
- Initial risk: S3 x P4 = U (for the disease output).
- Controls: RC-03 (the disease output, where the failure was measured, is not in production); RC-05 (no
  demographic input to any model, so no demographic shortcut; age and gender are refused at the API); labelling
  states children are not validated; open action A-03: subgroup evaluation of the cough check and cough type by
  age group and gender on new data before clinical use, with a minimum per-group sample size decided with a
  statistician.
- Verification: TC-INT-01, TC-PHI-02, TC-PHI-04.
- Residual: for the production outputs, unknown subgroup performance of cough type: S3 x P2 = **R**, accepted only
  with A-03 completed before clinical use.

### H-04 Time drift: performance falls as the population and disease change
- Hazardous situation: a model trained on 2020-2021 recordings is used years later; outputs degrade silently.
- Evidence: COVID-19 model trained before 2020-11-27 and tested on later recordings: ROC-AUC 0.559 (vs 0.609 with
  a random split).
- Initial risk: S3 x P4 = U (disease output); S3 x P3 = R (cough type).
- Controls: RC-03 (disease output removed); RC-06 monitoring per model version (cough-check rate, cough-type
  distribution, SNR distribution, rejection rates) with alert thresholds (11 section 4); RC-07 model change control:
  retraining only as a new semantic version with a model card and test report, rollback available; every output
  traceable to its model version (RC-12).
- Verification: TC-VER-01, TC-VER-05, TC-AUD-01.
- Residual: S3 x P2 = **R**, accepted with monitoring in place (REQ-ENV-06).

### H-05 Wrong cough type (model error, expert disagreement)
- Evidence: wet vs dry ROC-AUC 0.674 (0.59-0.75); the pulmonologists agreed with each other only 62% of the time
  (kappa 0.28).
- Hazardous situation: a clinician over-trusts "dry" for a productive cough.
- Initial risk: S3 x P3 = R.
- Controls: RC-02 indeterminate band; output titled "for clinician review", with the probability and the reasons;
  limitations in 01 section 3; RC-01 quality gate.
- Verification: TC-RSK-02, TC-INT-05.
- Residual: S3 x P2 = **R**, accepted (section 5).

### H-06 Non-cough accepted as a cough (speech, noise)
- Evidence: cough check ROC-AUC 0.988, sensitivity 98% at 90% specificity.
- Hazardous situation: cough type computed on a non-cough, misleading the clinician.
- Initial risk: S2 x P3 = R. Controls: cough check gates cough type; quality gate requires >= 2 cough segments
  (RC-01); AudioSet speech probability shown. Verification: TC-RSK-01. Residual: S2 x P2 = **A**.

### H-07 Wrong or unknown model version served
- Hazardous situation: a modified, half-written or overwritten model file is served; outputs unreproducible.
- Initial risk: S3 x P3 = R.
- Controls: RC-12 (file hashes in the model card, verified at start-up, production refuses unverified models;
  read-only files; training refuses to overwrite; version in every response and audit record).
- Verification: TC-VER-01..TC-VER-04, TC-AUD-01. Residual: S3 x P1 = **A**.

### H-08 Malformed, wrong-format or hostile input file
- Hazardous situation: wrong decoding gives a garbage signal and a wrong output; a hostile file exploits the
  decoder or hangs the service (denial of service for other users).
- Initial risk: S2 x P3 = R.
- Controls: RC-08 content-based format check, size limit, 415/400/413/422 with clear messages, decoding to one
  model format; RC-09 ffmpeg in a separate process with a 30 s limit; SOUP vulnerability monitoring (06).
- Verification: TC-FMT-01..TC-FMT-08, TC-INP-05. Residual: S2 x P1 = **A**.

### H-09 Loss, change or absence of the audit trail
- Hazardous situation: an output cannot be traced (which model, which input, who) after an incident; evidence is
  altered.
- Initial risk: S2 x P3 = R.
- Controls: RC-10 (record for every request, hash chain, append-only, fsync, no output without a record, daily
  verification, WORM storage in deployment).
- Verification: TC-AUD-01..TC-AUD-06. Residual: S2 x P1 = **A**.

### H-10 Personal data breach or re-identification
- Hazardous situation: audio (a voice biometric), location or identity leaks; harm to privacy.
- Initial risk: S3 x P3 = R.
- Controls: RC-11 (no audio by default, hash + features only, no file names, location and demographics refused,
  model inputs checked, retention and purge, erasure, ids must be opaque); encryption at rest and TLS
  (REQ-ENV-02/04); legal controls in 10 (processor agreement, DPIA, breach reporting within 72 hours).
- Verification: TC-PHI-01..TC-PHI-06. Residual: S3 x P1 = **R**, accepted (section 5).

### H-11 Use outside the intended population or setting (children, emergencies, lay self-diagnosis)
- Initial risk: S3 x P3 = R.
- Controls: RC-04 labelling and contraindications (01 section 3), result addressed to the clinician, disease
  outputs removed (RC-03). Residual: S3 x P2 = **R**, accepted.

## 4. Risk controls (summary)

| Id | Control | Implemented in | Verified by |
|---|---|---|---|
| RC-01 | Quality gate for cough type (>= 2 coughs, SNR >= 10 dB, cough detected) | `samd.quality_gate`, `api.Engine.analyse_file` | TC-RSK-01 |
| RC-02 | Indeterminate band 0.40-0.60 | `samd.cough_type_label` | TC-RSK-02 |
| RC-03 | No disease outputs in production; research mode only, labelled | `samd.allowed_tasks`, `Engine.scores`, `/examples` disabled, `/model-info` filtered | TC-INT-01..04 |
| RC-04 | Labelling: intended use, regulatory status, contraindications | `samd.labelling`, `/intended-use`, UI header | TC-INT-05, TC-DOC-01 |
| RC-05 | No demographic or location input to any model | `samd.check_model_inputs`, API refusal of extra fields | TC-PHI-02, TC-PHI-04 |
| RC-06 | Post-release monitoring per model version | 11 section 4 (deployment) | review at release |
| RC-07 | Model change control, rollback | 11, `versioning --use`, `COUGH_MODEL_VERSION` | TC-VER-05 |
| RC-08 | Content-based format check, size limit, single model input format | `formats.py`, `api.check_upload`, `audio.load_audio` | TC-FMT-* |
| RC-09 | Decoder isolation and 30 s limit | `audio.load_audio` | TC-INP-05 (code review) |
| RC-10 | Audit trail | `audit.py`, `api._audited` | TC-AUD-* |
| RC-11 | PHI minimisation, retention, erasure | `phi.py`, `api.analyze`, `api.erase` | TC-PHI-* |
| RC-12 | Model integrity and version traceability | `versioning.py`, `api.Engine`, `train.run` | TC-VER-* |

New hazards introduced by the controls (ISO 14971 7.5): the quality gate can make a user record several times
(S1, inconvenience: acceptable); refusing an unverified model makes the service unavailable (S1: a clinician
proceeds without the hint; preferred to serving an unknown model); the audit requirement makes the service fail
if storage fails (S1, same reasoning).

## 5. Overall residual risk and benefit-risk

All residual risks are A or R; none is U. The remaining R risks (H-02, H-03, H-04, H-05, H-10, H-11) are all of
the same kind: an imperfect acoustic hint read by a clinician. Benefit: a quick, objective record of cough presence
and character that the clinician can review, documented and traceable; the clinician remains the decision maker.
Given that the production outputs only inform and every one is reviewed, the benefit is judged to outweigh the
residual risk **for investigational use**, provided the open actions below are completed before any clinical use.

## 6. Open actions (must be closed before clinical use)

| Id | Action | Owner |
|---|---|---|
| A-01 | Clinical review of the hazard list and severities by a clinician | lead / clinician |
| A-02 | Cough-type performance per SNR band and per device type on new recordings | developer |
| A-03 | Subgroup evaluation (age groups incl. 50+, gender) of the cough check and cough type with an agreed minimum sample size | developer + statistician |
| A-04 | Prospective validation on Indian patients and devices (the training data is a global web-app population, 2020-2021) | lead |
| A-05 | Usability evaluation of the labelling and the "not assessed / indeterminate" messages (IEC 62366-1) | lead |
| A-06 | Set monitoring alert thresholds from the first month of real use | developer |

## 7. Approval

| Role | Name | Date | Signature |
|---|---|---|---|
| Author | Sairam | 2026-10-29 | |
| Reviewer | Chandan | | |
