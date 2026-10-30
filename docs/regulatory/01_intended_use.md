# 01 Intended use statement and labelling

This document is the controlled source of the intended use. The same text is in `cough/samd.py` and is returned
by `GET /intended-use` and in every `/analyze` response; test TC-DOC-01 checks that the code and this document
agree.

## 1. Intended use (short form)

Detects whether a recording contains coughs; optionally estimates wet vs dry cough type for clinician review. Does not diagnose COVID-19 or any other disease.

## 2. Intended use (full)

The software analyses a short cough recording made with a phone or computer microphone and (1) detects whether
the recording contains coughs, and (2) optionally estimates whether the cough sounds wet or dry, as supporting
information for review by a qualified clinician. It does not diagnose, screen for or exclude COVID-19 or any other
disease, and its output must not be used as the sole basis for a clinical decision.

| Element | Statement |
|---|---|
| Medical purpose | Supporting information: presence of cough sounds in a recording; acoustic cough character (wet / dry) |
| Significance of the information (IMDRF N12) | "Inform clinical management": the clinician combines it with history and examination; it does not drive or treat |
| State of the condition (IMDRF N12) | Non-serious: routine respiratory consultations; not emergency or critical care |
| Resulting IMDRF SaMD category | I (inform + non-serious) |
| Resulting CDSCO class (India, MDR-2017 Rule 4; CDSCO MDSW guidance 2026, Table 2) | A (low risk): non-serious condition x informs clinical management. Patients only record; the result is read by a clinician, so the lay-user escalation does not apply |
| IEC 62304 software safety class | B (non-serious injury possible from a software failure); a different scale from the CDSCO device class, see 02 section 2 |
| Intended users | Clinicians and clinical staff. Patients may make the recording under clinician instruction; the result is for the clinician |
| Intended patient population | Adults (18+) who can cough voluntarily |
| Use environment | Clinic or home, quiet room, phone or computer microphone about 20 cm away |
| Input | One recording, 1-10 s, 2-3 voluntary coughs, in a supported audio format (see `GET /formats`) |
| Outputs | Cough check (cough detected yes/no with probability); cough type wet / dry / indeterminate / not assessed, with the probability and the three acoustic features that moved it most |
| Principle of operation | Signal processing (cough segmentation) and machine-learning classifiers (LightGBM, logistic regression on pretrained PANNs embeddings, CNN), fused and calibrated; trained on COUGHVID v3 (crowdsourced, 2020-2021) |

## 3. Contraindications and limitations

- Not for diagnosis, screening or exclusion of COVID-19 or any disease.
- Not for emergency triage or for patients in respiratory distress.
- Not validated for children (under 18): performance was not evaluated for this group.
- Not validated for recordings longer than 10 s, recordings with speech or music, or more than one person.
- Wet / dry: test ROC-AUC 0.674 (95% CI 0.59-0.75) against the pulmonologists' majority vote; the experts
  themselves agreed on cough type only 62% of the time (Cohen's kappa 0.28). The estimate is a hint for the
  clinician, not a measurement.
- Recordings with fewer than 2 coughs or SNR below 10 dB are not assessed for cough type (risk control RC-01).
- Wet probabilities between 0.40 and 0.60 are reported as "indeterminate" (risk control RC-02).

## 4. Outputs outside the intended use

The COVID-19 and respiratory-symptom models are kept for research only. In production mode (the default) they are
not computed, not returned and not shown, and their performance is not shown in the interface (risk control
RC-03). The reasons are measured, not assumed (risk file 05): COVID-19 ROC-AUC 0.600 overall, 0.527 for age 50+,
0.559 on later recordings, sensitivity 20% at 91% specificity, and part of the signal is recording quality.

Research mode (`COUGH_MODE=research`) shows them labelled "RESEARCH USE ONLY" and is for internal evaluation by
the development team only. It must not be deployed to users.

## 5. Labelling (shown in the UI and returned by the API)

- Header of the web interface: the short intended use (section 1) and the regulatory status:
  "Not a medical device. Investigational software developed with IEC 62304 Class B intent; not licensed by CDSCO
  (India) or cleared or approved by any other regulator."
- Every result: "Not a medical device. This software does not diagnose COVID-19 or any disease. The cough type is
  supporting information for a clinician, not a diagnosis. If you are unwell, see a doctor."
- The cough type card is titled "Cough type (for clinician review)".
- No production output claims a diagnosis: the words "diagnose" / "diagnosis" appear only in negative statements
  ("does not diagnose", "not a diagnosis"), and "detects COVID" or "COVID test" appear nowhere.

CDSCO's guidance states that a disclaimer does not override what the software actually does: the label is not a
way around regulation. The software is regulated by its function, which is why the disease outputs are removed
from production (section 4) rather than only labelled. Until a CDSCO licence exists, the "Not a medical device" statement stays in the UI and in every response
(requirement REQ-LAB-01).
