# 10 Legal and privacy: India first (DPDP Act 2023, DPDP Rules 2025, CDSCO)

Status: engineering draft of the legal controls, for review by a lawyer before any deployment. It states what the
software and the organisation must do; it is not legal advice.

## 1. Roles

| Party | Role under the DPDP Act 2023 |
|---|---|
| Hospital / clinic using the software | Data Fiduciary: decides why and how the patient's data is processed |
| Our company (operating the service) | Data Processor: processes on the fiduciary's behalf, under a contract (s. 8(2)) |
| Patient | Data Principal |
| Cloud provider (MeitY-empanelled) | Sub-processor |

If our company offers the service directly to patients, it becomes the Data Fiduciary and takes on all fiduciary
duties below itself.

## 2. Applicable law and status (checked 2026-10-29)

| Instrument | Status | What it requires of us |
|---|---|---|
| Digital Personal Data Protection Act 2023 | in force through the DPDP Rules 2025 | lawful processing on consent (s. 6), notice (s. 5), security safeguards and breach reporting (s. 8), children's data (s. 9), rights of the person incl. erasure (s. 12), grievance redress (s. 13) |
| DPDP Rules 2025 (G.S.R. 846(E), notified 13 Nov 2025) | Board provisions in force; consent managers about 13 Nov 2026; main obligations from about 13 May 2027 | notice content, security measures (Rule 6), breach reporting with a full report within 72 hours, logs kept at least one year, verifiable parental consent for children |
| IT Act 2000 and the SPDI Rules 2011 | apply until the DPDP obligations replace them | medical records and biometric data are "sensitive personal data": written consent, a privacy policy, reasonable security practices |
| CERT-In Directions (28 April 2022) | in force | report cyber incidents to CERT-In within 6 hours; keep ICT system logs for 180 days in India |
| Medical Devices Rules 2017 (CDSCO) + CDSCO Guidance Document on Medical Device Software (2026) | in force; guidance listed 21 July 2026 | software with a medical purpose is regulated by its function (disclaimers do not override it); intended use here falls in CDSCO class A; an SBOM with vulnerability monitoring is expected; SaaS should be hosted on a MeitY-empanelled cloud |

We build to the DPDP obligations now, before May 2027, because they are stricter and the work is the same.

## 3. Processor agreement (India) / BAA (if deployed in the US)

India: the Data Fiduciary may engage us only under a valid contract (DPDP Act s. 8(2)). The agreement must cover:
1. Processing only on the fiduciary's documented instructions, only for the intended use (01).
2. The categories of data (section 2 of 09) and the retention periods (09 section 2.2).
3. Security safeguards of Rule 6: encryption, access control, activity logs, backups (09 section 2.4).
4. Breach: inform the fiduciary without delay, with the facts needed for its report to the Data Protection Board
   and to each affected person (full report within 72 hours) and for CERT-In (6 hours).
5. Sub-processors (cloud) only with the fiduciary's approval and on the same terms.
6. Help with the person's rights: access, correction, erasure (`DELETE /subjects/{id}`), grievance.
7. Return or deletion of all data at the end of the contract, except the audit records the law or the contract
   requires to keep.
8. Audits by the fiduciary; hosting in India.

US deployment only: a Business Associate Agreement under HIPAA 45 CFR 164.504(e) with the same substance:
permitted uses and disclosures, no other use, safeguards per the Security Rule, reporting of unauthorised uses and
breaches, subcontractors bound by the same terms, access / amendment / accounting of disclosures, records available
to HHS, return or destruction at termination, and termination for material breach.

## 4. Data protection impact assessment (DPIA)

The DPDP Act requires a periodic DPIA only from a Significant Data Fiduciary (s. 10). We carry one out anyway,
because the data is health data and voice audio, and a hospital fiduciary may itself be significant.

| Item | Assessment |
|---|---|
| Purpose | cough detection and cough type for clinician review (01); no other purpose (no training on users' data without separate consent) |
| Necessity and proportionality | only the audio is needed; audio is not stored by default; only numbers (features) and hashes are kept; location and demographics refused |
| Data principals | adult patients of the fiduciary; children excluded by the intended use (verifiable parental consent would otherwise be required) |
| Risks to people | re-identification from voice (audio); disclosure of a health condition; misuse of outputs as a diagnosis; loss of data |
| Measures | 09 section 2 (minimisation, retention, erasure, encryption, access control, WORM audit), labelling (01), disease outputs removed (05 RC-03) |
| Residual risk | low, with the measures in place; reviewed yearly and at every change of purpose, data or model |
| Sign-off | data protection officer / lead (to be signed) |

## 5. Consent and notice

Before the first recording, the fiduciary's app or the web interface shows a notice in clear language (and in the
patient's language, English or one of the languages of the Eighth Schedule when chosen) stating: what is processed
(the cough recording), why (cough detection and cough type for the clinician), that the audio is not kept (or for
how many days), what is kept (numbers and a hash), that the result is not a diagnosis, how to withdraw consent, how
to request erasure, and how to complain to the fiduciary and then to the Data Protection Board.

Consent is: free, specific, informed, unconditional and unambiguous, given by a clear action (a checkbox that is
not pre-ticked plus "Start recording"), recorded with time and notice version, and as easy to withdraw as to give.
Withdrawal stops further processing and triggers erasure of the stored features. Implementation: the fiduciary's
consent record (or a registered consent manager after Nov 2026) is checked by the gateway before it forwards a
request; the service refuses requests without ids (401).

## 6. Right to erasure and other rights

| Right (DPDP Act) | How it is met |
|---|---|
| Access to information (s. 11) | the request ids, outputs and stored features of a person can be exported from `features.jsonl` and the audit trail |
| Correction and erasure (s. 12) | `DELETE /subjects/{user_id}`; audit records kept as evidence (09 section 2.3) |
| Grievance redress (s. 13) | the fiduciary's grievance officer; we answer processor requests within 7 days |
| Nominate (s. 14) | handled by the fiduciary's account system |

## 7. Breach response

1. Detect (daily audit verification, access-log alerts, reports).
2. Contain, and inform the fiduciary immediately.
3. CERT-In within 6 hours of noticing a reportable cyber incident.
4. The fiduciary informs each affected person promptly and the Data Protection Board without delay, with the full
   report within 72 hours; we provide the facts.
5. Root cause, fix, problem report (02 section 8), risk file review (05).

## 8. Labelling: not a medical device in the UI until licensing

Every screen and response carries: "Not a medical device. Investigational software developed with IEC 62304 Class B
intent; not licensed by CDSCO (India) or cleared or approved by any other regulator." Because CDSCO judges software
by its function, the label is not used to avoid regulation: the disease outputs are removed, and before any use in
patient care the software will need a CDSCO licence for its class (class A: manufacturing licence through the State
Licensing Authority, forms MD-3 / MD-5; a test licence MD-12 / MD-13 for clinical evaluation).

## 9. Open legal actions

| Id | Action |
|---|---|
| L-01 | Lawyer review of sections 3-8 |
| L-02 | Final processor agreement template with the first hospital |
| L-03 | Privacy notice and consent text, translated |
| L-04 | Confirm the licence of the PANNs checkpoint and the ffmpeg build for commercial distribution (06) |
| L-05 | COUGHVID dataset (CC BY 4.0): attribution in the product documentation |
