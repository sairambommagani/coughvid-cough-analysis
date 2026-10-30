"""Intended use and the output rules that enforce it (SaMD controls, see docs/regulatory/).

Intended use (production mode) - document 01_intended_use.md is the controlled source of this text:
  The software analyses a short cough recording made on a phone or computer microphone and
  (1) detects whether the recording contains coughs, and (2) optionally estimates whether the cough sounds wet
  or dry, as supporting information for review by a clinician. It does not diagnose, screen for or exclude
  COVID-19 or any other disease, and its output must not be used as the sole basis for a clinical decision.

Two modes (environment variable COUGH_MODE):
  production (default)  only the intended-use outputs: cough check, cough type for clinician review.
                        COVID-19 and respiratory-symptom scores are NOT computed into the response, the test-set
                        browser is off, and every request needs a user id and an organisation id (audit).
  research              all model outputs, labelled "research use only", for internal evaluation.

Risk controls implemented here (risk file 05_risk_management_file.md):
  RC-01 quality gate: cough type is only given for a clear recording (cough detected, >= MIN_COUGHS coughs,
        SNR >= MIN_SNR_DB); otherwise the user is asked to record again (the COVID-19 model was weakest on
        0-1 cough recordings, and SNR acted as a shortcut feature).
  RC-02 indeterminate band: a wet probability inside INDETERMINATE is reported as "indeterminate", not wet/dry.
  RC-03 no disease outputs in production (mis-triage, age 50+ failure, time drift of the COVID-19 model).
  RC-04 labelling: every response and the UI carry the regulatory status and the intended-use statement.
"""
from __future__ import annotations

import os

SOFTWARE_NAME = "COUGHVID cough analysis"
INTENDED_USE = ("Detects whether a recording contains coughs; optionally estimates wet vs dry cough type for "
                "clinician review. Does not diagnose COVID-19 or any other disease.")
INTENDED_USE_FULL = (
    "The software analyses a short cough recording made with a phone or computer microphone and (1) detects "
    "whether the recording contains coughs, and (2) optionally estimates whether the cough sounds wet or dry, as "
    "supporting information for review by a qualified clinician. It does not diagnose, screen for or exclude "
    "COVID-19 or any other disease, and its output must not be used as the sole basis for a clinical decision.")
INTENDED_USERS = "Clinicians and clinical staff; patients may record under clinician instruction."
CONTRAINDICATIONS = [
    "Not for diagnosis, screening or exclusion of COVID-19 or any disease.",
    "Not for emergency triage or for patients in respiratory distress.",
    "Not validated for children (under 18): performance was not evaluated for this group.",
    "Not validated for recordings longer than 10 s, recordings with speech or music, or more than one person.",
]
REGULATORY_STATUS = ("Not a medical device. Investigational software developed with IEC 62304 Class B intent; not "
                     "licensed by CDSCO (India) or cleared or approved by any other regulator.")
SAFETY_CLASS = "B"

MIN_COUGHS = 2
MIN_SNR_DB = 10.0
INDETERMINATE = (0.40, 0.60)
PRODUCTION_TASKS = ("cough", "wet")            # tasks whose output may leave the system in production
RESEARCH_TASKS = ("cough", "wet", "covid", "symptomatic")

# never accepted as model input in production (identifiable or confounding metadata)
FORBIDDEN_INPUTS = ("latitude", "longitude", "lat", "lon", "location", "gps", "ip", "age", "gender", "name",
                    "email", "phone", "datetime", "uuid", "status", "fever_muscle_pain", "respiratory_condition")


def mode() -> str:
    m = os.environ.get("COUGH_MODE", "production").strip().lower()
    if m not in ("production", "research"):
        raise ValueError("COUGH_MODE must be 'production' or 'research'")
    return m


def is_production() -> bool:
    return mode() == "production"


def allowed_tasks() -> tuple[str, ...]:
    return PRODUCTION_TASKS if is_production() else RESEARCH_TASKS


def labelling() -> dict:
    return {"software": SOFTWARE_NAME, "mode": mode(), "intended_use": INTENDED_USE,
            "regulatory_status": REGULATORY_STATUS if is_production() else
            "RESEARCH USE ONLY - outputs outside the intended use. " + REGULATORY_STATUS}


def check_model_inputs(feature_names) -> None:
    """Fail closed if a model would use forbidden metadata (e.g. latitude/longitude) as input."""
    bad = [f for f in feature_names if f.lower() in FORBIDDEN_INPUTS or f.lower().startswith(("lat", "lon"))]
    if bad:
        raise RuntimeError(f"model uses forbidden inputs {bad}; production inference uses audio features only")


def quality_gate(coughs_found: int, snr_db: float, is_cough: bool) -> tuple[bool, str | None]:
    """RC-01: is the recording good enough for a cough-type estimate?"""
    if not is_cough:
        return False, "no clear cough found"
    if coughs_found < MIN_COUGHS:
        return False, f"only {coughs_found} cough found; at least {MIN_COUGHS} are needed"
    if snr_db < MIN_SNR_DB:
        return False, f"recording too noisy (SNR {snr_db:.1f} dB, at least {MIN_SNR_DB:.0f} dB needed)"
    return True, None


def cough_type_label(p_wet: float) -> str:
    """RC-02: wet / dry / indeterminate."""
    lo, hi = INDETERMINATE
    if lo <= p_wet <= hi:
        return "indeterminate"
    return "wet" if p_wet > hi else "dry"
