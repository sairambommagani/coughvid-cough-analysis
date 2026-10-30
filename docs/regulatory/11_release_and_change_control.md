# 11 Release, model change control and post-release monitoring

IEC 62304 5.8 (release), 6 (maintenance), 8 (configuration management); CDSCO software guidance (algorithm change
protocol, may be kept with the risk file).

## 1. Model versions (semantic versioning)

| Part | Increase when | Example |
|---|---|---|
| MAJOR | new architecture, new input features, or a change of the output / intended use | 1.4.2 -> 2.0.0 |
| MINOR | retrained: new data or hyperparameters, same architecture and outputs | 1.4.2 -> 1.5.0 |
| PATCH | change that does not alter what the model learned: threshold, calibration, bug fix | 1.4.2 -> 1.4.3 |

Name and folder: `coughvid_v<MAJOR.MINOR.PATCH>_fusion_<YYYYMMDD>`. Each version folder holds the model files and
the model card `version.json`:

| Model card field | Content |
|---|---|
| name, version, bump, parent | identity and lineage |
| created, git_commit, git_uncommitted_changes | when and from which code (a true flag means the release is not reproducible: not allowed for production) |
| data | SHA-256 of `meta.parquet` and `features.parquet`, combined `training_data_sha256`, number of recordings, split sizes |
| metrics_file, test_roc_auc | `metrics.json` (complete evaluation) and the ROC-AUC summary per task and model |
| file_sha256 | SHA-256 of every model file |
| soup | versions of numpy, pandas, scipy, scikit-learn, lightgbm, torch, librosa, soundfile, fastapi, ffmpeg |
| training_args, notes | how it was trained and what changed |

`artifacts/registry.json` lists every version; `artifacts/LATEST` names the served one.

Version 1.0.0 was registered on 2026-10-28, before file hashes existed. Its hashes were added once with
`python -m cough.versioning --add-hashes 1.0.0`; the card records this (`hashes_added`) so it is visible that they
were taken after registration. The command refuses any card that already has hashes.

No silent overwrite: `cough.train` always creates a new version folder; `train.run` refuses a folder that already
holds a model; registration makes the files read-only and cannot be repeated; the API checks every file hash at
start-up and in production refuses a version that fails.

## 2. Model change control (algorithm change protocol)

A retraining or any change to a model is released only if all of these hold, recorded in the problem report or
change request:
1. Reason for the change and the expected effect.
2. New version created by `python -m cough.train --bump <part> --notes "..."` from a clean git commit
   (`git_uncommitted_changes` false).
3. Same held-out test split as the parent (the split is part of the hashed metadata) or a documented new split.
4. Acceptance: cough check ROC-AUC >= 0.97; wet / dry ROC-AUC not below the lower 95% bound of the parent;
   calibration ECE not worse than the parent by more than 0.02; subgroup results reviewed (05 A-03).
5. Risk file reviewed for new hazards (05).
6. Full test suite passes (07).
7. Approval by the reviewer; then `python -m cough.versioning --use <version>` (or LATEST set by registration)
   and restart of the service.
Rollback: `python -m cough.versioning --use <previous version>` and restart, or start with
`COUGH_MODEL_VERSION=<version>`.

A MAJOR change, a change of the intended use, or a change of the outputs needs a regulatory assessment before
release (for CDSCO: whether it is a change requiring approval under MDR-2017).

## 3. Release checklist (software and model)

- [ ] All tests pass, 0 skipped (07); test report filled in.
- [ ] `python scripts/soup_versions.py` run; 06a and `requirements.lock.txt` committed; SOUP anomalies checked (06).
- [ ] Served model version verified (`/health` shows integrity "verified"); model card complete.
- [ ] `COUGH_MODE` is production (or unset) in the deployment configuration.
- [ ] Audit folder on WORM storage; `python -m cough.audit --verify` passes; purge job scheduled.
- [ ] Documents 01-11 updated; residual risk accepted (05).
- [ ] Known residual anomalies listed in the release notes.
- [ ] Git tag `release-<api version>-model-<model version>`; release notes.

## 4. Post-release monitoring (drift, RC-06)

Weekly from the audit trail, per model version and organisation:

| Signal | Why | Alert when (initial; set from the first month, 05 A-06) |
|---|---|---|
| requests, rejected requests by status (400/401/413/415/422) | input problems, new devices or formats | rejection rate doubles week on week |
| cough-check positive rate | population or device change | moves more than 10 points from the first month |
| cough type: wet / dry / indeterminate / not-assessed shares | model or population drift | any share moves more than 10 points |
| SNR distribution (median, share < 10 dB) | recording conditions (H-02) | median moves more than 3 dB |
| clinician-reported wrong outputs (problem reports with request ids) | real-world errors | any safety-relevant report |
| audit verification | integrity | any failure |

An alert opens a problem report (02 section 8); the reviewer decides on investigation, retraining (section 2) or
rollback.
