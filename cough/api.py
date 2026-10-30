"""FastAPI service: upload a cough recording -> cough check and (for clinician review) cough type.

  python -m uvicorn cough.api:app --port 8008   ->   http://localhost:8008/docs

SaMD controls (docs/regulatory/): intended use and output gating (cough/samd.py), input format check
(cough/formats.py), audit trail for every request (cough/audit.py), PHI minimisation, retention and erasure
(cough/phi.py), versioned models with integrity check (cough/versioning.py).

Not a medical device; investigational software developed with IEC 62304 Class B intent.

Environment
  COUGH_MODE              production (default) | research
  COUGH_ARTIFACTS         model root folder (default artifacts); COUGH_MODEL_VERSION=1.0.0 pins a version
  COUGH_AUDIT_DIR         audit trail folder (default audit)
  COUGH_DATA_DIR          feature store / retained audio (default data_store)
  COUGH_AUDIO_RETENTION_DAYS  0 (default) = audio never stored
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import phi, samd
from .audio import load_audio
from .audit import AuditLog, new_request_id, sha256_bytes, valid_id
from .cnn import CoughCNN, Normaliser, predict
from .features import READABLE, analyse
from .formats import SUPPORTED, extension_matches, sniff, supported_list
from .train import apply_fusion
from .versioning import card_sha256, resolve_artifacts, verify_integrity, version_info

ART_ROOT = Path(os.environ.get("COUGH_ARTIFACTS", "artifacts"))
ART = resolve_artifacts(ART_ROOT)            # the model version to serve (LATEST, or COUGH_MODEL_VERSION=1.2.0)
AUDIT_DIR = Path(os.environ.get("COUGH_AUDIT_DIR", "audit"))
DATA_DIR = Path(os.environ.get("COUGH_DATA_DIR", "data_store"))
MAX_BYTES = 20_000_000
PANNS_CKPT = Path(os.environ.get("COUGH_PANNS", "models/Cnn14_mAP=0.431.pth"))
DISCLAIMER = ("Not a medical device. This software does not diagnose COVID-19 or any disease. The cough type is "
              "supporting information for a clinician, not a diagnosis. If you are unwell, see a doctor.")


def readable(name: str) -> str:
    if name in READABLE:
        return READABLE[name]
    if m := re.match(r"mfcc(\d+)_(mean|std)", name):
        return f"timbre coefficient {m.group(1)} ({'average' if m.group(2) == 'mean' else 'variation'})"
    if m := re.match(r"dmfcc(\d+)_std", name):
        return f"timbre change {m.group(1)}"
    if m := re.match(r"band_(\d+)_(\d+)", name):
        return f"energy share {m.group(1)}-{m.group(2)} Hz"
    if m := re.match(r"contrast(\d+)_mean", name):
        return f"spectral contrast (band {m.group(1)})"
    return name.replace("_", " ")


class Engine:
    def __init__(self, art: Path):
        if not (art / "config.json").exists():
            raise FileNotFoundError(f"{art}/config.json missing - run `python -m cough.train` first")
        bad = verify_integrity(art)
        if bad and samd.is_production():                  # fail closed: never serve an unverified model
            raise RuntimeError(f"model integrity check failed for {art.name}: {bad}. Production serves only "
                               f"registered, unmodified model versions (python -m cough.versioning --list).")
        self.integrity = "verified" if not bad else f"NOT verified ({'; '.join(bad)})"
        self.cfg = json.loads((art / "config.json").read_text())
        self.version = version_info(art)
        self.card_sha256 = card_sha256(art)
        self.metrics = json.loads((art / "metrics.json").read_text())
        self.models = joblib.load(art / "models.joblib")
        ck = torch.load(art / "cnn.pt", map_location="cpu", weights_only=False)
        self.cnn = CoughCNN(len(ck["tasks"]))
        self.cnn.load_state_dict(ck["state"])
        self.cnn.eval()
        self.norm = Normaliser(ck["mean"], ck["std"])
        self.tasks = ck["tasks"]
        self.features = self.cfg["features"]
        samd.check_model_inputs(self.features)            # audio features only - never location / demographics
        self.examples = pd.read_parquet(art / "test_predictions.parquet")
        self.panns = None
        if self.cfg.get("has_panns") and PANNS_CKPT.exists():
            from .panns import load
            self.panns = load(PANNS_CKPT)

    def scores(self, path: str, tasks) -> tuple[dict, dict, dict]:
        x = load_audio(path)
        if len(x) < 1600:
            raise HTTPException(422, "recording shorter than 0.1 s")
        a = analyse(x)
        xf = np.array([[a["features"].get(c, 0.0) for c in self.features]], np.float32)
        logits = predict(self.cnn, a["logmel"][None], np.array([0]), self.norm)[0]
        emb, tags = None, None
        if self.panns is not None:
            from .panns import AUDIOSET, SR, embed
            x32 = load_audio(path, sr=SR)
            s, e = a["region"]
            w = x32[int(s / 16000 * SR):int(e / 16000 * SR)]
            emb, tg = embed(self.panns, [w if len(w) >= SR // 4 else x32[: 5 * SR]])
            tags = {k: float(tg[0, i]) for k, i in AUDIOSET.items()}
        out = {}
        for k, task in enumerate(self.tasks):
            if task not in self.models or task not in tasks:      # outside the intended use: not computed
                continue
            m = self.models[task]
            sc = {"handcrafted": m["handcrafted"].predict_proba(xf, raw_score=True), "cnn": logits[k:k + 1]}
            if "panns" in m["fusion"]["weights"]:
                if emb is None:                                   # PANNs not available: drop it from the fusion
                    f = dict(m["fusion"])
                    f["weights"] = {n: w for n, w in f["weights"].items() if n != "panns"} or {"handcrafted": 1.0}
                    out[task] = float(apply_fusion(f, sc)[0])
                    continue
                sc["panns"] = m["panns"][1].decision_function(m["panns"][0].transform(emb))
            out[task] = float(apply_fusion(m["fusion"], sc)[0])
        return out, a, {"x": xf, "tags": tags}

    def why(self, task: str, xf: np.ndarray, n: int = 3) -> list[str]:
        """Largest contributions of the handcrafted-feature model (LightGBM SHAP values)."""
        m = self.models[task]["handcrafted"]
        contrib = m.predict_proba(xf, pred_contrib=True)[0][:-1]
        order = np.argsort(-np.abs(contrib))[:n]
        return [f"{readable(self.features[i])} = {xf[0, i]:.3g} "
                f"({'raises' if contrib[i] > 0 else 'lowers'} the score)" for i in order]

    def analyse_file(self, path: str) -> tuple[dict, dict]:
        """(response, acoustic features). Only the outputs allowed in the current mode are computed."""
        t0 = time.time()
        p, a, extra = self.scores(path, samd.allowed_tasks())
        f = a["features"]
        segs = [[round(s / 16000, 2), round(e / 16000, 2)] for s, e in a["segments"]]
        cough_p = p.get("cough", 1.0)
        res = {"recording": {"duration_s": round(f["duration"], 2), "coughs_found": len(segs), "cough_segments_s": segs,
                             "snr_db": round(a["snr"], 1)},
               "cough_check": {"is_cough": bool(cough_p >= 0.5 and segs), "probability": round(cough_p, 3)}}
        if extra["tags"]:
            res["cough_check"]["audioset_cough_probability"] = round(extra["tags"]["cough"], 3)
            res["cough_check"]["audioset_speech_probability"] = round(extra["tags"]["speech"], 3)
        res["model_version"] = self.version.get("name")
        if not res["cough_check"]["is_cough"]:
            res["message"] = "No clear cough found. Please record 2-3 coughs in a quiet place, about 20 cm from the mic."
            res["disclaimer"] = DISCLAIMER
            return res, f
        if "wet" in p:
            ok, why_not = samd.quality_gate(len(segs), a["snr"], True)               # RC-01
            if ok:
                res["cough_type"] = {"label": samd.cough_type_label(p["wet"]),       # RC-02
                                     "wet_probability": round(p["wet"], 3),
                                     "indeterminate_band": list(samd.INDETERMINATE),
                                     "for": "clinician review only - not a diagnosis",
                                     "why": self.why("wet", extra["x"])}
            else:
                res["cough_type"] = {"label": "not assessed", "reason": why_not,
                                     "message": "Please record 2-3 coughs in a quiet place, about 20 cm from the mic."}
        for task, name in (("covid", "covid_19"), ("symptomatic", "respiratory_symptoms")):   # research mode only
            if task not in p:
                continue
            thr = self.models[task]["threshold_90spec"]
            r = self.metrics["results"].get(task, {}).get("fusion", {}).get("at_val_threshold", {})
            res[name] = {"probability": round(p[task], 3), "flag": "elevated" if p[task] >= thr else "not elevated",
                         "threshold": round(thr, 3), "research_use_only": True,
                         "how_reliable": (f"on unseen test recordings this flag caught {r.get('sensitivity', 0):.0%} of "
                                          f"positives with {r.get('specificity', 0):.0%} specificity "
                                          f"(ROC-AUC {self.metrics['results'][task]['fusion'].get('roc_auc', 0):.2f})"),
                         "why": self.why(task, extra["x"])}
        res["disclaimer"] = DISCLAIMER
        res["seconds"] = round(time.time() - t0, 2)
        return res, f


app = FastAPI(title="COUGHVID cough analysis API", version="1.1",
              description=samd.INTENDED_USE + " " + samd.REGULATORY_STATUS)
_cache: dict = {}


def eng() -> Engine:
    if "e" not in _cache:
        try:
            _cache["e"] = Engine(ART)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    return _cache["e"]


def audit() -> AuditLog:
    if "a" not in _cache:
        _cache["a"] = AuditLog(AUDIT_DIR)
    return _cache["a"]


def caller(user_id: Optional[str], org_id: Optional[str]) -> tuple[str | None, str | None]:
    """Identity of the caller (set by the authenticating gateway in deployment). Required in production."""
    if samd.is_production():
        if not (valid_id(user_id) and valid_id(org_id)):
            raise HTTPException(401, "X-User-Id and X-Org-Id headers are required (opaque account ids: letters, "
                                     "digits, . _ : - ; no names or e-mail addresses)")
    else:
        user_id = user_id if valid_id(user_id) else "research"
        org_id = org_id if valid_id(org_id) else "research"
    return user_id, org_id


def model_fields() -> dict:
    e = _cache.get("e")
    return {"model_version": e.version.get("name") if e else None, "model_card_sha256": e.card_sha256 if e else None}


@app.on_event("startup")
def warm_up():
    try:
        e = eng()
        audit()
        print(f"mode {samd.mode()} - model {e.version.get('name')} ({e.integrity}) - "
              f"interface: http://localhost:8008  (API docs: http://localhost:8008/docs)", flush=True)
    except Exception as ex:
        print(f"warm-up skipped: {getattr(ex, 'detail', ex)}", flush=True)


UI = Path(__file__).parent / "static" / "index.html"
MEDIA = {v[0]: v[1] for v in SUPPORTED.values()}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """Web interface: record or upload a cough (and, in research mode, pick a test-set recording)."""
    return HTMLResponse(UI.read_text(encoding="utf-8"))


@app.get("/intended-use")
def intended_use():
    """Intended use, users, contraindications and regulatory status (the labelling)."""
    return {**samd.labelling(), "intended_use_full": samd.INTENDED_USE_FULL, "intended_users": samd.INTENDED_USERS,
            "contraindications": samd.CONTRAINDICATIONS, "iec_62304_safety_class_intent": samd.SAFETY_CLASS,
            "outputs": list(samd.allowed_tasks())}


@app.get("/health")
def health():
    e = _cache.get("e")
    return {"status": "trained" if (ART / "config.json").exists() else "not trained", "mode": samd.mode(),
            "model_version": version_info(ART).get("name"), "integrity": e.integrity if e else "not loaded"}


@app.get("/model-info")
def model_info():
    e = eng()
    res = e.metrics.get("results", {})
    if samd.is_production():                      # performance of the intended-use outputs only
        res = {k: v for k, v in res.items() if k in samd.PRODUCTION_TASKS}
        return {"version": e.version, "labelling": samd.labelling(), "results": res}
    return {"version": e.version, "labelling": samd.labelling(), **e.metrics}


@app.get("/formats")
def formats():
    """Audio formats accepted by /analyze. Everything is converted to 16 kHz mono PCM before the models."""
    return {"supported": {k: v[2] for k, v in SUPPORTED.items()}, "max_mb": MAX_BYTES // 1_000_000,
            "converted_to": "16 kHz mono PCM (float32), peak-normalised"}


def check_upload(data: bytes, filename: str | None, content_type: str | None) -> dict:
    """The format layer in front of the models (see cough/formats.py). Raises the HTTP error to return."""
    if len(data) == 0:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES // 1_000_000} MB")
    info = sniff(data[:4096])
    if not info["supported"]:
        raise HTTPException(415, f"unsupported file: {info['reason']}. Supported audio formats: {supported_list()}")
    info["declared_type"] = content_type
    info["extension_matches"] = extension_matches(filename, info["format"])   # the file name itself is not kept
    return info


def _audited(event: str, rid: str, user, org, status: int, body: dict, **extra) -> JSONResponse:
    """Write the audit record, then return. If the audit write fails, the request fails (no unaudited output)."""
    user = user if valid_id(user) else None          # never write an invalid id (e.g. an e-mail) into the log
    org = org if valid_id(org) else None
    try:
        audit().write(event, rid, user_id=user, org_id=org, status=status, mode=samd.mode(), **model_fields(),
                      **extra, output=body)
    except Exception as ex:
        return JSONResponse({"detail": f"audit trail unavailable, request not processed ({ex})", "request_id": rid},
                            status_code=503, headers={"X-Request-ID": rid})
    return JSONResponse(body, status_code=status, headers={"X-Request-ID": rid})


@app.post("/analyze")
async def analyze(request: Request,
                  file: UploadFile = File(..., description="cough recording: webm, ogg, mp4, m4a, wav, mp3, aac, "
                                                           "flac, 3gp, amr (see GET /formats)"),
                  x_user_id: Optional[str] = Header(None), x_org_id: Optional[str] = Header(None)):
    rid = new_request_id()
    user, org, input_hash, fmt = x_user_id, x_org_id, None, None
    try:
        user, org = caller(x_user_id, x_org_id)
        form = await request.form()
        extra_fields = sorted(set(form.keys()) - {"file"}) + sorted(request.query_params.keys())
        if extra_fields:                                   # e.g. latitude / longitude: never accepted
            raise HTTPException(400, f"only the audio file is accepted; remove the fields {extra_fields} "
                                     f"(location and personal metadata never enter inference)")
        data = await file.read()
        input_hash = sha256_bytes(data) if data else None
        info = check_upload(data, file.filename, file.content_type)       # 1. format check before the models
        fmt = info["format"]
        e = eng()
        with tempfile.NamedTemporaryFile(suffix=info["extension"], delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        try:
            res, feats = e.analyse_file(path)                              # 2. decode to 16 kHz mono + models
        except ValueError as ex:
            raise HTTPException(422, f"could not decode the {fmt} file (corrupted or truncated?): {ex}")
        finally:
            os.unlink(path)                                                # audio is not kept (PHI default)
        phi.store_features(DATA_DIR, rid, user, org, input_hash, feats, res["model_version"])
        stored = phi.store_audio(DATA_DIR, rid, org, data, info["extension"])
        info.pop("supported", None)
        info.pop("reason", None)
        info["converted_to"] = "16 kHz mono PCM"
        body = {"request_id": rid, **samd.labelling(), "input": info, **res,
                "data_retention": (f"audio kept {phi.retention_days()} days for clinician review" if stored else
                                   "audio not stored; only its hash and acoustic features")}
        return _audited("inference", rid, user, org, 200, body, input_sha256=input_hash, input_format=fmt)
    except HTTPException as ex:
        body = {"detail": ex.detail, "request_id": rid}
        return _audited("rejected", rid, user, org, ex.status_code, body, input_sha256=input_hash, input_format=fmt)


@app.delete("/subjects/{user_id}")
def erase(user_id: str, x_user_id: Optional[str] = Header(None), x_org_id: Optional[str] = Header(None)):
    """Right to erasure: delete the stored audio and features of one person. The erasure is audited."""
    rid = new_request_id()
    who, org = caller(x_user_id, x_org_id)
    if not valid_id(user_id):
        raise HTTPException(400, "invalid user id")
    if samd.is_production() and who != user_id and not who.startswith("admin"):
        raise HTTPException(403, "a person can erase only their own data (or an admin account)")
    r = phi.erase_subject(DATA_DIR, user_id)
    return _audited("erasure", rid, who, org, 200, {"request_id": rid, **r}, subject=user_id)


def _research_only():
    if samd.is_production():
        raise HTTPException(404, "the test-set browser is available in research mode only")


@app.get("/examples/{uuid}/audio")
def example_audio(uuid: str):
    """The audio file of a test-set recording (research mode)."""
    _research_only()
    e = eng()
    row = e.examples[e.examples["uuid"] == uuid]
    if row.empty:
        raise HTTPException(404, "uuid not in the test set")
    path = Path(str(row.iloc[0]["path"]))
    if not path.exists():
        raise HTTPException(404, f"audio file not found: {path}")
    return FileResponse(path, media_type=MEDIA.get(path.suffix.lower(), "application/octet-stream"))


@app.get("/examples")
def examples(status: Optional[str] = Query(None, description="healthy, symptomatic or COVID-19"),
             n: int = Query(5, ge=1, le=50), seed: int = 0):
    """Test-set recordings (never used for training) with their labels and the model's scores (research mode)."""
    _research_only()
    ex = eng().examples
    if status:
        ex = ex[ex["status"].astype("string").str.lower() == status.lower()]
    ex = ex.sample(min(n, len(ex)), random_state=seed) if len(ex) else ex
    cols = [c for c in ex.columns if c != "path"]
    return [{k: (None if pd.isna(v) else (round(float(v), 3) if isinstance(v, (float, np.floating)) else v))
             for k, v in r.items()} for r in ex[cols].to_dict("records")]


@app.get("/examples/{uuid}/analyze")
def analyze_example(uuid: str):
    """Analyse a test-set recording by its uuid (research mode)."""
    _research_only()
    e = eng()
    row = e.examples[e.examples["uuid"] == uuid]
    if row.empty:
        raise HTTPException(404, "uuid not in the test set (use GET /examples)")
    r = row.iloc[0]
    path = str(r["path"])
    if not Path(path).exists():
        raise HTTPException(404, f"audio file not found: {path}")
    res, _ = e.analyse_file(path)
    res["true_label"] = {"status": None if pd.isna(r["status"]) else r["status"],
                         "status_SSL": None if pd.isna(r["status_SSL"]) else r["status_SSL"],
                         "expert_cough_type": None if pd.isna(r["exp_cough_type"]) else r["exp_cough_type"]}
    return {"uuid": uuid, **samd.labelling(), **res}
