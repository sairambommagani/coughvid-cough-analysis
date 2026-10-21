"""FastAPI service: upload a cough recording -> cough check, cough type, COVID-19 and respiratory-symptom risk,
with the reasons behind the scores.

  python -m uvicorn cough.api:app --port 8008   ->   http://localhost:8008/docs

Research prototype: trained on crowdsourced recordings with self-reported labels. Not a medical device.
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
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from .audio import load_audio
from .cnn import CoughCNN, Normaliser, predict
from .features import READABLE, analyse
from .train import apply_fusion

ART = Path(os.environ.get("COUGH_ARTIFACTS", "artifacts"))
PANNS_CKPT = Path(os.environ.get("COUGH_PANNS", "models/Cnn14_mAP=0.431.pth"))
DISCLAIMER = ("Research prototype trained on crowdsourced recordings with self-reported labels. It is not a medical "
              "device and does not diagnose COVID-19 or any disease; see a doctor or take a test if you are unwell.")


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
        self.cfg = json.loads((art / "config.json").read_text())
        self.metrics = json.loads((art / "metrics.json").read_text())
        self.models = joblib.load(art / "models.joblib")
        ck = torch.load(art / "cnn.pt", map_location="cpu", weights_only=False)
        self.cnn = CoughCNN(len(ck["tasks"]))
        self.cnn.load_state_dict(ck["state"])
        self.cnn.eval()
        self.norm = Normaliser(ck["mean"], ck["std"])
        self.tasks = ck["tasks"]
        self.features = self.cfg["features"]
        self.examples = pd.read_parquet(art / "test_predictions.parquet")
        self.panns = None
        if self.cfg.get("has_panns") and PANNS_CKPT.exists():
            from .panns import load
            self.panns = load(PANNS_CKPT)

    def scores(self, path: str) -> tuple[dict, dict, dict]:
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
            if task not in self.models:
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

    def analyse_file(self, path: str) -> dict:
        t0 = time.time()
        p, a, extra = self.scores(path)
        f = a["features"]
        segs = [[round(s / 16000, 2), round(e / 16000, 2)] for s, e in a["segments"]]
        cough_p = p.get("cough", 1.0)
        res = {"recording": {"duration_s": round(f["duration"], 2), "coughs_found": len(segs), "cough_segments_s": segs,
                             "snr_db": round(a["snr"], 1)},
               "cough_check": {"is_cough": bool(cough_p >= 0.5 and segs), "probability": round(cough_p, 3)}}
        if extra["tags"]:
            res["cough_check"]["audioset_cough_probability"] = round(extra["tags"]["cough"], 3)
            res["cough_check"]["audioset_speech_probability"] = round(extra["tags"]["speech"], 3)
        if not res["cough_check"]["is_cough"]:
            res["message"] = "No clear cough found. Please record 2-3 coughs in a quiet place, about 20 cm from the mic."
            res["disclaimer"] = DISCLAIMER
            return res
        if "wet" in p:
            res["cough_type"] = {"wet_probability": round(p["wet"], 3), "label": "wet" if p["wet"] >= 0.5 else "dry",
                                 "why": self.why("wet", extra["x"])}
        for task, name in (("covid", "covid_19"), ("symptomatic", "respiratory_symptoms")):
            if task not in p:
                continue
            thr = self.models[task]["threshold_90spec"]
            r = self.metrics["results"].get(task, {}).get("fusion", {}).get("at_val_threshold", {})
            res[name] = {"probability": round(p[task], 3), "flag": "elevated" if p[task] >= thr else "not elevated",
                         "threshold": round(thr, 3),
                         "how_reliable": (f"on unseen test recordings this flag caught {r.get('sensitivity', 0):.0%} of "
                                          f"positives with {r.get('specificity', 0):.0%} specificity "
                                          f"(ROC-AUC {self.metrics['results'][task]['fusion'].get('roc_auc', 0):.2f})"),
                         "why": self.why(task, extra["x"])}
        res["disclaimer"] = DISCLAIMER
        res["seconds"] = round(time.time() - t0, 2)
        return res


app = FastAPI(title="COUGHVID cough analysis API", version="1.0",
              description="Cough check, cough type (wet/dry), COVID-19 and respiratory-symptom risk from a cough "
                          "recording. Handcrafted-feature LightGBM, PANNs CNN14 embeddings and a log-mel CNN, fused "
                          "and calibrated on a validation split. " + DISCLAIMER)
_cache: dict = {}


def eng() -> Engine:
    if "e" not in _cache:
        try:
            _cache["e"] = Engine(ART)
        except FileNotFoundError as e:
            raise HTTPException(503, str(e))
    return _cache["e"]


@app.on_event("startup")
def warm_up():
    try:
        eng()
        print("models loaded - open http://localhost:8008/docs", flush=True)
    except Exception as ex:
        print(f"warm-up skipped: {ex}", flush=True)


@app.get("/health")
def health():
    return {"status": "trained" if (ART / "config.json").exists() else "not trained"}


@app.get("/model-info")
def model_info():
    return eng().metrics


@app.get("/examples")
def examples(status: Optional[str] = Query(None, description="healthy, symptomatic or COVID-19"),
             n: int = Query(5, ge=1, le=50), seed: int = 0):
    """Test-set recordings (never used for training) with their labels and the model's scores."""
    ex = eng().examples
    if status:
        ex = ex[ex["status"].astype("string").str.lower() == status.lower()]
    ex = ex.sample(min(n, len(ex)), random_state=seed) if len(ex) else ex
    cols = [c for c in ex.columns if c != "path"]
    return [{k: (None if pd.isna(v) else (round(float(v), 3) if isinstance(v, (float, np.floating)) else v))
             for k, v in r.items()} for r in ex[cols].to_dict("records")]


@app.get("/examples/{uuid}/analyze")
def analyze_example(uuid: str):
    """Analyse a test-set recording by its uuid (from /examples)."""
    e = eng()
    row = e.examples[e.examples["uuid"] == uuid]
    if row.empty:
        raise HTTPException(404, "uuid not in the test set (use GET /examples)")
    r = row.iloc[0]
    path = str(r["path"])
    if not Path(path).exists():
        raise HTTPException(404, f"audio file not found: {path}")
    res = e.analyse_file(path)
    res["true_label"] = {"status": None if pd.isna(r["status"]) else r["status"],
                         "status_SSL": None if pd.isna(r["status_SSL"]) else r["status_SSL"],
                         "expert_cough_type": None if pd.isna(r["exp_cough_type"]) else r["exp_cough_type"]}
    return {"uuid": uuid, **res}


@app.post("/analyze")
async def analyze(file: UploadFile = File(..., description="cough recording: webm, ogg, wav, mp3 or m4a")):
    suffix = Path(file.filename or "x.webm").suffix or ".webm"
    data = await file.read()
    if len(data) > 20_000_000:
        raise HTTPException(413, "file larger than 20 MB")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return {"file": file.filename, **eng().analyse_file(path)}
    except ValueError as ex:
        raise HTTPException(422, str(ex))
    finally:
        os.unlink(path)
