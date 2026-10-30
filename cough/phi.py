"""Protected health information (PHI) handling: what is stored, for how long, and how it is deleted.

Default (data minimisation):
  * the uploaded audio is NOT stored: it is decoded in memory / a temporary file that is deleted when the request
    ends;
  * stored per request: the SHA-256 of the input, the handcrafted acoustic features (numbers such as cough
    count, duration, spectral statistics - no audio, no speech content), the output, the model version, the
    user / organisation ids. File: <COUGH_DATA_DIR>/features.jsonl;
  * never stored and never accepted as model input: latitude / longitude or any other location, name, e-mail,
    phone, date of birth, free text (cough/samd.py FORBIDDEN_INPUTS). The original file name is not stored
    either (it can contain a person's name); only its detected format.

Audio retention (opt-in, documented): COUGH_AUDIO_RETENTION_DAYS=N > 0 keeps the audio for N days under
<COUGH_DATA_DIR>/audio/<org>/<request_id>.<ext> for clinician review, then `python -m cough.phi --purge`
(scheduled daily) deletes it. 0 (default) = never stored.

Right to deletion: DELETE /subjects/{user_id} (or `python -m cough.phi --erase USER`) removes that person's audio
and feature records. The audit trail keeps its records (hashes, ids, outputs - no audio, no features) because it
is required evidence; the erasure itself is recorded in the audit trail.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("COUGH_DATA_DIR", "data_store"))
_lock = threading.Lock()


def retention_days() -> int:
    try:
        return max(0, int(os.environ.get("COUGH_AUDIO_RETENTION_DAYS", "0")))
    except ValueError:
        return 0


def store_features(data_dir: Path, request_id: str, user_id: str, org_id: str, input_sha256: str,
                   features: dict, model_version: str) -> None:
    rec = {"request_id": request_id, "user_id": user_id, "org_id": org_id, "input_sha256": input_sha256,
           "model_version": model_version, "stored_at": time.time(),
           "features": {k: (None if v != v else round(float(v), 6)) for k, v in features.items()}}
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with _lock, open(data_dir / "features.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def store_audio(data_dir: Path, request_id: str, org_id: str, data: bytes, ext: str) -> Path | None:
    """Only when audio retention is switched on (COUGH_AUDIO_RETENTION_DAYS > 0)."""
    if retention_days() <= 0:
        return None
    d = Path(data_dir) / "audio" / org_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{request_id}{ext}"
    p.write_bytes(data)
    (d / f"{request_id}.meta.json").write_text(json.dumps({"stored_at": time.time(), "days": retention_days()}))
    return p


def _owner_requests(data_dir: Path, user_id: str) -> set[str]:
    p = Path(data_dir) / "features.jsonl"
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("user_id") == user_id:
                out.add(r["request_id"])
    return out


def erase_subject(data_dir: Path, user_id: str) -> dict:
    """Right to erasure: delete the person's feature records and stored audio."""
    data_dir = Path(data_dir)
    reqs = _owner_requests(data_dir, user_id)
    removed_audio = 0
    for meta in (data_dir / "audio").glob("*/*.meta.json") if (data_dir / "audio").exists() else []:
        rid = meta.name[: -len(".meta.json")]
        if rid in reqs:
            for f in meta.parent.glob(f"{rid}*"):
                f.unlink()
                removed_audio += f.suffix != ".json"
    p = data_dir / "features.jsonl"
    kept = 0
    if p.exists():
        with _lock:
            lines = [x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            keep = [x for x in lines if json.loads(x).get("user_id") != user_id]
            tmp = p.with_suffix(".tmp")
            tmp.write_text("".join(x + "\n" for x in keep), encoding="utf-8")
            os.replace(tmp, p)
            kept = len(keep)
    return {"user_id": user_id, "feature_records_deleted": len(reqs), "audio_files_deleted": removed_audio,
            "feature_records_remaining_for_others": kept}


def purge_expired(data_dir: Path, now: float | None = None) -> int:
    """Delete stored audio older than its retention period. Run daily."""
    now = now or time.time()
    n = 0
    root = Path(data_dir) / "audio"
    if not root.exists():
        return 0
    for meta in root.glob("*/*.meta.json"):
        m = json.loads(meta.read_text())
        if now - m["stored_at"] > m["days"] * 86400:
            rid = meta.name[: -len(".meta.json")]
            for f in meta.parent.glob(f"{rid}*"):
                f.unlink()
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="PHI store maintenance")
    ap.add_argument("--dir", default=str(DATA_DIR))
    ap.add_argument("--purge", action="store_true", help="delete audio past its retention period")
    ap.add_argument("--erase", metavar="USER_ID", help="delete all stored data of one person")
    a = ap.parse_args()
    if a.purge:
        print(f"purged {purge_expired(Path(a.dir))} recordings")
    if a.erase:
        print(json.dumps(erase_subject(Path(a.dir), a.erase), indent=2))


if __name__ == "__main__":
    main()
