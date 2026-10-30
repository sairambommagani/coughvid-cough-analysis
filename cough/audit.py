"""Audit trail: one record per inference (and per rejected request and per data deletion).

Each record: seq, request_id, timestamp (UTC, ISO 8601), event, user_id, org_id, model_version,
model_card_sha256, input_sha256, input format, output JSON (the exact response), status code, prev_hash, hash.

Append-only and tamper-evident:
  * the log is only ever opened in append mode, one JSON line per record, flushed and fsync'ed before the
    response is returned (no response without its audit record: if writing fails, the request fails);
  * every record contains the SHA-256 of the previous record (a hash chain), so editing, deleting or reordering
    any line breaks verification from that line on: `python -m cough.audit --verify`;
  * one file per UTC day (audit-YYYYMMDD.jsonl); the chain continues across files;
  * WORM in deployment: the log directory is shipped to storage with a write-once retention lock (e.g. AWS S3
    Object Lock in compliance mode, Azure immutable blob storage) - see docs/regulatory/09_audit_and_data.md.
    Locally the files are made read-only once the day is over (seal()).

The log holds no audio and no features: only hashes, ids, the model version and the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64
ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def new_request_id() -> str:
    return str(uuid.uuid4())


def valid_id(x: str | None) -> bool:
    """User / org ids must be opaque account ids (letters, digits, . _ : -), never names or e-mail addresses."""
    return bool(x) and bool(ID_RE.match(x)) and "@" not in x


class AuditLog:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq, self._prev = self._tail()

    def _files(self) -> list[Path]:
        return sorted(self.dir.glob("audit-*.jsonl"))

    def _tail(self) -> tuple[int, str]:
        for f in reversed(self._files()):
            lines = [x for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
            if lines:
                last = json.loads(lines[-1])
                return int(last["seq"]), last["hash"]
        return 0, GENESIS

    def write(self, event: str, request_id: str, **fields) -> dict:
        with self._lock:
            now = datetime.now(timezone.utc)
            rec = {"seq": self._seq + 1, "request_id": request_id, "timestamp": now.isoformat(timespec="milliseconds"),
                   "event": event, **fields, "prev_hash": self._prev}
            rec["hash"] = hashlib.sha256((self._prev + canonical(rec)).encode()).hexdigest()
            path = self.dir / f"audit-{now:%Y%m%d}.jsonl"
            with open(path, "a", encoding="utf-8") as f:            # append only
                f.write(canonical(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._seq, self._prev = rec["seq"], rec["hash"]
            return rec

    def seal(self, before_day: str | None = None) -> list[str]:
        """Make the files of finished days read-only (local stand-in for WORM storage)."""
        today = before_day or datetime.now(timezone.utc).strftime("%Y%m%d")
        done = []
        for f in self._files():
            if f.stem.split("-")[1] < today:
                f.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
                done.append(f.name)
        return done


def verify(directory: Path) -> dict:
    """Recompute the whole hash chain. Returns {"ok": bool, "records": n, "first_bad": {...} or None}."""
    prev, seq, n = GENESIS, 0, 0
    for f in sorted(Path(directory).glob("audit-*.jsonl")):
        for line_no, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            n += 1
            try:
                rec = json.loads(line)
            except ValueError:
                return {"ok": False, "records": n, "first_bad": {"file": f.name, "line": line_no, "why": "not JSON"}}
            h = rec.pop("hash", None)
            why = None
            if rec.get("prev_hash") != prev:
                why = "chain broken (a record before this one was changed, removed or reordered)"
            elif rec.get("seq") != seq + 1:
                why = "sequence gap"
            elif hashlib.sha256((prev + canonical(rec)).encode()).hexdigest() != h:
                why = "record content changed"
            if why:
                return {"ok": False, "records": n, "first_bad": {"file": f.name, "line": line_no, "seq": rec.get("seq"),
                                                                 "why": why}}
            prev, seq = h, rec["seq"]
    return {"ok": True, "records": n, "first_bad": None, "last_hash": prev}


def find(directory: Path, request_id: str) -> list[dict]:
    out = []
    for f in sorted(Path(directory).glob("audit-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if request_id in line:
                rec = json.loads(line)
                if rec.get("request_id") == request_id:
                    out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description="audit trail")
    ap.add_argument("--dir", default=os.environ.get("COUGH_AUDIT_DIR", "audit"))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--find", help="request id")
    ap.add_argument("--seal", action="store_true", help="make finished days read-only")
    a = ap.parse_args()
    if a.verify:
        r = verify(Path(a.dir))
        print(json.dumps(r, indent=2))
        raise SystemExit(0 if r["ok"] else 1)
    if a.find:
        print(json.dumps(find(Path(a.dir), a.find), indent=2))
    if a.seal:
        print(AuditLog(Path(a.dir)).seal())


if __name__ == "__main__":
    main()
