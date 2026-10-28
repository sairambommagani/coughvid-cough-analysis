"""Model versions: every training run gets a standard version name and a model card.

Standard used: semantic versioning MAJOR.MINOR.PATCH, applied to models
  MAJOR  new architecture, new input features or a change that breaks the API output   (1.4.2 -> 2.0.0)
  MINOR  retrained: new data, new hyperparameters, same architecture and output        (1.4.2 -> 1.5.0)
  PATCH  fix without changing what the model learned (threshold, calibration, bug fix) (1.4.2 -> 1.4.3)

Name of a version (and of its folder): <project>_v<MAJOR.MINOR.PATCH>_<model type>_<YYYYMMDD>
  e.g. artifacts/coughvid_v1.1.0_fusion_20261028/

  artifacts/
    registry.json                              every version: name, date, git commit, data, metrics, parent
    LATEST                                     name of the version the API serves
    coughvid_v1.0.0_fusion_20261022/           config.json, models.joblib, cnn.pt, metrics.json, version.json ...
    coughvid_v1.1.0_fusion_20261028/

  python -m cough.train --bump minor --notes "retrained with ..."   # new version folder, LATEST moves to it
  python -m cough.versioning --list
  python -m cough.versioning --adopt-legacy                         # old flat artifacts/ -> coughvid_v1.0.0_...
  COUGH_MODEL_VERSION=1.0.0 python -m uvicorn cough.api:app          # serve an older version (rollback)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path

PROJECT = "coughvid"
MODEL_TYPE = "fusion"
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
ARTIFACT_FILES = ("config.json", "metrics.json", "models.joblib", "cnn.pt", "test_predictions.parquet")


def parse(v: str) -> tuple[int, int, int]:
    m = SEMVER.match(str(v).lstrip("v"))
    if not m:
        raise ValueError(f"version '{v}' is not MAJOR.MINOR.PATCH")
    return tuple(int(x) for x in m.groups())


def bump(v: str | None, part: str) -> str:
    if v is None:
        return "1.0.0"
    major, minor, patch = parse(v)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError("bump must be major, minor or patch")


def version_name(version: str, day: str | None = None) -> str:
    parse(version)
    return f"{PROJECT}_v{version.lstrip('v')}_{MODEL_TYPE}_{day or time.strftime('%Y%m%d')}"


def load_registry(root: Path) -> dict:
    p = Path(root) / "registry.json"
    return json.loads(p.read_text()) if p.exists() else {"project": PROJECT, "versions": []}


def save_registry(root: Path, reg: dict):
    (Path(root) / "registry.json").write_text(json.dumps(reg, indent=2))


def latest_version(root: Path) -> str | None:
    vs = [v["version"] for v in load_registry(root)["versions"]]
    return max(vs, key=parse) if vs else None


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                           cwd=Path(__file__).resolve().parent)
        return r.stdout.strip() or None
    except Exception:
        return None


def git_dirty() -> bool | None:
    """True when the code had uncommitted changes at training time (the commit alone does not describe it)."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           timeout=5, cwd=Path(__file__).resolve().parent)
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None


def data_fingerprint(cache: Path) -> dict:
    """Which data the version was trained on: size and hash of the cached metadata."""
    p = Path(cache) / "meta.parquet"
    if not p.exists():
        return {}
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    try:
        import pandas as pd
        n = int(len(pd.read_parquet(p, columns=["uuid"])))
    except Exception:
        n = None
    return {"cache": str(cache), "meta_sha256": h, "recordings": n}


def new_version(root: Path, part: str = "minor", version: str | None = None) -> tuple[str, Path]:
    """Next version number and its (new, empty) folder."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.json").exists() and not load_registry(root)["versions"]:
        adopt_legacy(root)                     # old flat model becomes 1.0.0 first, so the new one is 1.1.0
    reg = load_registry(root)
    v = version.lstrip("v") if version else bump(latest_version(root), part)
    parse(v)
    if any(x["version"] == v for x in reg["versions"]):
        raise ValueError(f"version {v} already exists - use --bump or another --version")
    d = root / version_name(v)
    for old in root.glob(f"{PROJECT}_v{v}_{MODEL_TYPE}_*"):    # an unfinished (crashed) run of the same version
        if not (old / "version.json").exists():
            shutil.rmtree(old)
    d.mkdir(parents=True, exist_ok=False)
    return v, d


def register(root: Path, vdir: Path, version: str, part: str, cache: Path | None, args: dict, notes: str = "",
             parent: str | None = None) -> dict:
    """Write the model card (version.json) into the version folder, add it to the registry, point LATEST at it."""
    root, vdir = Path(root), Path(vdir)
    metrics = json.loads((vdir / "metrics.json").read_text()) if (vdir / "metrics.json").exists() else {}
    summary = {t: {m: r.get("roc_auc") for m, r in res.items() if isinstance(r, dict) and "roc_auc" in r}
               for t, res in metrics.get("results", {}).items() if isinstance(res, dict) and t != "covid_extra"}
    card = {"name": vdir.name, "project": PROJECT, "model_type": MODEL_TYPE, "version": version, "bump": part,
            "parent": parent, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
            "git_uncommitted_changes": git_dirty(),
            "data": data_fingerprint(cache) if cache else {}, "training_args": args, "notes": notes,
            "test_roc_auc": summary, "python": platform.python_version(),
            "files": sorted(p.name for p in vdir.iterdir() if p.is_file())}
    (vdir / "version.json").write_text(json.dumps(card, indent=2))
    reg = load_registry(root)
    reg["versions"] = [v for v in reg["versions"] if v["version"] != version] + [
        {k: card[k] for k in ("name", "version", "bump", "parent", "created", "git_commit", "notes", "test_roc_auc")}]
    reg["versions"].sort(key=lambda v: parse(v["version"]))
    save_registry(root, reg)
    (root / "LATEST").write_text(vdir.name)
    return card


def resolve_artifacts(root: Path) -> Path:
    """Folder of the version to serve: COUGH_MODEL_VERSION if set, else LATEST, else the old flat layout."""
    root = Path(root)
    want = os.environ.get("COUGH_MODEL_VERSION")
    if want:
        hits = [v["name"] for v in load_registry(root)["versions"] if v["version"] == want.lstrip("v")]
        if not hits:
            raise FileNotFoundError(f"model version {want} not in {root}/registry.json")
        return root / hits[0]
    if (root / "LATEST").exists():
        d = root / (root / "LATEST").read_text().strip()
        if (d / "config.json").exists():
            return d
    return root


def version_info(art: Path) -> dict:
    p = Path(art) / "version.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"name": "unversioned (trained before versioning; run python -m cough.versioning --adopt-legacy)"}


def previous_dir(root: Path) -> Path | None:
    """Artifacts of the current version (to resume the CNN from), new or old layout."""
    d = resolve_artifacts(root)
    return d if (d / "config.json").exists() else None


def adopt_legacy(root: Path, version: str = "1.0.0", cache: Path | None = None) -> dict:
    """Move a flat artifacts/ folder (trained before versioning) into its first version folder."""
    root = Path(root)
    if not (root / "config.json").exists():
        raise FileNotFoundError(f"no flat artifacts in {root}")
    day = time.strftime("%Y%m%d", time.localtime((root / "config.json").stat().st_mtime))
    vdir = root / version_name(version, day)
    vdir.mkdir(parents=True, exist_ok=False)
    for p in list(root.iterdir()):
        if p.name in ("registry.json", "LATEST") or p == vdir or p.name.startswith(f"{PROJECT}_v"):
            continue
        shutil.move(str(p), str(vdir / p.name))
    return register(root, vdir, version, "major", cache, {}, notes="first trained model (adopted from flat artifacts/)")


def main():
    ap = argparse.ArgumentParser(description="model versions")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--adopt-legacy", action="store_true", help="move flat artifacts/ into coughvid_v1.0.0_...")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--use", help="serve this version: point LATEST at it (rollback)")
    a = ap.parse_args()
    root = Path(a.artifacts)
    if a.adopt_legacy:
        card = adopt_legacy(root, cache=Path(a.cache))
        print(f"adopted -> {card['name']}")
    if a.use:
        hits = [v["name"] for v in load_registry(root)["versions"] if v["version"] == a.use.lstrip("v")]
        if not hits:
            raise SystemExit(f"version {a.use} not found")
        (root / "LATEST").write_text(hits[0])
        print(f"LATEST -> {hits[0]}")
    reg = load_registry(root)
    cur = (root / "LATEST").read_text().strip() if (root / "LATEST").exists() else None
    for v in reg["versions"]:
        mark = "*" if v["name"] == cur else " "
        auc = (v.get("test_roc_auc") or {}).get("covid", {}).get("fusion")
        print(f"{mark} {v['name']:42s} {v['bump']:5s} commit {v.get('git_commit') or '-':8s} "
              f"covid AUC {auc if auc is not None else '-'}  {v.get('notes', '')}")
    if not reg["versions"]:
        print("no versions yet")


if __name__ == "__main__":
    main()
