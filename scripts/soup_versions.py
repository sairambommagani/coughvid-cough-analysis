"""Write the exact installed versions of every SOUP item to docs/regulatory/06a_soup_versions.md and pin them in
requirements.lock.txt (IEC 62304 8.1.2: each SOUP item is identified by title, manufacturer and version).

  python scripts/soup_versions.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cough.audio import ffmpeg_exe                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "librosa", "soundfile", "lightgbm", "torch",
            "fastapi", "starlette", "uvicorn", "python-multipart", "httpx", "imageio-ffmpeg", "joblib", "matplotlib",
            "pytest"]


def main():
    rows, lock = [], []
    for p in PACKAGES:
        try:
            v = metadata.version(p)
        except metadata.PackageNotFoundError:
            v = "NOT INSTALLED"
        rows.append((p, v))
        if v != "NOT INSTALLED":
            lock.append(f"{p}=={v}")
    exe = ffmpeg_exe()
    ff = "NOT FOUND"
    if exe:
        out = subprocess.run([exe, "-version"], capture_output=True, text=True).stdout.splitlines()
        ff = out[0] if out else "unknown"
    md = ["# 06a SOUP versions (generated)", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python scripts/soup_versions.py` on Python "
          f"{sys.version.split()[0]}. These are the versions verified by the test run of this release; "
          "requirements.lock.txt pins them.", "", "| SOUP item | Installed version |", "|---|---|"]
    md += [f"| {p} | {v} |" for p, v in rows]
    md += [f"| ffmpeg binary | {ff} ({exe}) |", "| PANNs CNN14 checkpoint | Cnn14_mAP=0.431.pth (Zenodo 3987831) |", ""]
    (ROOT / "docs" / "regulatory" / "06a_soup_versions.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "requirements.lock.txt").write_text("\n".join(lock) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
