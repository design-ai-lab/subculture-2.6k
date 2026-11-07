"""Run manifests for the matched-text analyses: inputs, environment, outputs."""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in _Path(__file__).resolve().parents:
    if (_p / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from paths import DATASET, CODE_ROOT, OUTPUT, under
for _d in (
    CODE_ROOT / "results" / "laion_profile",
    CODE_ROOT / "results" / "separability",
    CODE_ROOT / "results" / "matched_text",
    CODE_ROOT / "dataset",
    CODE_ROOT / "scoring",
):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))


import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = DATASET
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return under(REPO_ROOT, path)


def file_record(path: Path) -> dict:
    record = {"path": rel(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    if path.suffix == ".csv":
        with open(path, "rb") as f:
            record["data_rows"] = sum(1 for _ in f) - 1
    return record


def write_manifest(out_path: Path, *, script: Path, purpose: str, inputs: list[Path],
                   outputs: list[Path], parameters: dict, checks: dict) -> None:
    now = datetime.now(timezone.utc)
    manifest = {
        "purpose": purpose,
        "script": file_record(Path(script)),
        "command": f"python3 -B {rel(Path(script))}",
        "run_utc": now.isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "packages": {p: metadata.version(p) for p in PACKAGES},
        },
        "parameters": parameters,
        "inputs": [file_record(Path(p)) for p in inputs],
        "outputs": [file_record(Path(p)) for p in outputs],
        "checks": checks,
    }
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
