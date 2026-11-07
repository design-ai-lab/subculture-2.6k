"""Locations for this code repository and a local copy of the dataset.

The dataset is not stored in this repository. Set SUBCULTURE_DATA to a download
of https://huggingface.co/datasets/DanLi26/subculture-2.6k, or leave it unset
when this folder sits beside the local export folder named dataset/.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent


def dataset_root() -> Path:
    env = os.environ.get("SUBCULTURE_DATA")
    if env:
        return Path(env).expanduser().resolve()
    sibling = CODE_ROOT.parent / "dataset"
    if (sibling / "croissant.json").is_file():
        return sibling.resolve()
    return (CODE_ROOT / "dataset").resolve()


DATASET = dataset_root()
OUTPUT = CODE_ROOT / "output"

LAION_HEAD_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/"
                  "raw/main/sac+logos+ava1-l14-linearMSE.pth")
LAION_HEAD = CODE_ROOT / "models" / "sac+logos+ava1-l14-linearMSE.pth"


def laion_head() -> Path:
    if not LAION_HEAD.is_file():
        raise FileNotFoundError(f"LAION-Aesthetics V2 head not found at {LAION_HEAD}. "
                                f"Download it from {LAION_HEAD_URL}")
    return LAION_HEAD


def under(root: Path, path: Path) -> str:
    path = Path(path).resolve()
    for base in (Path(root).resolve(), CODE_ROOT, DATASET):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name
