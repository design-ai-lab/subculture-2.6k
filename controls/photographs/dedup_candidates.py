"""
Photograph control — near-duplicate removal among harvest candidates, BEFORE
human curation. Needs the candidate image files, which are not in the dataset
release; fetch them first with fetch_candidates_openverse.py.

Same method as the corpus near-duplicate audit: cosine similarity on CLIP ViT-L/14
embeddings, threshold 0.95, connected components collapsed keeping the first
filename (alphabetical) per cluster. Within-class only — the two classes never
mix. Removed files are logged to candidates_dedup_log.csv (kept vs removed per
cluster) and deleted from candidates/. Rerun make_review_sheet.py afterwards.

Rationale: photographers upload bursts of near-identical shots; keeping them
would inflate n and violate independence in the photograph score comparison.
"""

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


import csv
from pathlib import Path

import numpy as np

from score_reference_set import embed_folder, l2, pick_device

HERE = Path(__file__).resolve().parent
TAU = 0.95


def main() -> None:
    import sys
    root = HERE / (sys.argv[1] if len(sys.argv) > 1 else "candidates")
    device = pick_device()
    names, classes, feats = embed_folder(root, device)
    x = l2(feats)
    names = np.array(names)
    classes = np.array(classes)

    removed, log_rows = set(), []
    for cls in np.unique(classes):
        m = classes == cls
        sub_names = names[m]
        order = np.argsort(sub_names)          # deterministic: keep alphabetical first
        sub_names = sub_names[order]
        sim = x[m][order] @ x[m][order].T
        n = len(sub_names)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= TAU:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            clusters.setdefault(find(i), []).append(i)
        for _croot, members in clusters.items():
            if len(members) < 2:
                continue
            keep, rest = members[0], members[1:]
            for r in rest:
                removed.add((cls, sub_names[r]))
                log_rows.append({"class": cls, "kept": sub_names[keep],
                                 "removed": sub_names[r],
                                 "cosine": round(float(sim[keep, r]), 4)})

    for cls, fn in sorted(removed):
        (root / cls / fn).unlink(missing_ok=True)

    with open(HERE / "candidates_dedup_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "kept", "removed", "cosine"])
        w.writeheader()
        w.writerows(sorted(log_rows, key=lambda r: (r["class"], r["kept"])))

    per_class = {}
    for cls, _ in removed:
        per_class[cls] = per_class.get(cls, 0) + 1
    print(f"[done] removed {len(removed)} near-duplicates at tau={TAU}: {per_class}")
    print("[done] candidates_dedup_log.csv written — rerun make_review_sheet.py next")


if __name__ == "__main__":
    main()
