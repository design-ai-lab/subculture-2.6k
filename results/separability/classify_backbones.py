"""
Audit backbone — subculture classification in each peer scorer's embedding space.

Uses the same protocol as results/separability/clip_subculture_analysis.py:
L2-normalized embeddings; LogisticRegression(max_iter=2000) on standardized
features; RandomForest(n_estimators=300, seed 42) on the normalized embeddings;
StratifiedKFold(5, shuffle=True, random_state=42); accuracy + per-class recall
from RF out-of-fold predictions. No figures (audit is text-only).

Spaces evaluated (whichever exist when the script runs):
    vitl_paper_check   data/clip_embeddings.npy — must reproduce the paper's
                       92.4% (LR) / 92.7% (RF); validates this code path
    vith_pick          output/embeddings/vith_pick.npy — PickScore's ViT-H/14 tower
    blip_ir            output/embeddings/blip_ir.npy   — ImageReward's BLIP tower

Outputs: output/backbone_accuracy_table.csv, output/backbone_per_class_recall.csv,
         output/backbone_report.txt (with a markdown table)
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


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO_ROOT = DATASET
SEED = 42
PAPER_BASELINE = {"LogReg": 0.924, "RandomForest": 0.927}  # ViT-L/14, as reported in the paper

SPACES = {
    "vitl_paper_check": REPO_ROOT / "data" / "clip_embeddings.npy",
    "vith_pick":        OUTPUT / "embeddings" / "vith_pick.npy",
    "blip_ir":          OUTPUT / "embeddings" / "blip_ir.npy",
}


def evaluate_space(name: str, emb: np.ndarray, y: np.ndarray):
    emb = emb.astype(np.float32)
    emb_norm = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    lr_X = StandardScaler().fit_transform(emb_norm)
    lr_sc = cross_val_score(LogisticRegression(max_iter=2000), lr_X, y, cv=cv, scoring="accuracy")
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf_sc = cross_val_score(rf, emb_norm, y, cv=cv, scoring="accuracy")

    rf_oof = cross_val_predict(
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        emb_norm, y, cv=cv, n_jobs=-1)
    labels_sorted = sorted(np.unique(y))
    cm = confusion_matrix(y, rf_oof, labels=labels_sorted)
    recall = {lab: cm[i, i] / cm[i].sum() for i, lab in enumerate(labels_sorted)}

    print(f"  {name:<18s} dim={emb.shape[1]:<5d} "
          f"LR={lr_sc.mean():.4f}±{lr_sc.std():.4f}  RF={rf_sc.mean():.4f}±{rf_sc.std():.4f}")
    return lr_sc, rf_sc, recall


def main() -> None:
    np.random.seed(SEED)
    out_dir = OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(REPO_ROOT / "data" / "clip_metadata.csv")
    y = meta["subculture"].values

    acc_rows, recall_rows = [], []
    for name, path in SPACES.items():
        if not path.exists():
            print(f"[skip] {name}: {path} not found")
            continue
        emb = np.load(path)
        assert emb.shape[0] == len(y), f"{name}: {emb.shape[0]} rows vs {len(y)} labels"
        lr_sc, rf_sc, recall = evaluate_space(name, emb, y)
        acc_rows.append({"space": name, "dim": emb.shape[1],
                         "logreg_acc": lr_sc.mean(), "logreg_std": lr_sc.std(),
                         "rf_acc": rf_sc.mean(), "rf_std": rf_sc.std()})
        recall_rows.append({"space": name, **{k: round(v, 3) for k, v in recall.items()}})

    acc = pd.DataFrame(acc_rows).set_index("space")
    rec = pd.DataFrame(recall_rows).set_index("space")
    acc.to_csv(out_dir / "backbone_accuracy_table.csv")
    rec.to_csv(out_dir / "backbone_per_class_recall.csv")

    md = ["| space | dim | LogReg acc | RF acc |", "|---|---|---|---|"]
    for name, r in acc.iterrows():
        md.append(f"| {name} | {int(r['dim'])} | {r['logreg_acc']:.3f} ± {r['logreg_std']:.3f} "
                  f"| {r['rf_acc']:.3f} ± {r['rf_std']:.3f} |")

    check = ""
    if "vitl_paper_check" in acc.index:
        r = acc.loc["vitl_paper_check"]
        ok = abs(r["logreg_acc"] - PAPER_BASELINE["LogReg"]) < 0.01 and \
             abs(r["rf_acc"] - PAPER_BASELINE["RandomForest"]) < 0.01
        check = (f"ViT-L self-check vs paper (92.4/92.7): LR {r['logreg_acc']:.4f}, "
                 f"RF {r['rf_acc']:.4f} -> {'OK' if ok else 'MISMATCH — do not trust this run'}")
        print(check)

    report = "\n".join([
        "Per-backbone separability",
        "Protocol: clip_subculture_analysis.py (L2-norm, LR max_iter=2000 on",
        "standardized features, RF 300 trees, StratifiedKFold 5, seed 42).",
        check, "",
        "=== accuracy (5-fold stratified CV) ===", acc.round(4).to_string(), "",
        "=== per-class recall (RF, out-of-fold) ===", rec.to_string(), "",
        "=== markdown table ===", *md, "",
    ])
    (out_dir / "backbone_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
