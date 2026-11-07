"""
CLIP-feature subculture separability (Section 5.1).

  - All six subcultures, stratified 5-fold CV.
  - Logistic regression + Random Forest classifiers.
  - Confusion matrix from RF (best classifier).
  - 2D UMAP scatter for qualitative geometry.
  - Comparison table: color-only vs CLIP vs color+CLIP combined.

Inputs:
  data/clip_embeddings.npy
  data/clip_metadata.csv
  data/color_analysis/*.json      (color features, for the combined model)

Outputs:
  output/clip_classification_report.txt
  output/confusion_matrix_clip_rf.csv
  output/feature_comparison_table.csv
  output/figures/clip_umap.png
  output/figures/clip_confusion_rf.png
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


import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = DATASET
EMB_PATH = REPO_ROOT / "data" / "clip_embeddings.npy"
META_PATH = REPO_ROOT / "data" / "clip_metadata.csv"
COLOR_DIR = REPO_ROOT / "data" / "color_analysis"

RESULTS_DIR = OUTPUT
FIG_DIR = RESULTS_DIR / "figures"
RPT_PATH = RESULTS_DIR / "clip_classification_report.txt"
CM_CSV = RESULTS_DIR / "confusion_matrix_clip_rf.csv"
CMP_CSV = RESULTS_DIR / "feature_comparison_table.csv"
UMAP_FIG = FIG_DIR / "clip_umap.png"
CM_FIG = FIG_DIR / "clip_confusion_rf.png"

SUBCULTURES = ["cottagecore", "cyberpunk", "dark_academia", "goblincore", "synthwave", "y2k"]
SEED = 42


def load_color_features(meta: pd.DataFrame) -> np.ndarray:
    """Build the 20-dim per-image color feature vector, aligned to
    the CLIP metadata row order. Missing images get NaN rows (we'll filter)."""
    by_filename: dict[tuple[str, str], list[float]] = {}
    for sub in SUBCULTURES:
        json_path = next(COLOR_DIR.glob(f"*{sub}*.json"), None)
        if json_path is None:
            json_path = COLOR_DIR / f"{sub}.json"
        if not json_path.exists():
            print(f"[warn] color json missing for {sub}: tried {json_path}")
            continue
        with json_path.open() as f:
            data = json.load(f)
        records = data.get("image_analysis", data) if isinstance(data, dict) else data
        for rec in records:
            colors = sorted(rec["colors"], key=lambda c: -c["percentage"])[:5]
            feat: list[float] = []
            for c in colors:
                r, g, b = c["rgb"]
                feat.extend([float(r), float(g), float(b), float(c["percentage"])])
            while len(feat) < 20:
                feat.extend([0.0, 0.0, 0.0, 0.0])
            by_filename[(sub, rec["filename"])] = feat

    rows = []
    missing = 0
    for _, m in meta.iterrows():
        key = (m["subculture"], m["filename"])
        if key in by_filename:
            rows.append(by_filename[key])
        else:
            rows.append([np.nan] * 20)
            missing += 1
    if missing:
        print(f"[warn] {missing} images had no color features (will drop from combined comparison)")
    return np.array(rows, dtype=np.float32)


def cv_score(X: np.ndarray, y: np.ndarray, clf, name: str) -> tuple[np.ndarray, float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"  {name:<28s}  acc = {scores.mean():.4f} ± {scores.std():.4f}   "
          f"folds = {[f'{s:.3f}' for s in scores]}")
    return scores, scores.mean(), scores.std()


def main() -> None:
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    emb = np.load(EMB_PATH).astype(np.float32)
    meta = pd.read_csv(META_PATH)
    y = meta["subculture"].values
    print(f"[info] CLIP embeddings: {emb.shape}   labels: {len(y)}")

    # L2-normalize CLIP for the probe (cosine-style geometry; standard for CLIP probes)
    emb_norm = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

    # --- 1. CLIP-only classification ---
    print("\n[1] CLIP-only classifiers (5-fold stratified CV)")
    clip_X = StandardScaler().fit_transform(emb_norm)
    clip_lr_scores, clip_lr_mean, clip_lr_std = cv_score(
        clip_X, y, LogisticRegression(max_iter=2000), "Logistic Regression")
    clip_rf_scores, clip_rf_mean, clip_rf_std = cv_score(
        emb_norm, y, RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "Random Forest")

    # --- 2. Confusion matrix from CLIP + RF (best CLIP classifier) ---
    print("\n[2] Confusion matrix (CLIP + Random Forest, out-of-fold predictions)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_oof = cross_val_predict(
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        emb_norm, y, cv=cv, n_jobs=-1)
    labels_sorted = sorted(np.unique(y))
    cm = confusion_matrix(y, rf_oof, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    cm_df.to_csv(CM_CSV)
    print(cm_df.to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CLIP + Random Forest — confusion matrix (5-fold OOF)")
    plt.xticks(rotation=20)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CM_FIG, dpi=150)
    plt.close()
    print(f"[done] wrote {CM_FIG}")

    # --- 3. Feature comparison: color vs CLIP vs combined ---
    print("\n[3] Feature comparison: color-only vs CLIP-only vs color+CLIP")
    color_feat = load_color_features(meta)
    valid = ~np.isnan(color_feat).any(axis=1)
    print(f"  Using {valid.sum()}/{len(meta)} images that have both feature sets")
    cf = color_feat[valid]
    ef = emb_norm[valid]
    yv = y[valid]

    color_X = StandardScaler().fit_transform(cf)
    clip_X_v = StandardScaler().fit_transform(ef)
    combo_X = np.hstack([color_X, clip_X_v])

    rows = []
    for name, X in [("color (20-d)", color_X), ("CLIP (768-d)", clip_X_v),
                    ("color+CLIP (788-d)", combo_X)]:
        for clf_name, clf in [
            ("LogReg", LogisticRegression(max_iter=2000)),
            ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        ]:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            sc = cross_val_score(clf, X, yv, cv=cv, scoring="accuracy")
            rows.append({
                "features": name, "classifier": clf_name,
                "accuracy_mean": sc.mean(), "accuracy_std": sc.std(),
                "fold_scores": ";".join(f"{s:.4f}" for s in sc),
            })
            print(f"  {name:<22s} {clf_name:<14s}  acc = {sc.mean():.4f} ± {sc.std():.4f}")
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(CMP_CSV, index=False)
    print(f"[done] wrote {CMP_CSV}")

    # --- 4. UMAP scatter ---
    print("\n[4] UMAP 2D projection of CLIP embeddings...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.15,
                       metric="cosine", random_state=SEED)
    proj = reducer.fit_transform(emb_norm)

    fig, ax = plt.subplots(figsize=(8, 7))
    palette = sns.color_palette("tab10", n_colors=len(labels_sorted))
    for color, sub in zip(palette, labels_sorted):
        mask = y == sub
        ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.6, label=sub, color=color)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("CLIP ViT-L/14 embeddings — UMAP (cosine, n=2,610)")
    ax.legend(markerscale=2, loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(UMAP_FIG, dpi=150)
    plt.close()
    print(f"[done] wrote {UMAP_FIG}")

    # --- 5. Report ---
    diag = np.diag(cm)
    per_class_recall = diag / cm.sum(axis=1)
    report = [
        "CLIP ViT-L/14 — subculture classification report",
        "=" * 64,
        f"N = {len(y)}   classes = {labels_sorted}",
        "",
        "CLIP-only classifiers (5-fold stratified CV):",
        f"  Logistic Regression : {clip_lr_mean:.4f} ± {clip_lr_std:.4f}",
        f"  Random Forest       : {clip_rf_mean:.4f} ± {clip_rf_std:.4f}",
        "",
        "Per-class recall (Random Forest, OOF):",
        *[f"  {lab:<14s}  {r:.3f}  ({d}/{cm[i].sum()})"
          for i, (lab, r, d) in enumerate(zip(labels_sorted, per_class_recall, diag))],
        "",
        "Comparison table (5-fold stratified CV, same images for all rows):",
        cmp_df.round(4).to_string(index=False),
    ]
    RPT_PATH.write_text("\n".join(report))
    print(f"[done] wrote {RPT_PATH}")


if __name__ == "__main__":
    main()
