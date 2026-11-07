"""
Cross-subculture color analysis for SubCulture-2.6K.

Takes per-image 5-color palettes (K-means in LAB, already extracted) for 6
subcultures and answers three questions:

    Q1. How different are the subcultures' color distributions?
         - 6x6 Earth Mover's Distance matrix in LAB space
         - 6x6 centroid Euclidean distance matrix in LAB (diagnostic)
    Q2. Are subcultures computationally distinguishable from color alone?
         - Logistic regression + random forest on a 20-dim per-image feature
           vector, stratified 5-fold CV
         - Accuracy + 6x6 confusion matrices, compared to majority baseline
    Q3. What do the aggregate palettes look like?
         - Per-subculture average palette visualization

Inputs:  data/color_analysis/{subculture}.json (6 files)
Outputs: output/*.csv, *.txt and output/figures/*.png

Run from the repo root:
    python3 results/separability/cross_subculture_color_analysis.py
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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wasserstein_distance_nd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SUBCULTURES = ["cottagecore", "cyberpunk", "dark_academia", "goblincore", "synthwave", "y2k"]

REPO_ROOT = DATASET
COLOR_DIR = REPO_ROOT / "data" / "color_analysis"
RESULTS_DIR = OUTPUT
FIG_DIR = RESULTS_DIR / "figures"

RANDOM_SEED = 42
EMD_SUMMARY_K = 50  # each subculture summarized as a 50-color weighted palette before EMD


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_palettes(color_dir: Path) -> dict[str, list[dict]]:
    """Return {subculture: [{filename, colors:[{rgb, percentage, ...}, ...]}, ...]}"""
    palettes = {}
    for sub in SUBCULTURES:
        path = color_dir / f"{sub}.json"
        with open(path) as f:
            data = json.load(f)
        entries = data["image_analysis"]
        # Defensive: each entry must have 5 colors; if fewer, skip (shouldn't happen)
        entries = [e for e in entries if len(e.get("colors", [])) == 5]
        palettes[sub] = entries
        print(f"  {sub}: {len(entries)} images")
    return palettes


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #

def image_to_feature_vector(entry: dict) -> np.ndarray:
    """20-dim vector: [R1,G1,B1,pct1, ..., R5,G5,B5,pct5], sorted by pct desc."""
    colors = sorted(entry["colors"], key=lambda c: -c["percentage"])
    feat = []
    for c in colors:
        r, g, b = c["rgb"]
        feat.extend([r, g, b, c["percentage"]])
    return np.asarray(feat, dtype=np.float32)


def build_feature_matrix(palettes: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack all images' 20-dim vectors; return (X, y, filenames)."""
    X, y, fnames = [], [], []
    for sub_idx, sub in enumerate(SUBCULTURES):
        for entry in palettes[sub]:
            X.append(image_to_feature_vector(entry))
            y.append(sub_idx)
            fnames.append(f"{sub}/{entry['filename']}")
    return np.vstack(X), np.asarray(y), fnames


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert Nx3 RGB (0-255) to Nx3 LAB via OpenCV (D65 illuminant, sRGB)."""
    rgb_img = rgb.reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    return lab.reshape(-1, 3).astype(np.float32)


def pooled_lab_colors(palettes: dict, sub: str) -> tuple[np.ndarray, np.ndarray]:
    """Pool all (color, weight) pairs across all images of one subculture.

    Returns:
        lab_colors: (N, 3) in LAB
        weights:    (N,) summing to 1.0, proportional to pixel percentage
    """
    rgbs, weights = [], []
    for entry in palettes[sub]:
        for c in entry["colors"]:
            rgbs.append(c["rgb"])
            weights.append(c["percentage"])
    rgbs = np.asarray(rgbs, dtype=np.uint8)
    lab = rgb_to_lab(rgbs)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    return lab, weights


# --------------------------------------------------------------------------- #
# Q1: Distance matrices
# --------------------------------------------------------------------------- #

def summarize_palette(lab: np.ndarray, weights: np.ndarray, k: int = EMD_SUMMARY_K
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Compress pooled weighted LAB samples to k centroids with summed weights.

    Standard preprocessing in the color-transfer literature: summarize each
    distribution as a k-color palette so EMD becomes a k-vs-k transport problem
    (orders of magnitude faster than sample-vs-sample). K-means is run with
    sample weights so high-percentage colors dominate cluster locations.
    """
    k = min(k, len(lab))
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    km.fit(lab, sample_weight=weights)
    centers = km.cluster_centers_
    cluster_w = np.zeros(k, dtype=np.float64)
    for label, w in zip(km.labels_, weights):
        cluster_w[label] += w
    cluster_w /= cluster_w.sum()
    return centers.astype(np.float64), cluster_w


def compute_emd_matrix(palettes: dict) -> pd.DataFrame:
    """Multi-dim Wasserstein distance between summarized LAB color palettes.

    Each subculture's pooled per-image palette colors are first clustered
    (weighted K-means, k=50) into a 50-color summary palette in LAB. EMD is
    then computed between pairs of 50-color palettes via
    scipy.stats.wasserstein_distance_nd (SciPy >= 1.13).
    """
    summaries = {}
    for sub in SUBCULTURES:
        lab, w = pooled_lab_colors(palettes, sub)
        summaries[sub] = summarize_palette(lab, w, k=EMD_SUMMARY_K)
    n = len(SUBCULTURES)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            (u, u_w), (v, v_w) = summaries[SUBCULTURES[i]], summaries[SUBCULTURES[j]]
            d = wasserstein_distance_nd(u, v, u_weights=u_w, v_weights=v_w)
            M[i, j] = M[j, i] = d
    return pd.DataFrame(M, index=SUBCULTURES, columns=SUBCULTURES)


def compute_centroid_distance_matrix(palettes: dict) -> pd.DataFrame:
    """Euclidean distance between weighted LAB centroids. Diagnostic baseline."""
    centroids = {}
    for sub in SUBCULTURES:
        lab, w = pooled_lab_colors(palettes, sub)
        centroids[sub] = (lab * w[:, None]).sum(axis=0)
    n = len(SUBCULTURES)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = np.linalg.norm(centroids[SUBCULTURES[i]] - centroids[SUBCULTURES[j]])
    return pd.DataFrame(M, index=SUBCULTURES, columns=SUBCULTURES)


# --------------------------------------------------------------------------- #
# Q2: Classification experiments
# --------------------------------------------------------------------------- #

def run_classifier(name: str, clf, X: np.ndarray, y: np.ndarray) -> dict:
    """Stratified 5-fold CV; return aggregated accuracy + confusion matrix."""
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    y_true_all, y_pred_all, fold_accs = [], [], []
    for train_idx, test_idx in kf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        fold_accs.append(accuracy_score(y[test_idx], y_pred))
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(y_pred.tolist())
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(SUBCULTURES))))
    return {
        "name": name,
        "mean_accuracy": float(np.mean(fold_accs)),
        "std_accuracy": float(np.std(fold_accs)),
        "fold_accuracies": fold_accs,
        "confusion_matrix": cm,
    }


def majority_baseline(y: np.ndarray) -> float:
    counts = np.bincount(y)
    return counts.max() / counts.sum()


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def plot_heatmap(df: pd.DataFrame, title: str, outpath: Path, fmt: str = ".2f") -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(df, annot=True, fmt=fmt, cmap="viridis", square=True, cbar_kws={"label": "distance"})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_confusion(cm: np.ndarray, title: str, outpath: Path) -> None:
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", square=True,
                xticklabels=SUBCULTURES, yticklabels=SUBCULTURES,
                cbar_kws={"label": "row-normalized count"})
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_subculture_palettes(palettes: dict, outpath: Path, k: int = 8) -> None:
    """For each subculture, show top-k most-weighted LAB-pooled colors as a strip."""
    fig, axes = plt.subplots(len(SUBCULTURES), 1, figsize=(10, 1.1 * len(SUBCULTURES)))
    for ax, sub in zip(axes, SUBCULTURES):
        # Aggregate RGB with their pixel-percentage weights
        rgbs, weights = [], []
        for entry in palettes[sub]:
            for c in entry["colors"]:
                rgbs.append(c["rgb"])
                weights.append(c["percentage"])
        rgbs = np.asarray(rgbs, dtype=np.float32) / 255.0
        weights = np.asarray(weights, dtype=np.float64)

        # Simple binning: k-means in RGB on weighted samples would be cleaner,
        # but a quantized histogram over the existing palette colors is enough
        # for a sanity-check figure. Group by hex bin.
        hex_keys = (np.round(rgbs * 15).astype(int) * 17).clip(0, 255)  # ~4-bit per channel
        df = pd.DataFrame({"r": hex_keys[:, 0], "g": hex_keys[:, 1], "b": hex_keys[:, 2], "w": weights})
        agg = df.groupby(["r", "g", "b"])["w"].sum().reset_index().sort_values("w", ascending=False).head(k)
        top_rgb = agg[["r", "g", "b"]].to_numpy() / 255.0
        top_w = agg["w"].to_numpy()
        top_w = top_w / top_w.sum()

        # Draw weighted strip
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xticks([])
        x = 0.0
        for rgb, w in zip(top_rgb, top_w):
            ax.add_patch(plt.Rectangle((x, 0), w, 1, color=rgb))
            x += w
        ax.set_ylabel(sub, rotation=0, ha="right", va="center", fontsize=10)
    plt.suptitle("Weighted dominant-color palette per subculture (top-k binned)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading palettes...")
    palettes = load_palettes(COLOR_DIR)

    print("\nBuilding per-image 20-dim feature matrix...")
    X, y, _ = build_feature_matrix(palettes)
    print(f"  X shape = {X.shape}, classes = {len(SUBCULTURES)}")
    baseline = majority_baseline(y)
    print(f"  majority-class baseline accuracy = {baseline:.4f}")

    # --- Q1: distance matrices ---
    print("\nQ1: pairwise EMD (Wasserstein_nd) in LAB space...")
    emd_df = compute_emd_matrix(palettes)
    emd_df.to_csv(RESULTS_DIR / "emd_distance_matrix.csv")
    plot_heatmap(emd_df, "Pairwise EMD in LAB (Wasserstein_nd)", FIG_DIR / "emd_heatmap.png")
    print(emd_df.round(2).to_string())

    print("\nQ1b: centroid Euclidean distance in LAB (diagnostic)...")
    cen_df = compute_centroid_distance_matrix(palettes)
    cen_df.to_csv(RESULTS_DIR / "centroid_distance_matrix.csv")
    plot_heatmap(cen_df, "LAB centroid Euclidean distance", FIG_DIR / "centroid_distance_heatmap.png")

    # --- Q2: classification ---
    print("\nQ2: classification (stratified 5-fold CV)...")
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
    )
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)

    results = [run_classifier("LogisticRegression", logreg, X, y),
               run_classifier("RandomForest", rf, X, y)]

    report_path = RESULTS_DIR / "color_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Majority-class baseline accuracy: {baseline:.4f}\n")
        f.write(f"Chance (uniform) accuracy: {1 / len(SUBCULTURES):.4f}\n\n")
        for r in results:
            line = (f"{r['name']}: mean_acc = {r['mean_accuracy']:.4f} "
                    f"(+- {r['std_accuracy']:.4f}), folds = "
                    f"{[round(a, 4) for a in r['fold_accuracies']]}\n")
            f.write(line)
            print("  " + line.strip())
            cm_df = pd.DataFrame(r["confusion_matrix"], index=SUBCULTURES, columns=SUBCULTURES)
            cm_df.to_csv(RESULTS_DIR / f"confusion_matrix_{r['name'].lower()}.csv")
            plot_confusion(r["confusion_matrix"],
                           f"{r['name']} — color features, 5-fold CV",
                           FIG_DIR / f"color_confusion_{r['name'].lower()}.png")

    # --- Q3: palette visualization ---
    print("\nQ3: per-subculture palette visualization...")
    plot_subculture_palettes(palettes, FIG_DIR / "subculture_palettes.png")

    print(f"\nDone. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
