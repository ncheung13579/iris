"""
Run all statistical validation analyses on cached data.

This script loads pre-computed activations and SAE features from
checkpoints/ and runs three analyses that strengthen the C3 results:

  1. 5-fold stratified cross-validation (mean ± std for all metrics)
  2. Per-category detection breakdown (detection rate by injection type)
  3. Confidence calibration (reliability diagram + Brier/ECE)

Usage:
    python scripts/run_statistical_analysis.py

All results are saved to results/metrics/ (JSON) and results/figures/ (PNG).
No GPU required — runs in under 60 seconds on CPU.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Enable imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import set_seed
from src.data.dataset import IrisDataset
from src.analysis.statistical import (
    run_cross_validation,
    run_per_category_breakdown,
    run_calibration_analysis,
    plot_cv_results,
    plot_per_category_heatmap,
    plot_calibration_diagram,
)

set_seed(42)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "iris_dataset_balanced.json"
ACTIVATIONS_PATH = PROJECT_ROOT / "checkpoints" / "j2_activations.npz"
FEATURES_PATH = PROJECT_ROOT / "checkpoints" / "feature_matrix.npy"
SENSITIVITY_PATH = PROJECT_ROOT / "checkpoints" / "sensitivity_scores.npy"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ---- Load cached data ----
    print("Loading cached data...")
    dataset = IrisDataset.load(DATA_PATH)
    texts = dataset.texts
    labels = np.array(dataset.labels)
    categories = [ex["category"] for ex in dataset.examples]

    act_data = np.load(ACTIVATIONS_PATH)
    activations = act_data["layer_0"]  # layer 0 matches the SAE
    features = np.load(FEATURES_PATH)
    sensitivity = np.load(SENSITIVITY_PATH)

    print(f"  Dataset: {len(texts)} examples")
    print(f"  Activations: {activations.shape}")
    print(f"  SAE features: {features.shape}")
    print(f"  Sensitivity scores: {sensitivity.shape}")

    # ---- 1. Cross-validation ----
    print("\n" + "=" * 60)
    print("ANALYSIS 1: 5-Fold Stratified Cross-Validation")
    print("=" * 60)

    cv_results = run_cross_validation(
        texts=texts,
        labels=labels,
        activations=activations,
        features=features,
        sensitivity_scores=sensitivity,
        n_folds=5,
        seed=42,
        top_k_values=[10, 50, 100],
    )

    # Save results
    # Convert numpy types for JSON serialization
    cv_save = {}
    for name, data in cv_results.items():
        cv_save[name] = {
            "mean": data["mean"],
            "std": data["std"],
            "ci_95_lower": data["ci_95_lower"],
            "ci_95_upper": data["ci_95_upper"],
            "per_fold": data["per_fold"],
        }

    cv_path = METRICS_DIR / "c3_cross_validation.json"
    cv_path.write_text(json.dumps(cv_save, indent=2))
    print(f"Saved CV results to {cv_path}")

    # Plot
    plot_cv_results(cv_results, save_path=str(FIGURES_DIR / "c3_cv_results.png"))

    # ---- 2. Per-category breakdown ----
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Per-Category Detection Breakdown")
    print("=" * 60)

    cat_results = run_per_category_breakdown(
        texts=texts,
        labels=labels,
        categories=categories,
        activations=activations,
        features=features,
        seed=42,
    )

    cat_path = METRICS_DIR / "c3_per_category.json"
    cat_path.write_text(json.dumps(cat_results, indent=2))
    print(f"Saved per-category results to {cat_path}")

    # Plot
    plot_per_category_heatmap(
        cat_results, save_path=str(FIGURES_DIR / "c3_per_category_heatmap.png")
    )

    # ---- 3. Confidence calibration ----
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Confidence Calibration")
    print("=" * 60)

    cal_results = run_calibration_analysis(
        texts=texts,
        labels=labels,
        activations=activations,
        features=features,
        seed=42,
    )

    cal_path = METRICS_DIR / "c3_calibration.json"
    # Convert for JSON serialization
    cal_save = {}
    for name, data in cal_results.items():
        cal_save[name] = {
            "brier_score": data["brier_score"],
            "ece": data["ece"],
            "bin_counts": data["bin_counts"],
        }
    cal_path.write_text(json.dumps(cal_save, indent=2))
    print(f"Saved calibration results to {cal_path}")

    # Plot
    plot_calibration_diagram(
        cal_results, save_path=str(FIGURES_DIR / "c3_calibration_diagram.png")
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)
    print(f"\nMetrics saved to:  {METRICS_DIR}/")
    print(f"Figures saved to:  {FIGURES_DIR}/")
    print("\nKey results:")
    for name in ["TF-IDF + LogReg", "Raw Activation + LogReg", "SAE Features (all) + LogReg"]:
        r = cv_results[name]
        print(f"  {name}:")
        print(f"    F1  = {r['mean']['f1']:.3f} +/- {r['std']['f1']:.3f} "
              f"[{r['ci_95_lower']['f1']:.3f}, {r['ci_95_upper']['f1']:.3f}]")
        print(f"    AUC = {r['mean']['roc_auc']:.3f} +/- {r['std']['roc_auc']:.3f} "
              f"[{r['ci_95_lower']['roc_auc']:.3f}, {r['ci_95_upper']['roc_auc']:.3f}]")


if __name__ == "__main__":
    main()
