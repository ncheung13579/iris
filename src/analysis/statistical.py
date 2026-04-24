"""
Statistical validation suite for the C3 detection experiment.

Adds three analyses that strengthen the single-split C3 results:

 1. Stratified k-fold cross-validation — replaces the single 70/30 split
 with k independent train/test folds, yielding mean ± std metrics and
 95% confidence intervals.

 2. Per-category detection breakdown — evaluates each injection category
 (override, extraction, roleplay, indirect, mixed) separately to
 reveal whether the detector has blind spots.

 3. Confidence calibration — reliability diagram, Brier score, and
 Expected Calibration Error (ECE) to check whether the detector's
 predicted probabilities are well-calibrated.

All functions operate on pre-computed numpy arrays (cached in checkpoints/)
so they run on CPU in seconds without needing GPT-2 or the SAE loaded.

Author: Nathan Cheung
York University | CSSD 2221 | Winter 2026
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
 brier_score_loss,
 f1_score,
 precision_score,
 recall_score,
 roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.baseline.classifiers import (
 train_activation_baseline,
 train_sae_feature_baseline,
 train_tfidf_baseline,
)


# Colorblind-friendly palette (consistent with detection.py)
_CB_PALETTE = [
 "#0072B2", # blue
 "#D55E00", # vermillion
 "#009E73", # bluish green
 "#E69F00", # orange
 "#56B4E9", # sky blue
 "#CC79A7", # reddish purple
 "#F0E442", # yellow
]


# -----------------------------------------------------------------------
# 1. Cross-validation
# -----------------------------------------------------------------------


def run_cross_validation(
 texts: List[str],
 labels: np.ndarray,
 activations: np.ndarray,
 features: np.ndarray,
 sensitivity_scores: np.ndarray,
 n_folds: int = 5,
 seed: int = 42,
 top_k_values: Optional[List[int]] = None,
) -> Dict[str, Dict[str, Any]]:
 """Run stratified k-fold cross-validation on all detection approaches.

 Trains and evaluates TF-IDF, raw activation, and SAE feature detectors
 on each fold independently. Reports mean, std, and 95% CI for each
 metric across folds.

 Args:
 texts: All prompt strings (N total).
 labels: All labels, shape (N,).
 activations: Raw residual-stream activations, shape (N, 768).
 features: SAE feature activations, shape (N, 6144).
 sensitivity_scores: Per-feature sensitivity, shape (6144,).
 n_folds: Number of CV folds (default 5).
 seed: Random seed.
 top_k_values: Optional list of K values for top-K ablation.

 Returns:
 Dict mapping approach name to {
 "mean": {metric: float},
 "std": {metric: float},
 "ci_95_lower": {metric: float},
 "ci_95_upper": {metric: float},
 "per_fold": [{metric: float}, ...]
 }
 """
 labels = np.asarray(labels)
 skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

 # Define approaches: (name, train_fn, get_train_data, get_test_data)
 # We build this list dynamically so top-K ablation is included
 abs_sens = np.abs(sensitivity_scores)
 ranked_indices = np.argsort(abs_sens)[::-1]

 approaches: List[Tuple[str, Any]] = [
 ("TF-IDF + LogReg", "tfidf"),
 ("Raw Activation + LogReg", "activation"),
 ("SAE Features (all) + LogReg", "sae_all"),
 ]
 if top_k_values:
 for k in top_k_values:
 approaches.append((f"SAE Top-{k} Features + LogReg", f"sae_top_{k}"))

 # Collect per-fold metrics
 fold_results: Dict[str, List[Dict[str, float]]] = {
 name: [] for name, _ in approaches
 }

 metrics_names = ["precision", "recall", "f1", "accuracy", "roc_auc"]

 for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
 print(f" Fold {fold_idx + 1}/{n_folds}...")

 train_texts = [texts[i] for i in train_idx]
 test_texts = [texts[i] for i in test_idx]
 train_labels = labels[train_idx]
 test_labels = labels[test_idx]
 train_act = activations[train_idx]
 test_act = activations[test_idx]
 train_feat = features[train_idx]
 test_feat = features[test_idx]

 for name, approach_type in approaches:
 if approach_type == "tfidf":
 lr_pipe, _ = train_tfidf_baseline(train_texts, train_labels, seed=seed)
 y_pred = lr_pipe.predict(test_texts)
 y_prob = lr_pipe.predict_proba(test_texts)[:, 1]
 elif approach_type == "activation":
 clf = train_activation_baseline(train_act, train_labels, seed=seed)
 y_pred = clf.predict(test_act)
 y_prob = clf.predict_proba(test_act)[:, 1]
 elif approach_type == "sae_all":
 clf = train_sae_feature_baseline(train_feat, train_labels, seed=seed)
 y_pred = clf.predict(test_feat)
 y_prob = clf.predict_proba(test_feat)[:, 1]
 elif approach_type.startswith("sae_top_"):
 k = int(approach_type.split("_")[-1])
 top_k_idx = ranked_indices[:k]
 clf = train_sae_feature_baseline(
 train_feat[:, top_k_idx], train_labels, seed=seed
 )
 y_pred = clf.predict(test_feat[:, top_k_idx])
 y_prob = clf.predict_proba(test_feat[:, top_k_idx])[:, 1]
 else:
 continue

 fold_metrics = {
 "precision": float(precision_score(test_labels, y_pred, zero_division=0)),
 "recall": float(recall_score(test_labels, y_pred, zero_division=0)),
 "f1": float(f1_score(test_labels, y_pred, zero_division=0)),
 "accuracy": float(np.mean(y_pred == test_labels)),
 "roc_auc": float(roc_auc_score(test_labels, y_prob)),
 }
 fold_results[name].append(fold_metrics)

 # Aggregate across folds
 cv_results: Dict[str, Dict[str, Any]] = {}
 for name, _ in approaches:
 folds = fold_results[name]
 agg: Dict[str, Any] = {"per_fold": folds, "mean": {}, "std": {}, "ci_95_lower": {}, "ci_95_upper": {}}

 for metric in metrics_names:
 values = np.array([f[metric] for f in folds])
 mean = float(values.mean())
 std = float(values.std(ddof=1))
 # 95% CI using t-distribution approximation (for small n)
 ci_half = 1.96 * std / np.sqrt(n_folds)
 agg["mean"][metric] = mean
 agg["std"][metric] = std
 agg["ci_95_lower"][metric] = mean - ci_half
 agg["ci_95_upper"][metric] = mean + ci_half

 cv_results[name] = agg

 # Print summary table
 print(f"\n{'='*80}")
 print(f"Cross-Validation Results ({n_folds}-fold)")
 print(f"{'='*80}")
 name_w = max(len(n) for n, _ in approaches)
 print(f"{'Approach'.ljust(name_w)} {'F1':>12} {'AUC':>12} {'Precision':>12} {'Recall':>12}")
 print("-" * (name_w + 54))
 for name, _ in approaches:
 r = cv_results[name]
 f1_str = f"{r['mean']['f1']:.3f}+/-{r['std']['f1']:.3f}"
 auc_str = f"{r['mean']['roc_auc']:.3f}+/-{r['std']['roc_auc']:.3f}"
 prec_str = f"{r['mean']['precision']:.3f}+/-{r['std']['precision']:.3f}"
 rec_str = f"{r['mean']['recall']:.3f}+/-{r['std']['recall']:.3f}"
 print(f"{name.ljust(name_w)} {f1_str:>12} {auc_str:>12} {prec_str:>12} {rec_str:>12}")
 print()

 return cv_results


# -----------------------------------------------------------------------
# 2. Per-category detection breakdown
# -----------------------------------------------------------------------


def run_per_category_breakdown(
 texts: List[str],
 labels: np.ndarray,
 categories: List[str],
 activations: np.ndarray,
 features: np.ndarray,
 train_ratio: float = 0.7,
 seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, float]]]:
 """Evaluate detection performance separately for each injection category.

 Trains detectors on the standard train split, then evaluates on the
 test split broken down by category. This reveals whether the detector
 has blind spots for specific injection types.

 Args:
 texts: All prompt strings (N total).
 labels: All labels, shape (N,).
 categories: Category string for each example.
 activations: Raw activations, shape (N, 768).
 features: SAE features, shape (N, 6144).
 train_ratio: Fraction used for training.
 seed: Random seed.

 Returns:
 Dict mapping category name to {approach_name: {metric: value}}.
 """
 labels = np.asarray(labels)
 n = len(labels)

 # Create stratified train/test split
 from sklearn.model_selection import train_test_split
 indices = np.arange(n)
 train_idx, test_idx = train_test_split(
 indices, train_size=train_ratio, stratify=labels, random_state=seed
 )

 train_texts = [texts[i] for i in train_idx]
 test_texts = [texts[i] for i in test_idx]
 train_labels = labels[train_idx]
 test_labels = labels[test_idx]
 train_act = activations[train_idx]
 test_act = activations[test_idx]
 train_feat = features[train_idx]
 test_feat = features[test_idx]
 test_cats = [categories[i] for i in test_idx]

 # Train all detectors on the training set
 lr_pipe, _ = train_tfidf_baseline(train_texts, train_labels, seed=seed)
 act_clf = train_activation_baseline(train_act, train_labels, seed=seed)
 sae_clf = train_sae_feature_baseline(train_feat, train_labels, seed=seed)

 detectors = {
 "TF-IDF + LogReg": (lr_pipe, test_texts),
 "Raw Activation + LogReg": (act_clf, test_act),
 "SAE Features (all) + LogReg": (sae_clf, test_feat),
 }

 # Get unique injection categories (skip "instruction" which is all normal)
 unique_cats = sorted(set(test_cats))

 # Evaluate per category
 category_results: Dict[str, Dict[str, Dict[str, float]]] = {}

 for cat in unique_cats:
 cat_mask = np.array([c == cat for c in test_cats])
 cat_labels = test_labels[cat_mask]

 # Skip categories with only one class (can't compute AUC)
 if len(set(cat_labels)) < 2:
 # For single-class categories, compute what we can
 cat_results: Dict[str, Dict[str, float]] = {}
 for det_name, (clf, X_test) in detectors.items():
 if isinstance(X_test, list):
 X_cat = [X_test[i] for i, m in enumerate(cat_mask) if m]
 else:
 X_cat = X_test[cat_mask]
 y_pred = clf.predict(X_cat)

 # For injection-only categories, recall = detection rate
 if cat_labels[0] == 1:
 detection_rate = float(np.mean(y_pred == 1))
 cat_results[det_name] = {
 "detection_rate": detection_rate,
 "n_samples": int(cat_mask.sum()),
 "class": "injection",
 }
 else:
 fp_rate = float(np.mean(y_pred == 1))
 cat_results[det_name] = {
 "false_positive_rate": fp_rate,
 "n_samples": int(cat_mask.sum()),
 "class": "normal",
 }
 category_results[cat] = cat_results
 else:
 cat_results = {}
 for det_name, (clf, X_test) in detectors.items():
 if isinstance(X_test, list):
 X_cat = [X_test[i] for i, m in enumerate(cat_mask) if m]
 else:
 X_cat = X_test[cat_mask]
 y_pred = clf.predict(X_cat)
 y_prob = clf.predict_proba(X_cat)[:, 1]

 cat_results[det_name] = {
 "f1": float(f1_score(cat_labels, y_pred, zero_division=0)),
 "precision": float(precision_score(cat_labels, y_pred, zero_division=0)),
 "recall": float(recall_score(cat_labels, y_pred, zero_division=0)),
 "roc_auc": float(roc_auc_score(cat_labels, y_prob)),
 "n_samples": int(cat_mask.sum()),
 }
 category_results[cat] = cat_results

 # Print summary
 print(f"\n{'='*70}")
 print("Per-Category Detection Breakdown")
 print(f"{'='*70}")
 for cat in unique_cats:
 cat_res = category_results[cat]
 first_det = list(cat_res.values())[0]
 n = first_det["n_samples"]
 cls = first_det.get("class", "mixed")
 print(f"\n {cat} (n={n}, class={cls}):")
 for det_name, metrics in cat_res.items():
 if "detection_rate" in metrics:
 print(f" {det_name}: detection_rate={metrics['detection_rate']:.3f}")
 elif "false_positive_rate" in metrics:
 print(f" {det_name}: FP_rate={metrics['false_positive_rate']:.3f}")
 else:
 print(f" {det_name}: F1={metrics['f1']:.3f}, "
 f"AUC={metrics['roc_auc']:.3f}, "
 f"recall={metrics['recall']:.3f}")
 print()

 return category_results


# -----------------------------------------------------------------------
# 3. Confidence calibration
# -----------------------------------------------------------------------


def compute_calibration(
 y_true: np.ndarray,
 y_prob: np.ndarray,
 n_bins: int = 10,
) -> Dict[str, Any]:
 """Compute calibration metrics for a binary classifier.

 A well-calibrated classifier's predicted probability should match
 the observed frequency. If the model says "80% chance of injection",
 roughly 80% of those prompts should actually be injections.

 Args:
 y_true: True labels, shape (N,).
 y_prob: Predicted probabilities for class 1, shape (N,).
 n_bins: Number of bins for the reliability diagram.

 Returns:
 Dict with keys: brier_score, ece, bin_edges, bin_accs, bin_confs,
 bin_counts.
 """
 y_true = np.asarray(y_true)
 y_prob = np.asarray(y_prob)

 # Brier score: mean squared error of probability predictions
 brier = float(brier_score_loss(y_true, y_prob))

 # Bin predictions for reliability diagram
 bin_edges = np.linspace(0, 1, n_bins + 1)
 bin_accs = []
 bin_confs = []
 bin_counts = []

 for i in range(n_bins):
 lo, hi = bin_edges[i], bin_edges[i + 1]
 mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
 count = mask.sum()
 bin_counts.append(int(count))

 if count > 0:
 bin_accs.append(float(y_true[mask].mean()))
 bin_confs.append(float(y_prob[mask].mean()))
 else:
 bin_accs.append(0.0)
 bin_confs.append((lo + hi) / 2)

 # Expected Calibration Error: weighted average of |acc - conf| per bin
 total = len(y_prob)
 ece = sum(
 (cnt / total) * abs(acc - conf)
 for cnt, acc, conf in zip(bin_counts, bin_accs, bin_confs)
 if cnt > 0
 )

 return {
 "brier_score": brier,
 "ece": float(ece),
 "bin_edges": bin_edges.tolist(),
 "bin_accs": bin_accs,
 "bin_confs": bin_confs,
 "bin_counts": bin_counts,
 }


def run_calibration_analysis(
 texts: List[str],
 labels: np.ndarray,
 activations: np.ndarray,
 features: np.ndarray,
 train_ratio: float = 0.7,
 seed: int = 42,
 n_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
 """Run calibration analysis on all detection approaches.

 Args:
 texts: All prompt strings.
 labels: All labels, shape (N,).
 activations: Raw activations, shape (N, 768).
 features: SAE features, shape (N, 6144).
 train_ratio: Fraction for training.
 seed: Random seed.
 n_bins: Number of bins for reliability diagram.

 Returns:
 Dict mapping approach name to calibration results.
 """
 labels = np.asarray(labels)
 n = len(labels)

 from sklearn.model_selection import train_test_split
 indices = np.arange(n)
 train_idx, test_idx = train_test_split(
 indices, train_size=train_ratio, stratify=labels, random_state=seed
 )

 train_texts = [texts[i] for i in train_idx]
 test_texts = [texts[i] for i in test_idx]
 train_labels = labels[train_idx]
 test_labels = labels[test_idx]

 # Train detectors
 lr_pipe, _ = train_tfidf_baseline(train_texts, train_labels, seed=seed)
 act_clf = train_activation_baseline(activations[train_idx], train_labels, seed=seed)
 sae_clf = train_sae_feature_baseline(features[train_idx], train_labels, seed=seed)

 # Get predictions
 detectors = {
 "TF-IDF + LogReg": lr_pipe.predict_proba(test_texts)[:, 1],
 "Raw Activation + LogReg": act_clf.predict_proba(activations[test_idx])[:, 1],
 "SAE Features (all) + LogReg": sae_clf.predict_proba(features[test_idx])[:, 1],
 }

 calibration_results: Dict[str, Dict[str, Any]] = {}
 for name, y_prob in detectors.items():
 cal = compute_calibration(test_labels, y_prob, n_bins=n_bins)
 calibration_results[name] = cal
 print(f"{name}: Brier={cal['brier_score']:.4f}, ECE={cal['ece']:.4f}")

 return calibration_results


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------


def plot_cv_results(
 cv_results: Dict[str, Dict[str, Any]],
 save_path: Optional[str] = None,
) -> None:
 """Bar chart of cross-validation F1 and AUC with error bars.

 Args:
 cv_results: Output of run_cross_validation().
 save_path: Optional path to save figure.
 """
 approaches = list(cv_results.keys())
 # Filter to main approaches (not top-K ablation) for cleaner plot
 main_approaches = [a for a in approaches if "Top-" not in a]

 fig, axes = plt.subplots(1, 2, figsize=(14, 5))

 for ax, metric, title in zip(
 axes, ["f1", "roc_auc"], ["F1 Score", "ROC AUC"]
 ):
 means = [cv_results[a]["mean"][metric] for a in main_approaches]
 stds = [cv_results[a]["std"][metric] for a in main_approaches]

 x = np.arange(len(main_approaches))
 colors = [_CB_PALETTE[i % len(_CB_PALETTE)] for i in range(len(main_approaches))]

 bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
 edgecolor="white", linewidth=0.5)

 # Value labels
 for bar, mean, std in zip(bars, means, stds):
 ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
 f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

 ax.set_xticks(x)
 ax.set_xticklabels([a.replace(" + LogReg", "") for a in main_approaches],
 rotation=15, ha="right", fontsize=9)
 ax.set_ylabel(title, fontsize=11)
 ax.set_title(f"{title} (5-fold CV)", fontsize=12)
 ax.set_ylim(0.8, 1.05)

 plt.tight_layout()
 if save_path:
 Path(save_path).parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(save_path, dpi=200, bbox_inches="tight")
 print(f"Saved CV results plot to {save_path}")
 plt.close(fig)


def plot_calibration_diagram(
 calibration_results: Dict[str, Dict[str, Any]],
 save_path: Optional[str] = None,
) -> None:
 """Reliability diagram (calibration curve) for all detectors.

 A perfectly calibrated classifier falls on the diagonal. Points
 above the diagonal indicate under-confidence (model says 60% but
 it's actually 80%). Points below indicate over-confidence.

 Args:
 calibration_results: Output of run_calibration_analysis().
 save_path: Optional path to save figure.
 """
 fig, ax = plt.subplots(figsize=(7, 7))

 # Perfect calibration line
 ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

 for i, (name, cal) in enumerate(calibration_results.items()):
 color = _CB_PALETTE[i % len(_CB_PALETTE)]
 confs = cal["bin_confs"]
 accs = cal["bin_accs"]
 counts = cal["bin_counts"]

 # Only plot bins that have samples
 mask = [c > 0 for c in counts]
 plot_confs = [c for c, m in zip(confs, mask) if m]
 plot_accs = [a for a, m in zip(accs, mask) if m]

 short_name = name.replace(" + LogReg", "")
 ax.plot(plot_confs, plot_accs, "o-", color=color, linewidth=2,
 markersize=6, label=f"{short_name} (ECE={cal['ece']:.3f})")

 ax.set_xlabel("Mean Predicted Probability", fontsize=12)
 ax.set_ylabel("Fraction of Positives", fontsize=12)
 ax.set_title("Calibration Diagram (Reliability Curve)", fontsize=13)
 ax.legend(loc="lower right", fontsize=10)
 ax.set_xlim(-0.02, 1.02)
 ax.set_ylim(-0.02, 1.02)
 ax.set_aspect("equal")

 plt.tight_layout()
 if save_path:
 Path(save_path).parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(save_path, dpi=200, bbox_inches="tight")
 print(f"Saved calibration diagram to {save_path}")
 plt.close(fig)


def plot_per_category_heatmap(
 category_results: Dict[str, Dict[str, Dict[str, float]]],
 save_path: Optional[str] = None,
) -> None:
 """Heatmap showing detection rate per injection category per detector.

 Args:
 category_results: Output of run_per_category_breakdown().
 save_path: Optional path to save figure.
 """
 # Only show injection categories (those with detection_rate)
 inject_cats = [
 cat for cat, res in category_results.items()
 if "detection_rate" in list(res.values())[0]
 ]
 if not inject_cats:
 print("No injection-only categories found for heatmap.")
 return

 det_names = list(category_results[inject_cats[0]].keys())
 short_names = [n.replace(" + LogReg", "") for n in det_names]

 # Build matrix: rows = categories, cols = detectors
 data = np.array([
 [category_results[cat][det]["detection_rate"] for det in det_names]
 for cat in inject_cats
 ])

 fig, ax = plt.subplots(figsize=(8, 5))
 im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")

 ax.set_xticks(np.arange(len(short_names)))
 ax.set_xticklabels(short_names, rotation=20, ha="right", fontsize=10)
 ax.set_yticks(np.arange(len(inject_cats)))
 ax.set_yticklabels(inject_cats, fontsize=10)

 # Annotate cells with values
 for i in range(len(inject_cats)):
 for j in range(len(det_names)):
 val = data[i, j]
 color = "white" if val < 0.75 else "black"
 ax.text(j, i, f"{val:.0%}", ha="center", va="center",
 fontsize=11, fontweight="bold", color=color)

 ax.set_title("Detection Rate by Injection Category", fontsize=13)
 fig.colorbar(im, ax=ax, label="Detection Rate", shrink=0.8)

 plt.tight_layout()
 if save_path:
 Path(save_path).parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(save_path, dpi=200, bbox_inches="tight")
 print(f"Saved per-category heatmap to {save_path}")
 plt.close(fig)
