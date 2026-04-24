"""
Three-way detection comparison pipeline for the C3 experiment.

This module orchestrates the core experiment of the project: comparing
prompt injection detection performance across three increasingly
sophisticated approaches:

 1. Classical text features (TF-IDF + Logistic Regression)
 2. Raw transformer activations (768-dim residual stream)
 3. SAE-decomposed features (6144-dim sparse feature vector)

The hypothesis is that SAE features outperform raw activations because
the SAE decomposes the entangled 768-dim representation into
interpretable, monosemantic directions — some of which correspond
directly to injection-relevant patterns. If this holds, it
demonstrates that SAEs not only aid interpretability but also improve
downstream detection.

See Design Document §4.4 and §6.2 (C3) for the experiment definition.

Author: Nathan Cheung
York University | CSSD 2221 | Winter 2026
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
 accuracy_score,
 f1_score,
 precision_score,
 recall_score,
 roc_auc_score,
 roc_curve,
)

# Import baseline classifiers — all model training/evaluation logic
# lives in src/baseline/classifiers.py so this module only handles
# orchestration, comparison, and visualization.
from src.baseline.classifiers import (
 evaluate_classifier,
 train_activation_baseline,
 train_sae_feature_baseline,
 train_tfidf_baseline,
)


# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong, 2011) — the same palette used
# throughout the project for consistency. Seven distinct colours that
# remain distinguishable under the three most common forms of colour
# vision deficiency.
# ---------------------------------------------------------------------------
_CB_PALETTE = [
 "#0072B2", # blue
 "#D55E00", # vermillion
 "#009E73", # bluish green
 "#E69F00", # orange
 "#56B4E9", # sky blue
 "#CC79A7", # reddish purple
 "#F0E442", # yellow
]


def _evaluate_approach(
 clf: Any,
 X_test: np.ndarray,
 y_test: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
 """Evaluate a trained classifier and return metrics + predicted probabilities.

 Factored out because all three approaches share the same evaluation
 logic — only the input feature space differs.

 Args:
 clf: A trained sklearn classifier with predict and predict_proba.
 X_test: Test feature matrix.
 y_test: True labels for the test set.

 Returns:
 Tuple of (metrics_dict, predicted_probabilities_for_class_1).
 """
 y_pred = clf.predict(X_test)
 y_prob = clf.predict_proba(X_test)[:, 1]

 metrics = {
 "precision": float(precision_score(y_test, y_pred, zero_division=0)),
 "recall": float(recall_score(y_test, y_pred, zero_division=0)),
 "f1": float(f1_score(y_test, y_pred, zero_division=0)),
 "accuracy": float(accuracy_score(y_test, y_pred)),
 "roc_auc": float(roc_auc_score(y_test, y_prob)),
 }
 return metrics, y_prob


def run_detection_comparison(
 train_texts: List[str],
 train_labels: np.ndarray,
 test_texts: List[str],
 test_labels: np.ndarray,
 train_activations: np.ndarray,
 test_activations: np.ndarray,
 train_features: np.ndarray,
 test_features: np.ndarray,
 sensitivity_scores: Optional[np.ndarray] = None,
 top_k_values: Optional[List[int]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
 """Run the three-way (or more) detection comparison experiment.

 Trains and evaluates all detection approaches on the same
 train/test split so that performance differences reflect the
 representation, not the data split.

 Design rationale for using logistic regression everywhere: it is
 the simplest linear classifier, and its weights are directly
 interpretable. Using the same classifier family isolates the
 effect of the *representation* (text features vs. raw activations
 vs. SAE features) from the effect of the *classifier*.

 Args:
 train_texts: Training set prompt strings.
 train_labels: Training labels (0 = normal, 1 = injection).
 test_texts: Test set prompt strings.
 test_labels: Test labels.
 train_activations: Raw residual-stream activations for training
 set, shape (N_train, 768).
 test_activations: Raw residual-stream activations for test set,
 shape (N_test, 768).
 train_features: SAE feature activations for training set,
 shape (N_train, 6144).
 test_features: SAE feature activations for test set,
 shape (N_test, 6144).
 sensitivity_scores: Optional array of shape (6144,) — per-feature
 injection-sensitivity scores. When provided together with
 top_k_values, additional experiments are run using only the
 top-K most sensitive features.
 top_k_values: Optional list of K values (e.g. [10, 50, 100]) for
 the feature-subset ablation (experiment A2).

 Returns:
 Tuple of:
 - comparison_results: dict mapping approach name to metrics dict
 (precision, recall, f1, accuracy, roc_auc).
 - predictions_dict: dict mapping approach name to predicted
 probabilities on the test set (needed for ROC plotting).
 """
 comparison_results: Dict[str, Dict[str, float]] = {}
 predictions_dict: Dict[str, np.ndarray] = {}

 # ---- Approach 1: TF-IDF + Logistic Regression ----
 # This is the classical NLP baseline. It operates on surface-level
 # text patterns (n-grams), which means it can catch obvious keyword
 # markers ("ignore previous") but will miss semantically equivalent
 # rephrasings.
 print("Training Approach 1: TF-IDF + Logistic Regression...")
 # train_tfidf_baseline returns (lr_pipeline, rf_pipeline).
 # Each pipeline includes its own TF-IDF vectorizer, so we can
 # call predict/predict_proba directly on raw text.
 lr_pipeline, rf_pipeline = train_tfidf_baseline(train_texts, train_labels)
 metrics, y_prob = _evaluate_approach(lr_pipeline, test_texts, test_labels)
 comparison_results["TF-IDF + LogReg"] = metrics
 predictions_dict["TF-IDF + LogReg"] = y_prob
 print(f" -> F1 = {metrics['f1']:.4f}, AUC = {metrics['roc_auc']:.4f}")

 # Also evaluate the Random Forest pipeline for completeness
 print("Training Approach 1b: TF-IDF + Random Forest...")
 metrics_rf, y_prob_rf = _evaluate_approach(rf_pipeline, test_texts, test_labels)
 comparison_results["TF-IDF + RandomForest"] = metrics_rf
 predictions_dict["TF-IDF + RandomForest"] = y_prob_rf
 print(f" -> F1 = {metrics_rf['f1']:.4f}, AUC = {metrics_rf['roc_auc']:.4f}")

 # ---- Approach 2: Raw activations + Logistic Regression ----
 # The 768-dim residual stream captures everything the model "knows"
 # at the chosen layer, but the representation is entangled — each
 # dimension encodes a superposition of many concepts. A linear
 # classifier can still pick up on injection-correlated directions,
 # but it's working with an opaque, compressed signal.
 print("Training Approach 2: Raw Activation + Logistic Regression...")
 act_clf = train_activation_baseline(train_activations, train_labels)
 metrics, y_prob = _evaluate_approach(act_clf, test_activations, test_labels)
 comparison_results["Raw Activation + LogReg"] = metrics
 predictions_dict["Raw Activation + LogReg"] = y_prob
 print(f" -> F1 = {metrics['f1']:.4f}, AUC = {metrics['roc_auc']:.4f}")

 # ---- Approach 3: Full SAE features + Logistic Regression ----
 # The SAE decomposes the entangled 768-dim vector into 6144
 # (mostly-zero) monosemantic features. If the decomposition
 # captures injection-relevant structure, the classifier should
 # find it easier to separate the classes — even though the
 # underlying information is the same.
 print("Training Approach 3: SAE Features (all) + Logistic Regression...")
 sae_clf = train_sae_feature_baseline(train_features, train_labels)
 metrics, y_prob = _evaluate_approach(sae_clf, test_features, test_labels)
 comparison_results["SAE Features (all) + LogReg"] = metrics
 predictions_dict["SAE Features (all) + LogReg"] = y_prob
 print(f" -> F1 = {metrics['f1']:.4f}, AUC = {metrics['roc_auc']:.4f}")

 # ---- Optional: top-K feature subsets ----
 # This corresponds to experiment A2 in the design document. By
 # restricting the SAE feature vector to only the K most
 # injection-sensitive features, we test whether a small, interpretable
 # feature set is sufficient for detection — and how performance
 # degrades as K shrinks.
 if sensitivity_scores is not None and top_k_values is not None:
 # Rank features by absolute sensitivity (most discriminative first)
 abs_sens = np.abs(sensitivity_scores)
 ranked_indices = np.argsort(abs_sens)[::-1]

 for k in top_k_values:
 top_k_idx = ranked_indices[:k]
 train_subset = train_features[:, top_k_idx]
 test_subset = test_features[:, top_k_idx]

 label = f"SAE Top-{k} Features + LogReg"
 print(f"Training Approach: {label}...")
 subset_clf = train_sae_feature_baseline(train_subset, train_labels)
 metrics, y_prob = _evaluate_approach(
 subset_clf, test_subset, test_labels
 )
 comparison_results[label] = metrics
 predictions_dict[label] = y_prob
 print(f" -> F1 = {metrics['f1']:.4f}, AUC = {metrics['roc_auc']:.4f}")

 # Print the comparison table at the end for quick reference
 print()
 print_comparison_table(comparison_results)

 return comparison_results, predictions_dict


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_roc_comparison(
 comparison_results: Dict[str, Dict[str, float]],
 test_labels: np.ndarray,
 predictions_dict: Dict[str, np.ndarray],
 save_path: Optional[str] = None,
) -> None:
 """Plot ROC curves for all detection approaches on a single figure.

 Each approach gets its own curve, colour-coded with the
 colorblind-friendly palette. AUC is shown in the legend so the
 reader can compare at a glance.

 Args:
 comparison_results: Output of run_detection_comparison() —
 maps approach name to metrics dict.
 test_labels: True test labels (0/1).
 predictions_dict: Maps approach name to predicted probabilities
 on the test set.
 save_path: If provided, save the figure to this path at 200 DPI.
 """
 fig, ax = plt.subplots(figsize=(8, 7))

 for i, (name, y_prob) in enumerate(predictions_dict.items()):
 color = _CB_PALETTE[i % len(_CB_PALETTE)]
 auc_val = comparison_results[name]["roc_auc"]

 fpr, tpr, _ = roc_curve(test_labels, y_prob)
 ax.plot(fpr, tpr, color=color, linewidth=2,
 label=f"{name} (AUC = {auc_val:.3f})")

 # Diagonal reference line — a classifier that guesses randomly
 ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
 label="Random (AUC = 0.500)")

 ax.set_xlabel("False Positive Rate", fontsize=12)
 ax.set_ylabel("True Positive Rate", fontsize=12)
 ax.set_title("ROC Comparison: Injection Detection Approaches", fontsize=14)
 ax.legend(loc="lower right", fontsize=9)
 ax.set_xlim([0, 1])
 ax.set_ylim([0, 1.02])

 plt.tight_layout()
 if save_path:
 save_path_obj = Path(save_path)
 save_path_obj.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(save_path_obj, dpi=200, bbox_inches="tight")
 print(f"Saved ROC comparison to {save_path_obj}")
 plt.show()


def plot_metrics_comparison(
 comparison_results: Dict[str, Dict[str, float]],
 save_path: Optional[str] = None,
) -> None:
 """Grouped bar chart comparing key metrics across detection approaches.

 Each approach is a group of four bars (precision, recall, F1,
 accuracy). This gives a quick visual overview of where each
 approach excels or falls short.

 Args:
 comparison_results: Output of run_detection_comparison().
 save_path: If provided, save the figure to this path at 200 DPI.
 """
 metrics_to_plot = ["precision", "recall", "f1", "accuracy"]
 approach_names = list(comparison_results.keys())
 n_approaches = len(approach_names)
 n_metrics = len(metrics_to_plot)

 # Build the data matrix: rows = approaches, cols = metrics
 data = np.array([
 [comparison_results[name][m] for m in metrics_to_plot]
 for name in approach_names
 ])

 fig, ax = plt.subplots(figsize=(max(10, n_approaches * 2.5), 6))

 # Bar positions — group by approach, separate bars per metric
 bar_width = 0.8 / n_metrics
 x = np.arange(n_approaches)

 for j, metric in enumerate(metrics_to_plot):
 offset = (j - n_metrics / 2 + 0.5) * bar_width
 color = _CB_PALETTE[j % len(_CB_PALETTE)]
 bars = ax.bar(
 x + offset, data[:, j], bar_width,
 label=metric.capitalize(), color=color, alpha=0.85,
 )
 # Add value labels on top of each bar for readability
 for bar in bars:
 height = bar.get_height()
 ax.text(
 bar.get_x() + bar.get_width() / 2, height + 0.01,
 f"{height:.2f}", ha="center", va="bottom", fontsize=8,
 )

 ax.set_xticks(x)
 ax.set_xticklabels(approach_names, rotation=20, ha="right", fontsize=10)
 ax.set_ylabel("Score", fontsize=12)
 ax.set_title("Detection Performance Comparison", fontsize=14)
 ax.set_ylim(0, 1.15) # room for value labels
 ax.legend(loc="upper left", fontsize=10)

 plt.tight_layout()
 if save_path:
 save_path_obj = Path(save_path)
 save_path_obj.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(save_path_obj, dpi=200, bbox_inches="tight")
 print(f"Saved metrics comparison to {save_path_obj}")
 plt.show()


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------


def print_comparison_table(
 comparison_results: Dict[str, Dict[str, float]],
) -> None:
 """Print a clean ASCII table comparing all detection approaches.

 The best value for each metric is highlighted with an asterisk (*).
 This makes it easy to see at a glance which approach wins on each
 criterion — and whether one approach dominates or there are
 trade-offs.

 Args:
 comparison_results: Maps approach name to metrics dict.
 """
 if not comparison_results:
 print("No results to display.")
 return

 metrics_order = ["precision", "recall", "f1", "accuracy", "roc_auc"]
 approach_names = list(comparison_results.keys())

 # Find the best (highest) value for each metric across approaches
 best_per_metric: Dict[str, float] = {}
 for metric in metrics_order:
 values = [comparison_results[name].get(metric, 0.0) for name in approach_names]
 best_per_metric[metric] = max(values)

 # Column widths — approach name column is as wide as the longest name
 name_width = max(len(name) for name in approach_names)
 name_width = max(name_width, len("Approach"))
 metric_width = 12 # enough for "Precision *" etc.

 # Header
 header_metrics = "".join(m.capitalize().ljust(metric_width) for m in metrics_order)
 header = f"{'Approach'.ljust(name_width)} {header_metrics}"
 separator = "-" * len(header)

 print(separator)
 print(header)
 print(separator)

 for name in approach_names:
 row = name.ljust(name_width) + " "
 for metric in metrics_order:
 val = comparison_results[name].get(metric, 0.0)
 # Mark the best performer with an asterisk so it stands out
 marker = " *" if abs(val - best_per_metric[metric]) < 1e-9 else " "
 row += f"{val:.4f}{marker}".ljust(metric_width)
 print(row)

 print(separator)
 print("(* = best for that metric)")
