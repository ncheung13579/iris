"""
Injection-sensitivity scoring for SAE features.

This module computes how strongly each SAE feature is associated with
prompt injection vs. normal prompts. The core metric is simple:

 sensitivity(feature_i) = mean_activation_on_injections(feature_i)
 - mean_activation_on_normal(feature_i)

Features with large positive sensitivity fire more on injections.
Features with large negative sensitivity are suppressed by injections.
Both directions are informative — they tell us the SAE learned to
distinguish the two classes.

See Design Document §4.3 for the formal definition.

Author: Nathan Cheung
York University | CSSD 2221 | Winter 2026
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch


def compute_feature_activations(
 sae: torch.nn.Module,
 activations: np.ndarray,
 device: Optional[torch.device] = None,
 batch_size: int = 256,
) -> np.ndarray:
 """
 Run the SAE encoder on all activations and return the feature matrix.

 We only need the encoder output (the sparse feature vector), not the
 full reconstruction. This is more memory-efficient than calling
 forward() which also computes the reconstruction and loss.

 Args:
 sae: Trained SparseAutoencoder instance.
 activations: Array of shape (N, d_input) — residual stream vectors.
 device: Torch device. If None, auto-detected.
 batch_size: Process this many activations at once.

 Returns:
 Array of shape (N, d_sae) — sparse feature activations for every
 input. Most values are zero (due to ReLU + sparsity pressure).
 """
 if device is None:
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 sae = sae.to(device)
 sae.eval()

 n = len(activations)
 # We don't know d_sae until we run the encoder, but we can read it
 # from the encoder weight shape: (d_sae, d_input)
 d_sae = sae.encoder.weight.shape[0]
 all_features = np.zeros((n, d_sae), dtype=np.float32)

 with torch.no_grad():
 for start in range(0, n, batch_size):
 end = min(start + batch_size, n)
 x = torch.from_numpy(activations[start:end]).to(device)

 # Run only the encoder + ReLU, not the full forward pass.
 # This gives us the sparse feature activations without
 # wasting compute on reconstruction.
 features = sae.relu(sae.encoder(x))

 all_features[start:end] = features.cpu().numpy()

 return all_features


def compute_sensitivity_scores(
 feature_matrix: np.ndarray,
 labels: np.ndarray,
) -> np.ndarray:
 """
 Compute injection-sensitivity score for each SAE feature.

 sensitivity(i) = mean_activation_on_injections(i) - mean_activation_on_normal(i)

 This is the simplest possible measure of class association. We chose
 it over more complex metrics (mutual information, AUROC per feature)
 because:
 1. It's interpretable — the sign tells you the direction, the
 magnitude tells you the strength.
 2. It's fast — just two group means per feature.
 3. It aligns with how the detector will use features — a logistic
 regression weight is essentially a scaled version of this.

 Args:
 feature_matrix: Array of shape (N, d_sae) — SAE feature activations.
 labels: Array of shape (N,) — 0 = normal, 1 = injection.

 Returns:
 Array of shape (d_sae,) — sensitivity score per feature.
 Positive = injection-associated, negative = normal-associated.
 """
 labels = np.asarray(labels)

 # Split by class
 normal_mask = labels == 0
 inject_mask = labels == 1

 # Mean activation per feature, separately for each class
 mean_normal = feature_matrix[normal_mask].mean(axis=0)
 mean_inject = feature_matrix[inject_mask].mean(axis=0)

 sensitivity = mean_inject - mean_normal

 # Summary statistics
 n_positive = np.sum(sensitivity > 0)
 n_negative = np.sum(sensitivity < 0)
 n_zero = np.sum(sensitivity == 0)

 print(f"Sensitivity scores computed for {len(sensitivity)} features:")
 print(f" Injection-associated (positive): {n_positive}")
 print(f" Normal-associated (negative): {n_negative}")
 print(f" Neutral (zero): {n_zero}")
 print(f" Max sensitivity: {sensitivity.max():.4f}")
 print(f" Min sensitivity: {sensitivity.min():.4f}")
 print(f" Mean |sensitivity|: {np.abs(sensitivity).mean():.4f}")

 return sensitivity


def get_top_features(
 sensitivity: np.ndarray,
 k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
 """
 Get the top-K most injection-sensitive features (both directions).

 We take the K features with the highest absolute sensitivity, which
 includes both injection-associated (positive) and normal-associated
 (negative) features. Both are useful:
 - Positive features fire on injections → direct detection signal.
 - Negative features are suppressed by injections → their absence
 is also a detection signal.

 Args:
 sensitivity: Array of shape (d_sae,) — per-feature scores.
 k: Number of top features to return.

 Returns:
 Tuple of (feature_indices, sensitivity_values), both of shape (k,),
 sorted by absolute sensitivity (most sensitive first).
 """
 # Sort by absolute value, descending
 abs_sensitivity = np.abs(sensitivity)
 top_indices = np.argsort(abs_sensitivity)[::-1][:k]
 top_values = sensitivity[top_indices]

 return top_indices, top_values


def get_top_activating_examples(
 feature_matrix: np.ndarray,
 feature_idx: int,
 texts: List[str],
 labels: List[int],
 k: int = 10,
) -> List[Dict]:
 """
 Find the top-K examples that most strongly activate a given feature.

 This is the core interpretability tool: by looking at what inputs
 cause a feature to fire strongly, we can infer what concept the
 feature represents. If a feature's top-activating examples are all
 injection prompts with "ignore previous instructions", that feature
 likely detects instruction override patterns.

 Args:
 feature_matrix: Array of shape (N, d_sae).
 feature_idx: Which feature to inspect.
 texts: List of prompt texts (same order as feature_matrix rows).
 labels: List of labels (0/1, same order).
 k: Number of top examples to return.

 Returns:
 List of dicts, each with keys:
 "text": str — the prompt text (truncated to 150 chars for display)
 "label": int — 0 or 1
 "label_str": str — "normal" or "injection"
 "activation": float — how strongly this feature fired
 """
 # Get activation values for this specific feature across all examples
 feature_acts = feature_matrix[:, feature_idx]

 # Sort by activation value, descending (strongest first)
 top_indices = np.argsort(feature_acts)[::-1][:k]

 results = []
 for idx in top_indices:
 results.append({
 "text": texts[idx][:150], # truncate for readability
 "label": int(labels[idx]),
 "label_str": "injection" if labels[idx] == 1 else "normal",
 "activation": float(feature_acts[idx]),
 })

 return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_sensitivity_distribution(
 sensitivity: np.ndarray,
 save_path: Optional[Path] = None,
) -> None:
 """
 Histogram of sensitivity scores across all features.

 This plot shows the overall distribution: most features should be
 near zero (injection-neutral), with tails of injection-associated
 (positive) and normal-associated (negative) features. A bimodal
 or heavy-tailed distribution suggests the SAE has learned
 injection-relevant structure.

 Args:
 sensitivity: Array of shape (d_sae,).
 save_path: Optional path to save the figure.
 """
 fig, ax = plt.subplots(figsize=(10, 5))

 ax.hist(sensitivity, bins=80, color="#4C72B0", alpha=0.8, edgecolor="white")
 ax.axvline(x=0, color="black", linestyle="-", linewidth=1)

 # Mark the top features
 top_pos = np.sort(sensitivity)[-5:] # top 5 positive
 top_neg = np.sort(sensitivity)[:5] # top 5 negative
 for val in top_pos:
 ax.axvline(x=val, color="#DD8452", linestyle="--", alpha=0.7)
 for val in top_neg:
 ax.axvline(x=val, color="#55A868", linestyle="--", alpha=0.7)

 ax.set_xlabel("Sensitivity Score (injection mean - normal mean)")
 ax.set_ylabel("Number of Features")
 ax.set_title("Distribution of Injection-Sensitivity Scores Across SAE Features")

 plt.tight_layout()
 if save_path:
 save_path = Path(save_path)
 save_path.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(save_path, dpi=200, bbox_inches="tight")
 print(f"Saved to {save_path}")
 plt.show()


def plot_top_features_bar(
 feature_indices: np.ndarray,
 sensitivity_values: np.ndarray,
 save_path: Optional[Path] = None,
) -> None:
 """
 Horizontal bar chart of the top injection-sensitive features.

 Color-coded: orange bars for injection-associated (positive sensitivity),
 blue bars for normal-associated (negative sensitivity).

 Args:
 feature_indices: Array of feature indices.
 sensitivity_values: Corresponding sensitivity scores.
 save_path: Optional path to save the figure.
 """
 fig, ax = plt.subplots(figsize=(10, 8))

 # Sort by value for visual clarity (most positive at top)
 sort_order = np.argsort(sensitivity_values)
 indices_sorted = feature_indices[sort_order]
 values_sorted = sensitivity_values[sort_order]

 # Color by sign: orange for injection-associated, blue for normal-associated
 colors = ["#DD8452" if v > 0 else "#4C72B0" for v in values_sorted]

 y_labels = [f"Feature {idx}" for idx in indices_sorted]
 ax.barh(y_labels, values_sorted, color=colors, alpha=0.8)
 ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

 ax.set_xlabel("Sensitivity Score")
 ax.set_title("Top 20 Most Injection-Sensitive SAE Features")

 # Add a legend
 from matplotlib.patches import Patch
 legend_elements = [
 Patch(facecolor="#DD8452", label="Injection-associated (+)"),
 Patch(facecolor="#4C72B0", label="Normal-associated (-)"),
 ]
 ax.legend(handles=legend_elements, loc="lower right")

 plt.tight_layout()
 if save_path:
 save_path = Path(save_path)
 save_path.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(save_path, dpi=200, bbox_inches="tight")
 print(f"Saved to {save_path}")
 plt.show()


def print_feature_dashboard(
 feature_idx: int,
 sensitivity: float,
 top_examples: List[Dict],
) -> None:
 """
 Print a human-readable dashboard for a single SAE feature.

 This is the key interpretability output: for each feature, we show
 its sensitivity score and the prompts that activate it most strongly.
 A human reviewer can then judge whether the feature captures a
 coherent, interpretable pattern.

 J3 pass criterion: at least 5 of the top 20 features show a clear,
 interpretable pattern (not random noise).

 Args:
 feature_idx: The SAE feature index.
 sensitivity: Its injection-sensitivity score.
 top_examples: Output of get_top_activating_examples().
 """
 direction = "INJECTION-associated" if sensitivity > 0 else "NORMAL-associated"
 print(f"\n{'='*70}")
 print(f"Feature {feature_idx} | sensitivity = {sensitivity:+.4f} | {direction}")
 print(f"{'='*70}")

 # Count how many of the top-activating examples are injections vs normal
 n_inject = sum(1 for ex in top_examples if ex["label"] == 1)
 n_normal = sum(1 for ex in top_examples if ex["label"] == 0)
 print(f"Top-{len(top_examples)} activating examples: "
 f"{n_inject} injection, {n_normal} normal\n")

 for i, ex in enumerate(top_examples, 1):
 label_tag = "[INJ]" if ex["label"] == 1 else "[NOR]"
 print(f" {i:2d}. {label_tag} (act={ex['activation']:.3f}) "
 f"{ex['text']}")
 print()
