"""
Separability metrics for Experiment J1.

This module computes how well normal vs. injection prompts separate
in the activation space at each transformer layer. If activations
from the two classes form distinct clusters, that's evidence that
the model internally distinguishes injection-like inputs — and
therefore that an SAE might learn features capturing this distinction.

Metrics computed:
 1. Silhouette score — measures how tight clusters are relative to
 their separation. Ranges from -1 (wrong clustering) through 0
 (overlapping) to +1 (perfect separation). The J1 pass criterion
 from the Design Document is silhouette > 0.1 at any layer.

 2. Cohen's d (effect size) — measures the standardized difference
 between the class means. Unlike silhouette, this works on the
 raw high-dimensional vectors without needing pairwise distances.
 d > 0.5 is a "medium" effect; d > 0.8 is "large."

 3. t-SNE / UMAP visualizations — 2D projections that let us visually
 inspect whether clusters exist. These are qualitative (projections
 can distort distances) but invaluable for sanity checking.

Author: Nathan Cheung
York University | CSSD 2221 | Winter 2026
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


def compute_silhouette_score(
 activations: np.ndarray,
 labels: np.ndarray,
 sample_size: int = 1000,
 seed: int = 42,
) -> float:
 """
 Compute the silhouette score for two-class clustering.

 Silhouette score measures:
 For each point:
 a = mean distance to points in the same class
 b = mean distance to points in the nearest other class
 s = (b - a) / max(a, b)
 Final score = mean of s across all points.

 We use a sample_size limit because silhouette computation is O(n²)
 in pairwise distances — with 1000 points in 768 dimensions, it's
 fast; with 10000 it would be slow and unnecessary for a go/no-go check.

 Args:
 activations: Array of shape (N, d_model) — activation vectors.
 labels: Array of shape (N,) — 0/1 class labels.
 sample_size: Max points to use (randomly sampled if N > this).
 seed: Random seed for sampling.

 Returns:
 Silhouette score (float between -1 and 1).
 """
 from sklearn.metrics import silhouette_score

 n = len(activations)

 # Subsample if the dataset is large (unlikely for J1 but future-proof)
 if n > sample_size:
 rng = np.random.RandomState(seed)
 indices = rng.choice(n, sample_size, replace=False)
 activations = activations[indices]
 labels = labels[indices]

 # silhouette_score needs at least 2 classes and 2 samples per class
 unique_labels = np.unique(labels)
 if len(unique_labels) < 2:
 print("Warning: only one class present, silhouette is undefined")
 return 0.0

 return silhouette_score(activations, labels)


def compute_cohens_d(
 activations: np.ndarray,
 labels: np.ndarray,
) -> float:
 """
 Compute Cohen's d between the two class centroids.

 Cohen's d = ||mean_1 - mean_0|| / pooled_std

 This measures how many standard deviations apart the class means
 are in the activation space. We compute the Euclidean distance
 between centroids and normalize by the pooled standard deviation
 of the activation norms.

 Why activation norms instead of per-dimension pooling?
 In 768 dimensions, per-dimension pooling gives a 768-dim vector
 of standard deviations. Cohen's d is defined for scalars, so
 we'd need to aggregate somehow. Using the norm reduces each
 activation to a scalar while preserving magnitude information.

 Args:
 activations: Array of shape (N, d_model).
 labels: Array of shape (N,) — 0/1 labels.

 Returns:
 Cohen's d (float, >= 0). Larger = more separated.
 """
 # Split activations by class
 acts_0 = activations[labels == 0]
 acts_1 = activations[labels == 1]

 # Compute class centroids (mean activation vector per class)
 mean_0 = acts_0.mean(axis=0)
 mean_1 = acts_1.mean(axis=0)

 # Euclidean distance between centroids
 centroid_distance = np.linalg.norm(mean_1 - mean_0)

 # Pooled standard deviation of activation norms.
 # We use norms (not raw activations) to get a scalar std.
 norms_0 = np.linalg.norm(acts_0, axis=1)
 norms_1 = np.linalg.norm(acts_1, axis=1)

 n0, n1 = len(norms_0), len(norms_1)
 # Pooled variance formula: weighted average of within-class variances
 pooled_var = ((n0 - 1) * norms_0.var(ddof=1) +
 (n1 - 1) * norms_1.var(ddof=1)) / (n0 + n1 - 2)
 pooled_std = np.sqrt(pooled_var)

 # Guard against division by zero (would mean all activations are identical)
 if pooled_std < 1e-10:
 return 0.0

 return centroid_distance / pooled_std


def compute_all_layers(
 activations_by_layer: Dict[int, np.ndarray],
 labels: List[int],
) -> Dict[int, Dict[str, float]]:
 """
 Compute separability metrics at every layer.

 This is the main function for J1. It iterates over all layers,
 computes silhouette score and Cohen's d at each, and returns
 a summary dict.

 Args:
 activations_by_layer: Dict mapping layer index -> (N, d_model) array.
 labels: List of integer labels (0 = normal, 1 = injection).

 Returns:
 Dict mapping layer index -> {"silhouette": float, "cohens_d": float}.
 """
 labels_arr = np.array(labels)
 results = {}

 for layer in sorted(activations_by_layer.keys()):
 acts = activations_by_layer[layer]

 sil = compute_silhouette_score(acts, labels_arr)
 d = compute_cohens_d(acts, labels_arr)

 results[layer] = {
 "silhouette": sil,
 "cohens_d": d,
 }

 print(f"Layer {layer:2d}: silhouette={sil:.4f}, Cohen's d={d:.4f}")

 return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_separability_by_layer(
 metrics: Dict[int, Dict[str, float]],
 save_path: Optional[Path] = None,
) -> None:
 """
 Bar chart of silhouette score and Cohen's d across layers.

 This visualization answers the J1 question at a glance: which
 layers (if any) show meaningful separation between normal and
 injection activations?

 The J1 pass criterion (silhouette > 0.1) is drawn as a dashed
 red line so we can immediately see if any layer passes.

 Args:
 metrics: Output of compute_all_layers().
 save_path: Optional path to save the figure. If None, only displays.
 """
 layers = sorted(metrics.keys())
 silhouettes = [metrics[l]["silhouette"] for l in layers]
 cohens_ds = [metrics[l]["cohens_d"] for l in layers]

 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

 # --- Silhouette scores ---
 bars1 = ax1.bar(layers, silhouettes, color="#4C72B0", alpha=0.8)
 # J1 pass criterion line
 ax1.axhline(y=0.1, color="red", linestyle="--", linewidth=1.5,
 label="J1 pass threshold (0.1)")
 ax1.set_xlabel("Layer")
 ax1.set_ylabel("Silhouette Score")
 ax1.set_title("Activation Separability by Layer (Silhouette)")
 ax1.set_xticks(layers)
 ax1.legend()
 # Color bars that pass the threshold differently
 for bar, val in zip(bars1, silhouettes):
 if val >= 0.1:
 bar.set_color("#2ca02c") # green for passing layers

 # --- Cohen's d ---
 bars2 = ax2.bar(layers, cohens_ds, color="#4C72B0", alpha=0.8)
 # Reference lines for effect size interpretation
 ax2.axhline(y=0.5, color="orange", linestyle="--", linewidth=1,
 label="Medium effect (0.5)")
 ax2.axhline(y=0.8, color="red", linestyle="--", linewidth=1,
 label="Large effect (0.8)")
 ax2.set_xlabel("Layer")
 ax2.set_ylabel("Cohen's d")
 ax2.set_title("Centroid Separation by Layer (Cohen's d)")
 ax2.set_xticks(layers)
 ax2.legend()

 plt.tight_layout()

 if save_path:
 save_path = Path(save_path)
 save_path.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(save_path, dpi=200, bbox_inches="tight")
 print(f"Saved figure to {save_path}")

 plt.show()


def plot_activation_tsne(
 activations: np.ndarray,
 labels: List[int],
 layer: int,
 save_path: Optional[Path] = None,
 seed: int = 42,
) -> None:
 """
 2D t-SNE visualization of activations at a single layer.

 t-SNE is a nonlinear dimensionality reduction that tries to
 preserve local structure: points that are close in 768-dim space
 should be close in the 2D plot. This is great for visualizing
 clusters, but the distances and cluster shapes can be misleading
 (t-SNE distorts global structure). Always pair with quantitative
 metrics (silhouette, Cohen's d).

 We use t-SNE over UMAP here because t-SNE is more widely known
 and the results are comparable for our dataset size (~1000 points).

 Args:
 activations: Array of shape (N, d_model) — one layer's activations.
 labels: List of integer labels (0 = normal, 1 = injection).
 layer: Layer index (used only for the plot title).
 save_path: Optional path to save the figure.
 seed: Random seed for t-SNE reproducibility.
 """
 from sklearn.manifold import TSNE

 labels_arr = np.array(labels)

 # Perplexity controls how many neighbors t-SNE considers.
 # 30 is the default and works well for ~1000 points.
 # For very small datasets (<100), lower perplexity would be needed.
 tsne = TSNE(
 n_components=2,
 perplexity=30,
 random_state=seed,
 n_iter=1000, # default; enough for convergence at this scale
 )

 # t-SNE is O(n²), but with 1000 points it runs in seconds
 embeddings = tsne.fit_transform(activations)

 fig, ax = plt.subplots(figsize=(8, 8))

 # Plot normal prompts (label=0) and injection prompts (label=1)
 # with distinct colors from a colorblind-friendly palette
 for label_val, label_name, color in [
 (0, "Normal", "#4C72B0"), # blue
 (1, "Injection", "#DD8452"), # orange
 ]:
 mask = labels_arr == label_val
 ax.scatter(
 embeddings[mask, 0],
 embeddings[mask, 1],
 c=color,
 label=label_name,
 alpha=0.6,
 s=20, # small dots — we have many points
 edgecolors="none",
 )

 ax.set_title(f"t-SNE of Residual Stream Activations — Layer {layer}")
 ax.set_xlabel("t-SNE 1")
 ax.set_ylabel("t-SNE 2")
 ax.legend(markerscale=2)

 # Remove axis ticks — t-SNE coordinates are arbitrary
 ax.set_xticks([])
 ax.set_yticks([])

 plt.tight_layout()

 if save_path:
 save_path = Path(save_path)
 save_path.parent.mkdir(parents=True, exist_ok=True)
 fig.savefig(save_path, dpi=200, bbox_inches="tight")
 print(f"Saved figure to {save_path}")

 plt.show()
