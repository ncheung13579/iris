"""
Attack taxonomy using SAE feature signatures.

Maps attack categories to their characteristic SAE feature patterns,
revealing that the SAE has learned structured representations of
different injection strategies — not just a binary classifier.

Key analyses:
  - Per-category mean feature vectors ("fingerprints")
  - Pairwise similarity between categories
  - Category-specific features (one-vs-rest)
  - Attack type classification from feature patterns
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_category_fingerprints(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    categories: List[str],
    top_k: int = 50,
    sensitivity_scores: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute mean feature vector per attack category.

    The "fingerprint" of a category is its mean activation pattern across
    the top-K most sensitive features. This reveals which features each
    attack type relies on.

    Args:
        feature_matrix: SAE feature activations, shape (N, d_sae).
        labels: Integer labels, shape (N,). 0=normal, 1=injection.
        categories: Category string per example, shape (N,).
        top_k: Number of top features to include in fingerprints.
        sensitivity_scores: Per-feature sensitivity, shape (d_sae,).
            If provided, restricts fingerprints to top-K most sensitive features.

    Returns:
        Dict mapping category name to mean feature vector.
        If sensitivity_scores provided, vectors are (top_k,) shaped.
        Otherwise, vectors are (d_sae,) shaped.
    """
    categories = np.array(categories)
    unique_cats = sorted(set(categories))

    # Select feature subset if sensitivity scores provided
    if sensitivity_scores is not None:
        abs_sens = np.abs(sensitivity_scores)
        top_indices = np.argsort(abs_sens)[::-1][:top_k]
        matrix = feature_matrix[:, top_indices]
    else:
        matrix = feature_matrix
        top_indices = None

    fingerprints = {}
    for cat in unique_cats:
        mask = categories == cat
        if mask.sum() > 0:
            fingerprints[cat] = matrix[mask].mean(axis=0)

    return fingerprints


def compute_category_similarity(
    fingerprints: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """Compute pairwise cosine similarity between category fingerprints.

    Reveals which attack types are most similar in feature space.
    Expected: override and extraction share features; mimicry looks
    like normal prompts (explaining its high evasion rate).

    Args:
        fingerprints: Dict mapping category name to mean feature vector.

    Returns:
        Tuple of (similarity_matrix, category_names).
        similarity_matrix shape: (n_categories, n_categories).
    """
    names = sorted(fingerprints.keys())
    vectors = np.array([fingerprints[name] for name in names])

    # Handle zero vectors (categories with no activation)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / norms

    sim_matrix = cosine_similarity(normalized)
    return sim_matrix, names


def identify_category_specific_features(
    feature_matrix: np.ndarray,
    categories: np.ndarray,
    top_k: int = 10,
) -> Dict[str, List[Tuple[int, float]]]:
    """Find features that are specific to each category (one-vs-rest).

    For each category, computes the difference between its mean activation
    and the mean activation of all other categories. Features with the
    largest positive difference are most specific to that category.

    Args:
        feature_matrix: SAE features, shape (N, d_sae).
        categories: Category labels, shape (N,).
        top_k: Number of top specific features to return per category.

    Returns:
        Dict mapping category name to list of (feature_index, specificity_score).
    """
    categories = np.array(categories)
    unique_cats = sorted(set(categories))
    result = {}

    for cat in unique_cats:
        mask = categories == cat
        if mask.sum() < 2:
            result[cat] = []
            continue

        mean_cat = feature_matrix[mask].mean(axis=0)
        mean_rest = feature_matrix[~mask].mean(axis=0)

        # Specificity = how much more this category activates each feature
        specificity = mean_cat - mean_rest
        top_indices = np.argsort(specificity)[::-1][:top_k]

        result[cat] = [
            (int(idx), float(specificity[idx]))
            for idx in top_indices
            if specificity[idx] > 0  # Only include features that are actually higher
        ]

    return result


def classify_attack_type(
    features: np.ndarray,
    fingerprints: Dict[str, np.ndarray],
) -> Tuple[str, float]:
    """Predict attack category from feature pattern using nearest fingerprint.

    Simple cosine similarity classifier — no training needed. This
    demonstrates that the SAE features contain enough structure to
    distinguish attack types, not just detect-vs-normal.

    Args:
        features: Single feature vector, shape (d_sae,) or (top_k,).
        fingerprints: Category fingerprints from compute_category_fingerprints().

    Returns:
        Tuple of (predicted_category, confidence_score).
    """
    if features.ndim == 2:
        features = features[0]

    best_cat = "unknown"
    best_sim = -1.0

    feat_norm = np.linalg.norm(features)
    if feat_norm == 0:
        return ("unknown", 0.0)
    feat_normalized = features / feat_norm

    for cat, fp in fingerprints.items():
        fp_norm = np.linalg.norm(fp)
        if fp_norm == 0:
            continue
        sim = float(np.dot(feat_normalized, fp / fp_norm))
        if sim > best_sim:
            best_sim = sim
            best_cat = cat

    return (best_cat, best_sim)


def build_taxonomy_heatmap_data(
    feature_matrix: np.ndarray,
    categories: np.ndarray,
    sensitivity_scores: np.ndarray,
    top_k: int = 50,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Build data for the taxonomy heatmap visualization.

    Returns a matrix where rows = top features, columns = categories,
    cells = mean activation. This is the core visualization showing
    that different attack types have distinct feature signatures.

    Args:
        feature_matrix: SAE features, shape (N, d_sae).
        categories: Category labels, shape (N,).
        sensitivity_scores: Per-feature sensitivity, shape (d_sae,).
        top_k: Number of top features for rows.

    Returns:
        Tuple of (heatmap_data, category_names, feature_indices).
        heatmap_data shape: (top_k, n_categories).
    """
    categories = np.array(categories)
    unique_cats = sorted(set(categories))

    # Top features by absolute sensitivity
    abs_sens = np.abs(sensitivity_scores)
    top_indices = np.argsort(abs_sens)[::-1][:top_k]

    heatmap = np.zeros((top_k, len(unique_cats)))

    for j, cat in enumerate(unique_cats):
        mask = categories == cat
        if mask.sum() > 0:
            heatmap[:, j] = feature_matrix[mask][:, top_indices].mean(axis=0)

    return heatmap, unique_cats, top_indices.tolist()
