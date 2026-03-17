"""
Standardized evaluation framework for IRIS defense metrics.

Provides consistent metric computation across all experiments
(detection, steering, red team, defense comparison) so that
v1 → v2 improvements are directly comparable.

Author: Nathan Cheung ()
York University | CSSD 2221 | Winter 2026
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard detection metrics.

    Args:
        y_true: Ground truth labels (0=normal, 1=injection).
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for class 1 (optional, for AUC).

    Returns:
        Dict with precision, recall, f1, accuracy, detection_rate, fpr, and optionally roc_auc.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    detection_rate = recall  # Same as recall: TP / total attacks
    fpr = fp / max(fp + tn, 1)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "detection_rate": detection_rate,
        "false_positive_rate": fpr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics


def compute_evasion_metrics(
    attacks: List[Dict],
    predictions: List[int],
) -> Dict[str, Any]:
    """Compute per-strategy evasion metrics.

    Args:
        attacks: Attack dicts with 'evasion_strategy' field.
        predictions: Model predictions (0=normal, 1=injection).

    Returns:
        Dict with overall and per-strategy evasion rates.
    """
    strategies = sorted(set(a.get("evasion_strategy", "unknown") for a in attacks))
    per_strategy = {}

    for strat in strategies:
        indices = [i for i, a in enumerate(attacks) if a.get("evasion_strategy") == strat]
        strat_preds = [predictions[i] for i in indices]
        evaded = sum(1 for p in strat_preds if p == 0)
        per_strategy[strat] = {
            "total": len(indices),
            "evaded": evaded,
            "evasion_rate": evaded / max(len(indices), 1),
        }

    total = len(predictions)
    total_evaded = sum(1 for p in predictions if p == 0)

    return {
        "overall_evasion_rate": total_evaded / max(total, 1),
        "total": total,
        "evaded": total_evaded,
        "per_strategy": per_strategy,
    }


def compute_steering_metrics(
    steering_results: List[Dict],
) -> Dict[str, float]:
    """Compute metrics for feature steering evaluation.

    Args:
        steering_results: List of result dicts from SteeringDefense.dampen().

    Returns:
        Dict with flip_rate, mean_prob_drop, and fidelity.
    """
    if not steering_results:
        return {"flip_rate": 0.0, "mean_prob_drop": 0.0, "n_samples": 0}

    n_flips = sum(1 for r in steering_results if r.get("flip", False))
    prob_drops = [
        r["orig_prob"] - r["steered_prob"]
        for r in steering_results
        if "orig_prob" in r and "steered_prob" in r
    ]

    return {
        "flip_rate": n_flips / len(steering_results),
        "mean_prob_drop": float(np.mean(prob_drops)) if prob_drops else 0.0,
        "n_samples": len(steering_results),
    }


def compute_latency(
    fn: Callable,
    *args: Any,
    n_runs: int = 10,
    **kwargs: Any,
) -> Dict[str, float]:
    """Measure function latency over multiple runs.

    Args:
        fn: Function to benchmark.
        *args: Positional arguments for fn.
        n_runs: Number of repetitions.
        **kwargs: Keyword arguments for fn.

    Returns:
        Dict with mean_ms, std_ms, min_ms, max_ms.
    """
    times = []
    for _ in range(n_runs):
        start = time.time()
        fn(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "n_runs": n_runs,
    }


def compute_defense_depth_score(
    defense_log: List[Dict],
) -> int:
    """Count how many defense layers an input had to bypass.

    A higher score means the attack penetrated deeper into the
    defense stack before being stopped. Score of 4 means it
    passed all layers (either legitimate or complete evasion).

    Args:
        defense_log: List of layer result dicts from DefenseStack.

    Returns:
        Number of layers passed (0-4).
    """
    passed = 0
    for layer in defense_log:
        if layer.get("details", {}).get("decision") == "SKIP":
            continue
        if layer.get("passed", False):
            passed += 1
        else:
            break  # Stopped at this layer
    return passed


def compare_defense_versions(
    v1_metrics: Dict[str, float],
    v2_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Compare v1 vs v2 defense metrics side by side.

    Args:
        v1_metrics: Metrics from defense v1.
        v2_metrics: Metrics from defense v2.

    Returns:
        Dict with side-by-side comparison and improvement deltas.
    """
    comparison = {}
    all_keys = set(v1_metrics.keys()) | set(v2_metrics.keys())

    for key in sorted(all_keys):
        v1_val = v1_metrics.get(key)
        v2_val = v2_metrics.get(key)
        if isinstance(v1_val, (int, float)) and isinstance(v2_val, (int, float)):
            delta = v2_val - v1_val
            # For evasion rate and FPR, lower is better (negative delta = improvement)
            improved = delta > 0
            if key in ("evasion_rate", "overall_evasion_rate", "false_positive_rate"):
                improved = delta < 0
            comparison[key] = {
                "v1": v1_val,
                "v2": v2_val,
                "delta": delta,
                "improved": improved,
            }
        else:
            comparison[key] = {"v1": v1_val, "v2": v2_val}

    return comparison


def print_metrics_table(
    metrics: Dict[str, float],
    title: str = "Metrics",
) -> None:
    """Print metrics in a clean ASCII table.

    Args:
        metrics: Dict of metric name to value.
        title: Table title.
    """
    print(f"\n{'=' * 45}")
    print(f"  {title}")
    print(f"{'=' * 45}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:30s}  {val:.4f}")
        elif isinstance(val, int):
            print(f"  {key:30s}  {val}")
        else:
            print(f"  {key:30s}  {val}")
    print(f"{'=' * 45}")
