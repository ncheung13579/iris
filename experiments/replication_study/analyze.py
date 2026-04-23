"""Analyze A/B/C feature activations — the Scaling Monosemanticity replication.

Three feature sets of interest:
  1. Self-question features: high activation on (A ∪ B), low on C
     -> What the SAE associates with self-directed queries regardless of intent
  2. Intent-discriminating features: high on B, low on A
     -> Features that separate malicious from benign identity questions
  3. FP-causing features: high on both A AND B, low on C
     -> Overlapping features that cause benign queries to look like injections

We also cross-reference with the existing agent_detector to see which features
the current classifier uses and whether those overlap with our findings.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
ACT_DIR = Path(__file__).parent / "activations"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from experiments.replication_study.prompt_sets import get_all_sets


def load_set(name):
    """Load features and prompts for a prompt set."""
    feats = np.load(ACT_DIR / f"{name}.npy")
    with open(ACT_DIR / f"{name}.prompts.json", encoding="utf-8") as f:
        prompts = json.load(f)
    return feats, prompts


def top_activating_prompts(feature_idx, feats, prompts, k=5):
    """Return the top-k prompts activating a specific feature in this set."""
    acts = feats[:, feature_idx]
    top_idx = np.argsort(acts)[::-1][:k]
    return [(float(acts[i]), prompts[i]) for i in top_idx]


def main():
    feats_A, prompts_A = load_set("A_benign_identity")
    feats_B, prompts_B = load_set("B_injection_identity")
    feats_C, prompts_C = load_set("C_mundane_control")

    print(f"A: {feats_A.shape}  B: {feats_B.shape}  C: {feats_C.shape}")

    # Mean activation per feature in each set (across all prompts in the set)
    mean_A = feats_A.mean(0)  # (d_sae,)
    mean_B = feats_B.mean(0)
    mean_C = feats_C.mean(0)

    # Also compute fraction of prompts in each set that activate each feature
    # (helps filter out single-outlier features)
    frac_A = (feats_A > 0).mean(0)
    frac_B = (feats_B > 0).mean(0)
    frac_C = (feats_C > 0).mean(0)

    d_sae = feats_A.shape[1]
    print(f"d_sae={d_sae}")
    print(f"Mean active fraction — A: {frac_A.mean():.3f}  B: {frac_B.mean():.3f}  C: {frac_C.mean():.3f}")

    # -------------------------------------------------------------------
    # Category 1: Self-question features — high in (A ∪ B), low in C
    # -------------------------------------------------------------------
    self_score = (mean_A + mean_B) / 2 - mean_C
    # Require the feature to actually fire on a reasonable fraction of self-Qs
    eligible = (frac_A + frac_B) / 2 > 0.5  # fires on ≥50% of self-questions
    self_score_eligible = np.where(eligible, self_score, -np.inf)
    top_self = np.argsort(self_score_eligible)[::-1][:20]

    # -------------------------------------------------------------------
    # Category 2: Intent-discriminating features — high in B, low in A
    # -------------------------------------------------------------------
    intent_score = mean_B - mean_A
    # Must fire on ≥60% of B and ≤20% of A for a cleaner discriminator
    clean_intent = (frac_B > 0.6) & (frac_A < 0.2)
    intent_score_clean = np.where(clean_intent, intent_score, -np.inf)
    top_intent = np.argsort(intent_score_clean)[::-1][:20]
    # Relaxed: just highest delta (may include features active on both)
    top_intent_loose = np.argsort(intent_score)[::-1][:20]

    # -------------------------------------------------------------------
    # Category 3: FP-causing features — high on BOTH A and B, low on C
    # -------------------------------------------------------------------
    overlap_score = np.minimum(mean_A, mean_B) - mean_C
    shared = (frac_A > 0.6) & (frac_B > 0.6) & (frac_C < 0.3)
    overlap_score_shared = np.where(shared, overlap_score, -np.inf)
    top_overlap = np.argsort(overlap_score_shared)[::-1][:20]

    # -------------------------------------------------------------------
    # Summary stats
    # -------------------------------------------------------------------
    print()
    print("=" * 70)
    print("CATEGORY 1: Self-question features (A ∪ B, not C)")
    print("=" * 70)
    print(f"{'idx':>6} {'muA':>6} {'muB':>6} {'muC':>6} {'Δ':>6} {'fA':>4} {'fB':>4} {'fC':>4}")
    for f_idx in top_self:
        if self_score_eligible[f_idx] == -np.inf:
            break
        print(f"{f_idx:>6} {mean_A[f_idx]:6.2f} {mean_B[f_idx]:6.2f} {mean_C[f_idx]:6.2f} "
              f"{self_score[f_idx]:6.2f} {frac_A[f_idx]:4.2f} {frac_B[f_idx]:4.2f} {frac_C[f_idx]:4.2f}")

    print()
    print("=" * 70)
    print("CATEGORY 2: Intent-discriminating features (B > A, strict)")
    print("=" * 70)
    print(f"{'idx':>6} {'muA':>6} {'muB':>6} {'muC':>6} {'Δ':>6} {'fA':>4} {'fB':>4} {'fC':>4}")
    any_clean = False
    for f_idx in top_intent:
        if intent_score_clean[f_idx] == -np.inf:
            break
        any_clean = True
        print(f"{f_idx:>6} {mean_A[f_idx]:6.2f} {mean_B[f_idx]:6.2f} {mean_C[f_idx]:6.2f} "
              f"{intent_score[f_idx]:6.2f} {frac_A[f_idx]:4.2f} {frac_B[f_idx]:4.2f} {frac_C[f_idx]:4.2f}")
    if not any_clean:
        print("  (none passed strict filter — relaxing)")
    print()
    print("  LOOSE (top 10 by raw intent delta):")
    print(f"{'idx':>6} {'muA':>6} {'muB':>6} {'muC':>6} {'Δ':>6} {'fA':>4} {'fB':>4} {'fC':>4}")
    for f_idx in top_intent_loose[:10]:
        print(f"{f_idx:>6} {mean_A[f_idx]:6.2f} {mean_B[f_idx]:6.2f} {mean_C[f_idx]:6.2f} "
              f"{intent_score[f_idx]:6.2f} {frac_A[f_idx]:4.2f} {frac_B[f_idx]:4.2f} {frac_C[f_idx]:4.2f}")

    print()
    print("=" * 70)
    print("CATEGORY 3: FP-causing features (high on BOTH A and B, low on C)")
    print("=" * 70)
    print(f"{'idx':>6} {'muA':>6} {'muB':>6} {'muC':>6} {'Δ':>6} {'fA':>4} {'fB':>4} {'fC':>4}")
    for f_idx in top_overlap:
        if overlap_score_shared[f_idx] == -np.inf:
            break
        print(f"{f_idx:>6} {mean_A[f_idx]:6.2f} {mean_B[f_idx]:6.2f} {mean_C[f_idx]:6.2f} "
              f"{overlap_score[f_idx]:6.2f} {frac_A[f_idx]:4.2f} {frac_B[f_idx]:4.2f} {frac_C[f_idx]:4.2f}")

    # -------------------------------------------------------------------
    # Show top-activating prompts for a few top features in each category
    # -------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Top prompts for the top-3 features in each category")
    print("=" * 70)

    def show_feature(f_idx, label):
        print(f"\n--- Feature {f_idx} ({label}) ---")
        print(f"  muA={mean_A[f_idx]:.2f}  muB={mean_B[f_idx]:.2f}  muC={mean_C[f_idx]:.2f}")
        print("  Top in A (benign identity):")
        for act, p in top_activating_prompts(f_idx, feats_A, prompts_A, 3):
            print(f"    {act:6.2f} | {p}")
        print("  Top in B (injection):")
        for act, p in top_activating_prompts(f_idx, feats_B, prompts_B, 3):
            print(f"    {act:6.2f} | {p}")
        print("  Top in C (mundane):")
        for act, p in top_activating_prompts(f_idx, feats_C, prompts_C, 3):
            print(f"    {act:6.2f} | {p}")

    for f_idx in top_self[:3]:
        show_feature(f_idx, "self-question")
    for f_idx in top_intent_loose[:3]:
        show_feature(f_idx, "intent-discriminator")
    for f_idx in top_overlap[:3]:
        if overlap_score_shared[f_idx] != -np.inf:
            show_feature(f_idx, "FP-causing overlap")

    # -------------------------------------------------------------------
    # Save summary JSON for later reuse
    # -------------------------------------------------------------------
    results = {
        "d_sae": int(d_sae),
        "n_A": len(prompts_A), "n_B": len(prompts_B), "n_C": len(prompts_C),
        "top_self_question_features": [int(i) for i in top_self[:20]],
        "top_intent_discriminators_strict": [int(i) for i in top_intent if intent_score_clean[i] != -np.inf][:20],
        "top_intent_discriminators_loose": [int(i) for i in top_intent_loose[:20]],
        "top_fp_overlap_features": [int(i) for i in top_overlap if overlap_score_shared[i] != -np.inf][:20],
        "feature_stats": {
            "mean_active_frac_A": float(frac_A.mean()),
            "mean_active_frac_B": float(frac_B.mean()),
            "mean_active_frac_C": float(frac_C.mean()),
        },
    }
    with open(RESULTS_DIR / "replication_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'replication_results.json'}")


if __name__ == "__main__":
    main()
