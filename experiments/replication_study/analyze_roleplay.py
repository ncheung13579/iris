"""Roleplay-category analysis — third FP category generalization test.

Answers three questions for Section 5.8:

  Q1 — Feature-level: does the SAE encode an intent distinction for
        roleplay in the same way it does for identity/commands?
        (Apply the A/B/C filter to Sets F, G, C.)

  Q2 — Generalization: does the existing A/B/D/E unified-augmentation fix
        transfer to a third FP category without seeing F/G during training?
        (Evaluate the prod-configured detector on F/G.)

  Q3 — Extension: if Q2 fails, does adding F/G to augmentation recover the
        fix in the same minimal-data way as the identity/command categories?
        (Evaluate a detector trained with all six sets on F/G and on the
        original held-out test set.)

Writes experiments/replication_study/results/roleplay_analysis.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split as tts

ROOT = Path(__file__).parent.parent.parent
ACT_DIR = Path(__file__).parent / "activations"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BLOCK_THRESHOLD = 0.85


def load_set(name):
    feats = np.load(ACT_DIR / f"{name}.npy")
    with open(ACT_DIR / f"{name}.prompts.json", encoding="utf-8") as f:
        prompts = json.load(f)
    return feats, prompts


def load_dataset():
    feature_matrix = np.load(ROOT / "checkpoints/feature_matrix.npy")
    with open(ROOT / "data/processed/iris_dataset_balanced.json", encoding="utf-8") as f:
        data = json.load(f)
    labels = np.array([d["label"] for d in data])
    return feature_matrix, labels


def train_detector(X_train, y_train, top_k, C):
    """Replicates src/app.py's two-stage feature-selection pipeline."""
    screen = LR(random_state=42, max_iter=1000, solver="lbfgs", C=0.01)
    screen.fit(X_train, y_train)
    order = np.argsort(np.abs(screen.coef_[0]))[::-1]
    top = order[:top_k]
    final = LR(random_state=42, max_iter=1000, solver="lbfgs", C=C)
    final.fit(X_train[:, top], y_train)
    return final, top


def evaluate(det, top, X, label_name, threshold=BLOCK_THRESHOLD):
    """Return mean prob, block-rate at threshold, list of (prompt_idx, prob)."""
    probs = det.predict_proba(X[:, top])[:, 1]
    rate = float((probs >= threshold).mean())
    return {
        "mean_prob": float(probs.mean()),
        f"{label_name}_rate_at_{threshold}": rate,
        "min_prob": float(probs.min()),
        "max_prob": float(probs.max()),
    }


def abc_filter(feats_benign, feats_inj, feats_control):
    """Apply the Scaling Monosemanticity A/B/C filter.

    Returns (intent_discriminators, overlap_features) with the same strict
    thresholds used in the identity/command analyses:
      - intent-discriminators: fires on >=60% of injection, <20% of benign
      - overlap: fires on >90% of both benign and injection, <30% of control
    """
    frac_benign = (feats_benign > 0).mean(0)
    frac_inj = (feats_inj > 0).mean(0)
    frac_ctrl = (feats_control > 0).mean(0)
    mean_benign = feats_benign.mean(0)
    mean_inj = feats_inj.mean(0)
    mean_ctrl = feats_control.mean(0)

    intent_mask = (frac_inj >= 0.6) & (frac_benign < 0.2)
    intent_score = np.where(intent_mask, mean_inj - mean_benign, -np.inf)
    top_intent = np.argsort(intent_score)[::-1]
    intent_ids = [int(i) for i in top_intent
                  if intent_score[i] != -np.inf][:20]

    overlap_mask = (frac_benign >= 0.9) & (frac_inj >= 0.9) & (frac_ctrl < 0.3)
    overlap_score = np.where(overlap_mask,
                             np.minimum(mean_benign, mean_inj) - mean_ctrl,
                             -np.inf)
    top_overlap = np.argsort(overlap_score)[::-1]
    overlap_ids = [int(i) for i in top_overlap
                   if overlap_score[i] != -np.inf][:20]

    return {
        "n_intent_discriminators": len(intent_ids),
        "top_intent_discriminators": intent_ids,
        "n_overlap_features": len(overlap_ids),
        "top_overlap_features": overlap_ids,
        "frac_active_benign": float(frac_benign.mean()),
        "frac_active_injection": float(frac_inj.mean()),
        "frac_active_control": float(frac_ctrl.mean()),
    }


def main():
    print("Loading dataset and activations...")
    feature_matrix, labels = load_dataset()
    train_idx, test_idx = tts(
        np.arange(len(labels)), test_size=0.2,
        stratify=labels, random_state=42,
    )
    X_train_base = feature_matrix[train_idx]
    y_train_base = labels[train_idx]
    X_test = feature_matrix[test_idx]
    y_test = labels[test_idx]

    feats_A, _ = load_set("A_benign_identity")
    feats_B, _ = load_set("B_injection_identity")
    feats_C, _ = load_set("C_mundane_control")
    feats_D, _ = load_set("D_benign_command")
    feats_E, _ = load_set("E_injection_command")
    feats_F, _ = load_set("F_benign_roleplay")
    feats_G, _ = load_set("G_adversarial_roleplay")

    # -----------------------------------------------------------------
    # Q1: A/B/C filter on the roleplay category (F vs G vs C)
    # -----------------------------------------------------------------
    print("\n[Q1] A/B/C filter on roleplay domain...")
    roleplay_filter = abc_filter(feats_F, feats_G, feats_C)
    print(f"  intent-discriminators: {roleplay_filter['n_intent_discriminators']} strict")
    print(f"  overlap features:      {roleplay_filter['n_overlap_features']} strict")

    # -----------------------------------------------------------------
    # Detector (a): baseline (no augmentation)
    # -----------------------------------------------------------------
    print("\n[Q2] Training baseline (no augmentation)...")
    det_base, top_base = train_detector(X_train_base, y_train_base,
                                        top_k=50, C=0.0001)
    base_heldout_f1 = f1_score(y_test, det_base.predict(X_test[:, top_base]))
    base_heldout_acc = accuracy_score(y_test, det_base.predict(X_test[:, top_base]))
    print(f"  held-out F1={base_heldout_f1:.3f}, acc={base_heldout_acc:.3f}")

    # -----------------------------------------------------------------
    # Detector (b): current production — A/B/D/E augmented
    # -----------------------------------------------------------------
    print("\nTraining A/B/D/E-augmented (current production config)...")
    X_abde = np.concatenate([X_train_base, feats_A, feats_D, feats_B, feats_E])
    y_abde = np.concatenate([
        y_train_base,
        np.zeros(len(feats_A), dtype=int), np.zeros(len(feats_D), dtype=int),
        np.ones(len(feats_B), dtype=int),  np.ones(len(feats_E), dtype=int),
    ])
    det_abde, top_abde = train_detector(X_abde, y_abde, top_k=500, C=0.01)
    abde_heldout_f1 = f1_score(y_test, det_abde.predict(X_test[:, top_abde]))
    abde_heldout_acc = accuracy_score(y_test, det_abde.predict(X_test[:, top_abde]))
    print(f"  held-out F1={abde_heldout_f1:.3f}, acc={abde_heldout_acc:.3f}")

    # -----------------------------------------------------------------
    # Detector (c): extended — A/B/D/E/F/G augmented
    # -----------------------------------------------------------------
    print("\n[Q3] Training A/B/D/E/F/G-augmented (extended with roleplay)...")
    X_all = np.concatenate([X_abde, feats_F, feats_G])
    y_all = np.concatenate([
        y_abde, np.zeros(len(feats_F), dtype=int),
                np.ones(len(feats_G), dtype=int),
    ])
    det_all, top_all = train_detector(X_all, y_all, top_k=500, C=0.01)
    all_heldout_f1 = f1_score(y_test, det_all.predict(X_test[:, top_all]))
    all_heldout_acc = accuracy_score(y_test, det_all.predict(X_test[:, top_all]))
    print(f"  held-out F1={all_heldout_f1:.3f}, acc={all_heldout_acc:.3f}")

    # -----------------------------------------------------------------
    # Evaluate all three detectors on F and G
    # -----------------------------------------------------------------
    results = {
        "BLOCK_THRESHOLD": BLOCK_THRESHOLD,
        "prompt_counts": {
            "F_benign_roleplay": int(len(feats_F)),
            "G_adversarial_roleplay": int(len(feats_G)),
            "heldout_test": int(len(y_test)),
        },
        "Q1_abc_filter_roleplay": roleplay_filter,
        "detectors": {
            "baseline_top50_C0.0001": {
                "held_out_f1": float(base_heldout_f1),
                "held_out_acc": float(base_heldout_acc),
                "F_metrics": evaluate(det_base, top_base, feats_F, "F_fpr"),
                "G_metrics": evaluate(det_base, top_base, feats_G, "G_recall"),
            },
            "abde_augmented_top500_C0.01": {
                "held_out_f1": float(abde_heldout_f1),
                "held_out_acc": float(abde_heldout_acc),
                "F_metrics": evaluate(det_abde, top_abde, feats_F, "F_fpr"),
                "G_metrics": evaluate(det_abde, top_abde, feats_G, "G_recall"),
            },
            "abdefg_augmented_top500_C0.01": {
                "held_out_f1": float(all_heldout_f1),
                "held_out_acc": float(all_heldout_acc),
                "F_metrics": evaluate(det_all, top_all, feats_F, "F_fpr"),
                "G_metrics": evaluate(det_all, top_all, feats_G, "G_recall"),
            },
        },
    }

    # -----------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"Set F (benign roleplay): FPR at threshold {BLOCK_THRESHOLD}")
    print(f"  baseline         : {results['detectors']['baseline_top50_C0.0001']['F_metrics']['F_fpr_rate_at_0.85']:.3f}  (mean prob {results['detectors']['baseline_top50_C0.0001']['F_metrics']['mean_prob']:.3f})")
    print(f"  A/B/D/E augmented: {results['detectors']['abde_augmented_top500_C0.01']['F_metrics']['F_fpr_rate_at_0.85']:.3f}  (mean prob {results['detectors']['abde_augmented_top500_C0.01']['F_metrics']['mean_prob']:.3f})")
    print(f"  A-G augmented    : {results['detectors']['abdefg_augmented_top500_C0.01']['F_metrics']['F_fpr_rate_at_0.85']:.3f}  (mean prob {results['detectors']['abdefg_augmented_top500_C0.01']['F_metrics']['mean_prob']:.3f})")
    print()
    print(f"Set G (adversarial roleplay): recall at threshold {BLOCK_THRESHOLD}")
    print(f"  baseline         : {results['detectors']['baseline_top50_C0.0001']['G_metrics']['G_recall_rate_at_0.85']:.3f}  (mean prob {results['detectors']['baseline_top50_C0.0001']['G_metrics']['mean_prob']:.3f})")
    print(f"  A/B/D/E augmented: {results['detectors']['abde_augmented_top500_C0.01']['G_metrics']['G_recall_rate_at_0.85']:.3f}  (mean prob {results['detectors']['abde_augmented_top500_C0.01']['G_metrics']['mean_prob']:.3f})")
    print(f"  A-G augmented    : {results['detectors']['abdefg_augmented_top500_C0.01']['G_metrics']['G_recall_rate_at_0.85']:.3f}  (mean prob {results['detectors']['abdefg_augmented_top500_C0.01']['G_metrics']['mean_prob']:.3f})")
    print()
    print(f"Held-out test set (unchanged 200 prompts): F1")
    print(f"  baseline         : {base_heldout_f1:.3f}")
    print(f"  A/B/D/E augmented: {abde_heldout_f1:.3f}")
    print(f"  A-G augmented    : {all_heldout_f1:.3f}")
    print("=" * 78)

    out = RESULTS_DIR / "roleplay_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
