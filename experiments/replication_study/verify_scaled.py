"""Verify the unified augmentation metrics hold with scaled (50-prompt) sets.

Re-computes the key metrics from results/unified_augmentation.json using
the current 50-prompt activation files and prints a comparison table.
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

BLOCK = 0.85


def load(name):
    return np.load(ACT_DIR / f"{name}.npy")


def train_detector(X_train, y_train, top_k, C):
    screen = LR(random_state=42, max_iter=1000, solver="lbfgs", C=0.01)
    screen.fit(X_train, y_train)
    order = np.argsort(np.abs(screen.coef_[0]))[::-1]
    top = order[:top_k]
    final = LR(random_state=42, max_iter=1000, solver="lbfgs", C=C)
    final.fit(X_train[:, top], y_train)
    return final, top


def rate(det, top, X, thresh=BLOCK):
    return float((det.predict_proba(X[:, top])[:, 1] >= thresh).mean())


def main():
    feature_matrix = np.load(ROOT / "checkpoints/feature_matrix.npy")
    with open(ROOT / "data/processed/iris_dataset_balanced.json", encoding="utf-8") as f:
        data = json.load(f)
    labels = np.array([d["label"] for d in data])
    train_idx, test_idx = tts(np.arange(len(labels)), test_size=0.2,
                              stratify=labels, random_state=42)

    A = load("A_benign_identity"); B = load("B_injection_identity")
    D = load("D_benign_command");  E = load("E_injection_command")
    F = load("F_benign_roleplay"); G = load("G_adversarial_roleplay")
    C = load("C_mundane_control")

    X0 = feature_matrix[train_idx]
    y0 = labels[train_idx]
    X_test = feature_matrix[test_idx]
    y_test = labels[test_idx]

    # Historical baseline used in the original replication study: ALL 10240
    # features with C=0.01 (pre-top-K-selection state). This is what produced
    # the 96% A_fpr in retrain_comparison.json. Not the current pre-augmentation
    # production config (top-50, C=0.0001), which has different failure shape.
    det_base_all = LR(random_state=42, max_iter=1000, solver="lbfgs", C=0.01)
    det_base_all.fit(X0, y0)
    top_all_feats = np.arange(X0.shape[1])

    # Also include the top-50, C=0.0001 configuration for completeness (this
    # is the pre-augmentation config currently in app.py).
    det_base, top_base = train_detector(X0, y0, top_k=50, C=0.0001)
    # Unified (A/B/D/E)
    X_abde = np.concatenate([X0, A, D, B, E])
    y_abde = np.concatenate([y0,
        np.zeros(len(A)), np.zeros(len(D)),
        np.ones(len(B)),  np.ones(len(E))])
    det_u, top_u = train_detector(X_abde, y_abde.astype(int), top_k=500, C=0.01)
    # Extended (A/B/D/E/F/G)
    X_all = np.concatenate([X_abde, F, G])
    y_all = np.concatenate([y_abde, np.zeros(len(F)), np.ones(len(G))])
    det_all, top_all = train_detector(X_all, y_all.astype(int), top_k=500, C=0.01)

    print(f"\n{'='*88}")
    print(f"{'Metric':<35}{'base_all10240':>15}{'base_top50':>13}{'unified':>10}{'A-G':>10}")
    print("=" * 88)

    for label, X in [("Set A FPR (benign identity)", A),
                     ("Set D FPR (benign command)", D),
                     ("Set F FPR (benign roleplay)", F)]:
        ra_base = rate(det_base_all, top_all_feats, X)
        r0 = rate(det_base, top_base, X)
        ru = rate(det_u, top_u, X)
        ra = rate(det_all, top_all, X)
        print(f"{label:<35}{ra_base:>15.3f}{r0:>13.3f}{ru:>10.3f}{ra:>10.3f}")

    for label, X in [("Set B recall (inj identity)", B),
                     ("Set E recall (inj command)", E),
                     ("Set G recall (inj roleplay)", G)]:
        ra_base = rate(det_base_all, top_all_feats, X)
        r0 = rate(det_base, top_base, X)
        ru = rate(det_u, top_u, X)
        ra = rate(det_all, top_all, X)
        print(f"{label:<35}{ra_base:>15.3f}{r0:>13.3f}{ru:>10.3f}{ra:>10.3f}")

    f1_all_base = f1_score(y_test, det_base_all.predict(X_test))
    f1_base = f1_score(y_test, det_base.predict(X_test[:, top_base]))
    f1_u = f1_score(y_test, det_u.predict(X_test[:, top_u]))
    f1_all = f1_score(y_test, det_all.predict(X_test[:, top_all]))
    print(f"{'Held-out F1 (original 200)':<35}{f1_all_base:>15.3f}{f1_base:>13.3f}{f1_u:>10.3f}{f1_all:>10.3f}")
    print("=" * 88)

    out = RESULTS_DIR / "unified_augmentation_scaled50.json"
    results = {
        "prompts_per_set": 50,
        "block_threshold": BLOCK,
        "baseline_top50_C1e-4": {
            "A_fpr": rate(det_base, top_base, A),
            "B_recall": rate(det_base, top_base, B),
            "D_fpr": rate(det_base, top_base, D),
            "E_recall": rate(det_base, top_base, E),
            "F_fpr": rate(det_base, top_base, F),
            "G_recall": rate(det_base, top_base, G),
            "held_out_f1": float(f1_base),
        },
        "unified_ABDE_top500_C1e-2": {
            "A_fpr": rate(det_u, top_u, A),
            "B_recall": rate(det_u, top_u, B),
            "D_fpr": rate(det_u, top_u, D),
            "E_recall": rate(det_u, top_u, E),
            "F_fpr": rate(det_u, top_u, F),
            "G_recall": rate(det_u, top_u, G),
            "held_out_f1": float(f1_u),
        },
        "extended_ABDEFG_top500_C1e-2": {
            "A_fpr": rate(det_all, top_all, A),
            "B_recall": rate(det_all, top_all, B),
            "D_fpr": rate(det_all, top_all, D),
            "E_recall": rate(det_all, top_all, E),
            "F_fpr": rate(det_all, top_all, F),
            "G_recall": rate(det_all, top_all, G),
            "held_out_f1": float(f1_all),
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
