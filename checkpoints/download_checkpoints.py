"""Verify IRIS checkpoint files are present.

The checkpoints required to run the dashboard are committed to this repo
via Git LFS. If you cloned with Git LFS installed, they were fetched
automatically. If you cloned before installing Git LFS, or `git clone`
created LFS pointer files instead of the real binaries, run:

    git lfs install
    git lfs pull

from the repo root. After that the files below should be present and
`python launch.py` should work.

Tracked-via-LFS (committed):
    sae_d10240_lambda1e-04.pt     (~300 MB) Trained sparse autoencoder
    feature_matrix.npy            (~40 MB)  SAE features on all 1,000 prompts
    sensitivity_scores.npy        (~80 KB)  Per-feature sensitivity scores
    expanded_feature_matrix.npy   (~40 MB)  v2-defense feature matrix
    expanded_sensitivity_scores.npy (~80 KB)  v2-defense sensitivity
    red_team_features.npy         (~8 MB)   Cached red-team features

Not tracked (only needed to retrain the SAE from scratch, which
requires a GPU and is out of scope for grading):
    j1_activations.npz, j2_activations.npz, expanded_activations.npz

Run this script to confirm the required files are present:

    python checkpoints/download_checkpoints.py
"""

import sys
from pathlib import Path

CHECKPOINT_DIR = Path(__file__).parent
REQUIRED_FILES = [
    "sae_d10240_lambda1e-04.pt",
    "feature_matrix.npy",
    "sensitivity_scores.npy",
]

OPTIONAL_FILES = [
    "expanded_feature_matrix.npy",
    "expanded_sensitivity_scores.npy",
    "red_team_features.npy",
]


def check_checkpoints() -> bool:
    """Verify all required checkpoint files are present and are real
    binaries (not LFS pointer files)."""
    all_present = True
    for fname in REQUIRED_FILES:
        path = CHECKPOINT_DIR / fname
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            # LFS pointer files are < 1 KB; real checkpoints are much larger.
            if path.stat().st_size < 1024:
                print(f"  [POINTER] {fname} ({size_mb:.3f} MB) "
                      f"-- looks like an LFS pointer, run `git lfs pull`")
                all_present = False
            else:
                print(f"  [OK] {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {fname}")
            all_present = False

    for fname in OPTIONAL_FILES:
        path = CHECKPOINT_DIR / fname
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if path.stat().st_size < 1024:
                print(f"  [POINTER] {fname} (run `git lfs pull`) (optional)")
            else:
                print(f"  [OK] {fname} ({size_mb:.1f} MB) (optional)")
        else:
            print(f"  [--] {fname} (optional, not required)")

    return all_present


if __name__ == "__main__":
    print("IRIS Checkpoint Verification")
    print("=" * 40)
    print(f"Checkpoint directory: {CHECKPOINT_DIR.resolve()}\n")

    if check_checkpoints():
        print("\nAll required checkpoints present. You can run: python launch.py")
    else:
        print("\nMissing or incomplete checkpoints. To obtain them:")
        print("  git lfs install")
        print("  git lfs pull")
        print("from the repo root.")
        sys.exit(1)
