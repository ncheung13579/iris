#!/usr/bin/env python3
"""
IRIS — Neural IDS for LLM Agent Pipelines
==========================================

One-command launcher. Run this script to install dependencies,
verify checkpoints, and launch the interactive dashboard.

Usage:
    python launch.py

The dashboard will open in your default browser. If running on
Google Colab, a public URL will be generated automatically.
"""

import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent

    # ── Step 1: Install dependencies ──────────────────────────────
    print("=" * 60)
    print("IRIS — Installing dependencies...")
    print("=" * 60)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "-r", str(root / "requirements.txt"),
    ])
    print()

    # ── Step 2: Verify checkpoints ────────────────────────────────
    print("=" * 60)
    print("IRIS — Verifying checkpoints...")
    print("=" * 60)
    required = {
        "checkpoints/sae_d6144_lambda1e-04.pt": "Trained SAE (run notebook 02)",
        "checkpoints/sensitivity_scores.npy": "Sensitivity scores (run notebook 05)",
        "checkpoints/feature_matrix.npy": "Feature matrix (run notebook 05)",
        "data/processed/iris_dataset_balanced.json": "Dataset (run notebook 02)",
    }
    missing = []
    for fpath, desc in required.items():
        p = root / fpath
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [OK]      {fpath} ({size_mb:.1f} MB)")
        else:
            print(f"  [MISSING] {fpath}")
            print(f"            -> {desc}")
            missing.append(fpath)

    if missing:
        print(f"\nMissing {len(missing)} required file(s).")
        print("Run the research notebooks (01-07) on Google Colab first,")
        print("or copy checkpoint files from Google Drive.")
        sys.exit(1)

    print("\nAll checkpoints verified.\n")

    # ── Step 3: Detect environment ────────────────────────────────
    in_colab = "google.colab" in sys.modules
    share = in_colab  # Public URL on Colab, local-only otherwise

    # ── Step 4: Launch ────────────────────────────────────────────
    print("=" * 60)
    print("IRIS — Launching Neural IDS Dashboard...")
    print("=" * 60)
    print()

    # Add project root to path so src/ imports work
    sys.path.insert(0, str(root))

    from src.app import IRISPipeline, build_app

    pipeline = IRISPipeline(str(root))
    pipeline.load()

    app = build_app(pipeline)
    app.launch(
        share=share,
        inbrowser=not in_colab,  # Auto-open browser locally
    )


if __name__ == "__main__":
    main()
