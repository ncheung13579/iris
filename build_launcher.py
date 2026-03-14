"""Builder script: generates notebooks/09_launch_app.ipynb."""
import json
from pathlib import Path

cells = []

def md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [source],
    })

def code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "execution_count": None,
        "outputs": [],
    })

# Cell 1: Title
md(
    "# IRIS Detection Dashboard\n\n"
    "**One-click launcher** for the IRIS prompt injection detection tool.\n\n"
    "This notebook installs dependencies, loads all pre-trained checkpoints, "
    "and launches an interactive Gradio web app with a public URL.\n\n"
    "**Requirements:** Run on Google Colab with GPU runtime (T4 recommended). "
    "Checkpoints and dataset must be in Google Drive at `/content/drive/MyDrive/iris/`.\n\n"
    "**Startup time:** ~60 seconds (model loading). After that, analysis is real-time."
)

# Cell 2: Setup (mount + install + path)
code(
    "# === Mount Google Drive and install dependencies ===\n"
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "!pip install -r /content/drive/MyDrive/iris/requirements.txt -q\n"
    "!pip install gradio -q\n"
    "\n"
    "import sys, os\n"
    "os.chdir('/content/drive/MyDrive/iris')\n"
    "sys.path.insert(0, '.')"
)

# Cell 3: Verify checkpoints
code(
    "# === Verify all checkpoints are present ===\n"
    "from pathlib import Path\n"
    "required = [\n"
    "    'checkpoints/sae_d6144_lambda1e-04.pt',\n"
    "    'checkpoints/sensitivity_scores.npy',\n"
    "    'checkpoints/feature_matrix.npy',\n"
    "    'data/processed/iris_dataset_balanced.json',\n"
    "    'results/metrics/c3_detection_comparison.json',\n"
    "    'results/metrics/c4_adversarial_evasion.json',\n"
    "]\n"
    "for f in required:\n"
    "    p = Path(f)\n"
    "    status = 'OK' if p.exists() else 'MISSING'\n"
    "    print(f'  [{status}] {f}')\n"
    "    assert p.exists(), f'Missing: {f}'\n"
    "print('\\nAll checkpoints verified.')"
)

# Cell 4: Launch
code(
    "# === Launch the IRIS Detection Dashboard ===\n"
    "from src.app import launch\n"
    "launch(project_root='.', share=True)"
)

# Assemble notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = Path("notebooks/09_launch_app.ipynb")
out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Generated {out} ({len(cells)} cells)")
