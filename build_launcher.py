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
    "# IRIS — Neural IDS for LLM Agent Pipelines\n\n"
    "**One-click launcher** for the IRIS Detection Dashboard.\n\n"
    "This notebook installs dependencies, loads all pre-trained models, "
    "and launches an interactive Gradio web app with a **public URL** "
    "you can open in any browser.\n\n"
    "### What you get\n"
    "| Tab | Description | Network Analogue |\n"
    "|---|---|---|\n"
    "| Live Analysis | Analyze any prompt in real time | Packet inspection |\n"
    "| Neural IDS Console | SOC-style monitoring dashboard | SIEM/Splunk |\n"
    "| Signature Management | Browse & toggle detection rules | Snort rule manager |\n"
    "| Red Team Lab | 5-level pentest challenge | Penetration test |\n"
    "| Evasion Lab | Adversarial evasion testing | IDS bypass lab |\n"
    "| System Analysis | STRIDE, kill chain, metrics | Threat intel |\n\n"
    "**Runtime:** T4 GPU recommended. Startup takes ~60 seconds (model loading). "
    "After that, analysis is real-time.\n\n"
    "**Instructions:** Select `Runtime > Run all` or run each cell in order."
)

# Cell 2: Setup
code(
    "# === Mount Google Drive and install dependencies ===\n"
    "from google.colab import drive\n"
    "drive.mount('/content/drive')\n"
    "!pip install -r /content/drive/MyDrive/iris/requirements.txt -q\n"
    "\n"
    "import sys, os\n"
    "os.chdir('/content/drive/MyDrive/iris')\n"
    "sys.path.insert(0, '.')\n"
    "print('Setup complete.')"
)

# Cell 3: Verify checkpoints
code(
    "# === Verify all required files are present ===\n"
    "from pathlib import Path\n"
    "required = [\n"
    "    'checkpoints/sae_d6144_lambda1e-04.pt',\n"
    "    'checkpoints/sensitivity_scores.npy',\n"
    "    'checkpoints/feature_matrix.npy',\n"
    "    'data/processed/iris_dataset_balanced.json',\n"
    "    'results/metrics/c3_detection_comparison.json',\n"
    "    'results/metrics/c4_adversarial_evasion.json',\n"
    "]\n"
    "all_ok = True\n"
    "for f in required:\n"
    "    p = Path(f)\n"
    "    status = 'OK' if p.exists() else 'MISSING'\n"
    "    if not p.exists(): all_ok = False\n"
    "    print(f'  [{status}] {f}')\n"
    "assert all_ok, 'Missing files! Run notebooks 01-07 first.'\n"
    "print('\\nAll files verified.')"
)

# Cell 4: Launch
code(
    "# === Launch the IRIS Detection Dashboard ===\n"
    "# This loads GPT-2, the SAE, and all detectors (~60 seconds),\n"
    "# then opens a Gradio web app with a public URL.\n"
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
