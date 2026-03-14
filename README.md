# IRIS — Interpretability Research for Injection Security

IRIS investigates whether sparse autoencoders (SAEs) trained on a language model's internal activations can detect prompt injection attacks by identifying injection-sensitive features in the model's representations.

**Course:** CSSD 2221 — Vulnerabilities & Classifications
**Term:** Winter 2026, York University (Lassonde School of Engineering)

---

## Quick Start

### Launch the Dashboard (fastest)

Open `notebooks/09_launch_app.ipynb` in Google Colab and run all cells. This installs dependencies, loads the pre-trained models, and launches an interactive Gradio web app with a public URL. The professor can analyze prompts, test evasion strategies, and review security analysis — all from a browser, no notebook interaction needed.

### Prerequisites

- Python 3.10+
- Google Colab account (for GPU access) or a local machine with a CUDA-capable GPU
- ~2 GB free disk space for model weights and cached activations

### Setup

```bash
git clone https://github.com/ncheung13579/iris.git
cd iris
pip install -r requirements.txt
```

### Running

Notebooks are numbered and designed to be run in order. Each notebook checks for prerequisites from earlier stages.

```
notebooks/
├── 01_data_exploration.ipynb     # Explore and validate the dataset
├── 02_classical_baseline.ipynb   # Text-based injection detection (sklearn)
├── 03_activation_analysis.ipynb  # Extract and analyze transformer activations
├── 04_sae_training.ipynb         # Train the sparse autoencoder
├── 05_feature_analysis.ipynb     # Identify injection-sensitive SAE features
├── 06_detection_pipeline.ipynb   # Build and evaluate the SAE-based detector
├── 07_adversarial_evasion.ipynb  # Test evasion attacks against the detector
├── 08_demo.ipynb                 # Full pipeline demo (loads pre-trained checkpoints)
└── 09_launch_app.ipynb           # One-click launcher for the Gradio dashboard
```

**To launch the interactive tool:** Run `09_launch_app.ipynb` on Colab (3 cells, ~60s startup).

**To verify results without retraining:** Start at `08_demo.ipynb`. It loads pre-trained checkpoints from `checkpoints/` and demonstrates the full pipeline in minutes.

**To replicate from scratch:** Run notebooks 01 through 07 in order. Full training takes approximately 60-90 minutes on a T4 GPU.

---

## Project Structure

```
iris/
├── CLAUDE.md              # AI development conventions and safety guardrails
├── README.md              # This file
├── requirements.txt       # Pinned dependencies
├── docs/                  # All project documentation
├── notebooks/             # Jupyter notebooks (exploration, training, demo)
├── src/                   # Reusable Python modules
│   ├── app.py             # Gradio dashboard (IRIS Detection Dashboard)
│   ├── data/              # Dataset loading and preprocessing
│   ├── model/             # TransformerLens wrapper
│   ├── sae/               # Sparse autoencoder architecture and training
│   ├── analysis/          # Feature analysis and detection pipeline
│   └── utils/             # Seeding, device management, checkpointing
├── checkpoints/           # Trained model weights (.pt files)
├── data/                  # Raw and processed datasets
└── results/               # Generated figures, metrics, and dashboards
```

See `docs/Design_Document.md` for full architectural details.

---

## Results Summary

### J1: Activation Separability — PASS

Layer 0 of GPT-2 Small shows strong separability between normal and injection prompts (silhouette score = 0.315, Cohen's d = 10.20). This confirms that injection-relevant signal exists in the residual stream and justifies training the SAE at layer 0.

### C3: Detection Comparison

| Approach | F1 | AUC |
|---|---|---|
| TF-IDF + Logistic Regression | 0.956 | 0.992 |
| TF-IDF + Random Forest | 0.966 | 0.988 |
| Raw Activations + LogReg | 0.915 | 0.966 |
| **SAE Features (all) + LogReg** | **0.946** | **0.973** |
| SAE Top-100 Features + LogReg | 0.905 | 0.957 |
| SAE Top-50 Features + LogReg | 0.834 | 0.924 |
| SAE Top-10 Features + LogReg | 0.715 | 0.800 |

The SAE-based detector achieves F1 = 0.946 using interpretable features, competitive with the black-box TF-IDF baseline (F1 = 0.956). Only 100 of 6144 SAE features are needed to reach F1 = 0.905.

### C4: Adversarial Evasion

| Strategy | Evasion Rate | Description |
|---|---|---|
| Encoded (l33t speak) | 0% (0/12) | Formatting tricks do not evade the detector |
| Subtle | 0% (0/12) | Short, casual injections are caught |
| Paraphrased | 23% (3/13) | Rewording partially evades detection |
| **Mimicry** | **100% (13/13)** | Educational-sounding injections fully evade |
| **Overall** | **32% (16/50)** | |

Mimicry attacks (injections disguised as educational questions) expose a blind spot — the detector relies on stylistic rather than semantic features at layer 0. See `docs/Project_Report.md` §5.4 for analysis.

---

## Checkpoints

Trained model weights are not checked into git due to size (~160 MB total). They are generated by running notebooks 01-05 on Google Colab and stored on Google Drive.

To verify checkpoint availability:
```bash
python checkpoints/download_checkpoints.py
```

Required files:
- `sae_d6144_lambda1e-04.pt` — Trained sparse autoencoder (113 MB)
- `j2_activations.npz` — Cached GPT-2 activations
- `sensitivity_scores.npy` — Per-feature injection sensitivity scores
- `feature_matrix.npy` — SAE feature activations for all 1000 prompts

---

## Documentation

| Document | Purpose |
|---|---|
| `docs/Project_Pitch_IRIS.md` | Professor-facing project proposal |
| `docs/Design_Document.md` | Architecture, design decisions, experiment plan |
| `docs/Project_Report.md` | Comprehensive writeup (background, methodology, results) |
| `docs/security/STRIDE_Analysis.md` | STRIDE threat model of the LLM agent pipeline |
| `docs/security/Kill_Chain.md` | Kill chain decomposition of prompt injection attacks |

---

## License

This project is academic coursework. All code is original or properly attributed.
