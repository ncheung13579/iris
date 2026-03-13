# IRIS — Interpretability Research for Injection Security

IRIS investigates whether sparse autoencoders (SAEs) trained on a language model's internal activations can detect prompt injection attacks by identifying injection-sensitive features in the model's representations.

**Course:** CSSD 2221 — Vulnerabilities & Classifications
**Term:** Winter 2026, York University (Lassonde School of Engineering)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Google Colab account (for GPU access) or a local machine with a CUDA-capable GPU
- ~2 GB free disk space for model weights and cached activations

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/iris.git
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
└── 08_demo.ipynb                 # Full pipeline demo (loads pre-trained checkpoints)
```

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

## Documentation

| Document | Purpose |
|---|---|
| `docs/Project_Pitch_IRIS.md` | Professor-facing project proposal |
| `docs/Design_Document.md` | Architecture, design decisions, experiment plan |
| `docs/Project_Report.md` | Comprehensive writeup (background, methodology, results) |
| `docs/security/STRIDE_Analysis.md` | STRIDE threat model of the LLM agent pipeline |

---

## License

This project is academic coursework. All code is original or properly attributed.
