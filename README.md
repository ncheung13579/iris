# IRIS — Neural IDS for LLM Agent Pipelines

IRIS is an interactive security tool that detects prompt injection attacks in LLM agent pipelines by monitoring neural activation patterns using Sparse Autoencoders (SAEs). It works like a network IDS — but instead of inspecting packets, it inspects the model's internal representations.

**Course:** CSSD 2221 — Introduction to Security
**Term:** Winter 2026, York University

---

## How to Run

```
python launch.py
```

That's it. The script installs all dependencies, verifies that the pre-trained models are present, and opens the IRIS dashboard in your browser.

**Requirements:** Python 3.10, 3.11, or 3.12 (3.13+ is not yet supported by PyTorch/numpy). An internet connection is needed for the first-time dependency install. A CUDA GPU is recommended for faster inference but not required — the tool works on CPU.

---

## The Tool

IRIS provides six interactive tabs:

### Tab 1: Live Analysis
Type any prompt and get an instant threat assessment. The tool runs it through GPT-2, decomposes the activations with the trained SAE, and classifies the result using two detection layers:
- **Layer 1 (Anomaly-based):** SAE neural feature patterns — like a behavioral IDS
- **Layer 2 (Signature-based):** TF-IDF text pattern matching — like Snort

Shows a verdict banner, threat probability, top 10 triggered signature IDs, and a natural-language alert explanation. Includes pre-loaded examples for normal traffic, obvious injections, encoded attacks, and mimicry.

### Tab 2: Neural IDS Console
A SOC-analyst-style monitoring dashboard. Process batches of prompts and see them logged with timestamps, signature IDs, severity ratings, and verdicts — like watching a Splunk/ELK SIEM feed. Export the session log as CSV for offline analysis.

### Tab 3: Signature Management
Browse all 6,144 learned detection signatures (SAE features). Each signature has an ID, direction (injection/normal), and confidence score — analogous to managing Snort/Suricata rule sets. Inspect any signature to see which prompts trigger it. Includes a signature ablation table showing how detection performance changes as you enable more signatures.

### Tab 4: Red Team Lab
A 5-level penetration testing exercise. Progress from crafting basic injections (Level 1) to advanced mimicry attacks (Level 4) and free-form APT-style attacks (Level 5). Each level explains the network security parallel. Generates a pentest report at the end.

### Tab 5: Evasion Lab
Side-by-side comparison of an original injection vs. a modified evasion attempt. See how signature activations change when you modify the attack — a live version of experiment C4. Analogous to testing IDS bypass techniques in a security lab.

### Tab 6: System Analysis
Static display of the project's security analysis: STRIDE threat model, kill chain decomposition, defense-in-depth architecture, and a concept mapping table linking every IRIS component to its network security analogue.

---

## Network Security Concept Mapping

| IRIS Component | Network Security Analogue | Function |
|---|---|---|
| SAE feature activations | Packet payload inspection | Deep content analysis |
| Sensitivity scores | IDS signature rules (Snort SIDs) | Pattern matching confidence |
| Feature thresholds | Firewall allow/deny rules | Binary pass/block decision |
| TF-IDF detector | Signature-based IDS (Snort) | Known-pattern matching |
| SAE detector | Anomaly-based IDS (behavioral) | Deviation from baseline |
| Dual-detector consensus | Defense-in-depth | Multiple detection layers |
| Evasion Lab | Penetration testing | Adversarial robustness |
| Mimicry evasion | Zero-day exploit | No existing signature |
| Top-K feature selection | Ruleset tuning | Reduce alert fatigue |
| IDS Console log | SIEM (Splunk/ELK) | Centralized monitoring |

---

## Key Results

| Metric | Value |
|---|---|
| Activation separability (J1) | PASS (Cohen's d = 10.2) |
| SAE detection F1 (C3) | 0.946 (vs TF-IDF: 0.956) |
| SAE detection AUC (C3) | 0.973 (vs TF-IDF: 0.992) |
| Evasion rate — encoded (C4) | 0% |
| Evasion rate — subtle (C4) | 0% |
| Evasion rate — paraphrased (C4) | 23% |
| Evasion rate — mimicry (C4) | 100% (zero-day equivalent) |
| Overall evasion rate (C4) | 32% |

---

## Project Structure

```
iris/
├── launch.py                  # <-- Run this
├── src/
│   ├── app.py                 # IRIS Detection Dashboard (Gradio)
│   ├── data/                  # Dataset loading and preprocessing
│   ├── model/                 # GPT-2 wrapper (TransformerLens)
│   ├── sae/                   # Sparse autoencoder architecture
│   ├── analysis/              # Feature analysis, detection, adversarial
│   ├── baseline/              # TF-IDF and activation baselines
│   └── utils/                 # Seeding, device management
├── notebooks/                 # Research notebooks (background material)
├── checkpoints/               # Pre-trained models (included)
├── results/                   # Figures and metrics
├── docs/                      # Report, STRIDE, kill chain
└── requirements.txt
```

## Research Notebooks

The `notebooks/` directory documents the full research process behind the tool. These are background material — the tool itself runs independently of them.

| Notebook | Purpose |
|---|---|
| 01 | Data exploration + activation separability (J1) |
| 02 | SAE training iterations (J2) |
| 03 | Feature inspection + interpretability (J3) |
| 04 | Formal SAE evaluation (C1) |
| 05 | Injection-sensitivity analysis (C2) |
| 06 | Detection pipeline comparison (C3) |
| 07 | Adversarial evasion testing (C4) |
| 08 | Full pipeline demo |

## Documentation

| Document | Purpose |
|---|---|
| `docs/Tutorial.md` | Step-by-step guide to using the dashboard |
| `docs/Project_Report.md` | Comprehensive project report |
| `docs/Design_Document.md` | Architecture and experiment plan |
| `docs/security/STRIDE_Analysis.md` | STRIDE threat model |
| `docs/security/Kill_Chain.md` | Kill chain decomposition |

The tutorial is also available inside the dashboard under the **Guide** tab.
