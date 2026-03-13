# IRIS — Design Document

## Interpretability Research for Injection Security

**Version:** 0.1 — Pre-implementation
**Date:** March 2026
**Author:** Nathan

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline Design](#3-data-pipeline-design)
4. [Model and Analysis Pipeline](#4-model-and-analysis-pipeline)
5. [Major Design Decisions](#5-major-design-decisions)
6. [Experiment Plan](#6-experiment-plan)
7. [Directory Structure](#7-directory-structure)
8. [Reproducibility Plan](#8-reproducibility-plan)
9. [Risk Register](#9-risk-register)
10. [Timeline and Milestones](#10-timeline-and-milestones)

---

## 1. Introduction

### 1.1 Purpose

This design document describes the architecture, design decisions, experiment plan, and reproducibility strategy for IRIS (Interpretability Research for Injection Security). It serves as the planning artifact before implementation begins and will be updated as the project evolves.

### 1.2 Scope

IRIS investigates whether sparse autoencoders (SAEs) trained on a language model's internal activations can detect prompt injection attacks by identifying injection-sensitive features. The project encompasses: dataset curation, classical detection baseline, transformer activation analysis, SAE training and feature analysis, a proof-of-concept injection detector, and a STRIDE security analysis of the LLM agent pipeline.

### 1.3 Constraints

**Compute:** Google Colab free tier — T4 GPU, 15-30 GPU hours/week, 12-hour session limit, ~15 GB RAM. All training must fit within these limits. Model checkpoints are saved to Google Drive to survive session disconnects.

**Model size:** GPT-2 Small (124M parameters). This is the largest model that fits comfortably in Colab's memory when storing activations for SAE training. Larger models (GPT-2 Medium, GPT-2 Large) are out of scope for the free tier.

**Timeline:** 4 weeks. This constrains the scope of experiments — we prioritize depth on one model at one layer over breadth across many configurations.

**Presentation requirement:** All code must be explainable by the student without AI assistance. This means every design decision should have a clear rationale that can be articulated from first principles.

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           IRIS PIPELINE                                 │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────────────┐  │
│  │   STAGE 1    │   │   STAGE 2    │   │        STAGE 3            │  │
│  │  Data Layer  │──→│ Baseline     │──→│  Transformer Analysis     │  │
│  │              │   │ Detection    │   │                           │  │
│  │ • Curation   │   │ • TF-IDF    │   │ • Load GPT-2 Small       │  │
│  │ • Cleaning   │   │ • Log. Reg  │   │   via TransformerLens     │  │
│  │ • Splitting  │   │ • Rand.For. │   │ • Extract residual stream │  │
│  │ • Labeling   │   │ • Metrics   │   │   activations per layer   │  │
│  └──────────────┘   └──────────────┘   │ • Attention analysis     │  │
│                                         └─────────┬─────────────────┘  │
│                                                   │                    │
│                                                   ▼                    │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────────────┐  │
│  │   STAGE 6    │   │   STAGE 5    │   │        STAGE 4            │  │
│  │  Security    │←──│  Detection   │←──│  SAE Feature Analysis     │  │
│  │  Analysis    │   │  Pipeline    │   │                           │  │
│  │              │   │              │   │ • Train SAE on            │  │
│  │ • STRIDE     │   │ • Feature    │   │   residual stream acts.  │  │
│  │ • Kill chain │   │   monitoring │   │ • Identify injection-     │  │
│  │ • Defense    │   │ • Threshold  │   │   sensitive features      │  │
│  │   in depth   │   │   tuning     │   │ • Feature dashboards     │  │
│  │ • Evasion    │   │ • Compare vs │   │ • Top-activating         │  │
│  │   analysis   │   │   baseline   │   │   example visualization  │  │
│  └──────────────┘   └──────────────┘   └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

**Data module (`src/data/`):** Dataset loading, preprocessing, train/val/test splitting, and batch iteration. Responsible for ensuring consistent tokenization and label encoding across all pipeline stages.

**Baseline module (`src/baseline/`):** Classical ML classifiers (logistic regression, random forest) on text features. Self-contained — depends only on the data module and scikit-learn.

**Model module (`src/model/`):** TransformerLens wrapper for GPT-2 Small. Handles model loading, activation caching, and hook management. All other modules access the transformer through this interface rather than calling TransformerLens directly.

**SAE module (`src/sae/`):** Sparse autoencoder architecture definition, training loop, and feature extraction. Takes cached activations as input, produces trained SAE weights and feature activation matrices.

**Analysis module (`src/analysis/`):** Feature interpretation, injection-sensitivity scoring, visualization generation, and detection pipeline. Consumes SAE features and produces the final detection decisions and dashboards.

**Security module (`docs/security/`):** Not code — written analysis. STRIDE threat model, kill chain mapping, defense-in-depth proposal. Produced during Week 4 as a markdown document.

---

## 3. Data Pipeline Design

### 3.1 Dataset Schema

Each example in the dataset has the following fields:

| Field | Type | Description |
|---|---|---|
| `text` | string | The prompt text |
| `label` | int | 0 = normal, 1 = injection |
| `category` | string | Subcategory (e.g., "qa", "instruction", "override", "extraction", "roleplay") |
| `source` | string | Origin dataset or "synthetic" |
| `token_count` | int | Number of tokens after GPT-2 tokenization |

### 3.2 Data Sources

**Normal prompts:** Alpaca dataset (instruction-following), OpenAssistant conversations (multi-turn), and manually curated examples spanning coding, writing, analysis, and factual Q&A.

**Injection prompts:** Published prompt injection benchmarks (HuggingFace datasets), supplemented with synthetic examples covering: direct instruction override ("Ignore previous instructions and..."), indirect injection (instructions embedded in simulated retrieved documents), context manipulation (role-playing that shifts behavior), and extraction attacks (attempts to reveal system prompts).

### 3.3 Preprocessing

All prompts are tokenized using GPT-2's tokenizer (via TransformerLens). Prompts are truncated or padded to a fixed context length (128 tokens for initial experiments — short enough for fast iteration, long enough to contain meaningful injection patterns). A system prompt prefix is prepended to all examples to simulate a realistic agent setup: "You are a helpful assistant. Answer the user's question.\n\nUser: {prompt}\n\nAssistant:"

### 3.4 Splitting Strategy

Standard 70/15/15 train/validation/test split, stratified by label and category. The test set is held out until final evaluation — all hyperparameter tuning uses the validation set. This prevents information leakage from the test set into model selection decisions.

---

## 4. Model and Analysis Pipeline

### 4.1 Activation Extraction

Using TransformerLens, we extract the residual stream activation at a target layer for each prompt. The residual stream at layer $l$ is a tensor of shape `(batch, seq_len, d_model)` where `d_model = 768` for GPT-2 Small. For classification purposes, we take the activation at the final token position (the last non-padding token), producing a vector of shape `(768,)` per prompt. This is the standard approach — the final token's residual stream accumulates information from the full context.

**Layer selection:** We initially extract activations from all 12 layers and compute a simple separability metric (e.g., the silhouette score between injection and normal classes in the activation space). The layer with the highest separability is selected for SAE training. If multiple layers show strong separability, we train SAEs on 2-3 layers and compare.

### 4.2 SAE Architecture

The SAE follows the standard architecture from the mechanistic interpretability literature:

```
Input:   x ∈ R^768          (residual stream activation)
Encoder: f = ReLU(W_enc · x + b_enc)   where W_enc ∈ R^(d_sae × 768)
Decoder: x̂ = W_dec · f + b_dec         where W_dec ∈ R^(768 × d_sae)

Loss = ||x - x̂||² + λ · ||f||₁
```

**Expansion factor:** d_sae = 8 × d_model = 6144. This is the standard multiplier in the literature (Anthropic uses 8x to 32x). We start with 8x and can increase if features are not sufficiently monosemantic.

**Sparsity coefficient (λ):** Tuned on the validation set. Starting value: 1e-3. The target is approximately 50-100 active features per prompt out of 6144 total (~1-2% sparsity).

**Decoder weight normalization:** After each gradient step, decoder columns are normalized to unit norm. This is standard practice — it prevents the SAE from "cheating" by scaling decoder weights up and encoder weights down, which would reduce the effective sparsity penalty.

### 4.3 Injection-Sensitivity Scoring

After SAE training, each of the 6144 features is scored for injection sensitivity:

```
sensitivity(feature_i) = mean_activation_on_injections(feature_i) 
                        - mean_activation_on_normal(feature_i)
```

Features with high positive sensitivity activate strongly on injections. Features with high negative sensitivity are suppressed by injections. Both are informative. Features near zero are injection-neutral.

The top-K most injection-sensitive features (positive and negative) form the basis of the detector.

### 4.4 Detection Pipeline

The detector is a simple logistic regression classifier trained on the SAE feature activation vector (not the raw residual stream). This tests whether the SAE decomposition produces features that are more discriminative than raw activations. We compare three detection approaches:

1. **Classical baseline:** Logistic regression on TF-IDF text features
2. **Raw activation baseline:** Logistic regression on the 768-dim residual stream vector
3. **SAE feature detector:** Logistic regression on the 6144-dim SAE feature vector (or a subset of the top-K injection-sensitive features)

If the SAE features outperform raw activations, that's evidence that the decomposition has uncovered structure that the raw representation obscures.

---

## 5. Major Design Decisions

### 5.1 Why GPT-2 Small?

GPT-2 Small (124M parameters) is the standard model for mechanistic interpretability research. TransformerLens has extensive support for it, the model fits in Colab memory with room for activation caching, and there is abundant prior work to compare against. Larger models would be more realistic targets for prompt injection but exceed our compute constraints.

### 5.2 Why TransformerLens (Not Raw HuggingFace)?

TransformerLens provides clean hook access to every internal component (residual stream, attention patterns, MLP outputs) with a consistent API. Raw HuggingFace models require manual hook registration and careful handling of model internals. Since the project's core contribution is in the analysis — not in the model infrastructure — using TransformerLens avoids unnecessary complexity and aligns with the tooling used in published interpretability research.

### 5.3 Why a Fixed System Prompt Prefix?

All prompts are wrapped in a consistent system prompt template. This simulates the real-world setting where prompt injections occur: the model has a system-level instruction, and the attacker's goal is to override it via user input. Without this prefix, there's no trust boundary to cross — and without a trust boundary, there's no injection vulnerability.

### 5.4 Why Final-Token Activations?

The residual stream at the final token position is the standard representation used for sequence-level classification with autoregressive models. This token has "seen" the entire input (via causal attention) and its activation accumulates information about the full prompt. Alternative approaches (mean-pooling across all token positions, or using specific token positions) are possible extensions but add complexity.

### 5.5 Why Compare Against Raw Activation Baselines?

The SAE is not inherently useful just because it's an SAE. If a logistic regression on raw 768-dim activations detects injections just as well as one on 6144-dim SAE features, then the SAE decomposition adds complexity without adding detection capability. The raw activation baseline is the honest test of whether the SAE is finding structure that matters.

### 5.6 Why Logistic Regression as the Detection Classifier?

Simplicity and interpretability. A logistic regression on SAE features produces a weight per feature — the weight tells you *how much each SAE feature contributes to the injection detection decision*. This makes the detector explainable: "This prompt was flagged because SAE features #237 (instruction-boundary detection) and #1042 (imperative verb pattern) were abnormally active." A more complex classifier (neural network, random forest) might achieve higher detection accuracy but would obscure this interpretability.

---

## 6. Experiment Plan

### 6.1 Junction Experiments (Week 1 — Go/No-Go)

These experiments determine whether Path B is viable. If they fail, we pivot to Path A (CNN/OCT, already approved).

**Experiment J1: Activation separability.** Extract residual stream activations at all 12 layers for 500 normal and 500 injection prompts. Compute silhouette score and visualize with t-SNE/UMAP. **Pass criterion:** At least one layer shows visible clustering or silhouette score > 0.1.

**Experiment J2: SAE sanity check.** Train a small SAE (expansion factor 4x, 3072 features) on 5000 activation vectors from the best layer. **Pass criterion:** Reconstruction loss converges below 0.1 of the input variance, and average sparsity is below 10% active features.

**Experiment J3: Feature inspection.** For the top 20 most injection-sensitive features, examine top-10 activating prompts. **Pass criterion:** At least 5 features show a clear, interpretable pattern (not random noise).

### 6.2 Core Experiments (Weeks 2-3)

**Experiment C1: Full SAE training.** Train the production SAE (8x expansion, 6144 features) on the full activation dataset. Report reconstruction loss, sparsity statistics, and dead feature percentage.

**Experiment C2: Feature analysis.** Compute injection-sensitivity scores for all features. Visualize the distribution. Identify and characterize the top 50 injection-sensitive features (positive and negative).

**Experiment C3: Detection comparison.** Train and evaluate all three detection approaches (classical text, raw activation, SAE features). Report precision, recall, F1, and ROC-AUC on the held-out test set.

**Experiment C4: Adversarial evasion.** Craft 50 injection prompts specifically designed to evade the SAE-based detector (e.g., paraphrased injections, multi-lingual injections, injections that mimic normal instruction style). Measure evasion rate and analyze which SAE features the evasions exploit.

### 6.3 Analysis Experiments (Week 4)

**Experiment A1: Layer comparison.** If time permits, train SAEs on 2-3 different layers and compare which layer produces the most injection-discriminative features.

**Experiment A2: Ablation study.** Vary the number of SAE features used by the detector (top 10, 50, 100, all) and measure the impact on detection accuracy. This determines the minimum feature set needed for effective detection.

---

## 7. Directory Structure

```
iris/
├── CLAUDE.md                    # Claude Code safety guardrails and conventions
├── README.md                    # Quick-start guide
├── requirements.txt             # Pinned dependencies
├── .gitignore
│
├── docs/
│   ├── Project_Pitch_IRIS.md    # Professor-facing pitch
│   ├── Design_Document.md       # This document
│   ├── Project_Report.md        # Comprehensive writeup (evolves over 4 weeks)
│   └── security/
│       ├── STRIDE_Analysis.md   # Full STRIDE threat model
│       └── Kill_Chain.md        # Kill chain decomposition
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classical_baseline.ipynb
│   ├── 03_activation_analysis.ipynb
│   ├── 04_sae_training.ipynb
│   ├── 05_feature_analysis.ipynb
│   ├── 06_detection_pipeline.ipynb
│   ├── 07_adversarial_evasion.ipynb
│   └── 08_demo.ipynb            # Presentation demo notebook
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # Dataset class and loading utilities
│   │   ├── preprocessing.py     # Tokenization, padding, label encoding
│   │   └── sources.py           # Functions to fetch/curate data sources
│   ├── model/
│   │   ├── __init__.py
│   │   └── transformer.py       # TransformerLens wrapper, activation caching
│   ├── sae/
│   │   ├── __init__.py
│   │   ├── architecture.py      # SAE class definition
│   │   └── training.py          # SAE training loop
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── features.py          # Injection-sensitivity scoring
│   │   ├── detection.py         # Detection pipeline and evaluation
│   │   └── visualization.py     # Feature dashboards, plots
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Seeding, device management, checkpoint I/O
│
├── checkpoints/                 # Saved model weights (gitignored, stored on Drive)
│   ├── sae_layer_N.pt
│   └── detector.pt
│
├── data/                        # Processed datasets (gitignored if large)
│   ├── raw/
│   └── processed/
│
└── results/
    ├── figures/
    ├── metrics/
    └── feature_dashboards/
```

---

## 8. Reproducibility Plan

**Random seeds:** All randomness (PyTorch, NumPy, Python, CUDA) is seeded with a single global seed (42) set at the beginning of every notebook and script.

**Pinned dependencies:** `requirements.txt` lists exact versions of all packages (e.g., `transformer-lens==2.x.x`, `torch==2.x.x`).

**Checkpoints:** Trained model weights (.pt files) are saved to Google Drive and included in the submission. The professor can skip training and load checkpoints directly to verify results.

**Data versioning:** The curated dataset is saved as a JSON file with a SHA-256 hash. The hash is recorded in the project report so that exact reproduction is verifiable.

**Notebook ordering:** Notebooks are numbered and designed to be run sequentially. Each notebook's first cell checks for prerequisites (required checkpoints, processed data files) and prints a clear error if something is missing.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SAE features show no injection sensitivity | Medium | High — project loses its core contribution | Junction experiments (J1-J3) in Week 1; pivot to Path A if no signal |
| Colab GPU quota exhausted mid-training | Medium | Medium — delays but doesn't block | Save checkpoints every epoch; use smaller batch sizes; schedule training at off-peak hours |
| Dataset quality is poor (too easy or too noisy) | Medium | High — detector results are meaningless | Manual inspection of 100 random examples during curation; ensure diverse injection types; verify classical baseline achieves reasonable but imperfect performance (60-80%, not 99%) |
| GPT-2 Small is too simple to exhibit injection-relevant features | Low-Medium | High — SAE finds nothing interesting | GPT-2 Small is known to follow instructions at a basic level; system prompt framing helps |
| Presentation risk: can't explain SAE architecture clearly | Medium | High — professor evaluates understanding | Study Part V of ML guide thoroughly; practice explaining SAE loss function, sparsity penalty, and feature interpretation without notes |
| Scope creep | Medium | Medium — incomplete deliverables | Strict milestone enforcement; stretch goals are clearly labeled |

---

## 10. Timeline and Milestones

### Week 1: Foundation

| Day | Task | Done When |
|---|---|---|
| 1-2 | Dataset curation and preprocessing | JSON dataset file exists with ≥ 5000 examples per class |
| 2-3 | Classical baseline (sklearn) | Logistic regression and random forest metrics reported |
| 3-4 | TransformerLens setup + activation extraction | Activations cached for all 12 layers |
| 4-5 | Junction experiments J1-J3 | Go/no-go decision made |

### Week 2: Core Implementation

| Day | Task | Done When |
|---|---|---|
| 1-2 | Full SAE training | Converged SAE checkpoint saved, reconstruction quality verified |
| 3-4 | Feature analysis + injection-sensitivity scoring | Top 50 features characterized, feature dashboards generated |
| 5 | Detection pipeline | Three-way comparison (text, raw activation, SAE) metrics reported |

### Week 3: Analysis and Depth

| Day | Task | Done When |
|---|---|---|
| 1-2 | Adversarial evasion experiments | Evasion results documented |
| 3-4 | STRIDE threat model + security analysis | Written analysis complete |
| 5 | Comprehensive project document draft | Full writeup exists (will be revised in Week 4) |

### Week 4: Synthesis and Presentation

| Day | Task | Done When |
|---|---|---|
| 1-2 | Demo notebook | 08_demo.ipynb runs end-to-end, loads checkpoints, produces all visualizations |
| 3 | Project document finalization | Document reviewed and complete |
| 4-5 | Presentation preparation | Can explain every component without notes |

---
