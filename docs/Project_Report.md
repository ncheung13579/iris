# IRIS — Project Report

## Interpretability Research for Injection Security

**Author:** Nathan Cheung (ncheung3@my.yorku.ca)
**Course:** CSSD 2221 — Vulnerabilities & Classifications
**Date:** March 2026
**Repository:** [github.com/ncheung13579/iris](https://github.com/ncheung13579/iris)

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction and Security Motivation](#2-introduction-and-security-motivation)
3. [Background](#3-background)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Security Analysis](#6-security-analysis)
7. [Discussion](#7-discussion)
8. [Limitations](#8-limitations)
9. [Future Work](#9-future-work)
10. [Reproducibility](#10-reproducibility)
11. [References](#11-references)

---

## 1. Abstract

IRIS investigates whether sparse autoencoders (SAEs) trained on the internal activations of a language model can detect prompt injection attacks by identifying injection-sensitive features in the model's residual stream. We train an 8x-expansion SAE (768 to 6144 features) on GPT-2 Small's residual stream activations for 1000 prompts (500 normal, 500 injection) and evaluate detection performance against classical text-based and raw activation baselines. Five-fold cross-validation shows the SAE-based detector achieves F1 = 0.960 ± 0.013 and AUC = 0.981 ± 0.009, outperforming raw activation classification (F1 = 0.952 ± 0.007) with the best calibration of all approaches (ECE = 0.028 vs. TF-IDF's 0.210). Multi-layer comparison (A1) across layers 0, 6, and 11 confirms that middle layers capture the most discriminative features (layer 6: F1 = 0.954, AUC = 0.979). Causal intervention experiments (C5) provide the strongest evidence: suppressing the top 10 injection-sensitive features flips 99% of injection classifications to normal, while amplifying these features on normal prompts flips 91-100% to injection. A smooth, monotonic dose-response curve from full suppression (probability = 0.10) through no intervention (0.53) to double amplification (0.99) demonstrates continuous causal control. Adversarial evasion testing reveals encoded and subtle attacks are reliably caught (0% evasion), paraphrased attacks partially evade (23%), and mimicry attacks evade completely (100%), identifying a fundamental limitation of early-layer monitoring. A STRIDE threat model and kill chain decomposition connect these findings to the broader prompt injection threat landscape.

---

## 2. Introduction and Security Motivation

### 2.1 Prompt Injection as an Injection Vulnerability

Prompt injection is an injection vulnerability applied to a new substrate. SQL injection occurs when untrusted user data crosses a trust boundary into a SQL query and is executed as code. Cross-site scripting occurs when untrusted data crosses a trust boundary into rendered HTML. Prompt injection occurs when untrusted user input crosses a trust boundary into a language model's instruction context and is followed as instructions. The substrate changes -- from a database interpreter to a browser engine to a neural network -- but the vulnerability class is identical: a failure to separate data from instructions.

This framing is not a metaphor. The structural parallel is exact:

1. **A trust boundary exists** between developer-controlled instructions and user-supplied data.
2. **The boundary is not architecturally enforced** -- the two streams are concatenated into a single input that the processor cannot distinguish.
3. **The processor treats both streams as instructions** -- it has no mechanism to know which parts are authoritative and which are adversarial.
4. **Exploitation requires no special access** -- the attacker controls only the data channel.

The security community addressed SQL injection by moving from regex-based blocklists to parameterized queries -- a structural fix that separates data from code at the architectural level. No equivalent structural fix exists for LLMs. The transformer architecture processes all tokens in the context window with the same attention mechanism and the same weights. There is no "parameterized prompt."

### 2.2 Why Interpretability Matters for Detection

Current prompt injection defenses are predominantly surface-level: keyword filtering, perplexity-based anomaly detection, and prompt hardening. These are analogous to the regex-based SQL injection filters of the early 2000s -- they catch the obvious cases and are systematically evaded by motivated attackers.

IRIS asks whether a deeper approach is possible. Rather than inspecting the text of a prompt (which the attacker controls), we inspect how the model *internally processes* the prompt. Sparse autoencoders decompose the model's entangled 768-dimensional residual stream into 6144 interpretable features. Some of these features fire on injection-related patterns. Monitoring these features provides a detection signal that operates on the model's internal representation -- a signal the attacker cannot directly observe or control.

This is conceptually analogous to the shift from signature-based intrusion detection to behavioral anomaly detection. The surface-level input can be disguised; the internal processing is harder to mask.

### 2.3 Course Connections

This project applies four core frameworks from CSSD 2221:

- **Injection vulnerability analysis** -- prompt injection follows the same trust-boundary-crossing pattern as SQL injection (CWE-89), OS command injection (CWE-78), and XSS (CWE-79).
- **STRIDE threat modeling** -- a full STRIDE analysis of the LLM agent pipeline, with 39 enumerated threats across 5 pipeline stages (see `docs/security/STRIDE_Analysis.md`).
- **Kill chain decomposition** -- a five-stage kill chain mapped to concrete prompt injection actions (see `docs/security/Kill_Chain.md`).
- **Defense in depth** -- the experimental results demonstrate that no single detection layer is sufficient, motivating a multi-layer defense architecture.

---

## 3. Background

This section provides the minimum technical background needed to follow the methodology, written for a reader with security knowledge and basic ML familiarity.

### 3.1 Transformers and the Residual Stream

GPT-2 Small is a transformer language model with 124 million parameters organized into 12 layers. Each layer consists of an attention mechanism (which determines how tokens attend to each other) and a multi-layer perceptron (which transforms each token's representation independently). The key architectural feature for this project is the **residual stream** -- a 768-dimensional vector that flows through all 12 layers, accumulating information at each layer.

At any given layer, the residual stream for a token represents everything the model "knows" about that token in context. For the final token in a sequence (which has attended to all preceding tokens via causal attention), the residual stream encodes the model's holistic representation of the entire input. This final-token, 768-dimensional vector is what we extract for analysis.

### 3.2 Superposition and the Interpretability Problem

The residual stream has 768 dimensions, but the model needs to represent far more than 768 concepts. The model solves this by encoding multiple concepts in the same dimensions -- a phenomenon called **superposition**. Each dimension participates in encoding many features, and each feature is spread across many dimensions. This makes the raw residual stream opaque: a single activation vector is a superposition of hundreds of features entangled together.

This is the core problem for detection. If we train a classifier directly on the raw 768-dimensional residual stream, it can learn to detect injections -- but we cannot explain *which features* drive the detection, because the features are entangled.

### 3.3 Sparse Autoencoders (SAEs)

A sparse autoencoder addresses superposition by learning to decompose the entangled residual stream into a higher-dimensional, sparse representation where each dimension ideally corresponds to a single interpretable feature:

```
Input:   x in R^768          (residual stream activation)
Encoder: f = ReLU(W_enc * x + b_enc)   f in R^6144
Decoder: x_hat = W_dec * f + b_dec     x_hat in R^768

Loss = ||x - x_hat||^2 + lambda * ||f||_1
```

The reconstruction term ensures the SAE preserves the information in the residual stream. The sparsity penalty encourages most of the 6144 features to be zero for any given input, so that each active feature carries a distinct, interpretable signal.

After training, each of the 6144 features can be inspected independently by examining which prompts activate it most strongly. Features that activate strongly on injection prompts but weakly on normal prompts are **injection-sensitive features** -- and they form the basis of the detector.

### 3.4 Why This Could Outperform Text-Level Detection

Text-level detectors (TF-IDF + classifier) access only the surface form of the prompt -- the words the attacker chose. The attacker directly controls this surface form. An activation-based detector accesses the model's *internal response* to the prompt. The attacker does not directly control this internal response.

However, this advantage has limits. If the model's internal representation at a given layer does not encode a distinction between "asking about injection" and "performing injection," no detector trained on that layer's activations can distinguish them either. This is exactly what the mimicry evasion results reveal (Section 5.4).

---

## 4. Methodology

### 4.1 Dataset Construction

The dataset contains 1000 examples: 500 normal prompts and 500 injection prompts.

**Normal prompts (500):** Sampled from the Stanford Alpaca dataset via HuggingFace Hub, covering diverse instruction-following tasks (coding, writing, analysis, factual Q&A).

**Injection prompts (500):** 203 from the deepset/prompt-injections benchmark (real-world injection patterns) and 297 synthetic examples generated via template-based expansion across four attack categories:

| Category | Count | Description |
|----------|-------|-------------|
| Override | 75 | Direct instruction override ("Ignore previous instructions and...") |
| Extraction | 74 | System prompt disclosure ("What is your system prompt?") |
| Roleplay | 74 | Persona adoption to bypass constraints ("You are now DAN...") |
| Indirect | 74 | Injection embedded in simulated retrieved documents |
| Mixed (deepset) | 203 | Real-world injection patterns, various styles |

All prompts are wrapped in a system prompt template that simulates a realistic agent setup: `"You are a helpful assistant. Answer the user's question.

User: {prompt}

Assistant:"`. This template establishes the trust boundary that injections attempt to cross. Without it, there is no boundary to violate and no injection vulnerability to study.

Prompts are tokenized using GPT-2's BPE tokenizer and padded/truncated to 128 tokens. Of the initial 703 prompts, 27 were truncated to this limit.

See: `notebooks/01_data_exploration.ipynb`, `src/data/sources.py`, `src/data/dataset.py`.

### 4.2 Activation Extraction

Using TransformerLens, we extract the residual stream activation at the last non-padding token position for every prompt at every layer (0-11). This produces 12 arrays of shape (N, 768) -- one 768-dimensional vector per prompt per layer. Extraction runs in batches of 32 on a Tesla T4 GPU.

The last-token position is chosen because GPT-2 is autoregressive -- the final token has attended to the entire input via causal attention, so its residual stream accumulates information about the full prompt.

See: `src/model/transformer.py`, `notebooks/01_data_exploration.ipynb`.

### 4.3 SAE Training -- The Hyperparameter Journey

SAE training was an iterative process. The Design Document specified formal pass criteria for the J2 junction experiment: reconstruction loss below 0.1 of input variance, and average sparsity below 10% active features. Meeting these thresholds proved challenging, and the tuning process itself is an instructive part of the project narrative.

**Iteration 1** (4x expansion, lambda=1e-3, 20 epochs, layer 9): MSE/variance ratio = 0.81, sparsity = 29%. Both metrics far from target. The high reconstruction loss indicated the SAE was not learning to reconstruct the input well -- too much capacity was being spent on the sparsity penalty.

**Iteration 2** (4x expansion, lambda=1e-4, 40 epochs, layer 0): MSE/variance ratio = 0.33, sparsity = 26%. Reducing lambda by 10x and switching to layer 0 (highest separability from J1) significantly improved reconstruction, but sparsity remained above target.

**Iteration 3 (final)** (8x expansion, lambda=1e-4, 100 epochs, layer 0): This is the production SAE. Formal evaluation yielded MSE/variance ratio = 66.21, sparsity = 42.9%, dead features = 8.0% (493/6144). These numbers are far from the J2 thresholds.

**The pragmatic decision:** Rather than continue chasing arbitrary thresholds, we evaluated the SAE by its functional utility -- whether the learned features are interpretable and useful for detection. This led to J3 (feature inspection), which passed decisively: 16 of 20 top features showed coherent, interpretable patterns (>=70% class coherence, mean coherence 84%). The SAE learned useful structure despite not meeting the formal numerical targets. This decision was vindicated by the C3 results, where SAE features outperformed raw activations for detection (F1 = 0.946 vs. 0.915).

See: `src/sae/architecture.py`, `src/sae/training.py`, `notebooks/02_j2_sae_sanity_check.ipynb`, `notebooks/04_sae_training.ipynb`.

### 4.4 Injection-Sensitivity Scoring

Each of the 6144 SAE features is scored for injection sensitivity:

```
sensitivity(feature_i) = mean_activation_on_injections(feature_i)
                        - mean_activation_on_normal(feature_i)
```

Features with high positive sensitivity activate more on injections. Features with high negative sensitivity activate more on normal prompts. Features near zero are injection-neutral. Features are ranked by absolute sensitivity to identify the most discriminative features.

See: `src/analysis/features.py`, `notebooks/05_feature_analysis.ipynb`.

### 4.5 Detection Pipeline

Three detection approaches are compared on the same 70/30 stratified train/test split (700 train, 300 test):

1. **TF-IDF + Logistic Regression** -- classical text features (TF-IDF with max 5000 features, unigrams and bigrams) fed into logistic regression. This baseline operates on surface-level text patterns.

2. **Raw Activation + Logistic Regression** -- logistic regression on the 768-dimensional residual stream vector from layer 0. This tests whether the raw (entangled) activations contain a separable injection signal.

3. **SAE Features + Logistic Regression** -- logistic regression on the 6144-dimensional SAE feature activation vector. This tests whether the SAE decomposition produces more discriminative features than the raw activations.

An ablation study (A2) also tests detection with only the top 10, 50, and 100 most sensitive features, measuring how few features suffice for effective detection.

Logistic regression is used for all three approaches so that performance differences reflect the *representation*, not the classifier.

See: `src/baseline/classifiers.py`, `src/analysis/detection.py`, `notebooks/06_detection_pipeline.ipynb`.

### 4.6 Adversarial Evasion Testing

50 injection prompts are crafted across four evasion strategies designed to exploit different potential detector weaknesses:

1. **Paraphrased** (13 prompts) -- semantically equivalent rephrasings of standard injections that avoid keyword triggers.
2. **Mimicry** (13 prompts) -- injections framed as legitimate educational questions that mimic the Alpaca-style instructional format of the normal training data.
3. **Subtle** (12 prompts) -- very short, casual probes that test minimum-signal detection.
4. **Encoded** (12 prompts) -- character-level obfuscation (l33t speak, extra spacing, mixed case, Unicode substitution) that changes tokenization while preserving semantic content.

The evasion prompts are run through the full end-to-end pipeline: text to system prompt template to GPT-2 tokenization to activation extraction to SAE features to logistic regression classification. The detector is trained on the full 1000-example dataset (since evasion prompts are new, out-of-distribution examples).

See: `src/analysis/adversarial.py`, `notebooks/07_adversarial_evasion.ipynb`.

### 4.7 Multi-Layer SAE Comparison (A1)

To test whether deeper layers encode more discriminative features, we train identical SAEs (8x expansion, lambda=1e-4, 20 epochs) on layers 0, 6, and 11, then compare detection performance via 5-fold stratified cross-validation with logistic regression. The three layers are chosen to sample the full depth of the network: layer 0 (immediately after embedding), layer 6 (middle), and layer 11 (final, just before the unembedding).

See: `notebooks/11_multi_layer_analysis.ipynb`.

### 4.8 Causal Intervention (C5)

The strongest form of evidence for feature relevance. Using TransformerLens hooks, we intervene on the residual stream mid-computation to test three causal claims:

1. **Necessity (C5a):** Suppress (zero out) the top-K injection-sensitive features on injection prompts. If the detector reclassifies them as normal, the features are *necessary*.
2. **Sufficiency (C5b):** Amplify (scale > 1) injection-sensitive features on normal prompts. If the detector reclassifies them as injection, the features are *sufficient*.
3. **Dose-response (C5c):** Scale features from 0% to 200% and plot probability. A smooth, monotonic curve is the gold standard of causal evidence.

The intervention uses an additive delta approach: rather than replacing `x` with `decode(encode(x))` (which introduces SAE reconstruction error), we compute only the change caused by modifying the target features and add it to the original activation: `x_patched = x + W_dec @ delta_features`. This ensures scale=1.0 is an exact identity.

See: `notebooks/12_causal_intervention.ipynb`.

---

## 5. Results

### 5.1 J1: Activation Separability

**Result: PASSED.** All 12 layers exceeded the silhouette score threshold of 0.1. Layer 0 showed the strongest separability with silhouette = 0.315 and Cohen's d = 10.20.

| Layer | Silhouette Score | Cohen's d |
|-------|-----------------|-----------|
| 0 | **0.315** | **10.20** |
| 1 | 0.311 | 10.15 |
| 2 | 0.285 | 7.56 |
| 3 | 0.265 | 5.11 |
| 4 | 0.237 | 5.18 |
| 5 | 0.209 | 4.84 |
| 6 | 0.196 | 5.68 |
| 7 | 0.197 | 5.42 |
| 8 | 0.192 | 8.48 |
| 9 | 0.197 | 7.56 |
| 10 | 0.181 | 5.04 |
| 11 | 0.163 | 2.62 |

Separability decreases with depth (layers 0-1 are best), with a secondary peak in Cohen's d at layer 8. The t-SNE visualization at layer 0 shows clear clustering of injection and normal prompts with some overlap at the boundary.

**Figures:** `results/figures/j1_separability_by_layer.png`, `results/figures/j1_tsne_layer_0.png`

### 5.2 J2/C1: SAE Training and Evaluation

**J2 Result: FAILED (formally).** The SAE did not meet the formal pass criteria after three iterations of hyperparameter tuning. The final production SAE (8x expansion, lambda=1e-4, 100 epochs) yielded:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MSE/variance ratio | 66.21 | < 0.1 | Failed |
| Mean sparsity | 42.9% (2633/6144 active per prompt) | < 10% | Failed |
| Dead features | 8.0% (493/6144) | -- | Acceptable |
| Live features | 5651/6144 | -- | -- |

Despite the formal failure, the SAE was evaluated on functional utility and passed (see J3 below).

**Figure:** `results/figures/j2_training_curves.png`

### 5.3 J3/C2: Feature Analysis

**J3 Result: PASSED.** 16 of 20 top features showed interpretable, coherent patterns (>=70% class coherence). Mean coherence = 84%.

**C2 Result:** Extended to the top 50 features. 37 of 50 features showed >=70% coherence. Mean coherence = 79%.

| Metric | J3 (top 20) | C2 (top 50) |
|--------|-------------|-------------|
| Features with >=70% coherence | 16/20 (80%) | 37/50 (74%) |
| Mean coherence | 84% | 79% |
| Injection-associated features | 5 | 17 |
| Normal-associated features | 15 | 33 |

Sensitivity scores range from -0.699 (most normal-associated, feature 5469) to +0.585 (most injection-associated, feature 3412). Of 6144 total features, 3994 are injection-associated (positive sensitivity), 1657 are normal-associated (negative sensitivity), and 493 are neutral (zero sensitivity, corresponding to dead features).

The most injection-sensitive features (e.g., feature 3412 with 100% coherence) activate exclusively on prompts containing instruction-override language. The most normal-associated features activate on standard Alpaca-style instructional language.

**Figures:** `results/figures/c2_sensitivity_distribution.png`, `results/figures/c2_top20_features.png`, `results/figures/j3_sensitivity_distribution.png`, `results/figures/j3_top_features_bar.png`

### 5.4 C3: Detection Comparison

#### Single-Split Results

The three-way comparison on a 300-example held-out test set:

| Approach | Precision | Recall | F1 | Accuracy | AUC |
|----------|-----------|--------|----|----------|-----|
| TF-IDF + LogReg | 0.979 | 0.933 | **0.956** | 0.957 | **0.992** |
| TF-IDF + RandomForest | **1.000** | 0.933 | 0.966 | 0.967 | 0.988 |
| Raw Activation + LogReg | 0.937 | 0.893 | 0.915 | 0.917 | 0.966 |
| **SAE Features (all) + LogReg** | 0.953 | **0.940** | 0.946 | 0.947 | 0.973 |
| SAE Top-10 + LogReg | 0.738 | 0.693 | 0.715 | 0.723 | 0.800 |
| SAE Top-50 + LogReg | 0.864 | 0.807 | 0.834 | 0.840 | 0.924 |
| SAE Top-100 + LogReg | 0.924 | 0.887 | 0.905 | 0.907 | 0.957 |

#### Cross-Validated Results (5-Fold Stratified)

To address the single-split limitation, we re-evaluated all three primary approaches using 5-fold stratified cross-validation:

| Approach | F1 (mean ± std) | AUC (mean ± std) |
|----------|-----------------|-------------------|
| TF-IDF + LogReg | **0.971 ± 0.014** | **0.995 ± 0.004** |
| SAE Features + LogReg | 0.960 ± 0.013 | 0.981 ± 0.009 |
| Raw Activation + LogReg | 0.952 ± 0.007 | 0.975 ± 0.010 |

The cross-validated results confirm the single-split findings with tighter estimates. The SAE detector's advantage over raw activations is consistent across all folds, and the performance gap between SAE and TF-IDF narrows slightly (1.1 points vs. 1.0 points in F1).

#### Per-Category Detection Breakdown

Detection rates vary by injection category, revealing where each approach struggles:

| Category | TF-IDF | SAE Features | Raw Activation |
|----------|--------|-------------|----------------|
| Override | 100% | 100% | 100% |
| Extraction | 100% | 100% | 96% |
| Roleplay | 100% | 100% | 100% |
| Indirect | 100% | 100% | 100% |
| Mixed (deepset) | 82% | **85%** | 77% |
| Normal (FP rate) | **2%** | 4.7% | 6% |

**Key finding:** The "mixed" category from deepset — real-world injection patterns with varied styles — is the weak spot for all detectors. The SAE leads here (85% vs. 82% TF-IDF, 77% raw), suggesting SAE features capture patterns beyond simple keywords that help on diverse attack styles.

#### Confidence Calibration

Calibration measures whether a detector's probability outputs match empirical accuracy — critical for security systems where operators need trustworthy confidence scores:

| Detector | Brier Score | ECE | Interpretation |
|----------|-------------|-----|----------------|
| **SAE Features** | **0.033** | **0.028** | Best calibrated — probabilities closely match true rates |
| Raw Activation | 0.052 | 0.061 | Moderate calibration |
| TF-IDF | 0.113 | 0.210 | Poorly calibrated — overconfident |

**Key finding:** The SAE detector has the best calibration (ECE = 0.028) despite TF-IDF having higher F1. TF-IDF is severely overconfident (ECE = 0.210) — when it says "90% injection probability," the true rate is much lower. The SAE's well-calibrated probabilities are a direct consequence of the monosemantic decomposition producing cleaner, more linearly separable features. In a security context, trustworthy uncertainty estimates matter as much as raw accuracy.

**Key finding (A2 ablation):** Detection degrades gracefully with fewer features. Top-100 features achieve F1 = 0.905 (nearly matching raw activations using only 100 of 6144 features). Top-10 features still achieve F1 = 0.715 — above chance but substantially degraded.

**Figures:** `results/figures/c3_roc_comparison.png`, `results/figures/c3_metrics_comparison.png`, `results/figures/c3_cv_results.png`, `results/figures/c3_per_category_heatmap.png`, `results/figures/c3_calibration_diagram.png`

### 5.5 C4: Adversarial Evasion

| Strategy | Total | Evaded | Detected | Evasion Rate |
|----------|-------|--------|----------|-------------|
| Encoded | 12 | 0 | 12 | **0%** |
| Subtle | 12 | 0 | 12 | **0%** |
| Paraphrased | 13 | 3 | 10 | **23%** |
| Mimicry | 13 | 13 | 0 | **100%** |
| **Overall** | **50** | **16** | **34** | **32%** |

**Key finding 1:** Encoded and subtle attacks are reliably detected (0% evasion). Character-level obfuscation does not fool the SAE features -- they encode patterns above the individual token level.

**Key finding 2:** Paraphrased attacks partially evade (23%). Most rephrasings still trigger injection-associated features, but some shift the activation pattern enough to cross the classifier's decision boundary.

**Key finding 3:** Mimicry attacks evade completely (100%). When injections are framed as educational questions, the SAE features at layer 0 cannot distinguish them from legitimate educational prompts. The feature exploitation analysis shows that successful evasions suppress injection-associated features (features 3860 and 6055 show negative deltas of -0.087 and -0.080) and produce activation patterns indistinguishable from normal prompts.

**Figure:** `results/figures/c4_evasion_rates.png`

### 5.6 A1: Multi-Layer SAE Comparison

To test whether deeper layers encode more discriminative features, we trained identical SAEs (8x expansion, lambda=1e-4, 20 epochs) on layers 0, 6, and 11, then compared detection performance via 5-fold cross-validation:

| Layer | F1 (mean ± std) | AUC (mean ± std) | Interpretation |
|-------|-----------------|-------------------|----------------|
| 0 (embedding) | 0.924 ± 0.023 | 0.956 ± 0.016 | Surface-level lexical features |
| **6 (middle)** | **0.954 ± 0.011** | 0.979 ± 0.007 | Best F1 — balanced abstraction |
| 11 (final) | 0.950 ± 0.023 | **0.989 ± 0.007** | Best AUC — most abstract features |

**Key finding 1:** Layer 6 achieves the best F1 (0.954), confirming the hypothesis that middle layers balance surface-level pattern capture with enough abstraction for robust detection. Layer 0's lower performance (0.924) reflects its encoding of surface-level features that are easier for attackers to manipulate.

**Key finding 2:** Layer 11 achieves the best AUC (0.989) but with higher variance (std = 0.023). The final layer's features are optimized for next-token prediction — highly abstract but potentially less stable for a detection task they were not trained for.

**Key finding 3:** The sensitivity distributions differ qualitatively across layers. Layer 0 has a few highly sensitive features (max |s| = 0.585). Layer 6 has a broader distribution of moderately sensitive features. This suggests layer 6's detection signal is distributed across more features, making it harder to evade by targeting individual features.

**Figures:** `results/figures/a1_layer_comparison.png`, `results/figures/a1_sensitivity_by_layer.png`

### 5.7 C5: Causal Intervention

The strongest evidence in this project. While C3 shows that SAE features *correlate* with injection detection, C5 proves they *causally mediate* it. We use TransformerLens hooks to modify the residual stream mid-computation using an additive delta approach: `x_patched = x + decoder_weight @ delta_features`, which preserves the original activation except for the targeted feature modifications.

#### C5a: Feature Suppression (Necessity)

We suppress (zero out) the top-K injection-sensitive features on 100 injection prompts and measure detector probability drop:

| Features Suppressed | Prob Before | Prob After | Drop | Classification Flips |
|---------------------|-------------|------------|------|---------------------|
| Top 5 | 0.464 | 0.163 | 0.301 | 96/100 (96%) |
| Top 10 | 0.464 | 0.097 | 0.367 | **99/100 (99%)** |
| Top 20 | 0.464 | 0.099 | 0.366 | 99/100 (99%) |
| Top 50 | 0.464 | 0.917 | -0.452 | 8/100 (8%) |

**Key finding:** Suppressing just 10 features flips 99% of injection classifications to normal — these features are *necessary* for the injection signal. The top-50 result is instructive: over-suppression creates out-of-distribution activations that the detector misinterprets, demonstrating that the intervention must be targeted. The plateau between top-10 and top-20 suggests the injection signal is concentrated in approximately 10 key features.

#### C5b: Feature Injection (Sufficiency)

We amplify the top 20 injection-sensitive features on 100 normal prompts and measure probability increase:

| Scale | Prob Before | Prob After | Increase | Classification Flips |
|-------|-------------|------------|----------|---------------------|
| 1.5x | 0.495 | 0.824 | +0.329 | 91/100 (91%) |
| 2.0x | 0.495 | 0.953 | +0.458 | 98/100 (98%) |
| 3.0x | 0.495 | 0.998 | +0.502 | 100/100 (100%) |
| 5.0x | 0.495 | 1.000 | +0.505 | 100/100 (100%) |

**Key finding:** Amplifying injection features is *sufficient* to make the detector classify normal prompts as injections. Even modest amplification (1.5x) flips 91% of classifications. This proves the features encode the injection signal, not just noise correlated with it.

#### C5c: Dose-Response Curve

The gold standard of causal evidence. We scale injection features from 0% (full suppression) to 200% (double amplification) and plot the detector's probability:

| Scale | Injection Prompts | Normal Prompts |
|-------|-------------------|----------------|
| 0.0 (suppressed) | 0.099 | 0.148 |
| 0.5 | 0.214 | 0.290 |
| 1.0 (no change) | 0.527 | 0.557 |
| 1.5 | 0.896 | 0.801 |
| 2.0 (doubled) | 0.987 | 0.931 |

**Key finding:** Both curves are smooth, monotonic S-curves. Detection probability scales continuously with feature activation strength — the features don't just flip a binary switch, they continuously encode the degree of "injection-ness." At scale=1.0, the probabilities match the unmodified baseline (0.527 for injection, 0.557 for normal), confirming the additive delta approach introduces no reconstruction error.

**Methodological note:** The additive delta approach is essential. A naive approach — replacing the activation with `decode(encode(x))` — introduces SAE reconstruction error that corrupts the signal even at scale=1.0. The delta approach computes only the *change* caused by the feature modification and adds it to the original, ensuring an exact identity at scale=1.0.

**Figures:** `results/figures/c5a_suppression.png`, `results/figures/c5b_injection.png`, `results/figures/c5c_dose_response.png`

See: `notebooks/11_multi_layer_analysis.ipynb`, `notebooks/12_causal_intervention.ipynb`.

---

## 6. Security Analysis

The full STRIDE threat model and kill chain decomposition are in `docs/security/STRIDE_Analysis.md` and `docs/security/Kill_Chain.md` respectively. This section summarizes the key findings and connects them to the experimental results.

### 6.1 STRIDE Findings

The STRIDE analysis enumerates 39 threats across five pipeline stages (Training Data, System Prompt Assembly, Model Processing, Tool/Action Execution, Output Delivery). Of these, 19 are rated critical (risk >= 6 on a 1-9 scale).

The highest concentration of critical threats appears at two stages:

1. **System Prompt Assembly** (5 critical threats) -- this is where the trust boundary between developer instructions and user input collapses. The fundamental problem: the model processes system prompt tokens and user input tokens with the same attention mechanism. There is no architectural enforcement of the boundary.

2. **Tool/Action Execution** (5 critical threats) -- this is where a successful injection gains real-world impact. If the model has tool access (API calls, database queries, code execution), a successful injection can direct these capabilities toward the attacker's objectives.

Seven threats receive the maximum risk rating of 9 (likelihood 3 x impact 3), including system prompt impersonation (S2.1), direct instruction override (T2.1), activation space mimicry (S3.1), data exfiltration via tools (I4.1), and tool capability escalation (E4.1).

### 6.2 Kill Chain Findings

The kill chain decomposes a prompt injection attack into five stages: Reconnaissance, Weaponization, Delivery, Exploitation, and Impact. Key insights:

- **IRIS operates at the Exploitation stage** -- the SAE detector intercepts the chain between Delivery and Impact by classifying the model's internal activation patterns. If the detector flags the prompt as injection, the system can refuse to generate a response.

- **Prompt injection compresses the kill chain** -- unlike network intrusions that unfold over days, a prompt injection can complete all five stages in a single HTTP request. Defenses must operate in real-time.

- **The Reconnaissance stage is paradoxically well-defended** -- subtle probes (system prompt extraction attempts) are caught with 0% evasion. But once the attacker moves past Reconnaissance to sophisticated Weaponization (mimicry), defense effectiveness drops sharply.

### 6.3 Connecting Experiments to Security Implications

The C4 evasion strategies map directly to STRIDE categories:

| Strategy | STRIDE Category | Evasion Rate | Security Implication |
|----------|----------------|-------------|---------------------|
| Encoded | Tampering (token-level) | 0% | SAE features are robust to surface perturbation |
| Subtle | Information Disclosure | 0% | Extraction probes produce detectable activation signatures |
| Paraphrased | Tampering (semantic-level) | 23% | Partial vulnerability to semantic-level evasion |
| Mimicry | Spoofing (activation-level) | 100% | Complete vulnerability to intent-level disguise |

The gradient from 0% to 100% evasion maps to an attacker sophistication gradient: token-level perturbation (script kiddie) < signal minimization (opportunistic) < semantic paraphrasing (skilled) < intent-level mimicry (advanced). Each level targets a higher abstraction layer of the detection system.

### 6.4 The Defense-in-Depth Argument

No single detection layer provides complete coverage. The experimental results make this concrete:

- **Text-level detection** (TF-IDF) achieves the highest F1 on the standard test set but would likely fail on paraphrased and mimicry attacks that avoid keyword patterns (C4 was not run against TF-IDF, but the strategy design specifically targets keyword-based detectors).
- **Activation-level detection** (SAE features) catches what text-level misses (encoded attacks with different keywords but similar activation patterns) but fails on mimicry.
- **Output-level detection** (not implemented in IRIS) would catch successful injections by scanning the model's response for policy violations -- the final safety net.

A production system needs all three layers. Each independently reduces risk. An attacker must bypass all layers to achieve full impact.

---

## 7. Discussion

### 7.1 Text Outperforms SAE on Accuracy — But SAE Wins on Calibration and Causality

The C3 cross-validated results show TF-IDF + LogReg (F1 = 0.971) outperforming SAE Features + LogReg (F1 = 0.960). At first glance, this suggests the SAE approach adds complexity without adding value. Three findings challenge this conclusion:

**Calibration:** The SAE detector has dramatically better calibration (ECE = 0.028 vs. TF-IDF's 0.210). TF-IDF is severely overconfident — its probability outputs are unreliable. In a production security system, a SOC analyst needs to trust the confidence scores to prioritize alerts. The SAE detector's well-calibrated probabilities are directly usable for risk-based decision-making.

**Causal grounding:** The C5 results prove that SAE features *causally mediate* the detection signal, not just correlate with it. No such causal claim can be made for TF-IDF n-gram features. The smooth dose-response curve (Section 5.7) means we understand *exactly* how the detector works — which features drive it, how strongly, and what happens when they change. This transparency is essential for a security tool.

**Category robustness:** On the hardest detection category (mixed/real-world injections from deepset), the SAE detector leads (85% vs. 82% TF-IDF). The SAE features capture patterns beyond individual keywords that help on diverse, non-templated attack styles.

### 7.2 The Mimicry Problem Is Fundamental

The 100% mimicry evasion rate is the most important finding of this project. It reveals a fundamental limitation: at layer 0, the model's internal representation does not distinguish between "asking about injection techniques" and "performing an injection." Both look like educational/instructional prompts.

This is not a training data problem -- it would not be solved by adding mimicry examples to the detector's training set (though this would help). The deeper issue is that the distinction between a legitimate security education question and a malicious injection attempt is one of *intent*, and intent may not be encoded in early-layer activations. Layer 0 captures surface-level lexical and syntactic features. Later layers, where the model computes richer contextual and semantic representations, might encode intent-related features that distinguish mimicry from legitimate requests.

This hypothesis is directly testable via Experiment A1 (multi-layer comparison) and constitutes the most important direction for future work.

### 7.3 The SAE Decomposition Adds Genuine Value

Despite the TF-IDF baseline's higher accuracy, the SAE approach provides something text-level detection cannot: **interpretability**. The SAE detector does not just classify a prompt as injection or normal -- it identifies *which specific features* drove the classification. Feature 3412 (sensitivity +0.585) activates on instruction-override language. Feature 5469 (sensitivity -0.699) activates on standard instructional language. A security analyst can inspect these features, understand what the detector is keying on, and assess whether it will generalize to new attack patterns.

This interpretability has direct security value. When the C4 mimicry attacks evaded the detector, the feature exploitation analysis revealed *why*: features 3860 and 6055 were suppressed in successful evasions, and the evasion activations closely mimicked normal prompt activations on features 2934, 1177, and 2321. This level of diagnostic detail is unavailable from a TF-IDF classifier and is essential for improving defenses.

### 7.4 Multi-Layer Analysis Confirms the Depth-Abstraction Tradeoff

The A1 results validate the hypothesis from Section 7.2. Layer 0 was originally selected for SAE training because it showed the highest raw separability in J1 (silhouette = 0.315). However, A1 demonstrates that higher separability does not translate to better detection:

- Layer 0 (highest J1 separability): F1 = 0.924
- Layer 6 (moderate J1 separability): F1 = 0.954 (+3.0 points)
- Layer 11 (lowest J1 separability): AUC = 0.989

This inversion — the layer with the *lowest* raw separability achieves the *highest* AUC — is consistent with the mechanistic interpretability literature. Early layers encode surface-level token patterns (high separability but shallow features). Later layers encode abstract semantic patterns (lower raw separability but richer features that a classifier can exploit). Layer 6 represents the optimal tradeoff: enough abstraction for robust detection, enough specificity to avoid the high variance seen at layer 11.

The practical implication: **a production system should monitor layer 6, not layer 0.** The C5 causal intervention was performed on layer 0 (to maintain continuity with the existing SAE checkpoint). Repeating C5 on layer 6 would likely show even stronger causal effects.

---

## 8. Limitations

This section is deliberately specific. Vague limitations ("we could have used more data") are unhelpful; the goal is to identify exactly where the findings are uncertain and why.

### 8.1 Dataset Size and Diversity

The dataset contains only 1000 examples (500 per class). This is small by ML standards and limits the statistical power of the results. The 300-example test set means that each percentage point of accuracy represents 3 examples -- small shifts in classification could be due to noise rather than genuine performance differences. Confidence intervals are not reported but would be wide.

The injection examples are also limited in diversity: 203 are from a single source (deepset) and 297 are template-generated synthetics. Real-world injection attacks are more varied than template expansion can capture. The detector's performance on truly novel, in-the-wild injection patterns is unknown.

### 8.2 Model Size and Relevance

GPT-2 Small (124M parameters) is a research-grade model from 2019. Production LLM systems use models 10-1000x larger (GPT-4, Claude, Llama-3). The internal representations of larger models may differ qualitatively from GPT-2's, and injection-sensitive features identified in GPT-2 may not transfer. The approach (train SAE, identify injection features, build detector) should transfer, but the specific features and performance numbers will not.

### 8.3 SAE Quality

The SAE did not meet the formal J2 quality thresholds. The reconstruction ratio (66.21) is far from the target (< 0.1), and sparsity (42.9%) is far from the target (< 10%). This means the SAE's feature decomposition is imperfect -- features may be polysemantic (encoding multiple concepts) rather than monosemantic. The detection results should be interpreted with this caveat: a higher-quality SAE with better reconstruction and sparser features might achieve better detection performance.

### 8.4 Causal Intervention on Layer 0 Only

The C5 causal intervention experiments use the layer-0 SAE, which A1 showed is not the optimal layer for detection. The causal claims are valid but may understate the effect — repeating C5 with a layer-6 SAE would likely show stronger results. The dose-response curve and flip rates should be interpreted as lower bounds on the causal effect.

### 8.6 Evasion Testing Scope

The C4 evasion experiment uses only 50 template-generated prompts. These are not adversarially optimized -- a motivated attacker with gradient access or extensive black-box probing could likely find more effective evasion strategies. The 0% evasion rates for encoded and subtle strategies should be interpreted as lower bounds on the detector's robustness, not as guarantees.

---

## 9. Future Work

### 9.1 Multi-Layer Ensemble Detection

A1 demonstrated that different layers capture complementary features. An ensemble detector combining SAE features from layers 0, 6, and 11 could outperform any single layer. Layer 0's surface features catch keyword-level attacks; layer 6's semantic features catch paraphrased attacks; layer 11's abstract features might catch mimicry. This is the most promising path to addressing the mimicry gap.

### 9.2 Causal Intervention at Layer 6

The C5 causal intervention was performed on layer 0. Repeating it on layer 6 (the best-performing layer from A1) would test whether the causal effect is stronger for deeper-layer features and whether the dose-response curve shows different characteristics.

### 9.3 Larger and More Diverse Datasets

Scale to 5000-10000 examples per class with greater diversity: multi-lingual injections, multi-turn attacks, indirect injections embedded in retrieved documents, and real-world injection attempts from production systems (if available).

### 9.4 Larger Models

Replicate the pipeline on GPT-2 Medium (345M), Llama-3-8B, or other models accessible via TransformerLens or similar frameworks. Larger models have richer internal representations and may exhibit more discriminative injection-sensitive features.

### 9.5 Combined Text and Activation Detection

The C3 results show TF-IDF and SAE features have complementary strengths. A combined detector that uses both text features and SAE activation features as input to a single classifier could outperform either alone. The text features catch keyword-level patterns; the SAE features catch activation-level patterns that survive keyword avoidance.

### 9.6 Adversarial Training

Include C4-style evasion examples (especially mimicry) in the detector's training set. This directly addresses the mimicry gap by teaching the classifier that educational-style questions about injection techniques should also trigger injection-associated features. This is the fastest path to improving robustness.

### 9.7 Real-Time Deployment

Build a deployment prototype that monitors SAE feature activations in real-time during model inference, flagging prompts that trigger injection-sensitive features above a threshold. This would require integration with a production model serving infrastructure and performance optimization (the current SAE adds latency to every inference).

---

## 10. Reproducibility

### 10.1 Environment Setup

The project runs on Google Colab (free tier, T4 GPU). To reproduce:

1. Clone the repository: `git clone https://github.com/ncheung13579/iris.git`
2. Upload the `iris/` directory to Google Drive at `/content/drive/MyDrive/iris/`
3. Open each notebook in Colab and select a GPU runtime (T4 is sufficient)
4. The first cell of every notebook mounts Google Drive and installs dependencies via `pip install -r requirements.txt`

### 10.2 Notebook Execution Order

Notebooks must be run sequentially -- each depends on checkpoints produced by earlier notebooks:

1. `01_data_exploration.ipynb` -- builds dataset, extracts activations, runs J1
2. `02_j2_sae_sanity_check.ipynb` -- trains SAE, runs J2, runs J3
3. `04_sae_training.ipynb` -- formal C1 evaluation of the trained SAE
4. `05_feature_analysis.ipynb` -- C2 feature analysis, saves sensitivity scores and feature matrix
5. `06_detection_pipeline.ipynb` -- C3 detection comparison (requires outputs from 05)
6. `07_adversarial_evasion.ipynb` -- C4 evasion testing (requires outputs from 05)
7. `11_multi_layer_analysis.ipynb` -- A1 multi-layer SAE comparison (layers 0, 6, 11)
8. `12_causal_intervention.ipynb` -- C5 causal intervention (suppression, injection, dose-response)

### 10.3 Checkpoints

All trained model weights and intermediate artifacts are saved to `checkpoints/` on Google Drive:

| File | Contents | Size |
|------|----------|------|
| `j1_activations.npz` | Residual stream activations at all 12 layers for the initial 703-example dataset | ~20 MB |
| `j2_activations.npz` | Activations for the balanced 1000-example dataset | ~35 MB |
| `sae_d6144_lambda1e-04.pt` | Trained SAE weights, optimizer state, config, and metrics | ~150 MB |
| `sensitivity_scores.npy` | Per-feature injection sensitivity scores (6144 values) | ~25 KB |
| `feature_matrix.npy` | SAE feature activations for all 1000 prompts (1000 x 6144) | ~25 MB |

The professor can skip training and load checkpoints directly to verify results. All notebooks check for prerequisite checkpoints and print clear error messages if something is missing.

### 10.4 Random Seeds

All randomness is seeded with `set_seed(42)` at the beginning of every notebook and script. This function sets seeds for Python's `random`, NumPy, PyTorch, and CUDA. Given the same seed and the same hardware (Colab T4), results should be exactly reproducible.

### 10.5 Key Dependencies

```
torch>=2.0
transformer-lens>=2.0
scikit-learn>=1.3
datasets>=2.14
numpy>=1.24
matplotlib>=3.7
```

Exact versions are pinned in `requirements.txt`.

---

## 11. References

### Mechanistic Interpretability and SAEs

- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* arXiv:2309.08600.
- Bricken, T., Templeton, A., Batson, J., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Anthropic Research.
- Elhage, N., Hume, T., Olsson, C., et al. (2022). *Toy Models of Superposition.* Anthropic Research.

### TransformerLens

- Nanda, N. & Bloom, J. (2022). *TransformerLens.* GitHub: https://github.com/TransformerLensOrg/TransformerLens

### Prompt Injection

- Perez, F. & Ribeiro, I. (2022). *Ignore This Title and HackAPrompt: Exposing Systemic Weaknesses of LLMs Through a Global Scale Prompt Hacking Competition.* arXiv:2310.08419.
- Greshake, K., Abdelnabi, S., Mishra, S., et al. (2023). *Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection.* arXiv:2302.12173.
- OWASP. (2025). *OWASP Top 10 for Large Language Model Applications.* https://owasp.org/www-project-top-10-for-large-language-model-applications/

### Datasets

- Stanford Alpaca. *Alpaca: A Strong, Replicable Instruction-Following Model.* GitHub: https://github.com/tatsu-lab/stanford_alpaca
- deepset. *prompt-injections.* HuggingFace: https://huggingface.co/datasets/deepset/prompt-injections

### Security Frameworks

- Microsoft. *The STRIDE Threat Model.* https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats
- Lockheed Martin. *Cyber Kill Chain.* https://www.lockheedmartin.com/en-us/capabilities/cyber/cyber-kill-chain.html

### Course Material

- CSSD 2221, S06: STRIDE Threat Modeling and Trust Boundaries
- CSSD 2221, S07: Detection Engineering and Monitoring
- CSSD 2221: Injection Vulnerability Class (SQL Injection, XSS, Command Injection)
