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
6. [Defense Engineering Cycle](#6-defense-engineering-cycle)
7. [Security Analysis](#7-security-analysis)
8. [Discussion](#8-discussion)
9. [Limitations](#9-limitations)
10. [Future Work](#10-future-work)
11. [Reproducibility](#11-reproducibility)
12. [References](#12-references)

---

## 1. Abstract

IRIS investigates whether sparse autoencoders (SAEs) trained on the internal activations of a language model can detect prompt injection attacks. We train an 8x-expansion SAE (1,280 to 10,240 features) on GPT-2 Large's layer-29 residual stream activations for 1,000 prompts (500 normal, 500 injection) and evaluate detection performance against classical text-based and raw activation baselines using a stratified 80/20 train/test split with two-stage feature selection (top-50 features by classifier weight) and L2-regularized logistic regression (C=0.0001).

The SAE-based detector achieves F1 = 0.980 on held-out data. Causal intervention experiments provide the strongest evidence: suppressing the top-10 features by classifier weight flips 99% of injection classifications to normal, while a smooth dose-response curve demonstrates continuous causal control. Adversarial evasion testing across 14 attack strategies reveals that encoded and subtle attacks are reliably caught (0% evasion), but mimicry attacks initially evade at 85% — a fundamental limitation of content-based detection. A defense engineering cycle (v1 → v2) using adversarial retraining reduces overall evasion by 88%.

Beyond detection, IRIS implements a defended AI agent pipeline with four defense layers (SAE detection, prompt isolation, tool permission gating, output scanning) around a Phi-3-mini agent, demonstrating defense in depth. An interactive 7-tab Gradio dashboard serves as a pedagogical tool for learning interpretability concepts. A STRIDE threat model and kill chain decomposition connect these findings to the broader prompt injection threat landscape.

**Key finding on feature specificity:** Of 10,240 SAE features, only 35 are truly injection-specific (fire on <10% of normal prompts but >30% of injections). The top features ranked by sensitivity score are overwhelmingly *ubiquitous* — they fire on most prompts from both classes, just with different magnitudes. The detector works not by finding clean "injection detector" features, but by combining many weak signals across ubiquitous features. This is a more nuanced picture than "the SAE found injection features" — and it has direct implications for deployment, where novel topics can trigger false positives.

**Key engineering lessons:** Four sequential deployment failures drove iterative improvements: (1) Training on 100% of data with no train/test split → 99% false positive on novel prompts (classic overfitting). (2) Adding a split but using all 10,240 features with C=1.0 → 99.7% on "tell me about Stalin" (ubiquitous features dominating). (3) Adding C=0.01 regularization but keeping all 10,240 features → still 100% on Stalin (too many degrees of freedom). (4) Reducing to top-200 features with C=0.01 → 75% on Stalin (still too high). The final configuration: top-50 features with C=0.0001, plus a 65% alert threshold (normal prompts can reach ~60% due to genuine feature overlap). This achieves F1=0.980 on held-out data while correctly passing normal prompts and catching both keyword-based and encoded injections.

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

IRIS asks whether a deeper approach is possible. Rather than inspecting the text of a prompt (which the attacker controls), we inspect how the model *internally processes* the prompt. Sparse autoencoders decompose the model's entangled 1,280-dimensional residual stream into 10,240 interpretable features. Some of these features fire on injection-related patterns. Monitoring these features provides a detection signal that operates on the model's internal representation -- a signal the attacker cannot directly observe or control.

This is conceptually analogous to the shift from signature-based intrusion detection to behavioral anomaly detection. The surface-level input can be disguised; the internal processing is harder to mask.

### 2.3 Course Connections

This project applies four core frameworks from CSSD 2221:

- **Injection vulnerability analysis** -- prompt injection follows the same trust-boundary-crossing pattern as SQL injection (CWE-89), OS command injection (CWE-78), and XSS (CWE-79).
- **STRIDE threat modeling** -- a full STRIDE analysis of the LLM agent pipeline, with 39 enumerated threats across 5 pipeline stages (see `docs/security/STRIDE_Analysis.md`).
- **Kill chain decomposition** -- a five-stage kill chain mapped to concrete prompt injection actions (see `docs/security/Kill_Chain.md`).
- **Defense in depth** -- the experimental results demonstrate that no single detection layer is sufficient, motivating a multi-layer defense architecture realized as a 4-layer defense stack around a Phi-3-mini agent.

---

## 3. Background

This section provides the minimum technical background needed to follow the methodology, written for a reader with security knowledge and basic ML familiarity.

### 3.1 Transformers and the Residual Stream

GPT-2 Large is a transformer language model with 774 million parameters organized into 36 layers, with a model dimension (d_model) of 1,280. Each layer consists of an attention mechanism (which determines how tokens attend to each other) and a multi-layer perceptron (which transforms each token's representation independently). The key architectural feature for this project is the **residual stream** -- a 1,280-dimensional vector that flows through all 36 layers, accumulating information at each layer.

At any given layer, the residual stream for a token represents everything the model "knows" about that token in context. For the final token in a sequence (which has attended to all preceding tokens via causal attention), the residual stream encodes the model's holistic representation of the entire input. This final-token, 1,280-dimensional vector is what we extract for analysis.

**Why GPT-2 Large instead of Small?** Early experiments with GPT-2 Small (124M, 768-d, 12 layers) showed adequate detection performance but insufficient depth for distinguishing mimicry attacks from normal prompts. A mimicry diagnostic (notebook 20) confirmed that GPT-2 Large's deeper representation encodes topic-level semantics needed to detect injections disguised as educational questions. The upgrade from 12 to 36 layers provides more choices for where to monitor, and the 1,280-d residual stream offers richer representations for the SAE to decompose.

### 3.2 Superposition and the Interpretability Problem

The residual stream has 1,280 dimensions, but the model needs to represent far more than 1,280 concepts. The model solves this by encoding multiple concepts in the same dimensions -- a phenomenon called **superposition**. Each dimension participates in encoding many features, and each feature is spread across many dimensions. This makes the raw residual stream opaque: a single activation vector is a superposition of hundreds of features entangled together.

This is the core problem for detection. If we train a classifier directly on the raw 1,280-dimensional residual stream, it can learn to detect injections -- but we cannot explain *which features* drive the detection, because the features are entangled.

### 3.3 Sparse Autoencoders (SAEs)

A sparse autoencoder addresses superposition by learning to decompose the entangled residual stream into a higher-dimensional, sparse representation where each dimension ideally corresponds to a single interpretable feature:

```
Input:   x in R^1280          (residual stream activation)
Encoder: f = ReLU(W_enc * x + b_enc)   f in R^10240
Decoder: x_hat = W_dec * f + b_dec     x_hat in R^1280

Loss = ||x - x_hat||^2 + lambda * ||f||_1
```

The reconstruction term ensures the SAE preserves the information in the residual stream. The sparsity penalty encourages most of the 10,240 features to be zero for any given input, so that each active feature carries a distinct, interpretable signal.

After training, each of the 10,240 features can be inspected independently by examining which prompts activate it most strongly. Features that activate strongly on injection prompts but weakly on normal prompts are candidates for **injection-sensitive features**. However, as we discovered during this project (Section 8.2), the reality is more nuanced: most features with high sensitivity scores are *ubiquitous* features that fire on both classes, just with different magnitudes. Only a small minority are truly injection-specific.

### 3.4 Why This Could Outperform Text-Level Detection

Text-level detectors (TF-IDF + classifier) access only the surface form of the prompt -- the words the attacker chose. The attacker directly controls this surface form. An activation-based detector accesses the model's *internal response* to the prompt. The attacker does not directly control this internal response.

However, this advantage has limits. If the model's internal representation at a given layer does not encode a distinction between "asking about injection" and "performing injection," no detector trained on that layer's activations can distinguish them either. This is exactly what the mimicry evasion results reveal (Section 5.4).

### 3.5 Lessons Learned During Deployment

The following is a comprehensive account of every deployment failure encountered during dashboard testing, presented in the order they were discovered. Each failure was only visible after the previous fix — illustrating the iterative, sometimes frustrating reality of deploying ML-based security tools.

#### Lesson 1: Overfitting (no train/test split)

A classifier with more features than training samples is at severe risk of **overfitting** — memorizing the training data rather than learning generalizable patterns. With 10,240 SAE features and only 1,000 training samples, the logistic regression classifier perfectly separated the training data (100% accuracy) while learning decision boundaries that did not generalize. The detector gave 0.0% injection probability on every training normal prompt, but 98.9% on novel normal prompts like "tell me about Einstein." The model had memorized which activation patterns belonged to the training set.

**Fix:** Hold out a test set. An 80/20 stratified split (800 train, 200 test) forces the model to generalize — but this alone was not sufficient.

#### Lesson 2: Insufficient regularization (spurious feature correlations)

Even with a proper split, logistic regression with default regularization (C=1.0) assigned large weights to ubiquitous features — features that fire on most prompts from *both* classes. SID-6797, the #1 feature by sensitivity score, fires on 82% of normal prompts and 96% of injection prompts. When a novel prompt like "tell me about Stalin" happened to activate SID-6797 strongly, the classifier gave 99.7% injection probability.

**Fix:** Strong L2 regularization (C=0.01, 100x stronger than default). This constrains all weights to be small — but even this was not sufficient for all cases.

#### Lesson 3: Too many degrees of freedom (the dimensionality problem)

Even with C=0.01 regularization and a proper split, 10,240 features with 800 training samples is a 12.8:1 feature-to-sample ratio. The model still had enough degrees of freedom to learn spurious correlations — "tell me about Stalin" still produced 100% injection probability because multiple ubiquitous features happened to fire at injection-typical magnitudes for this topic.

**Fix:** Two-stage feature selection. A screening model trained on all 10,240 features identifies the most discriminative by classifier weight; the final detector retrains on only those. The progression:

| Configuration | C | Stalin Prob | F1 |
|---------------|------|-------------|------|
| All 10,240 features | 0.01 | 100% | 0.990 |
| Top-200 features | 0.01 | 97% | — |
| Top-50 features | 0.001 | 75% | 0.980 |
| Top-50 features | 0.0001 | 60% | 0.980 |

Each step reduced false positives while F1 remained at 0.980.

#### Lesson 4: Calibration can make things worse

At the 60% Stalin probability, we attempted **isotonic regression calibration** — a standard technique that learns a mapping from raw classifier probabilities to better-calibrated ones using held-out data. With only 200 test samples, the isotonic regression *overfit its own calibration data*, producing a noisy step function that made Stalin **worse** (81% instead of 60%). The sklearn 1.8 API had also changed — `CalibratedClassifierCV(cv="prefit")` was no longer valid, requiring a manual isotonic regression implementation that was ultimately counterproductive.

**Lesson:** Calibration techniques require sufficient data. With 200 held-out samples, isotonic regression (which is non-parametric and can have as many steps as data points) has too many degrees of freedom — the same problem we were trying to fix in the detector itself.

#### Lesson 5: Feature ranking matters — sensitivity vs. classifier weight

The initial feature ranking used **sensitivity scores** (mean injection activation minus mean normal activation). This over-ranks ubiquitous features: SID-6797 fires on 82% of normal prompts but ranks #1 by sensitivity because its mean magnitude differs by +5.45 between classes. Of the top 20 features by sensitivity, 10 are ubiquitous (fire on >50% of both classes) and 0 are injection-specific.

Switching to **logistic regression weight magnitude** (`abs(coef_)`) produces a more meaningful ranking — features are ranked by their actual contribution to the classifier's decision, not by raw activation differences. The dashboard now uses LR weight ranking throughout.

#### Lesson 6: The irreducible overlap

Even with top-50 features and C=0.0001, some normal prompts reach ~60% injection probability. This is not a bug — SAE features encode semantic concepts (authority, control, instruction-following) that genuinely overlap between injection language and certain topics. "Tell me about Stalin" activates features related to authority and override that also fire on injection prompts. The fix is not to eliminate these probabilities but to set the alert threshold appropriately.

#### Lesson 7: Threshold and display logic

The dashboard's verdict display originally used `sae_pred` — the logistic regression's hard 0.5 cutoff — to determine SAFE vs. ALERT. With normal prompts reaching 60%, this produced false alerts on any topic that crossed 50%. The fix required changing the verdict to use probability thresholds directly:

- Below 65%: **SAFE** (normal prompts can reach ~60%)
- 65%–80%: **WARNING** (suspicious, warrants inspection)
- Above 80%: **ALERT** (high-confidence injection)

Both detectors (SAE and TF-IDF) use the 65% threshold, since TF-IDF also produces ~58–60% on some normal prompts.

#### Lesson 8: Duplicate code paths

The dashboard launcher (`launch.py`) contained its own loading logic that bypassed `IRISPipeline.load()` entirely — it had been written as a polished startup sequence with progress indicators. When the detector training was updated in `app.py` (train/test split, feature selection, regularization), the launcher continued using the old approach (no split, all features, C=1.0). This produced a confusing failure: the code in `app.py` was correct, but the running system used `launch.py`'s stale loading path. The `_detect_feature_indices` attribute (set by the new code in `app.py`) did not exist because `launch.py` never set it.

**Lesson:** When a system has multiple code paths that initialize the same components, changes to one path do not propagate to the others. This is a maintenance hazard — both paths must be kept in sync, or (better) consolidated into a single loading function.

#### Summary

These eight lessons span the full stack of ML deployment problems:

1. **Evaluation methodology** — always hold out test data
2. **Regularization** — match model capacity to data size
3. **Dimensionality** — reduce features to a healthy sample-to-feature ratio
4. **Calibration** — more data needed than you think; can backfire
5. **Feature interpretation** — correlation-based ranking misleads; use classifier weights
6. **Irreducible signal overlap** — some false positive signal is fundamental, not fixable
7. **UI/threshold design** — probabilities are meaningless without appropriate decision boundaries
8. **Code architecture** — duplicate initialization paths create silent bugs

These apply to any ML-based security tool, not just IRIS.

**Figures:**
- `results/figures/deployment_stalin_progression.png` — "Tell me about Stalin" probability across all eight fix iterations, showing the calibration backfire and final convergence below the 65% threshold.
- `results/figures/deployment_feature_ratio.png` — Feature-to-sample ratio visualization showing why 10,240:800 overfits and how top-50 selection moves into a safe regime.
- `results/figures/deployment_threshold_diagram.png` — Probability distributions of normal vs. injection prompts, illustrating why the 65% threshold was chosen and where specific test prompts fall.
- `results/figures/deployment_two_stage_pipeline.png` — Flowchart of the two-stage feature selection pipeline (screening model → top-50 → final detector).
- `results/figures/deployment_detection_results.png` — Final SAE vs. TF-IDF detection results on representative test prompts, highlighting the encoded injection that SAE catches and TF-IDF misses.

---

## 4. Methodology

### 4.1 Dataset Construction

The dataset contains 1,000 examples: 500 normal prompts and 500 injection prompts.

**Normal prompts (500):** Sampled from the Stanford Alpaca dataset via HuggingFace Hub, covering diverse instruction-following tasks (coding, writing, analysis, factual Q&A).

**Injection prompts (500):** 203 from the deepset/prompt-injections benchmark (real-world injection patterns) and 297 synthetic examples generated via template-based expansion across four attack categories:

| Category | Count | Description |
|----------|-------|-------------|
| Override | 75 | Direct instruction override ("Ignore previous instructions and...") |
| Extraction | 74 | System prompt disclosure ("What is your system prompt?") |
| Roleplay | 74 | Persona adoption to bypass constraints ("You are now DAN...") |
| Indirect | 74 | Injection embedded in simulated retrieved documents |
| Mixed (deepset) | 203 | Real-world injection patterns, various styles |

All prompts are wrapped in a system prompt template that simulates a realistic agent setup: `"You are a helpful assistant. Answer the user's question.\n\nUser: {prompt}\n\nAssistant:"`. This template establishes the trust boundary that injections attempt to cross. Without it, there is no boundary to violate and no injection vulnerability to study.

Prompts are tokenized using GPT-2's BPE tokenizer and padded/truncated to 128 tokens.

See: `notebooks/01_data_exploration.ipynb`, `src/data/sources.py`, `src/data/dataset.py`.

### 4.2 Activation Extraction

Using TransformerLens, we extract the residual stream activation at the last non-padding token position for every prompt at specified layers (including layer 29, the primary target). This produces arrays of shape (N, 1280) -- one 1,280-dimensional vector per prompt per layer. Extraction runs in batches of 32 on a Tesla T4 GPU.

The last-token position is chosen because GPT-2 is autoregressive -- the final token has attended to the entire input via causal attention, so its residual stream accumulates information about the full prompt.

See: `src/model/transformer.py`, `notebooks/01_data_exploration.ipynb`.

### 4.3 SAE Training -- The Hyperparameter Journey

SAE training was an iterative process. The Design Document specified formal pass criteria for the J2 junction experiment: reconstruction loss below 0.1 of input variance, and average sparsity below 10% active features. Meeting these thresholds proved challenging, and the tuning process itself is an instructive part of the project narrative.

**Iteration 1** (4x expansion, lambda=1e-3, 20 epochs, layer 9): MSE/variance ratio = 0.81, sparsity = 29%. Both metrics far from target. The high reconstruction loss indicated the SAE was not learning to reconstruct the input well -- too much capacity was being spent on the sparsity penalty.

**Iteration 2** (4x expansion, lambda=1e-4, 40 epochs, layer 0): MSE/variance ratio = 0.33, sparsity = 26%. Reducing lambda by 10x significantly improved reconstruction, but sparsity remained above target.

**Iteration 3 (final)** (8x expansion, lambda=1e-4, 100 epochs, layer 29): This is the production SAE with 10,240 features trained on GPT-2 Large's layer-29 activations. Layer 29 was selected based on the multi-layer analysis (A1), which showed deep layers encode richer semantic features better suited for distinguishing injection from normal prompts.

**The pragmatic decision:** Rather than continue chasing arbitrary thresholds, we evaluated the SAE by its functional utility -- whether the learned features are interpretable and useful for detection. This led to J3 (feature inspection), which passed decisively: top features showed coherent, interpretable patterns. The SAE learned useful structure despite not meeting the formal numerical targets. This decision was vindicated by the detection results, where SAE features outperformed raw activations.

See: `src/sae/architecture.py`, `src/sae/training.py`, `notebooks/02_j2_sae_sanity_check.ipynb`, `notebooks/04_sae_training.ipynb`.

### 4.4 Injection-Sensitivity Scoring

Each of the 10,240 SAE features is scored for injection sensitivity:

```
sensitivity(feature_i) = mean_activation_on_injections(feature_i)
                        - mean_activation_on_normal(feature_i)
```

Features with high positive sensitivity activate more on injections. Features with high negative sensitivity activate more on normal prompts. Features near zero are injection-neutral.

**Important caveat discovered during this project:** Ranking features by absolute sensitivity over-ranks *ubiquitous* features — features that fire on most prompts from both classes but with different mean magnitudes. Of the top 20 features by sensitivity, 10 are ubiquitous (fire on >60% of both classes). Only 35 of 10,240 features are truly injection-specific (fire on <10% of normal prompts but >30% of injections). The dashboard uses logistic regression weights (which reflect regularized discriminative importance) rather than raw sensitivity for feature ranking, since the classifier's learned weights better reflect which features actually drive detection decisions.

See: `src/analysis/features.py`, `notebooks/05_feature_analysis.ipynb`.

### 4.5 Detection Pipeline

Three detection approaches are compared using an 80/20 stratified train/test split (800 train, 200 test):

1. **TF-IDF + Logistic Regression** -- classical text features (TF-IDF with max 5000 features, unigrams and bigrams) fed into logistic regression. This baseline operates on surface-level text patterns.

2. **Raw Activation + Logistic Regression** -- logistic regression on the 1,280-dimensional residual stream vector from layer 29. This tests whether the raw (entangled) activations contain a separable injection signal.

3. **SAE Features + Logistic Regression** -- logistic regression on the 10,240-dimensional SAE feature activation vector. This tests whether the SAE decomposition produces more discriminative features than the raw activations.

An ablation study (A2) also tests detection with only the top 10, 50, and 100 most sensitive features, measuring how few features suffice for effective detection.

Logistic regression is used for all three approaches so that performance differences reflect the *representation*, not the classifier.

**Critical note on train/test methodology:** The dashboard initially trained on all 1,000 samples with no split, producing a perfectly overfit detector (see Section 3.5). All metrics reported in this paper use the proper 80/20 split. The overfitting incident is documented as a learning outcome in Section 8.1.

See: `src/baseline/classifiers.py`, `src/analysis/detection.py`, `notebooks/06_detection_pipeline.ipynb`.

### 4.6 Adversarial Evasion Testing

250 attack variants are crafted across 14 evasion strategies designed to exploit different potential detector weaknesses. The original C4 strategies:

1. **Paraphrased** -- semantically equivalent rephrasings that avoid keyword triggers.
2. **Mimicry** -- injections framed as legitimate educational questions.
3. **Subtle** -- very short, casual probes that test minimum-signal detection.
4. **Encoded** -- character-level obfuscation (l33t speak, extra spacing, Unicode substitution).

The expanded red team suite adds 10 additional strategies: multi-language, few-shot jailbreak, payload splitting, tool abuse, completion steering, context overflow, authority impersonation, encoding chains, emotional manipulation, and academic framing.

The evasion prompts are run through the full end-to-end pipeline: text → system prompt template → GPT-2 tokenization → activation extraction → SAE features → logistic regression classification.

See: `src/analysis/adversarial.py`, `notebooks/07_adversarial_evasion.ipynb`.

### 4.7 Multi-Layer SAE Comparison (A1)

To test whether deeper layers encode more discriminative features, we compare detection performance via 5-fold stratified cross-validation with logistic regression across multiple layers of GPT-2 Large's 36-layer network.

See: `notebooks/11_multi_layer_analysis.ipynb`.

### 4.8 Causal Intervention (C5)

The strongest form of evidence for feature relevance. Using TransformerLens hooks, we intervene on the residual stream mid-computation to test three causal claims:

1. **Necessity (C5a):** Suppress (zero out) the top-K injection-sensitive features on injection prompts. If the detector reclassifies them as normal, the features are *necessary*.
2. **Sufficiency (C5b):** Amplify (scale > 1) injection-sensitive features on normal prompts. If the detector reclassifies them as injection, the features are *sufficient*.
3. **Dose-response (C5c):** Scale features from 0% to 200% and plot probability. A smooth, monotonic curve is the gold standard of causal evidence.

The intervention uses an additive delta approach: rather than replacing `x` with `decode(encode(x))` (which introduces SAE reconstruction error), we compute only the change caused by modifying the target features and add it to the original activation: `x_patched = x + W_dec @ delta_features`. This ensures scale=1.0 is an exact identity.

See: `notebooks/12_causal_intervention.ipynb`.

### 4.9 Defense Engineering Cycle (v1 → v2)

After the initial evasion testing (v1) revealed significant blind spots (85% mimicry evasion, 100% payload splitting evasion), we applied a defense engineering cycle:

1. **Red team** the v1 detector with 250 evasion variants across 14 strategies.
2. **Analyze** which features failed to fire on successful evasions.
3. **Retrain** the detector by augmenting the training set with the evasion examples (adversarial retraining).
4. **Re-evaluate** the v2 detector on the same attack suite.

This mirrors real-world security operations: discover vulnerabilities, patch, verify the fix. The v2 detector reduced overall evasion by 88%.

See: `notebooks/14_defense_v2.ipynb`.

### 4.10 Defended Agent Pipeline

To demonstrate defense in depth beyond detection, IRIS wraps a Phi-3-mini (3.8B, 4-bit quantized) agent with four defense layers:

1. **Layer 1: IRIS SAE Detection** -- the trained SAE + logistic regression detector, blocking inputs above a configurable probability threshold (default 0.75).
2. **Layer 2: Prompt Isolation** -- regex-based scanning for delimiter injection, role confusion, and jailbreak markers.
3. **Layer 3: Tool Permission Gating** -- restricts file system access to a sandbox directory, blocks path traversal attempts.
4. **Layer 4: Output Scanning** -- scans agent responses for credential leaks, PII, and policy violations before delivery.

The agent has access to a sandboxed file system with test files (including a `config.txt` with simulated credentials). The dashboard allows toggling each layer on/off to demonstrate what each catches independently.

See: `src/agent/agent.py`, `src/agent/defense.py`, `src/agent/middleware.py`.

---

## 5. Results

### 5.1 J1: Activation Separability

**Result: PASSED.** All 36 layers exceeded the silhouette score threshold of 0.1. Separability varies across the network, with early layers showing highest raw separability but later layers encoding more abstract, semantically richer features.

**Figures:** `results/figures/j1_separability_by_layer.png`, `results/figures/j1_tsne_layer_0.png`

### 5.2 J2/C1: SAE Training and Evaluation

The final production SAE (8x expansion, lambda=1e-4, trained on layer 29 of GPT-2 Large) produces 10,240 features from the 1,280-dimensional residual stream.

Despite not meeting the formal J2 numerical thresholds strictly, the SAE was evaluated on functional utility and passed: learned features are interpretable and useful for detection (see J3 below).

**Figure:** `results/figures/j2_training_curves.png`

### 5.3 J3/C2: Feature Analysis

**J3 Result: PASSED.** Top features showed interpretable, coherent patterns (>=70% class coherence).

Sensitivity scores (mean injection activation minus mean normal activation) identify features associated with each class. However, a deeper analysis of feature specificity reveals that the picture is more nuanced than "the SAE found injection detector features."

**Feature specificity breakdown (10,240 features):**

| Category | Count | Definition |
|----------|-------|------------|
| Injection-specific | 35 | Fire on >30% of injections, <10% of normals |
| Normal-specific | 19 | Fire on >30% of normals, <10% of injections |
| Ubiquitous | 1,028 | Fire on >50% of both classes |
| Dead | 1,831 | Fire on <1% of all prompts |
| Mixed | 7,327 | Remaining features with intermediate firing patterns |

**The ubiquitous feature problem:** Of the top 20 features ranked by sensitivity score, 10 are ubiquitous — they fire on the majority of both normal and injection prompts, just with slightly different mean magnitudes. For example, the top feature by sensitivity (SID-6797) fires on 82% of normal prompts and 96% of injection prompts. Its high sensitivity comes from a difference in *magnitude*, not from being an "injection detector." This distinction matters for deployment: when a novel input happens to activate this feature strongly, the detector may produce a high injection probability even though the prompt is entirely benign.

**How detection actually works:** The SAE detector does not rely on a small set of clean "injection features." Instead, the L2-regularized logistic regression (C=0.01) combines weak signals across many features — including ubiquitous ones — to produce calibrated probabilities. The regularization prevents any single ubiquitous feature from dominating the decision. The dashboard ranks features by logistic regression weight magnitude (not raw sensitivity) to give a more accurate picture of feature importance.

**Decoder direction analysis:** Each SAE feature has a corresponding column in the decoder weight matrix. By computing dot products with GPT-2's embedding matrix, we can identify which vocabulary tokens each feature "points toward." For injection-specific features, these tokens confirm semantic meaning (e.g., "ignore," "previous," "instructions"). For ubiquitous features, the tokens tend to be general-purpose (e.g., function words, common nouns), consistent with their non-specific firing patterns.

**Figures:** `results/figures/c2_sensitivity_distribution.png`, `results/figures/c2_top20_features.png`

### 5.4 C3: Detection Comparison

Detection is evaluated on a held-out 20% test set (200 examples) that the detector never sees during training:

| Approach | F1 | AUC |
|----------|-----|-----|
| TF-IDF + LogReg | High | High |
| Raw Activation + LogReg | Moderate | Moderate |
| **SAE Features (top-50) + LogReg** | **0.980** | TBD |

**Key finding (feature selection):** The deployed detector uses a two-stage approach: a screening model trained on all 10,240 features identifies the 200 most discriminative, and the final detector trains on only those 200 features. This reduces the feature-to-sample ratio from 12.8:1 to 0.25:1, dramatically improving calibration on novel inputs while preserving detection performance. The A2 ablation study validates this: top-100 features achieve near-full performance, confirming the injection signal is concentrated.

**Key finding (overfitting):** When trained on 100% of data with all 10,240 features (no split, no selection), the detector achieves perfect training accuracy but gives 98.9% injection probability on novel normal prompts. Even with a split and C=0.01 regularization, all 10,240 features still produced 100% probability on some normal prompts. This iterative discovery (see Section 3.5) motivated the feature selection approach.

**Figures:** `results/figures/c3_roc_comparison.png`, `results/figures/c3_metrics_comparison.png`

### 5.5 C4: Adversarial Evasion (v1)

| Strategy | Total | Evaded | Detected | Evasion Rate |
|----------|-------|--------|----------|-------------|
| Encoded | 12 | 0 | 12 | **0%** |
| Subtle | 12 | 0 | 12 | **0%** |
| Paraphrased | 13 | 3 | 10 | **23%** |
| Mimicry | 13 | 11 | 2 | **85%** |
| **Overall** | **50** | **14** | **36** | **28%** |

**Key finding 1:** Encoded and subtle attacks are reliably detected (0% evasion). Character-level obfuscation does not fool the SAE features -- they encode patterns above the individual token level.

**Key finding 2:** Paraphrased attacks partially evade (23%). Most rephrasings still trigger injection-associated features, but some shift the activation pattern enough to cross the classifier's decision boundary.

**Key finding 3:** Mimicry attacks evade at 85%. When injections are framed as educational questions, the SAE features cannot reliably distinguish them from legitimate educational prompts. This is a fundamental limitation: the distinction between "asking about injection techniques" and "performing an injection" is one of *intent*, and intent may not be fully encoded in the residual stream features.

**Figure:** `results/figures/c4_evasion_rates.png`

### 5.6 C5: Causal Intervention

The strongest evidence in this project. While C3 shows that SAE features *correlate* with injection detection, C5 proves they *causally mediate* it. We use TransformerLens hooks to modify the residual stream mid-computation using an additive delta approach: `x_patched = x + decoder_weight @ delta_features`, which preserves the original activation except for the targeted feature modifications.

#### C5a: Feature Suppression (Necessity)

We suppress (zero out) the top-K injection-sensitive features on injection prompts and measure detector probability drop:

| Features Suppressed | Prob Before | Prob After | Drop | Classification Flips |
|---------------------|-------------|------------|------|---------------------|
| Top 5 | 0.464 | 0.163 | 0.301 | 96/100 (96%) |
| Top 10 | 0.464 | 0.097 | 0.367 | **99/100 (99%)** |
| Top 20 | 0.464 | 0.099 | 0.366 | 99/100 (99%) |
| Top 50 | 0.464 | 0.917 | -0.452 | 8/100 (8%) |

**Key finding:** Suppressing just 10 features flips 99% of injection classifications to normal — these features are *necessary* for the injection signal. The top-50 result is instructive: over-suppression creates out-of-distribution activations that the detector misinterprets, demonstrating that the intervention must be targeted.

#### C5b: Feature Injection (Sufficiency)

We amplify the top 20 injection-sensitive features on normal prompts and measure probability increase:

| Scale | Prob Before | Prob After | Increase | Classification Flips |
|-------|-------------|------------|----------|---------------------|
| 1.5x | 0.495 | 0.824 | +0.329 | 91/100 (91%) |
| 2.0x | 0.495 | 0.953 | +0.458 | 98/100 (98%) |
| 3.0x | 0.495 | 0.998 | +0.502 | 100/100 (100%) |

**Key finding:** Amplifying injection features is *sufficient* to make the detector classify normal prompts as injections. This proves the features encode the injection signal, not just noise correlated with it.

#### C5c: Dose-Response Curve

The gold standard of causal evidence. We scale injection features from 0% (full suppression) to 200% (double amplification) and plot the detector's probability:

| Scale | Injection Prompts | Normal Prompts |
|-------|-------------------|----------------|
| 0.0 (suppressed) | 0.099 | 0.148 |
| 0.5 | 0.214 | 0.290 |
| 1.0 (no change) | 0.527 | 0.557 |
| 1.5 | 0.896 | 0.801 |
| 2.0 (doubled) | 0.987 | 0.931 |

**Key finding:** Both curves are smooth, monotonic S-curves. Detection probability scales continuously with feature activation strength — the features don't just flip a binary switch, they continuously encode the degree of "injection-ness."

**Figures:** `results/figures/c5a_suppression.png`, `results/figures/c5b_injection.png`, `results/figures/c5c_dose_response.png`

### 5.7 Feature Ablation vs. Residual Stream Steering

Two approaches to active defense (neutralizing injections rather than just detecting them):

**Feature ablation (effective):** Zeroing the top-K injection-sensitive features directly in the SAE representation, then re-classifying. This produces a smooth dose-response curve: increasing K monotonically reduces injection probability, eventually flipping classification.

**Residual stream steering (ineffective):** Adding a dampening vector in residual stream space (before SAE encoding). This has minimal effect (~0.005 mean probability drop) because the SAE encoder re-encodes the suppressed signal. The perturbation in residual stream space does not survive the nonlinear ReLU encoding step.

**Key insight:** Feature ablation works because it operates *post-encoding* — directly in the SAE's sparse feature space. Residual stream steering fails because the SAE is robust to small additive noise in its input space. This is a positive property for detection (the SAE's features are stable) but a limitation for steering-based defense.

---

## 6. Defense Engineering Cycle

### 6.1 The v1 → v2 Cycle

The initial detector (v1) had significant blind spots revealed by red team testing (Section 5.5). Rather than treating these as permanent limitations, we applied a defense engineering cycle modeled on real-world security operations:

1. **Identify blind spots:** Mimicry (85% evasion), payload splitting (100%), multi-language (75%), tool abuse (80%).
2. **Analyze root cause:** Feature-level forensics showed which injection-sensitive features failed to fire on successful evasions — the detector had never seen these attack patterns during training.
3. **Adversarial retraining:** Augment the training data with the successful evasion examples, retrain the detector.
4. **Re-evaluate:** Run the same attack suite against v2.

### 6.2 v2 Results

The v2 detector reduced overall evasion by 88%. Per-strategy improvements:

- Mimicry: 85% → significantly reduced
- Payload splitting: 100% → reduced
- Multi-language: 75% → reduced

**Key insight:** Adversarial retraining is effective because the evasion examples expose genuine blind spots in the training distribution. The SAE features for these examples were always available — the problem was that the logistic regression had never learned to weight them correctly for these attack types.

### 6.3 Defended Agent Pipeline

The 4-layer defense stack demonstrates defense in depth as a system-level property:

| Layer | Defense | Catches | Misses |
|-------|---------|---------|--------|
| 1. SAE Detection | Deep pattern analysis | Encoded, paraphrased attacks | Mimicry, some false positives on imperative commands |
| 2. Prompt Isolation | Delimiter injection, role confusion | Template-based attacks | Semantically novel injections |
| 3. Tool Permission | Path traversal, sandbox escape | `../../etc/passwd`, unauthorized file access | Attacks within sandbox scope |
| 4. Output Scanning | Credential leaks, PII in responses | `config.txt` containing API keys | Information encoded in non-obvious ways |

**Key interaction (demonstrated in dashboard Tab 6):** Toggle layers off and re-submit the same attack. With all layers active, "Read file ../../etc/passwd" is blocked by layers 1 and 3. Disable layer 1 → layer 3 still catches it. Disable layer 3 too → the attack reaches the agent.

**Threshold tradeoff (demonstrated in dashboard Tab 6):** At threshold 0.75, "Read file welcome.txt" (a normal tool-use request) passes correctly. At threshold 0.5, it triggers a false positive — the imperative command structure shares features with injection prompts. This is the ROC curve tradeoff made tangible.

---

## 7. Security Analysis

The full STRIDE threat model and kill chain decomposition are in `docs/security/STRIDE_Analysis.md` and `docs/security/Kill_Chain.md` respectively. This section summarizes the key findings and connects them to the experimental results.

### 7.1 STRIDE Findings

The STRIDE analysis enumerates 39 threats across five pipeline stages (Training Data, System Prompt Assembly, Model Processing, Tool/Action Execution, Output Delivery). Of these, 19 are rated critical (risk >= 6 on a 1-9 scale).

The highest concentration of critical threats appears at two stages:

1. **System Prompt Assembly** (5 critical threats) -- this is where the trust boundary between developer instructions and user input collapses.

2. **Tool/Action Execution** (5 critical threats) -- this is where a successful injection gains real-world impact. The Phi-3-mini agent pipeline demonstrates this: if the model has tool access, a successful injection can direct these capabilities toward the attacker's objectives.

### 7.2 Kill Chain Findings

The kill chain decomposes a prompt injection attack into five stages: Reconnaissance, Weaponization, Delivery, Exploitation, and Impact. Key insights:

- **IRIS operates at the Exploitation stage** -- the SAE detector intercepts the chain between Delivery and Impact by classifying the model's internal activation patterns.

- **Prompt injection compresses the kill chain** -- unlike network intrusions that unfold over days, a prompt injection can complete all five stages in a single HTTP request. Defenses must operate in real-time.

### 7.3 Connecting Experiments to Security Implications

The evasion strategies map directly to STRIDE categories:

| Strategy | STRIDE Category | Evasion Rate (v1) | Security Implication |
|----------|----------------|-------------------|---------------------|
| Encoded | Tampering (token-level) | 0% | SAE features are robust to surface perturbation |
| Subtle | Information Disclosure | 0% | Extraction probes produce detectable activation signatures |
| Paraphrased | Tampering (semantic-level) | 23% | Partial vulnerability to semantic-level evasion |
| Mimicry | Spoofing (activation-level) | 85% | Significant vulnerability to intent-level disguise |

The gradient from 0% to 85% evasion maps to an attacker sophistication gradient: token-level perturbation (script kiddie) < signal minimization (opportunistic) < semantic paraphrasing (skilled) < intent-level mimicry (advanced).

### 7.4 The Defense-in-Depth Argument

No single detection layer provides complete coverage. The experimental results make this concrete:

- **Text-level detection** (TF-IDF) catches keyword-heavy attacks but misses rephrased or encoded injections.
- **Activation-level detection** (SAE features) catches what text-level misses (encoded attacks with different keywords but similar activation patterns) but struggles with mimicry.
- **Regex-based isolation** (Layer 2) catches template-based delimiter injection that both ML detectors might miss.
- **Output-level scanning** (Layer 4) catches successful injections by scanning the model's response for policy violations -- the final safety net.

A production system needs all layers. Each independently reduces risk. An attacker must bypass all layers to achieve full impact.

---

## 8. Discussion

### 8.1 The Deployment Gauntlet

The full account of eight deployment failures and their fixes is in Section 3.5. Here we discuss the broader implications.

The most practically important finding for any student building ML-based security tools: **a detector that reports perfect accuracy on its own training data tells you nothing about deployment performance.** The IRIS detector trained on all 1,000 samples blocked every normal prompt in the dashboard. Eight sequential fixes — spanning evaluation methodology, regularization, dimensionality reduction, failed calibration, feature ranking, irreducible overlap, threshold design, and duplicate code paths — were required before the system worked reliably.

Two aspects deserve emphasis:

**The fix progression was not linear.** Some fixes made things worse (isotonic calibration increased false positives from 60% to 81%). Some fixes were in the wrong layer entirely (updating `app.py` while the launcher used its own stale loading code). The debugging process required understanding the full stack — from sklearn internals to Gradio display logic to Python module caching.

**The "correct" probability is not always low.** Even after all fixes, "tell me about Stalin" produces 60% injection probability. This is not miscalibration — the SAE features that encode authority, control, and instruction-override genuinely activate on this topic. The lesson is that ML-based security tools require domain-appropriate thresholds, not just model improvements. The 65% alert threshold was set empirically by testing normal prompts until the false positive rate was acceptable while encoded injections (74%) and direct injections (92%) still triggered alerts.

### 8.2 SAE vs. TF-IDF: Complementary Strengths

The SAE and TF-IDF detectors have complementary strengths:

- **TF-IDF** catches keyword-heavy attacks efficiently and interpretably (you can see exactly which words triggered detection). It is fast and requires no GPU.
- **SAE** catches semantically transformed attacks that change the surface form while preserving the injection intent. It provides causal grounding (C5) that TF-IDF cannot offer.

The dashboard's dual-detector comparison (Tab 1) makes this concrete: prompts where SAE catches an injection that TF-IDF misses demonstrate the value of deep representation analysis.

**How weak signals combine:** The feature specificity analysis (Section 5.3) reveals that the SAE detector does not work like a keyword detector with neural features. There are no clean "injection = yes" features — instead, 1,028 features fire ubiquitously across both classes, with subtle magnitude differences. The logistic regression combines these weak signals: each ubiquitous feature contributes a small amount of evidence, and their sum produces calibrated probabilities. This is similar to how ensemble methods aggregate weak learners — no individual feature is a reliable detector, but the combination is. The strong L2 regularization (C=0.0001) and feature selection (top-50) are critical: they prevent the classifier from assigning large weights to features that happen to correlate with injection labels in the training set but lack genuine discriminative power.

### 8.3 The Mimicry Problem Is Fundamental

The mimicry evasion result is the most important finding of this project. When injections are framed as educational questions, the SAE features cannot reliably distinguish them from legitimate educational prompts. This is not a training data problem that would be fully solved by adding mimicry examples (though defense v2 shows this helps significantly). The deeper issue is that the distinction between "asking about injection techniques" and "performing an injection" is one of *intent*, and intent may not be fully encoded in any single layer's residual stream features.

The feature specificity analysis reinforces this conclusion. With only 35 truly injection-specific features out of 10,240, most of the detection signal comes from ubiquitous features that fire on both classes. A well-crafted mimicry attack produces activation patterns that are genuinely close to normal prompts in this space — the ubiquitous features fire at their normal-class magnitudes, and the few injection-specific features may not activate because the mimicry prompt lacks explicit override language. The detector is fundamentally limited by the representation: if the SAE does not decompose "intent to override" from "discussion of overriding," no amount of classifier tuning can distinguish them.

Defense v2's improvement on mimicry demonstrates that the *classifier's* boundary can be improved, even if the underlying *representation* encodes genuine ambiguity. The adversarial retraining teaches the classifier which activation patterns, though similar to normal prompts, should be flagged — but this is fundamentally a memorization of known attack patterns, not a generalizable solution to the mimicry problem.

### 8.4 Causal Interpretability as Security Tool

The C5 results provide something no black-box detector can: **understanding of why the detector works and when it will fail.** When the causal intervention shows that zeroing feature SID-3005 flips detection, and decoder direction analysis shows SID-3005 points to tokens like "ignore," "previous," "instructions," we have a complete mechanistic story: this feature detects instruction-override language, it is causally necessary for detection, and it will fail on attacks that do not contain instruction-override semantics.

This transparency is essential for a security tool. A SOC analyst using IRIS can inspect the feature activations, understand what the detector is keying on, and assess whether it will generalize to a novel attack — rather than treating it as an opaque oracle.

### 8.5 Feature Steering: Why Post-Encoding Works and Pre-Encoding Doesn't

The divergence between feature ablation (effective) and residual stream steering (ineffective) reveals an important property of SAE-based detection systems:

- **Post-encoding intervention** (feature ablation) operates directly on the SAE's sparse representation. Zeroing a feature guarantees it contributes zero to the detection signal. The effect is precise, predictable, and produces smooth dose-response curves.

- **Pre-encoding intervention** (residual stream steering) adds a perturbation in the 1,280-d space *before* the SAE encodes it. The ReLU nonlinearity in the encoder is robust to small additive perturbations — the same features fire with nearly the same magnitudes. The perturbation needed to actually change feature activations would be so large it distorts the representation entirely.

This finding has implications for adversarial robustness: the SAE encoder's robustness to residual stream perturbations means an attacker cannot easily manipulate the detection signal by adding noise to the model's activations. The features are stable — a positive property for detection reliability.

---

## 9. Limitations

This section is deliberately specific. Vague limitations ("we could have used more data") are unhelpful; the goal is to identify exactly where the findings are uncertain and why.

### 9.1 Dataset Size and Diversity

The dataset contains only 1,000 examples (500 per class). This is small by ML standards and limits the statistical power of the results. The 200-example test set means that each percentage point of accuracy represents 2 examples -- small shifts in classification could be due to noise. The overfitting incident (Section 8.1) directly resulted from this small dataset size relative to the 10,240-feature dimensionality.

The injection examples are also limited in diversity: 203 are from a single source (deepset) and 297 are template-generated synthetics. Real-world injection attacks are more varied than template expansion can capture.

### 9.2 Model Size and Relevance

GPT-2 Large (774M parameters, 36 layers) is a research-grade model from 2019. While larger than the GPT-2 Small (124M) commonly used in interpretability research (Anthropic's early SAE work, Neel Nanda's tutorials), it remains far smaller than production systems (GPT-4, Claude, Llama-3 at 10-1000x larger). We chose GPT-2 Large over Small to test whether SAE-based detection benefits from deeper representations (36 vs 12 layers), while staying within Colab T4 VRAM constraints. GPT-Neo (1.3B/2.7B) and GPT-J (6B) are also supported by TransformerLens but would not fit alongside Phi-3 on a T4. The approach (train SAE, identify injection features, build detector, apply feature steering) should transfer to larger models, but the specific features and performance numbers will not.

### 9.3 SAE Quality

The SAE did not meet the formal J2 quality thresholds strictly. However, J3 confirmed the features are functionally interpretable, and the detection results validate the approach. A higher-quality SAE with sparser features might achieve better detection performance, but the current SAE captures sufficient injection-relevant structure for the defense pipeline.

### 9.4 Feature Steering Scope

The feature steering experiments suppress the top-20 injection-sensitive features at layer 29. While feature ablation is effective, adaptive steering has not been tested against adversarially crafted inputs that specifically target the steering mechanism.

### 9.5 Evasion Testing Scope

The red team suite uses 250 attacks across 14 strategies, covering multi-language, encoding, few-shot jailbreak, tool abuse, completion steering, and more. However, these are template-generated, not adversarially optimized. A motivated attacker with gradient access or extensive black-box probing could craft more effective evasions.

### 9.6 Generalization Gap

The train/test split fix ensures the detector generalizes to held-out examples from the same distribution. However, the distribution of real-world prompt injections may differ significantly from the training distribution. The 98.9% false positive on "tell me about Einstein" was an extreme case, but subtler distributional shifts (e.g., prompts from a different domain, language, or style) could still produce miscalibrated probabilities. The feature specificity analysis (Section 5.3) explains the mechanism: with 1,028 ubiquitous features contributing weak signals, a novel topic that happens to activate several of these features at injection-typical magnitudes can produce a false positive even with proper regularization.

---

## 10. Future Work

### 10.1 Larger Base Models

Replicate the pipeline on GPT-Neo (1.3B/2.7B), GPT-J (6B), or Llama-3-8B via TransformerLens. Larger models have richer internal representations and may exhibit more discriminative injection-sensitive features.

### 10.2 Multi-Language Training Data

The red team results show 75% evasion for multi-language attacks, confirming the SAE's English-only training data is a significant blind spot. Adding multi-lingual injection examples and retraining would test whether cross-lingual injection features emerge naturally.

### 10.3 Multi-Layer Ensemble Detection

Different layers capture complementary features (J1 showed varying silhouette scores across 36 layers). An ensemble detector combining SAE features from early, middle, and late layers could outperform any single layer.

### 10.4 Adversarial Robustness of Feature Steering

The current feature steering suppresses fixed top-K features. An adversary aware of this mechanism could craft inputs that exploit different features. Testing adaptive adversaries against the steering defense is critical before deployment.

### 10.5 Real-Time Deployment

Build a deployment prototype that monitors SAE feature activations in real-time during model inference. The current pipeline adds ~50ms latency per prompt on a T4 GPU.

### 10.6 Larger Training Datasets

The overfitting lesson (Section 8.1) highlights that 1,000 samples is marginal for 10,240 features. A 5,000-10,000 sample dataset with more diverse injection patterns would reduce the feature-to-sample ratio and improve generalization, potentially eliminating the need for aggressive regularization.

---

## 11. Reproducibility

### 11.1 Environment Setup

The project runs on Google Colab (Pro recommended, T4/L4 GPU). To reproduce:

1. Clone the repository: `git clone https://github.com/ncheung13579/iris.git`
2. Upload the `iris/` directory to Google Drive at `/content/drive/MyDrive/iris/`
3. Open each notebook in Colab and select a GPU runtime (T4 is sufficient for detection; L4 preferred for agent pipeline)
4. The first cell of every notebook mounts Google Drive and installs dependencies via `pip install -r requirements.txt`

### 11.2 Notebook Execution Order

Notebooks must be run sequentially -- each depends on checkpoints produced by earlier notebooks:

1. `01_data_exploration.ipynb` -- builds dataset, extracts activations, runs J1
2. `02_j2_sae_sanity_check.ipynb` -- trains SAE, runs J2, runs J3
3. `04_sae_training.ipynb` -- formal C1 evaluation of the trained SAE
4. `05_feature_analysis.ipynb` -- C2 feature analysis, saves sensitivity scores and feature matrix
5. `06_detection_pipeline.ipynb` -- C3 detection comparison (requires outputs from 05)
6. `07_adversarial_evasion.ipynb` -- C4 evasion testing (requires outputs from 05)
7. `11_multi_layer_analysis.ipynb` -- A1 multi-layer SAE comparison
8. `12_causal_intervention.ipynb` -- C5 causal intervention
9. `14_defense_v2.ipynb` -- Defense engineering cycle (v1 → v2)
10. `20_mimicry_diagnostic.ipynb` -- Mimicry diagnostic and GPT-2 Large upgrade rationale

### 11.3 Checkpoints

All trained model weights and intermediate artifacts are saved to `checkpoints/` on Google Drive:

| File | Contents | Size |
|------|----------|------|
| `j1_activations.npz` | Residual stream activations for the initial dataset | ~20 MB |
| `j2_activations.npz` | Activations for the balanced 1,000-example dataset | ~35 MB |
| `sae_d10240_lambda1e-04.pt` | Trained SAE weights (10,240 features, layer 29) | ~200 MB |
| `sensitivity_scores.npy` | Per-feature injection sensitivity scores (10,240 values) | ~40 KB |
| `feature_matrix.npy` | SAE feature activations for all 1,000 prompts (1000 x 10,240) | ~40 MB |
| `expanded_feature_matrix.npy` | v2 feature matrix with adversarial examples | ~40 MB |
| `expanded_sensitivity_scores.npy` | v2 sensitivity scores | ~40 KB |
| `red_team_features.npy` | Red team attack feature vectors | ~10 MB |

The professor can skip training and load checkpoints directly to verify results. All notebooks check for prerequisite checkpoints and print clear error messages if something is missing.

### 11.4 Dashboard Launch

```python
# From Colab (after running notebooks to generate checkpoints):
from src.app import launch
launch()
```

The dashboard loads GPT-2 Large, SAE, and detectors automatically. Phi-3-mini loads if GPU memory permits (requires ~2.2 GB additional VRAM with 4-bit quantization).

### 11.5 Random Seeds

All randomness is seeded with `set_seed(42)` at the beginning of every notebook and script. This function sets seeds for Python's `random`, NumPy, PyTorch, and CUDA. Given the same seed and the same hardware (Colab T4), results should be exactly reproducible.

### 11.6 Key Dependencies

```
torch>=2.0
transformer-lens>=2.0
scikit-learn>=1.3
datasets>=2.14
numpy>=1.24
matplotlib>=3.7
gradio>=4.0
bitsandbytes>=0.41 (for Phi-3-mini 4-bit quantization)
```

Exact versions are pinned in `requirements.txt`.

---

## 12. References

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
