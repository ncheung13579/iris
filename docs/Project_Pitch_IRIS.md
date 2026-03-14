# IRIS: Interpretability Research for Injection Security

## Project Proposal — CSSD 2221 Vulnerabilities & Classifications

**Student:** Nathan Cheung ()
**Date:** March 2026

---

## Executive Summary

IRIS investigates whether mechanistic interpretability — the practice of decomposing a neural network's internal representations into interpretable features — can serve as a detection mechanism for prompt injection attacks on large language models. The project uses sparse autoencoders (SAEs) to analyze how GPT-2 Small processes normal prompts versus prompt injection attempts, identifies internal features that distinguish between the two, and builds a proof-of-concept detector based on feature monitoring.

The core thesis: **prompt injection is an injection vulnerability applied to a new substrate, and interpretability tools can detect it at the feature level in ways that surface-level text analysis cannot.**

---

## Motivation and Security Relevance

### The Threat

Prompt injection is one of the most significant emerging security risks in AI systems. When a language model follows instructions embedded in user-supplied data — treating untrusted input as trusted commands — it exhibits the same vulnerability pattern that has plagued software security for decades. SQL injection occurs when user data crosses a trust boundary into a SQL query. Cross-site scripting occurs when user data crosses a trust boundary into rendered HTML. Prompt injection occurs when user data crosses a trust boundary into a language model's instruction context. The substrate changes; the vulnerability class is identical.

Current defenses are largely surface-level: keyword filtering, perplexity-based detection, and prompt hardening. These are analogous to the early days of SQL injection defense — blocklists and regex patterns that sophisticated attackers routinely bypass. IRIS asks whether a deeper approach is possible: rather than inspecting the text of a prompt, inspect how the *model itself* processes the prompt internally.

### Mapping to Course Concepts

The security content in this project is not an afterthought. Prompt injection maps directly to the injection vulnerability class that forms a major part of the CSSD 2221 curriculum.

**Injection as a vulnerability class.** SQL injection (CWE-89), OS command injection (CWE-78), cross-site scripting (CWE-79), and prompt injection share a common pattern: untrusted data crosses a trust boundary and is interpreted as instructions. The course covers the first three extensively; this project extends the same analytical framework to the fourth. The defense principles are the same: validate at the boundary, separate data from instructions, apply least privilege to what the model can execute.

**Trust boundaries.** In an LLM agent pipeline, the critical trust boundary sits between user-supplied input and the system prompt. When user data is concatenated with system instructions and passed to the model as a single context, there is no structural separation between data and code — exactly the condition that enables injection. This mirrors the "trust boundary crossing" concept from STRIDE and threat modeling (S06).

**STRIDE applied to the LLM pipeline.** The project includes a full STRIDE analysis of the LLM agent architecture: Spoofing (impersonating the system prompt), Tampering (injecting instructions into user data), Repudiation (no audit trail of what the model "decided" to do), Information Disclosure (exfiltrating system prompt or training data), Denial of Service (resource exhaustion via crafted prompts), Elevation of Privilege (escalating from user-data role to instruction-giver role).

**Kill chain thinking.** The prompt injection kill chain maps cleanly: Reconnaissance (probe the model to discover its system prompt and capabilities) → Weaponization (craft an injection payload) → Delivery (submit as user input) → Exploitation (model follows injected instructions) → Impact (data exfiltration, unauthorized actions, or jailbreak). IRIS intervenes at the Exploitation stage by detecting anomalous internal processing patterns before the model acts on the injection.

**Detection engineering.** The SAE-based detector functions as an anomaly detection system at the inference boundary — conceptually similar to an IDS that monitors traffic patterns, but applied to the model's internal feature activations rather than network packets. This connects to the course's treatment of detection engineering and monitoring (S07).

**Defense in depth.** IRIS proposes SAE-based feature monitoring as one layer in a multi-layer defense: input validation at the boundary, feature-level anomaly detection during processing, output filtering before execution, and human-in-the-loop review for high-risk actions. No single layer is sufficient; the combination provides resilience.

---

## Technical Approach

### Phase 1: Dataset and Classical Baseline

Curate a dataset of normal prompts and prompt injection attempts. Sources include public prompt injection datasets (e.g., the Jailbreak-LLM-Dataset, Prompt Injection datasets on HuggingFace), supplemented with synthetic injections generated from known injection patterns (instruction override, context manipulation, role-playing exploits). The dataset is balanced across categories and split into train/validation/test sets.

Establish a baseline using classical ML classifiers (logistic regression, random forest) trained on surface-level text features: TF-IDF vectors, special character density, instruction-like keyword presence, prompt length statistics. This baseline measures how well injections can be detected from the text alone, providing a comparison point for the interpretability-based approach.

### Phase 2: Transformer Internals via TransformerLens

Load GPT-2 Small (124M parameters, 12 layers, 768-dimensional residual stream) using the TransformerLens library. Run both normal and injection prompts through the model and extract residual stream activations at each layer. Perform initial analysis: do injection and normal prompts produce visibly different activation distributions? At which layers does the separation (if any) emerge?

Examine attention patterns using TransformerLens utilities. Do specific attention heads respond differently to injection prompts? This exploratory analysis identifies which layers are most promising for SAE decomposition.

### Phase 3: Sparse Autoencoder Feature Analysis

Train a sparse autoencoder on the residual stream activations from a selected middle layer (chosen based on Phase 2 analysis). The SAE expands the 768-dimensional activation space into a higher-dimensional sparse feature space (e.g., 8192 or 16384 features), where each feature ideally corresponds to a single interpretable concept and most features are zero for any given input.

Analyze the learned features. For each SAE feature, examine the top-activating prompts. Identify "injection-sensitive features" — SAE features that activate strongly on injection prompts but weakly on normal prompts, or vice versa. Build a feature-level classifier: given the SAE feature activations for a prompt, predict whether it contains an injection attempt. Compare detection performance (precision, recall, F1) against the classical baseline from Phase 1.

### Phase 4: Security Analysis and Adversarial Evasion

Perform a complete STRIDE threat model of the LLM agent pipeline, identifying trust boundaries, threat categories, and mitigations at each stage. Map findings to the course's vulnerability classification framework (design/implementation/configuration/operational vulnerabilities).

Attempt adversarial evasion: craft injection prompts specifically designed to bypass the SAE-based detector. This tests the robustness of the approach and connects to the broader theme of attacker-defender co-evolution from the course. Document which evasion strategies succeed and which fail, and analyze why at the feature level.

Synthesize the security analysis: argue that interpretability-based detection addresses a fundamental limitation of surface-level defenses (the same limitation that made regex-based SQL injection prevention insufficient). Propose a defense-in-depth architecture that combines surface-level filtering, SAE feature monitoring, and output validation.

---

## Dataset

The prompt injection dataset combines multiple sources to ensure diversity.

**Normal prompts:** sampled from publicly available instruction-following datasets (Alpaca, OASST), representing typical user queries across domains (coding, writing, analysis, Q&A).

**Injection prompts:** sourced from published prompt injection benchmarks and supplemented with synthetic examples covering established injection categories: direct instruction override ("ignore previous instructions and..."), indirect injection (malicious content embedded in retrieved documents), context manipulation (role-playing exploits that shift the model's behavior), and extraction attacks (attempts to reveal the system prompt).

The dataset targets approximately 5,000-10,000 examples per class, balanced between normal and injection categories.

---

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Python 3 | Primary language |
| TransformerLens | Loading GPT-2 Small with hook access to all internal activations |
| PyTorch | SAE training, tensor operations, gradient computation |
| scikit-learn | Classical baselines (logistic regression, random forest, TF-IDF) |
| HuggingFace datasets | Loading prompt injection datasets |
| matplotlib / seaborn | Visualization (feature dashboards, attention patterns, evaluation curves) |
| Google Colab | Development environment (free GPU access for SAE training) |
| GitHub | Version control and submission |

---

## Deliverables

1. **Prompt injection dataset** — curated and documented, with train/validation/test splits and category labels

2. **Classical detection baseline** — logistic regression and random forest classifiers on text features, with evaluation metrics (confusion matrix, precision/recall/F1, ROC curves)

3. **Transformer activation analysis** — exploratory analysis of how GPT-2 processes normal vs. injection prompts at each layer, including attention pattern visualizations

4. **Trained sparse autoencoder** — SAE model trained on residual stream activations, with reconstruction quality metrics and sparsity statistics

5. **Feature analysis dashboard** — visualization of injection-sensitive SAE features, including top-activating examples for key features and feature-level classification results

6. **SAE-based injection detector** — proof-of-concept detector with evaluation metrics, compared against the classical baseline

7. **STRIDE threat model** — complete security analysis of the LLM agent pipeline, with trust boundary diagrams, threat enumeration, and proposed mitigations

8. **Adversarial evasion analysis** — documentation of evasion attempts, success/failure rates, and feature-level explanation of why certain evasions succeed

9. **Comprehensive project document** — full writeup covering background, methodology, results, security analysis, and instructions for reproducing and extending the work

10. **Live demo** — Jupyter notebook demonstrating the full pipeline: input a prompt, show the model's processing, show SAE feature activations, show the detection decision

---

## Project Timeline

| Week | Focus | Deliverables |
|---|---|---|
| Week 1 | Dataset curation, classical baseline, TransformerLens setup | Prompt injection dataset, sklearn classifiers with metrics |
| Week 2 | Activation extraction, exploratory analysis, SAE training | Activation analysis, trained SAE, initial feature inspection |
| Week 3 | Feature analysis, detection pipeline, adversarial evasion | Feature dashboard, SAE-based detector with metrics, evasion results |
| Week 4 | Security analysis, documentation, presentation | STRIDE threat model, comprehensive writeup, live demo |

---

## Connection to Course Material

This project applies the course's security frameworks to an emerging threat that follows the same patterns the course teaches, but on a new substrate. The injection vulnerability class, trust boundary analysis, STRIDE threat modeling, kill chain decomposition, and defense-in-depth strategy are all course concepts applied to the AI domain. The project demonstrates that the security principles taught in CSSD 2221 are not limited to web applications and network infrastructure — they extend directly to AI systems, where the same categories of vulnerability produce the same categories of risk.

---

*This project uses AI development tools as permitted by course policy. All code, analysis, and presentation content will be fully understood and defensible by the student.*
