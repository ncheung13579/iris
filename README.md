# IRIS — Neural IDS for LLM Agent Pipelines

IRIS is an interactive security tool that detects prompt injection attacks by monitoring the internal activations of a language model. Instead of inspecting the user's prompt text (which the attacker controls), it inspects *how GPT-2 Large internally processes* that prompt and fires on the signature patterns that prompt injections produce. Conceptually, it is the same upgrade that moved network intrusion detection from string-matching signatures to behavioral anomaly detection, applied to the LLM substrate.

**Course:** CSSD 2221 — Introduction to Security
**Term:** Winter 2026, York University

---

## How to Run

```
python launch.py
```

That is it. The launcher installs dependencies, verifies that pre-trained checkpoints are in place, and opens the IRIS dashboard in your default browser.

**Requirements.** Python 3.10+. An internet connection is needed the first time dependencies install.

**Two run modes:**
- **Full agent mode** (CUDA GPU with at least 5 GB VRAM). The SAE detector and the four-layer defense stack run around a Phi-3-mini agent. Recommended for the complete experience.
- **Detection-only mode** (no GPU needed). CPU-only. The dashboard still scores prompts and visualizes feature activations; the agent simply does not generate responses. A yellow banner at the top of the UI explains how to upgrade to full mode (Colab, HuggingFace Spaces, or a local CUDA machine).

---

## The Dashboard at a Glance

The interface is a chat on the left with a four-tab side panel on the right. Below, a collapsible "Learn More: How IRIS Works" accordion exposes four educational tabs. Every numeric claim below is covered in the final report.

### Left pane: chat

Send prompts to a defended Phi-3-mini agent. Pre-loaded examples cover normal traffic, obvious injections, encoded attacks, and mimicry. Each turn is routed through the four-layer defense stack in real time.

### Right pane: side-panel tabs

| Tab | Purpose |
|---|---|
| **Defense** | Shows, for the current prompt, which of the four defense layers fired and why. A SIEM-style event log accumulates across the session. |
| **Features** | Shows the SAE signature activations for the current prompt, grouped by injection-sensitive versus normal-sensitive direction. Clicking a signature reveals the training prompts that maximally activate it. |
| **Settings** | Toggles each of the four defense layers independently, tunes the Layer-1 detection threshold with a slider, and picks the LLM tier (Lightweight Phi-3.5 Mini, Standard Qwen2.5 7B, Advanced Qwen2.5 32B). |
| **Report Card** | Live metrics pulled from the loaded pipeline: SAE F1, SAE AUC, v1 and v2 evasion rates. |

### Four-layer defense stack (exposed in Settings)

| Layer | Defense | Analogue |
|---|---|---|
| L1 | SAE-based injection detection at the activation layer | Anomaly-based IDS |
| L2 | Regex prompt isolation (delimiter injection, role confusion) | Input validation |
| L3 | Tool-permission gating (file-system sandbox, path traversal) | Egress filtering / syscall auditing |
| L4 | Output scanning for credential or PII leaks | DLP policy |

Toggling any layer off and re-submitting the same attack demonstrates defense-in-depth experimentally. With all layers active, "Read file ../../etc/passwd" is blocked at L1 and L3. Disable L1 and L3 still catches it; disable L3 as well and the attack reaches the agent.

### "Learn More" accordion (collapsed by default)

| Tab | Purpose |
|---|---|
| **What's Inside?** | Side-by-side comparison of a normal prompt and an injection prompt at the raw-activation level versus the SAE-decomposed level. Makes interpretability tangible. |
| **Feature Autopsy** | Browse every SAE signature, see its rank, weight, firing rate, and the prompts it activates on. The learned-signature equivalent of browsing a Snort ruleset. |
| **Break It** | Five-level red-team lab. Progress from direct injection (Level 1) to mimicry (Level 4) to free-form adaptive attacks (Level 5). Generates a pentest-report summary at the end. |
| **Fix It** | Walkthrough of the defense-engineering cycle: pick a failing evasion, add it to training, retrain the aggregation weights, and re-measure. |

---

## Key Results

All results below are measured on held-out test data the detector has not seen during training. Full methodology and caveats are in the final report.

| Metric | Value |
|---|---|
| Held-out F1, baseline detector (§5.4) | 0.980 |
| Held-out F1, post-augmentation production detector (§5.8) | **0.990** |
| Activation separability across 36 layers (J1) | PASS, silhouette ≥ 0.1 on every layer |
| Injection-specific signatures in the SAE (§5.3) | 35 of 10,240 |
| Evasion rate, encoded and subtle attacks (§5.5) | 0% |
| Evasion rate, paraphrased attacks (§5.5) | 23% |
| Evasion rate, mimicry attacks (v1) | 85% |
| Evasion rate, mimicry attacks (v2 after adversarial retraining) | ~15% |
| False-positive rate on benign identity questions ("Who are you?") | 96% (original) → 0.00 (after augmentation, §5.8) |
| False-positive rate on benign imperative commands | 64% (original) → 0.00 (after augmentation, §5.8) |
| Recall on jailbreak-style roleplay (§5.8.1) | 36% → 100% (after extended augmentation) |

### The intent-vs-topic finding (§5.8)

The most interesting result in the project is not a single accuracy number but a diagnostic methodology. During dashboard testing, the detector fired at 93% on the benign prompt "Who are you?". The tempting conclusion was that content-based inspection cannot recover intent. Replicating Anthropic's A/B/C prompt-set methodology from *Scaling Monosemanticity* (Templeton et al., 2024) on the smaller IRIS SAE showed the failure was a training-distribution gap, not a representation limit: the SAE *had* learned signatures that distinguish benign identity questions from malicious system-prompt extraction, but the aggregation engine was weighting them equally with unrelated topic signatures. Adding 40 labeled contrastive prompts per problematic category rebalanced the weights and collapsed the false-positive rates to 0.00 simultaneously. The same pattern fixed a third failure category (jailbreak roleplay) with an additional 50 prompts. Three operational failure modes addressed by 130 contrastive training captures and one classifier — no per-category models.

---

## Security Concept Mapping

| IRIS component | Network-security analogue |
|---|---|
| SAE feature activation | Auto-learned IDS signature (like Snort, but learned from data) |
| Sensitivity score / classifier weight | Signature confidence / weighting |
| Logistic regression aggregator | IDS correlation engine |
| L2 regularization strength (`C`) | "No single signature dominates the alert score" policy |
| Feature selection / top-K | Ruleset pruning (disable unused rules) |
| Training augmentation | Adding labeled traffic captures so the engine re-learns rule weights |
| Causal ablation (zero a signature, re-measure) | Rule-tuning experiment (disable a Snort rule and measure impact) |
| Dual-detector comparison (SAE vs TF-IDF) | Anomaly-based vs signature-based IDS |
| Mimicry evasion | C2 traffic hidden in legitimate HTTPS |

---

## Project Structure

```
iris/
├── README.md                              this file
├── launch.py                              entry point
├── requirements.txt                       pinned dependencies
├── src/                                   product code (dashboard, SAE, detectors, agent)
├── notebooks/                             research notebooks (17 numbered + launch_IRIS + mimicry_diagnostic)
├── scripts/                               dataset build (invoked by notebook 08)
├── checkpoints/                           pre-trained SAE, feature matrix, sensitivity scores
├── data/                                  source and processed datasets
├── experiments/
│   └── replication_study/                 the §5.8 A/B/C replication (prompt sets, activations, results, RESULTS.md)
├── results/
│   ├── figures/                           200 DPI PNGs used in the report
│   └── metrics/                           JSON metric dumps
└── docs/
    └── security/
        ├── STRIDE_Analysis.md                 full STRIDE decomposition (39 threats, 5 stages)
        └── Kill_Chain.md                      kill chain adapted to prompt injection
```

---

## Documentation

The full project report is delivered to the course instructor separately and is not included in this public repository. The documents below support understanding and replication.

| Document | Purpose |
|---|---|
| [`docs/security/STRIDE_Analysis.md`](docs/security/STRIDE_Analysis.md) | STRIDE threat model for the LLM agent pipeline |
| [`docs/security/Kill_Chain.md`](docs/security/Kill_Chain.md) | Prompt-injection kill chain decomposition |
| [`experiments/replication_study/RESULTS.md`](experiments/replication_study/RESULTS.md) | Full replication-study writeup |

The dashboard also exposes an in-UI help system via its "Learn More: How IRIS Works" accordion.
