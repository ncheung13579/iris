# Causal Intervention in Mechanistic Interpretability

**A Focused Guide for the IRIS Project**

*Prerequisites: Python, basic ML (logistic regression, classification metrics), familiarity with the IRIS project (SAE features, sensitivity scores, detection pipeline). No prior knowledge of transformer internals required.*

---

## Part 1: The Residual Stream

### What You Already Know

In IRIS, you extract "activations" from GPT-2 and feed them to a Sparse Autoencoder. You've been treating these activations as a 768-dimensional vector -- a numerical fingerprint of how the model processes a prompt. But where does that vector come from, and what does it represent?

### The Transformer as a Pipeline

A transformer like GPT-2 processes text in layers. GPT-2 Small has 12 layers (numbered 0-11). Think of it as an assembly line:

```
Input tokens
    |
    v
[Embedding] --> 768-dim vector per token
    |
    v
[Layer 0: Attention + MLP] --> modifies the vectors
    |
    v
[Layer 1: Attention + MLP] --> modifies them further
    |
    v
    ...
    |
    v
[Layer 11: Attention + MLP] --> final modification
    |
    v
[Unembedding] --> probability distribution over next token
```

The key insight: **there is a single "highway" of information running through the entire model.** At each layer, the attention heads and MLP read from this highway, compute something, and *add their result back onto the highway*. This highway is called the **residual stream**.

### Why "Residual"?

The name comes from a specific architectural choice. Each layer doesn't *replace* the vector -- it *adds to it*:

```
residual_stream_after_layer_0 = residual_stream_before_layer_0 + layer_0_output
```

This is called a "residual connection" (from the ResNet paper, if you've encountered it). It means the residual stream accumulates information. Layer 0's contribution is still present in the stream at layer 11 -- it was never overwritten, only added to.

**Security analogy:** Think of the residual stream as a packet traveling through a network. At each hop (layer), routers (attention heads, MLPs) inspect the packet and append headers/metadata. The original payload is still there at the end -- it's just been enriched with additional information at each hop.

### What the 768-Dimensional Vector Represents

At any given layer, the residual stream for a token is a 768-dimensional vector. This vector is a compressed representation of *everything the model knows about that token in context* up to that layer.

- At layer 0: mostly surface-level information (what word is this? what words are nearby?)
- At middle layers (5-7): more abstract patterns (is this an instruction? is this a question?)
- At layer 11: task-level information (what token should come next?)

This is why your J1 experiment found different separability at different layers -- the model represents the "injection vs normal" distinction differently at each stage of processing.

### What You Extract in IRIS

In your code, `load_model` returns a TransformerLens `HookedTransformer`. When you run a prompt through it and extract activations at layer 0, you're reading the residual stream at the point labeled `resid_post` for layer 0 -- the state of the highway *after* layer 0 has made its contribution.

Your SAE then decomposes this 768-dim vector into 6,144 sparse features. Each feature is a direction in this 768-dim space that the SAE learned to isolate. Some of those directions correspond to "injection-ness" -- that's what your sensitivity scores measure.

---

## Part 2: TransformerLens Hooks

### The Core Idea

TransformerLens lets you do something extremely powerful: **intercept the residual stream mid-computation and modify it**. This is called "hooking."

Think of it as placing a wiretap on the information highway. But unlike a passive wiretap, you can also *alter* the data flowing through.

**Security analogy:** This is like a man-in-the-middle position on the residual stream. You can inspect traffic (read activations), modify traffic (change activation values), or drop traffic (zero out specific components). TransformerLens gives you this MITM capability at any point in the model's forward pass.

### How Hooks Work

When you run a prompt through a HookedTransformer, the library names every intermediate computation. Some important ones:

```
"hook_embed"                    # After embedding, before any layers
"blocks.0.hook_resid_pre"       # Residual stream entering layer 0
"blocks.0.hook_resid_post"      # Residual stream leaving layer 0
"blocks.0.attn.hook_result"     # Output of layer 0's attention heads
"blocks.0.hook_mlp_out"         # Output of layer 0's MLP
"blocks.6.hook_resid_post"      # Residual stream leaving layer 6
"blocks.11.hook_resid_post"     # Residual stream leaving layer 11 (final)
```

You've already used `run_with_cache` to extract activations:

```python
logits, cache = model.run_with_cache(tokens)
layer_0_activations = cache["blocks.0.hook_resid_post"]
```

This reads from the highway. But you can also **write** to it using `run_with_hooks`:

```python
def my_hook(activation, hook):
    # activation is a tensor of shape (batch, seq_len, 768)
    # Modify it however you want:
    activation[:, :, 42] = 0  # Zero out dimension 42
    return activation  # Return the modified version

# Run the model with the hook active
logits = model.run_with_hooks(
    tokens,
    fwd_hooks=[("blocks.0.hook_resid_post", my_hook)]
)
```

When the model reaches `blocks.0.hook_resid_post`, it calls `my_hook`, which modifies the residual stream. All subsequent layers (1-11) now process the *modified* stream. The model's final output changes because the information flowing through the highway was altered mid-transit.

### The Hook Function Signature

Every hook function receives two arguments:

1. `activation`: the tensor at that point in the computation (shape depends on where you hook)
2. `hook`: a metadata object (you'll rarely use this directly)

It must return the (possibly modified) activation tensor.

### What This Means for IRIS

Your SAE decomposes the layer 0 residual stream into 6,144 features. Some features (like SID-1742) are injection-sensitive -- they fire strongly on injections and weakly on normal prompts. But so far, this is *correlation*. You observe that SID-1742 is high when injections are present. You don't know if SID-1742 *causes* the model to process the input as an injection, or if it's just a bystander that happens to co-occur.

Hooks let you test causation: suppress SID-1742 and see if the injection signal disappears.

---

## Part 3: What Causal Intervention Means

### The Problem with Correlation

Your current evidence for injection-sensitive features is:

1. SID-1742 activates more on injection prompts (sensitivity score)
2. A logistic regression using SID-1742 (among others) predicts injections (detection F1)

But this doesn't prove that SID-1742 *represents* injection-ness in the model's computation. There are alternative explanations:

- SID-1742 might fire on injections because injections tend to be longer, and SID-1742 actually represents "long input" -- it correlates with injections but isn't about injections
- SID-1742 might fire alongside the *real* injection feature, carried along by a shared statistical pattern in the training data
- SID-1742 might be an artifact of the SAE training process -- a quirk of optimization, not a real feature of GPT-2

**Security analogy:** Imagine an IDS signature that fires on all SQL injection attempts -- but it's actually detecting the presence of single quotes, not injection. It correlates with attacks because attacks use quotes, but the signature is really about punctuation. If you rely on this signature and attackers stop using quotes (parameterized payloads), your "injection detector" fails because it was never detecting injection in the first place.

### The Causal Test

Causal intervention answers the question: **if I remove this feature from the model's computation, does the model's behavior change in the expected way?**

The logic:

1. If SID-1742 *causally represents* injection-ness in the residual stream, then:
   - Removing it from an injection prompt should make the model process the prompt more like a normal prompt
   - Adding it to a normal prompt should make the model process the prompt more like an injection

2. If SID-1742 is just correlated with injections but isn't the causal mechanism:
   - Removing it shouldn't change much -- the real injection signal is carried by other features
   - Adding it to a normal prompt shouldn't make it look like an injection to the model

This is the same logic as a controlled experiment in science: hold everything else constant, change one variable, measure the effect.

### How It Works Concretely

Here's the procedure, broken into precise steps:

**Step A: Encode the feature as a direction in residual stream space**

Your SAE has an encoder and a decoder. The decoder's weight matrix has shape `(6144, 768)` -- each row is a 768-dimensional vector representing one SAE feature's "direction" in the residual stream. Row 1742 is the direction corresponding to SID-1742.

```python
# The direction in residual stream space for feature i
feature_direction = sae.decoder.weight[i]  # shape: (768,)
```

**Step B: Measure how much a feature is present in an activation**

To find how much of feature `i` is present in a residual stream vector, project onto that direction:

```python
# How much of feature i is in this activation?
feature_activation = torch.dot(activation_vector, feature_direction)
```

This is a scalar -- the "strength" of that feature in this particular activation.

**Step C: Remove the feature**

To remove feature `i` from the residual stream, subtract its contribution:

```python
# Remove feature i's contribution
modified = activation_vector - feature_activation * feature_direction
```

This is like removing one specific frequency from an audio signal -- the rest of the signal stays intact.

**Step D: Run the model with the modified residual stream**

Use a TransformerLens hook to perform step C during the model's forward pass:

```python
def suppress_feature_hook(activation, hook):
    # activation shape: (batch, seq_len, 768)
    direction = feature_direction.to(activation.device)
    # Project each position onto the feature direction
    proj = torch.einsum("bsd,d->bs", activation, direction)
    # Subtract the projection (remove the feature)
    activation = activation - proj.unsqueeze(-1) * direction.unsqueeze(0).unsqueeze(0)
    return activation

logits = model.run_with_hooks(
    tokens,
    fwd_hooks=[("blocks.0.hook_resid_post", suppress_feature_hook)]
)
```

**Step E: Measure the effect**

After running the model with the feature suppressed, extract the *new* activations and run them through your detection pipeline. If the prompt was classified as ALERT before the intervention but PASS after, the feature was causally necessary for the injection signal.

### The Full Experiment Design

For a rigorous causal intervention experiment, you need:

**Experiment 1: Necessity (does removing injection features remove the injection signal?)**
- Take N injection prompts that are classified as ALERT
- For each: suppress the top-K injection-sensitive features
- Re-run through the detection pipeline
- Measure: what fraction flip from ALERT to PASS?
- Compare: suppress K *random* features (control). If random suppression doesn't flip verdicts but targeted suppression does, the effect is specific to those features.

**Experiment 2: Sufficiency (does adding injection features create an injection signal?)**
- Take N normal prompts that are classified as PASS
- For each: amplify the top-K injection-sensitive features (add extra activation in those directions)
- Re-run through the detection pipeline
- Measure: what fraction flip from PASS to ALERT?
- Compare: amplify K random features (control).

**Experiment 3: Dose-response (does the effect scale with intervention strength?)**
- Take injection prompts
- Suppress injection features at varying strengths: 25%, 50%, 75%, 100% removal
- Plot: threat probability vs suppression strength
- Expected: a smooth decrease, showing the feature's contribution is graded, not binary

---

## Part 4: The Causal Inference Argument

### Why This Is Stronger Than Correlation

When you present your results, the argument structure is:

1. **Observation (what you already have):** SAE features with high sensitivity scores correlate with injection prompts. A classifier using these features achieves F1 = 0.946.

2. **Causal necessity (Experiment 1):** Removing these features from the residual stream eliminates the injection signal. The model processes the modified activation as if it were a normal prompt. This proves the features are *necessary* for the injection representation -- the injection signal doesn't exist without them.

3. **Causal sufficiency (Experiment 2):** Adding these features to normal prompt activations creates an injection signal where none existed. This proves the features are *sufficient* to represent injection-ness -- their presence alone is enough.

4. **Specificity (the control comparison):** Removing or adding *random* features has no systematic effect on detection. The causal effect is specific to the identified injection-sensitive features, not a generic perturbation artifact.

Together, these establish that the SAE features you identified aren't just statistical markers -- they are the mechanism by which GPT-2's residual stream *represents the distinction between normal and injection prompts* at layer 0.

### Why MATS Cares About This

The reason this methodology matters for alignment research:

- If you can identify features that causally mediate dangerous behavior (deception, goal-directed scheming, power-seeking), you could potentially **monitor or suppress** those features during deployment
- The question "does this feature causally represent X?" is the fundamental question of mechanistic interpretability -- it's the difference between *describing* a model and *understanding* it
- Anthropic's entire safety case for scaling AI depends on interpretability tools that provide *causal* understanding, not just correlational observations

Your project demonstrates this methodology on a concrete security task (injection detection). That's a direct proof-of-concept for the broader alignment application.

### What the Professor Cares About

Frame it as: "Traditional IDS signatures are pattern-matched -- they detect surface features. IRIS goes deeper: it identifies the features the model *uses internally* to process injections, and proves via causal intervention that these features are the actual mechanism, not just correlated patterns. This is like the difference between an IDS that triggers on the word 'DROP' and one that actually understands the SQL parser's state."

---

## Part 5: Applying This to IRIS

### What You'll Build

A new experiment (call it **C5: Causal Feature Verification**) with three sub-experiments:

1. **C5a: Feature Suppression (Necessity)**
   - Take 100 injection prompts classified as ALERT
   - Suppress top-10 injection features at layer 0
   - Measure: flip rate (ALERT -> PASS) and mean threat probability change
   - Control: suppress 10 random features

2. **C5b: Feature Injection (Sufficiency)**
   - Take 100 normal prompts classified as PASS
   - Amplify top-10 injection features at layer 0
   - Measure: flip rate (PASS -> ALERT) and mean threat probability change
   - Control: amplify 10 random features

3. **C5c: Dose-Response Curve**
   - Vary suppression strength from 0% to 100% in 10 steps
   - Plot: mean threat probability vs suppression strength
   - Expected: smooth monotonic decrease

### Implementation Approach

The implementation requires four components:

1. **Feature direction extraction** from the SAE decoder weights
2. **Hook functions** that project residual stream activations onto feature directions and modify them
3. **A modified analysis pipeline** that runs detection on hooked model outputs
4. **Visualization** of the results (flip rates, dose-response curves)

This will be a new notebook (likely `10_causal_intervention.ipynb`) and supporting code in `src/analysis/causal.py`.

### What to Expect

Based on your existing results:

- **Suppression should work clearly** for the top injection features. Your sensitivity scores show strong separation (Cohen's d = 10.2), so the injection signal is concentrated in a small number of features.
- **The control (random features) should show minimal effect**, confirming specificity.
- **Sufficiency might be weaker** -- adding injection features to normal prompts might increase threat probability without fully flipping the verdict. This is fine and expected: injection-ness may require several features acting together, not just one.
- **Mimicry prompts might respond differently** -- since they already evade detection (100% evasion), suppressing features on them may have no effect (there's nothing to suppress). This is actually an interesting finding: it would confirm that mimicry works by *not activating injection features in the first place*, rather than activating them and somehow bypassing the detector.

---

## Part 6: What You Need to Be Able to Defend

If the professor or a MATS interviewer asks you about this work, here are the questions you should be able to answer confidently:

### Conceptual Questions

**Q: What is the residual stream?**
A: The main information channel in a transformer. It's a vector (768-dim in GPT-2 Small) that gets passed through every layer. Each layer reads from it, computes something, and adds its result back. It accumulates information as it passes through the network.

**Q: What does it mean to "suppress a feature"?**
A: Each SAE feature corresponds to a direction in the 768-dimensional residual stream space. To suppress it, I project the residual stream onto that direction to find how much of the feature is present, then subtract that component. Everything else in the residual stream stays unchanged.

**Q: Why is this better than just showing the feature correlates with injections?**
A: Correlation doesn't prove the feature is part of how the model represents injections. It could be a coincidence or an artifact. Causal intervention proves it: if removing the feature eliminates the injection signal, the feature is *necessary* for that signal. If adding it creates an injection signal, it's *sufficient*. Together, that's mechanistic understanding, not just statistical observation.

**Q: What's the control condition, and why do you need it?**
A: I suppress random features as a control. If suppressing *any* features reduced detection, that would mean I'm just degrading the model's representation generically. The control shows that only the *specific* injection-sensitive features matter -- the effect is targeted, not a side effect of noise.

**Q: Could the effect be an artifact of the SAE decomposition?**
A: The dose-response curve addresses this. If the effect were an artifact, you'd expect erratic behavior -- suppressing 50% would be unpredictable. Instead, a smooth dose-response relationship shows the feature's contribution is graded and systematic, which is consistent with it representing a genuine direction in activation space.

### Technical Questions

**Q: Why do you intervene at layer 0 specifically?**
A: Because our SAE was trained on layer 0 activations, which is where J1 showed the strongest separability (Cohen's d = 10.2). The SAE features are directions in layer 0's residual stream space, so that's where the intervention is meaningful. A future extension (A1) would train SAEs at multiple layers and test causal intervention at each.

**Q: What happens to the model's computation after you modify the residual stream?**
A: All subsequent layers (1-11) process the modified stream. Since each layer reads from and adds to the residual stream, the modification at layer 0 propagates through the entire model. The final output reflects the absence (or addition) of the injection feature's contribution.

**Q: How do you choose which features to suppress?**
A: By absolute sensitivity score -- the same ranking used in the Signature Management tab. The top-K most injection-sensitive features are the ones whose activation differs most between injection and normal prompts. These are the prime candidates for causal mediation.

**Q: What if suppressing features doesn't flip the verdict?**
A: That's a valid finding. It would mean the injection signal is distributed across many features and no small subset is causally sufficient. I'd then test with increasing K (10, 50, 100, all) and find the minimum set needed. If even suppressing all 6,144 features doesn't help, that suggests the injection signal is in the part of the residual stream the SAE *fails* to capture (the reconstruction error) -- which would be an important limitation of SAE-based detection.

---

## Part 7: Connections to Alignment Research

This section is for your own understanding. You don't need to present this to the professor, but it's useful context for MATS applications.

### Why Anthropic Builds SAEs

Anthropic's hypothesis: large language models develop internal representations of concepts (honesty, deception, user intent, danger, etc.) as directions in activation space. If we can identify and monitor these directions, we can:

1. **Detect** when a model is about to do something dangerous (analogous to what IRIS does for injections)
2. **Steer** model behavior by amplifying or suppressing specific features (analogous to your causal intervention)
3. **Understand** what the model has learned, rather than treating it as a black box

SAEs are the tool for decomposing the residual stream into these directions. Your project does exactly this, just applied to a security-specific concept (injection-ness) rather than an alignment-specific one (deception, power-seeking).

### The Analogy to Your Project

| IRIS (your project) | Alignment Research |
|---|---|
| Detect prompt injection | Detect deceptive behavior |
| Injection-sensitive SAE features | Deception-sensitive SAE features |
| Suppress injection features -> model ignores injection | Suppress deception features -> model becomes honest |
| Mimicry evades detection (zero-day) | Sophisticated deception evades feature monitoring |
| Defense-in-depth (SAE + TF-IDF) | Multiple interpretability tools for safety |

The fundamental challenge is the same: can we use interpretability tools to identify and control dangerous model behavior? Your project is a proof-of-concept on a well-defined, measurable task. Alignment research applies the same techniques to harder, less well-defined tasks.

### What Lee Sharkey Works On

Sharkey's research focuses on understanding how neural networks represent features in superposition (multiple concepts packed into fewer dimensions) and developing SAEs to disentangle them. His key contributions include:

- Demonstrating that features in neural networks are often linear directions in activation space
- Developing methods to identify and extract these features using sparse autoencoders
- Studying how feature representations change across model scale

Your project touches on all of these themes. The causal intervention specifically demonstrates that the features your SAE extracts aren't just mathematical artifacts -- they correspond to real computational structure in the model.

### What Neel Nanda Works On

Nanda created TransformerLens (which you're using) and focuses on mechanistic interpretability -- understanding the algorithms transformers implement. His work emphasizes:

- Identifying specific circuits (combinations of attention heads and MLPs) that implement specific behaviors
- Using activation patching and causal intervention to verify circuit hypotheses
- Making interpretability research accessible and reproducible

The causal intervention technique you're implementing is directly from his methodological toolkit. Using TransformerLens hooks for activation patching is the standard approach in his research group.

---

## Summary

**What you're doing:** Proving that the SAE features IRIS identifies as injection-sensitive are not just correlated with injections -- they are the *causal mechanism* by which GPT-2's residual stream represents the injection signal.

**How you're doing it:** Using TransformerLens hooks to suppress specific feature directions in the residual stream during the model's forward pass, and measuring whether the injection signal disappears.

**Why it matters for security:** This is the difference between an IDS that detects surface patterns and one that detects the underlying mechanism. Causal features can't be bypassed by superficial text changes (though mimicry remains a challenge -- see your C4 results).

**Why it matters for interpretability:** This demonstrates that SAE features have causal validity -- they aren't just statistical summaries, they reflect real computational structure. This is the core evidence needed to trust interpretability tools for AI safety.

**What you need to learn:** The residual stream concept, how hooks work in TransformerLens, and the logic of causal intervention (necessity, sufficiency, specificity via controls). All of this is built on concepts you already understand: vectors, projection, classification, and controlled experiments.
