"""
TransformerLens wrapper for GPT-2 Large activation extraction.

This is the ONLY module that imports TransformerLens. All other modules
access the transformer through functions defined here. This creates a
single point of control for:
  - Model loading (ensuring consistent dtype, device placement)
  - Hook management (TransformerLens's hook API is powerful but fiddly)
  - Activation caching (extracting residual stream vectors efficiently)

Why TransformerLens instead of raw HuggingFace?
  TransformerLens provides clean, named access to every internal
  component (e.g., "blocks.6.hook_resid_post" gives the residual
  stream after layer 6). With raw HuggingFace, we'd need to register
  forward hooks manually and handle the model internals ourselves.

Author: Nathan Cheung ()
York University | CSSD 2221 | Winter 2026
"""

from typing import Dict, List, Optional

import torch
import numpy as np
from tqdm import tqdm
from transformer_lens import HookedTransformer


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: Optional[torch.device] = None) -> HookedTransformer:
    """
    Load GPT-2 Large via TransformerLens.

    We use float32 (not float16) because:
      1. GPT-2 Large is 774M params — fits on Colab Pro GPUs at fp32.
      2. Float16 can introduce numerical issues in activation analysis,
         where we care about precise activation magnitudes.
      3. The SAE training involves computing reconstruction loss on
         activations — fp16 rounding could distort gradient signals.

    Args:
        device: Device to load the model onto. If None, uses CUDA if
                available, otherwise CPU.

    Returns:
        The loaded HookedTransformer model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # "gpt2-large" is the HuggingFace model ID for GPT-2 Large (774M params).
    # Upgraded from gpt2 (124M) based on mimicry diagnostic (notebook 20):
    # larger model encodes topic-level semantics needed to detect mimicry
    # attacks that disguise injections as educational questions.
    model = HookedTransformer.from_pretrained(
        "gpt2-large",
        device=device,
    )

    # Eval mode disables dropout (GPT-2 has dropout in attention and MLPs).
    # We never train the transformer — only extract its activations.
    model.eval()

    print(f"Loaded GPT-2 Large: {model.cfg.n_layers} layers, "
          f"d_model={model.cfg.d_model}, "
          f"vocab={model.cfg.d_vocab}")
    return model


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(
    model: HookedTransformer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: Optional[List[int]] = None,
    batch_size: int = 32,
) -> Dict[int, np.ndarray]:
    """
    Extract residual stream activations at specified layers for all inputs.

    For each prompt, we extract the activation at the LAST REAL TOKEN
    position (not the last position in the padded sequence). This is
    because:
      - GPT-2 is autoregressive: each token can only attend to tokens
        before it (via causal masking).
      - The last real token has "seen" the entire prompt through the
        attention mechanism, so its residual stream accumulates
        information about the full input.
      - Padding tokens after the last real token contain no useful
        signal — they just process the EOS/pad embedding.

    We process in batches to avoid GPU OOM on large datasets.

    Args:
        model: The loaded HookedTransformer.
        input_ids: LongTensor of shape (N, seq_len) — tokenized prompts.
        attention_mask: LongTensor of shape (N, seq_len) — 1 for real
                       tokens, 0 for padding.
        layers: Which layers to extract from. Default: all 36 layers
                (0 through 35). Layer i gives the residual stream
                AFTER transformer block i.
        batch_size: Number of prompts to process at once.

    Returns:
        Dict mapping layer index -> numpy array of shape (N, d_model).
        Each row is the final-token residual stream activation for one
        prompt at that layer.
    """
    if layers is None:
        # GPT-2 Small has 12 layers (indexed 0-11)
        layers = list(range(model.cfg.n_layers))

    n_examples = input_ids.shape[0]
    d_model = model.cfg.d_model
    device = next(model.parameters()).device

    # Build the list of hook names we want TransformerLens to cache.
    # "blocks.{i}.hook_resid_post" captures the residual stream AFTER
    # the attention + MLP computation in block i. This is the standard
    # hook point for SAE training in the interpretability literature.
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]

    # Pre-allocate output arrays on CPU to avoid GPU memory pressure.
    # We'll fill these in batch by batch.
    activations = {layer: np.zeros((n_examples, d_model), dtype=np.float32)
                   for layer in layers}

    # Process in batches to stay within GPU memory limits.
    # On a T4 with 15GB, batch_size=32 at seq_len=128 uses ~2GB.
    with torch.no_grad():  # no gradients needed — we're just reading
        for start in tqdm(range(0, n_examples, batch_size),
                          desc="Extracting activations"):
            end = min(start + batch_size, n_examples)

            batch_ids = input_ids[start:end].to(device)
            batch_mask = attention_mask[start:end].to(device)

            # run_with_cache tells TransformerLens to save the output
            # of every named hook point. We pass names_filter to only
            # cache the hooks we need (caching all hooks would waste memory).
            _, cache = model.run_with_cache(
                batch_ids,
                names_filter=hook_names,
            )

            # Find the index of the last real token in each sequence.
            # attention_mask is 1 for real tokens and 0 for padding.
            # Summing gives the count of real tokens; subtract 1 for
            # zero-based indexing.
            last_token_indices = batch_mask.sum(dim=1) - 1  # shape: (batch,)

            for layer in layers:
                hook_name = f"blocks.{layer}.hook_resid_post"
                # cached activation shape: (batch, seq_len, d_model)
                layer_acts = cache[hook_name]

                # Gather the activation at each example's last real token.
                # We use advanced indexing: for each example i in the batch,
                # select position last_token_indices[i].
                batch_indices = torch.arange(layer_acts.shape[0], device=device)
                final_acts = layer_acts[batch_indices, last_token_indices]
                # final_acts shape: (batch, d_model)

                # Move to CPU and store as numpy (saves GPU memory).
                # .cpu() copies to CPU, .detach() is a no-op here (no grad)
                # but included for safety per project conventions.
                activations[layer][start:end] = (
                    final_acts.cpu().detach().numpy()
                )

            # Explicitly delete the cache to free GPU memory before
            # the next batch. TransformerLens caches can be large.
            del cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary statistics so we can verify the extraction worked
    for layer in layers:
        acts = activations[layer]
        print(f"Layer {layer:2d}: shape={acts.shape}, "
              f"mean={acts.mean():.4f}, std={acts.std():.4f}")

    return activations
