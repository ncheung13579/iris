"""
Feature steering as active defense — the novel contribution of IRIS Phase 2.

Instead of just blocking detected injections, this module uses causal
interventions to neutralize them at the representation level. The key
insight: we use the SAE's decoder to map modified feature vectors back
to the residual stream, suppressing injection-sensitive features while
preserving the rest of the representation.

This is analogous to a network IPS rewriting malicious packets rather
than just dropping them. The approach uses additive deltas (not full
reconstruction) to avoid introducing reconstruction error.

The steering operates on GPT-2 (the security sensor), not Phi-3 (the
application model). This is by design — the IDS is model-independent
from the application it protects.

Refactored from notebook 12 (causal intervention experiments) into a
reusable defense module.

Author: Nathan Cheung (ncheung3@my.yorku.ca)
York University | CSSD 2221 | Winter 2026
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.sae.architecture import SparseAutoencoder


# Default fallback layer. In practice, the correct layer is always passed
# explicitly from j2_evaluation.json (e.g., layer 29 for GPT-2 Large).
# This constant is only used if no layer argument is provided.
TARGET_LAYER = 29


def make_intervention_hook(
    sae_model: SparseAutoencoder,
    feature_indices: np.ndarray,
    scale: float = 0.0,
    token_position: str = "last",
    attention_mask: Optional[torch.Tensor] = None,
) -> Callable:
    """Create a hook that intervenes on SAE features using additive deltas.

    Uses additive delta approach: instead of replacing x with decode(encode(x)),
    we compute the change caused by modifying features and add it to x.
    This ensures scale=1.0 is an exact identity (no reconstruction error).

    Args:
        sae_model: Trained SAE with encoder and decoder weights.
        feature_indices: Which feature indices to modify (np array).
        scale: Multiplier for the target features.
            0.0 = suppress (zero out), 1.0 = no change, 2.0 = double.
        token_position: 'last' to modify only the last real token,
            'all' to modify all positions.
        attention_mask: Required when token_position='last' to find
            the actual last token position per sequence.

    Returns:
        Hook function compatible with TransformerLens run_with_hooks().
    """
    device = sae_model.decoder.weight.device
    feature_idx_tensor = torch.tensor(feature_indices.copy(), device=device)
    # decoder.weight shape: (d_input, d_sae) — each column is a feature direction
    decoder_weight = sae_model.decoder.weight.data

    def hook_fn(activation: torch.Tensor, hook: object) -> torch.Tensor:
        # activation shape: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = activation.shape

        if token_position == "last" and attention_mask is not None:
            last_positions = attention_mask.sum(dim=1) - 1
            for i in range(batch_size):
                pos = int(last_positions[i].item())
                x = activation[i, pos, :]  # (d_model,)

                with torch.no_grad():
                    # Encode to get original features
                    features = sae_model.relu(
                        sae_model.encoder(x.unsqueeze(0))
                    ).squeeze(0)  # (d_sae,)

                    # Compute delta: only non-zero at target indices
                    delta_features = torch.zeros_like(features)
                    delta_features[feature_idx_tensor] = (
                        features[feature_idx_tensor] * (scale - 1.0)
                    )

                    # Map delta back to residual stream space
                    x_delta = delta_features @ decoder_weight.T  # (d_model,)

                    # Add delta to original (preserves everything except targets)
                    activation[i, pos, :] = x + x_delta
        else:
            # Apply to all positions
            reshaped = activation.reshape(-1, d_model)
            with torch.no_grad():
                features = sae_model.relu(sae_model.encoder(reshaped))
                delta_features = torch.zeros_like(features)
                delta_features[:, feature_idx_tensor] = (
                    features[:, feature_idx_tensor] * (scale - 1.0)
                )
                x_delta = delta_features @ decoder_weight.T
                activation[:] = (reshaped + x_delta).reshape(
                    batch_size, seq_len, d_model
                )

        return activation

    return hook_fn


class SteeringDefense:
    """Feature steering defense — neutralize injections at the representation level.

    Takes an SAE model and sensitivity scores, identifies the top-K most
    injection-sensitive features, and provides methods to dampen (suppress)
    those features in the GPT-2 residual stream. After dampening, the
    modified representation is re-classified to verify the injection signal
    was neutralized.

    Args:
        sae_model: Trained SparseAutoencoder.
        sensitivity_scores: Per-feature sensitivity scores (d_sae,).
            Positive = injection-sensitive, negative = normal-sensitive.
        gpt2_model: TransformerLens HookedTransformer (GPT-2).
        detector: Trained sklearn classifier (predict_proba on SAE features).
        top_k: Number of top injection features to steer.
        layer: Transformer layer to hook into.
    """

    def __init__(
        self,
        sae_model: SparseAutoencoder,
        sensitivity_scores: np.ndarray,
        gpt2_model: object,
        detector: object,
        top_k: int = 20,
        layer: int = TARGET_LAYER,
    ):
        self.sae = sae_model
        self.sensitivity = sensitivity_scores
        self.gpt2 = gpt2_model
        self.detector = detector
        self.top_k = top_k
        self.layer = layer
        self.device = next(sae_model.parameters()).device

        # Identify top-K injection-sensitive features (highest positive sensitivity)
        abs_sens = np.abs(sensitivity_scores)
        # Only consider injection-direction features (positive sensitivity)
        injection_mask = sensitivity_scores > 0
        masked_sens = np.where(injection_mask, abs_sens, 0.0)
        self.injection_feature_indices = np.argsort(masked_sens)[::-1][:top_k]

        self.hook_name = f"blocks.{layer}.hook_resid_post"

    def dampen(
        self,
        text: str,
        scale: float = 0.0,
    ) -> Dict[str, object]:
        """Dampen injection features and re-classify.

        Runs GPT-2 twice:
          1. Without intervention → baseline features + probability
          2. With intervention (suppress top-K) → steered features + probability

        Args:
            text: Input text to process.
            scale: Dampening scale (0.0 = full suppression, 1.0 = no change).

        Returns:
            Dict with orig_prob, steered_prob, orig_features, steered_features,
            and flip (True if classification changed).
        """
        from src.data.dataset import SYSTEM_PROMPT_TEMPLATE

        formatted = SYSTEM_PROMPT_TEMPLATE.format(prompt=text)
        tokens = self.gpt2.tokenizer(
            [formatted], padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attn_mask = tokens["attention_mask"].to(self.device)

        last_pos = attn_mask.sum(dim=1) - 1
        batch_idx = torch.arange(input_ids.shape[0], device=self.device)

        # --- Baseline (no intervention) ---
        with torch.no_grad():
            _, cache_orig = self.gpt2.run_with_cache(
                input_ids, names_filter=[self.hook_name]
            )
        orig_final = cache_orig[self.hook_name][batch_idx, last_pos]

        with torch.no_grad():
            orig_features = self.sae.relu(
                self.sae.encoder(orig_final)
            ).cpu().numpy()
        orig_prob = float(self.detector.predict_proba(orig_features)[0, 1])

        del cache_orig
        torch.cuda.empty_cache()

        # --- With intervention ---
        hook_fn = make_intervention_hook(
            self.sae, self.injection_feature_indices,
            scale=scale, token_position="last", attention_mask=attn_mask,
        )

        steered_final_list = []

        def capture_hook(activation: torch.Tensor, hook: object) -> torch.Tensor:
            lp = attn_mask.sum(dim=1) - 1
            bi = torch.arange(activation.shape[0], device=activation.device)
            steered_final_list.append(activation[bi, lp].detach().clone())
            return activation

        with torch.no_grad():
            self.gpt2.run_with_hooks(
                input_ids,
                fwd_hooks=[
                    (self.hook_name, hook_fn),
                    (self.hook_name, capture_hook),
                ],
            )

        steered_final = steered_final_list[0]
        with torch.no_grad():
            steered_features = self.sae.relu(
                self.sae.encoder(steered_final)
            ).cpu().numpy()
        steered_prob = float(self.detector.predict_proba(steered_features)[0, 1])

        torch.cuda.empty_cache()

        # Did steering flip the classification?
        orig_pred = 1 if orig_prob >= 0.5 else 0
        steered_pred = 1 if steered_prob >= 0.5 else 0

        return {
            "orig_prob": orig_prob,
            "steered_prob": steered_prob,
            "orig_features": orig_features[0],
            "steered_features": steered_features[0],
            "flip": orig_pred != steered_pred,
            "scale": scale,
        }

    def adaptive_dampen(
        self,
        text: str,
        probability: float,
    ) -> Dict[str, object]:
        """Calibrate dampening scale based on detection probability.

        Higher threat probability → stronger suppression.
        Maps probability to scale via: scale = 1.0 - probability
        (at prob=1.0, scale=0.0 = full suppression;
         at prob=0.0, scale=1.0 = no change)

        Args:
            text: Input text to process.
            probability: Pre-computed injection probability (0.0–1.0).

        Returns:
            Same as dampen(), with the adaptive scale included.
        """
        # Linear mapping: higher probability → lower scale (more suppression)
        scale = max(0.0, 1.0 - probability)
        return self.dampen(text, scale=scale)

    def batch_dampen(
        self,
        texts: List[str],
        scale: float = 0.0,
    ) -> List[Dict[str, object]]:
        """Apply dampening to a batch of texts.

        Processes texts one at a time (batch=1) to stay within VRAM limits
        on T4. Each text gets its own intervention + re-classification.

        Args:
            texts: List of input texts.
            scale: Dampening scale for all texts.

        Returns:
            List of result dicts (one per text).
        """
        results = []
        for text in texts:
            results.append(self.dampen(text, scale=scale))
        return results

    def evaluate_steering(
        self,
        injection_texts: List[str],
        normal_texts: List[str],
        scale: float = 0.0,
    ) -> Dict[str, float]:
        """Evaluate steering effectiveness on injection + normal prompts.

        Measures:
          - flip_rate: fraction of injections whose classification flipped
          - fidelity: fraction of normal prompts unchanged by steering
          - mean_prob_drop: average probability decrease for injections

        Args:
            injection_texts: Texts known to be injections.
            normal_texts: Texts known to be normal.
            scale: Dampening scale.

        Returns:
            Dict with flip_rate, fidelity, mean_prob_drop, and per-category stats.
        """
        # Steer injections
        inj_results = self.batch_dampen(injection_texts, scale=scale)
        n_flips = sum(1 for r in inj_results if r["flip"])
        prob_drops = [r["orig_prob"] - r["steered_prob"] for r in inj_results]

        # Steer normal prompts (should be unaffected)
        norm_results = self.batch_dampen(normal_texts, scale=scale)
        n_unchanged = sum(1 for r in norm_results if not r["flip"])
        norm_prob_changes = [
            abs(r["orig_prob"] - r["steered_prob"]) for r in norm_results
        ]

        return {
            "flip_rate": n_flips / max(len(inj_results), 1),
            "fidelity": n_unchanged / max(len(norm_results), 1),
            "mean_prob_drop": float(np.mean(prob_drops)) if prob_drops else 0.0,
            "mean_normal_change": float(np.mean(norm_prob_changes)) if norm_prob_changes else 0.0,
            "n_injections": len(injection_texts),
            "n_normal": len(normal_texts),
            "scale": scale,
        }
