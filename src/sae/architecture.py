"""
Sparse Autoencoder (SAE) architecture for mechanistic interpretability.

This module defines the SAE that learns a sparse, overcomplete decomposition
of GPT-2 Small's residual stream activations. The goal is to find monosemantic
features — individual neurons in the SAE that correspond to human-interpretable
concepts — some of which may be sensitive to prompt injection attacks.

Architecture (from Design Document §4.2):
    Input:   x ∈ R^768          (residual stream activation)
    Encoder: f = ReLU(W_enc · x + b_enc)   W_enc ∈ R^(d_sae × 768)
    Decoder: x̂ = W_dec · f + b_dec         W_dec ∈ R^(768 × d_sae)
    Loss = ||x - x̂||² + λ · ||f||₁
"""

from typing import Dict

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """A sparse autoencoder for decomposing transformer residual stream activations.

    The SAE learns an overcomplete basis (d_sae > d_input) where only a small
    fraction of basis elements activate for any given input. This sparsity
    pressure, enforced by the L1 penalty on feature activations, encourages
    each learned feature to represent a single interpretable concept
    (monosemanticity) rather than entangling multiple concepts in one neuron
    (polysemanticity).

    Args:
        d_input: Dimension of the input activations (1280 for GPT-2 Large).
        expansion_factor: How many times larger d_sae is than d_input.
            4x for quick experiments (J2), 8x (10240) for production.
        sparsity_coeff: Lambda weight for the L1 sparsity penalty in the loss.
            Higher values produce sparser features but worse reconstruction.
            Start with 1e-3 and tune on validation set.

    Example:
        >>> sae = SparseAutoencoder(d_input=768, expansion_factor=8)
        >>> x = torch.randn(32, 768)  # batch of activations
        >>> output = sae(x)
        >>> output["loss"].backward()
    """

    def __init__(
        self,
        d_input: int = 1280,
        expansion_factor: int = 8,
        sparsity_coeff: float = 1e-3,
    ) -> None:
        super().__init__()

        self.d_input = d_input
        self.d_sae = expansion_factor * d_input
        self.sparsity_coeff = sparsity_coeff

        # --- Encoder ---
        # Maps from residual stream space (768) to sparse feature space (d_sae).
        # The bias is initialized to zero; the weight uses Kaiming uniform,
        # which is appropriate for ReLU activations (accounts for the fact that
        # ReLU kills ~half the gradient signal).
        self.encoder = nn.Linear(d_input, self.d_sae)

        # --- Decoder ---
        # Maps from sparse feature space back to residual stream space.
        # No activation function — the reconstruction is a linear combination
        # of learned feature directions, weighted by their activations.
        self.decoder = nn.Linear(self.d_sae, d_input)

        # ReLU enforces non-negative feature activations.  This is critical:
        # it means each feature is either "off" (0) or "on" (positive value),
        # making sparsity well-defined and interpretable.
        self.relu = nn.ReLU()

        # Apply sensible initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming uniform for encoder, Xavier for decoder.

        Why two different schemes?
        - Encoder feeds into ReLU, so Kaiming (He) initialization is optimal —
          it accounts for the variance-halving effect of ReLU.
        - Decoder has no activation function, so Xavier (Glorot) is appropriate —
          it preserves variance through linear layers.
        Biases start at zero, which is standard practice.
        """
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the SAE encode-decode pass and compute all loss components.

        Args:
            x: Input activations of shape (batch_size, d_input). These are
               residual stream vectors extracted from GPT-2 at a chosen layer.

        Returns:
            Dict with keys:
                x_hat: Reconstructed activations, shape (batch_size, d_input).
                features: Sparse feature activations, shape (batch_size, d_sae).
                    Most values are zero due to ReLU + L1 pressure.
                loss: Total training loss = MSE + lambda * L1.
                mse_loss: Reconstruction loss (how well the SAE preserves info).
                sparsity_loss: L1 penalty on feature activations (encourages
                    the SAE to use few features per input).
        """
        # Encode: project to overcomplete space and apply ReLU for sparsity
        features = self.relu(self.encoder(x))

        # Decode: reconstruct the original activation as a linear combination
        # of learned feature directions (decoder columns), weighted by
        # the sparse feature activations
        x_hat = self.decoder(features)

        # --- Loss computation ---
        # MSE measures reconstruction quality: can we recover the original
        # activation from the sparse representation?  We want this to be low.
        mse_loss = torch.mean((x - x_hat) ** 2)

        # L1 penalty on feature activations encourages sparsity.  Without this,
        # the SAE would learn a dense representation (essentially PCA), which
        # defeats the purpose — we want each feature to fire rarely so it
        # corresponds to a specific concept.
        sparsity_loss = torch.mean(torch.abs(features))

        # Total loss balances reconstruction fidelity against sparsity.
        # Lambda (sparsity_coeff) controls the tradeoff:
        #   - Too low: dense features, poor interpretability
        #   - Too high: very sparse but bad reconstruction (information lost)
        #   - Sweet spot (~1e-3): ~1-2% of features active per input
        loss = mse_loss + self.sparsity_coeff * sparsity_loss

        return {
            "x_hat": x_hat,
            "features": features,
            "loss": loss,
            "mse_loss": mse_loss,
            "sparsity_loss": sparsity_loss,
        }

    @torch.no_grad()
    def normalize_decoder_weights(self) -> None:
        """Normalize each decoder column to unit norm.

        WHY this is necessary:
        The L1 penalty penalizes the magnitude of feature activations (encoder
        output). The SAE could "cheat" by making decoder columns very large
        and encoder outputs very small — the reconstruction stays the same
        (large_decoder * small_activation = same product), but the L1 penalty
        shrinks because activations are smaller. This is a degenerate solution
        that defeats the sparsity objective.

        By constraining decoder columns to unit norm after each gradient step,
        we remove this degree of freedom. The encoder must produce activations
        whose magnitude honestly reflects how much each feature contributes
        to the reconstruction.

        This is standard practice in the SAE literature (Anthropic, 2023;
        Cunningham et al., 2023).
        """
        # decoder.weight has shape (d_input, d_sae) — each COLUMN is a
        # feature direction in residual stream space.  We normalize along
        # dim=0 (the d_input dimension) so each column has unit L2 norm.
        norms = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)

        # Clamp to avoid division by zero for any dead feature whose decoder
        # column might be all zeros (unlikely but defensive).
        norms = torch.clamp(norms, min=1e-8)

        self.decoder.weight.data /= norms

    @torch.no_grad()
    def compute_sparsity_stats(
        self, features: torch.Tensor, threshold: float = 0.0
    ) -> Dict[str, float]:
        """Compute sparsity statistics for a batch of feature activations.

        These metrics help monitor SAE training health:
        - Active fraction tells you if sparsity is in the right range (~1-2%).
        - Dead features are wasted capacity — features that never activate.
          A high dead feature count means the SAE is underutilizing its
          capacity, possibly because lambda is too high or training hasn't
          converged.

        Args:
            features: Feature activations of shape (batch_size, d_sae).
                Typically the "features" value from forward()'s output dict.
            threshold: Activation threshold to consider a feature "active".
                Default 0.0 means any positive activation counts (since ReLU
                already zeroes negatives, this catches all non-zero features).

        Returns:
            Dict with keys:
                active_fraction: Mean fraction of features active per input.
                    Target: 0.01 - 0.02 (1-2% of d_sae features).
                active_features_per_input: Mean count of active features per
                    input (e.g., ~50-100 out of 6144 at good sparsity).
                dead_feature_count: Number of features that never activated
                    across the entire batch. Monitor this during training —
                    dead features should decrease as training progresses.
                dead_feature_fraction: dead_feature_count / d_sae.
        """
        # Boolean mask: True where a feature is active (above threshold)
        active_mask = features > threshold  # shape: (batch_size, d_sae)

        # Per-input: what fraction of the d_sae features fired?
        active_per_input = active_mask.float().sum(dim=1)  # (batch_size,)
        active_fraction = (active_per_input / self.d_sae).mean().item()
        mean_active_count = active_per_input.mean().item()

        # Across the batch: which features never activated for ANY input?
        # A feature is "dead" if its column in active_mask is all False.
        ever_active = active_mask.any(dim=0)  # (d_sae,) — True if activated at least once
        dead_count = int((~ever_active).sum().item())

        return {
            "active_fraction": active_fraction,
            "active_features_per_input": mean_active_count,
            "dead_feature_count": dead_count,
            "dead_feature_fraction": dead_count / self.d_sae,
        }
