"""
SAE training loop and evaluation for the IRIS project.

This module handles:
  - Training a SparseAutoencoder on cached residual-stream activations
  - Evaluating reconstruction quality against the J2 sanity-check criterion
    (reconstruction loss < 0.1 × input variance, sparsity < 10%)

Design note — why a standalone training function instead of a Trainer class?
The SAE training procedure is simple enough (fixed dataset, no data augmentation,
no learning-rate scheduling) that a plain function keeps the code shorter and
easier for the student to follow during presentation.

Author: Nathan Cheung (ncheung3@my.yorku.ca)
York University | CSSD 2221 | Winter 2026
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Local imports — SAE architecture and shared helpers
from src.sae.architecture import SparseAutoencoder
from src.utils.helpers import save_checkpoint, set_seed


# ───────────────────────────── defaults ──────────────────────────────
# These match the J2 sanity-check experiment described in the Design
# Document §6.1.  They are intentionally conservative (small expansion,
# short training) so the experiment finishes in minutes on a Colab T4.

DEFAULT_EXPANSION_FACTOR: int = 4       # 3072 features for d_input=768
DEFAULT_SPARSITY_COEFF: float = 1e-3    # λ in L = MSE + λ·‖f‖₁
DEFAULT_LR: float = 3e-4                # Adam learning rate
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_EPOCHS: int = 20
DEFAULT_LOG_EVERY: int = 50             # print stats every N batches


# ───────────────────────────── training ──────────────────────────────

def train_sae(
    activations: torch.Tensor,
    d_input: int,
    expansion_factor: int = DEFAULT_EXPANSION_FACTOR,
    sparsity_coeff: float = DEFAULT_SPARSITY_COEFF,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    device: Optional[torch.device] = None,
    seed: int = 42,
    checkpoint_dir: Optional[Path] = None,
    log_every: int = DEFAULT_LOG_EVERY,
) -> Dict[str, Any]:
    """Train a sparse autoencoder on pre-cached activation vectors.

    Args:
        activations: Tensor of shape (N, d_input) — one row per prompt.
            These are the residual-stream activations extracted from a
            specific transformer layer by src.model.transformer.
        d_input: Dimensionality of each activation vector (768 for GPT-2
            Small).
        expansion_factor: Ratio d_sae / d_input.  4× gives 3072 features,
            8× gives 6144.  We use 4× for the quick J2 check and 8× for
            the full training run (experiment C1).
        sparsity_coeff: Weight of the L1 penalty on the latent code.
            Higher values → sparser features but worse reconstruction.
        lr: Adam learning rate.
        batch_size: Mini-batch size.
        epochs: Number of full passes over the dataset.
        device: Torch device.  If None, auto-detected (GPU if available).
        seed: Random seed — used to re-seed at the start so the training
            is fully reproducible even when called mid-notebook.
        checkpoint_dir: Where to save the final checkpoint.  If None,
            defaults to ``checkpoints/``.
        log_every: Print loss statistics every this many batches.

    Returns:
        Dict with keys:
            ``model``  — the trained SparseAutoencoder (on ``device``)
            ``history`` — dict mapping metric name to list-of-floats
                          (one entry per epoch)
            ``final_metrics`` — dict of scalar summary metrics
    """
    # ------------------------------------------------------------------
    # 0.  Setup: seeding, device, data
    # ------------------------------------------------------------------

    # Re-seed everything so this function is reproducible regardless of
    # what random state the caller left behind.
    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure activations live on the correct device as float32.
    # .clone() avoids mutating the caller's tensor when we shuffle later.
    activations = activations.to(device=device, dtype=torch.float32).clone()
    n_samples = activations.shape[0]
    assert activations.shape == (n_samples, d_input), (
        f"Expected shape (N, {d_input}), got {activations.shape}"
    )

    # ------------------------------------------------------------------
    # 1.  Compute input variance — needed to interpret J2 criterion
    # ------------------------------------------------------------------
    # J2 says: "reconstruction loss < 0.1 of the input variance".
    # Input variance = mean over the dataset of ‖x − mean(x)‖².
    # We print this upfront so the user knows what the 0.1 threshold
    # means in absolute MSE terms.
    input_mean = activations.mean(dim=0)
    input_variance = ((activations - input_mean) ** 2).mean().item()
    j2_threshold = 0.1 * input_variance
    print(f"Dataset: {n_samples} activation vectors of dim {d_input}")
    print(f"Input variance (mean per-element MSE from mean): {input_variance:.6f}")
    print(f"J2 target — reconstruction MSE must be < {j2_threshold:.6f} "
          f"(= 0.1 × input variance)")
    print()

    # ------------------------------------------------------------------
    # 2.  Create the SAE and optimizer
    # ------------------------------------------------------------------
    sae = SparseAutoencoder(d_input=d_input, expansion_factor=expansion_factor)
    sae = sae.to(device)

    # Adam is the standard optimizer for SAE training in the literature
    # (Bricken et al. 2023, Cunningham et al. 2023).  Weight decay is
    # omitted — the L1 penalty on the *activations* (not weights)
    # already provides regularization.
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # 3.  Training loop
    # ------------------------------------------------------------------
    n_batches = (n_samples + batch_size - 1) // batch_size  # ceil division

    # History: per-epoch aggregate metrics for plotting later.
    history: Dict[str, List[float]] = {
        "total_loss": [],
        "mse_loss": [],
        "l1_loss": [],
        "mean_sparsity": [],  # fraction of active features (> 0)
    }

    # We create a separate Generator for shuffling so that the shuffle
    # order is reproducible without disturbing other random state.
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    for epoch in range(epochs):
        # Shuffle the dataset each epoch.  A fresh permutation avoids
        # the network memorising batch-position patterns.
        perm = torch.randperm(n_samples, generator=rng)
        activations_shuffled = activations[perm]

        epoch_total_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_l1_loss = 0.0
        epoch_active_frac = 0.0
        batches_in_epoch = 0

        sae.train()

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            x = activations_shuffled[start:end]

            # Forward pass — the SAE returns a dict with x_hat, features,
            # loss, mse_loss, and sparsity_loss (see architecture.py)
            output = sae(x)
            loss = output["loss"]
            mse = output["mse_loss"]
            l1 = output["sparsity_loss"]
            f = output["features"]

            # ── Backward pass ──
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ── Decoder weight normalization ──
            # After each gradient step we normalise decoder columns to
            # unit norm.  Without this, the network can shrink encoder
            # weights and inflate decoder weights, effectively reducing
            # the L1 penalty without genuinely learning sparser codes.
            # This is standard practice (Bricken et al. 2023).
            sae.normalize_decoder_weights()

            # ── Accumulate stats ──
            batch_active_frac = (f > 0).float().mean().item()
            epoch_total_loss += loss.item()
            epoch_mse_loss += mse.item()
            epoch_l1_loss += l1.item()
            epoch_active_frac += batch_active_frac
            batches_in_epoch += 1

            # ── Periodic logging ──
            if (batch_idx + 1) % log_every == 0 or batch_idx == n_batches - 1:
                print(
                    f"  Epoch {epoch + 1}/{epochs}  "
                    f"Batch {batch_idx + 1}/{n_batches}  "
                    f"loss={loss.item():.6f}  "
                    f"mse={mse.item():.6f}  "
                    f"l1={l1.item():.6f}  "
                    f"active={batch_active_frac:.3%}"
                )

        # ── End-of-epoch summary ──
        avg_total = epoch_total_loss / batches_in_epoch
        avg_mse = epoch_mse_loss / batches_in_epoch
        avg_l1 = epoch_l1_loss / batches_in_epoch
        avg_active = epoch_active_frac / batches_in_epoch

        history["total_loss"].append(avg_total)
        history["mse_loss"].append(avg_mse)
        history["l1_loss"].append(avg_l1)
        history["mean_sparsity"].append(avg_active)

        j2_status = "PASS" if avg_mse < j2_threshold else "FAIL"
        sparsity_status = "PASS" if avg_active < 0.10 else "FAIL"
        print(
            f"[Epoch {epoch + 1}/{epochs}]  "
            f"loss={avg_total:.6f}  mse={avg_mse:.6f}  l1={avg_l1:.6f}  "
            f"active={avg_active:.3%}  "
            f"J2-recon={j2_status}  J2-sparse={sparsity_status}"
        )
        print()

    # ------------------------------------------------------------------
    # 4.  Save checkpoint
    # ------------------------------------------------------------------
    if checkpoint_dir is None:
        checkpoint_dir = Path("checkpoints")

    d_sae = d_input * expansion_factor
    checkpoint_path = Path(checkpoint_dir) / (
        f"sae_d{d_sae}_lambda{sparsity_coeff:.0e}.pt"
    )

    config = {
        "d_input": d_input,
        "expansion_factor": expansion_factor,
        "sparsity_coeff": sparsity_coeff,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "n_samples": n_samples,
    }
    final_metrics = {
        "final_total_loss": history["total_loss"][-1],
        "final_mse_loss": history["mse_loss"][-1],
        "final_l1_loss": history["l1_loss"][-1],
        "final_mean_sparsity": history["mean_sparsity"][-1],
        "input_variance": input_variance,
        "j2_threshold": j2_threshold,
    }
    save_checkpoint(
        path=checkpoint_path,
        model=sae,
        optimizer=optimizer,
        config=config,
        metrics=final_metrics,
        epoch=epochs,
    )

    return {
        "model": sae,
        "history": history,
        "final_metrics": final_metrics,
    }


# ──────────────────────────── evaluation ─────────────────────────────

def evaluate_sae(
    model: SparseAutoencoder,
    activations: torch.Tensor,
    device: Optional[torch.device] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, float]:
    """Evaluate a trained SAE and check the J2 pass/fail criteria.

    Computes reconstruction quality, sparsity statistics, and dead
    features (features that never fire across the entire dataset).

    Args:
        model: A trained SparseAutoencoder instance.
        activations: Tensor of shape (N, d_input).
        device: Torch device.  If None, auto-detected.
        batch_size: Batch size for evaluation (no gradients, so can be
            larger than training batch size if memory allows).

    Returns:
        Dict with keys:
            ``mean_mse``        — average reconstruction MSE
            ``input_variance``  — variance of the input data
            ``j2_ratio``        — mean_mse / input_variance (must be < 0.1)
            ``j2_pass``         — bool, True if j2_ratio < 0.1
            ``mean_sparsity``   — fraction of features active (> 0) per sample
            ``sparsity_pass``   — bool, True if mean_sparsity < 0.10
            ``dead_features``   — count of features that never activated
            ``dead_feature_pct``— percentage of dead features
            ``total_features``  — d_sae
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    activations = activations.to(device=device, dtype=torch.float32)
    n_samples = activations.shape[0]

    model = model.to(device)
    model.eval()

    # We need to track per-feature activation across the whole dataset
    # to find dead features.
    d_sae = model.encoder.weight.shape[0]  # rows of encoder = d_sae
    feature_ever_active = torch.zeros(d_sae, dtype=torch.bool, device=device)

    total_mse = 0.0
    total_active_frac = 0.0
    n_batches = 0

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x = activations[start:end]

            output = model(x)
            f = output["features"]

            mse = output["mse_loss"].item()
            active_frac = (f > 0).float().mean().item()

            # Track which features fired at least once
            # Any feature that is > 0 in any sample in this batch
            batch_active = (f > 0).any(dim=0)
            feature_ever_active |= batch_active

            total_mse += mse
            total_active_frac += active_frac
            n_batches += 1

    # ── Aggregate metrics ──
    mean_mse = total_mse / n_batches

    # Input variance: mean squared deviation from the dataset mean
    input_mean = activations.mean(dim=0)
    input_variance = ((activations - input_mean) ** 2).mean().item()

    j2_ratio = mean_mse / input_variance if input_variance > 0 else float("inf")
    j2_pass = j2_ratio < 0.1

    mean_sparsity = total_active_frac / n_batches
    sparsity_pass = mean_sparsity < 0.10

    dead_count = int((~feature_ever_active).sum().item())
    dead_pct = 100.0 * dead_count / d_sae

    metrics = {
        "mean_mse": mean_mse,
        "input_variance": input_variance,
        "j2_ratio": j2_ratio,
        "j2_pass": j2_pass,
        "mean_sparsity": mean_sparsity,
        "sparsity_pass": sparsity_pass,
        "dead_features": dead_count,
        "dead_feature_pct": dead_pct,
        "total_features": d_sae,
    }

    # ── Print a clear summary ──
    print("=" * 60)
    print("SAE Evaluation — J2 Sanity Check")
    print("=" * 60)
    print(f"  Samples evaluated:     {n_samples}")
    print(f"  SAE features (d_sae):  {d_sae}")
    print()
    print(f"  Mean reconstruction MSE:  {mean_mse:.6f}")
    print(f"  Input variance:           {input_variance:.6f}")
    print(f"  J2 ratio (MSE / var):     {j2_ratio:.4f}  "
          f"(need < 0.1)  {'PASS' if j2_pass else '** FAIL **'}")
    print()
    print(f"  Mean sparsity (active %): {mean_sparsity:.3%}  "
          f"(need < 10%)  {'PASS' if sparsity_pass else '** FAIL **'}")
    print(f"  Dead features:            {dead_count}/{d_sae} "
          f"({dead_pct:.1f}%)")
    print("=" * 60)

    if j2_pass and sparsity_pass:
        print("J2 OVERALL: PASS — SAE meets both criteria.")
    else:
        reasons = []
        if not j2_pass:
            reasons.append("reconstruction too high")
        if not sparsity_pass:
            reasons.append("features not sparse enough")
        print(f"J2 OVERALL: ** FAIL ** — {', '.join(reasons)}.")
        print("Consider adjusting sparsity_coeff (lambda) or training longer.")

    return metrics
