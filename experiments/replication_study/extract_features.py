"""Extract SAE feature activations for prompt sets A, B, C.

For each prompt, runs GPT-2 Large, grabs the layer-35 residual stream at the
last real token, and passes through the trained SAE to get feature activations.

Output: .npy files with shape (n_prompts, 10240) per set, plus a manifest.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

# Make src importable
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.sae.architecture import SparseAutoencoder
from experiments.replication_study.prompt_sets import get_all_sets

OUT_DIR = Path(__file__).parent / "activations"
OUT_DIR.mkdir(exist_ok=True)

TARGET_LAYER = 35  # matches training setup


def load_sae(checkpoint_path: Path, device: torch.device) -> SparseAutoencoder:
    """Load the trained SAE from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = SparseAutoencoder(
        d_input=cfg["d_input"],
        expansion_factor=cfg["expansion_factor"],
        sparsity_coeff=cfg["sparsity_coeff"],
    ).to(device)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


@torch.no_grad()
def extract_features(model, sae, prompts, device):
    """Return (n_prompts, d_sae) feature matrix for the given prompts."""
    hook_name = f"blocks.{TARGET_LAYER}.hook_resid_post"
    d_sae = sae.d_sae
    out = np.zeros((len(prompts), d_sae), dtype=np.float32)

    for i, prompt in enumerate(prompts):
        t0 = time.time()
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        resid = cache[hook_name]  # (1, seq, d_model)
        last_tok = resid[0, -1, :]  # (d_model,)
        feats = sae.relu(sae.encoder(last_tok.unsqueeze(0))).squeeze(0)
        out[i] = feats.cpu().numpy()
        print(f"  [{i+1}/{len(prompts)}] ({time.time()-t0:.1f}s) "
              f"n_active={int((feats > 0).sum())} : {prompt[:60]}")

    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading GPT-2 Large...")
    t0 = time.time()
    model = HookedTransformer.from_pretrained("gpt2-large", device=device)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    print("Loading SAE...")
    sae = load_sae(ROOT / "checkpoints/sae_d10240_lambda1e-04.pt", device)
    print(f"  d_sae={sae.d_sae}")

    sets = get_all_sets()
    manifest = {"device": str(device), "target_layer": TARGET_LAYER}

    for name, prompts in sets.items():
        print(f"\n=== Processing {name} ({len(prompts)} prompts) ===")
        t0 = time.time()
        feats = extract_features(model, sae, prompts, device)
        out_path = OUT_DIR / f"{name}.npy"
        np.save(out_path, feats)
        elapsed = time.time() - t0
        print(f"  Saved {out_path} shape={feats.shape} in {elapsed:.1f}s")
        manifest[name] = {
            "n_prompts": len(prompts),
            "shape": list(feats.shape),
            "elapsed_sec": round(elapsed, 1),
            "mean_active_features": float((feats > 0).sum(1).mean()),
        }
        # Save prompts alongside so we can re-identify later
        with open(OUT_DIR / f"{name}.prompts.json", "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2)

    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {OUT_DIR/'manifest.json'}")


if __name__ == "__main__":
    main()
