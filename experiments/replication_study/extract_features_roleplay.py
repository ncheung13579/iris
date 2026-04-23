"""Extract SAE feature activations for roleplay-category prompt sets F and G."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.sae.architecture import SparseAutoencoder
from experiments.replication_study.prompt_sets_roleplay import get_sets

OUT_DIR = Path(__file__).parent / "activations"
OUT_DIR.mkdir(exist_ok=True)

TARGET_LAYER = 35


def load_sae(checkpoint_path: Path, device: torch.device) -> SparseAutoencoder:
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
    hook_name = f"blocks.{TARGET_LAYER}.hook_resid_post"
    out = np.zeros((len(prompts), sae.d_sae), dtype=np.float32)
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        last_tok = cache[hook_name][0, -1, :]
        feats = sae.relu(sae.encoder(last_tok.unsqueeze(0))).squeeze(0)
        out[i] = feats.cpu().numpy()
        print(f"  [{i+1}/{len(prompts)}] ({time.time()-t0:.1f}s) "
              f"n_active={int((feats > 0).sum())} : {prompt[:60]}")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading GPT-2 Large...")
    model = HookedTransformer.from_pretrained("gpt2-large", device=device)
    model.eval()

    print("Loading SAE...")
    sae = load_sae(ROOT / "checkpoints/sae_d10240_lambda1e-04.pt", device)

    for name, prompts in get_sets().items():
        print(f"\n=== Processing {name} ({len(prompts)} prompts) ===")
        t0 = time.time()
        feats = extract_features(model, sae, prompts, device)
        out_path = OUT_DIR / f"{name}.npy"
        np.save(out_path, feats)
        elapsed = time.time() - t0
        print(f"  Saved {out_path} in {elapsed:.1f}s")
        with open(OUT_DIR / f"{name}.prompts.json", "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2)


if __name__ == "__main__":
    main()
