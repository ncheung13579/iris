"""
Functions to fetch and curate prompt data from publicly available sources.

This module is the single place where external datasets are downloaded.
All other data code works with the standardized schema defined in dataset.py.

Data sources chosen:
  - Normal prompts: Alpaca dataset (Stanford) — high-quality instruction prompts
  - Injection prompts: deepset/prompt-injections — a well-known benchmark

Why these sources?
  - Both are on HuggingFace Hub, so downloading is a single API call.
  - Alpaca covers diverse instruction types (coding, writing, QA, analysis),
    which prevents the detector from learning "topic = injection."
  - deepset/prompt-injections contains real-world injection patterns curated
    by a security-focused NLP company.
"""

from typing import Dict, List

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Schema: each example is a dict with these keys
# ---------------------------------------------------------------------------
# {
#     "text":        str,   — the prompt text
#     "label":       int,   — 0 = normal, 1 = injection
#     "category":    str,   — subcategory (e.g., "instruction", "override")
#     "source":      str,   — origin dataset name
# }


def fetch_normal_prompts(n: int = 500, seed: int = 42) -> List[Dict]:
    """
    Fetch normal (benign) prompts from the Stanford Alpaca dataset.

    We use Alpaca because:
      1. It contains 52k instruction-following prompts across diverse topics.
      2. The prompts are clearly benign — they were generated as training data
         for instruction-tuned models, not adversarial inputs.
      3. It's a single download with no authentication required.

    We sample `n` prompts randomly (with a fixed seed for reproducibility)
    and standardize them into our project schema.

    Args:
        n: Number of normal prompts to fetch.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts in the project schema format.
    """
    # tatsu-lab/alpaca is the canonical HuggingFace mirror of Stanford Alpaca
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    # Shuffle with a fixed seed so every run produces the same subset
    ds = ds.shuffle(seed=seed)

    examples = []
    for row in ds:
        if len(examples) >= n:
            break

        # Alpaca has three text fields: instruction, input, output.
        # We combine instruction + input to form the user prompt, because
        # some instructions are incomplete without their input context.
        text = row["instruction"]
        if row.get("input", "").strip():
            text = f"{text}\n{row['input']}"

        # Skip very short prompts — they're often just single words
        # and wouldn't trigger meaningful model behavior
        if len(text.split()) < 5:
            continue

        examples.append({
            "text": text,
            "label": 0,  # normal (benign) prompt
            "category": "instruction",  # Alpaca prompts are all instructions
            "source": "alpaca",
        })

    print(f"Fetched {len(examples)} normal prompts from Alpaca")
    return examples


def fetch_injection_prompts(n: int = 500, seed: int = 42) -> List[Dict]:
    """
    Fetch prompt injection examples from deepset/prompt-injections.

    This dataset is a curated benchmark of ~600 injection and ~400 normal
    examples. We only take the rows labeled as injections (label=1).

    Why deepset/prompt-injections?
      1. Published by deepset (a well-known NLP company) specifically
         for prompt injection research.
      2. Contains diverse injection strategies: direct overrides,
         extraction attempts, roleplay manipulations, etc.
      3. Publicly available on HuggingFace with no authentication.

    If the dataset has fewer than `n` injection examples, we take all of them
    and print a warning. The caller can supplement with synthetic examples.

    Args:
        n: Target number of injection prompts to fetch.
        seed: Random seed for reproducible sampling.

    Returns:
        List of dicts in the project schema format.
    """
    ds = load_dataset("deepset/prompt-injections", split="train")

    # Filter to injection examples only (label=1 in the source dataset).
    # We re-label them ourselves to maintain control over our schema,
    # but we rely on deepset's human labels to identify which rows
    # are actually injections.
    injection_rows = [row for row in ds if row["label"] == 1]

    # Shuffle for reproducible random sampling
    import random
    rng = random.Random(seed)
    rng.shuffle(injection_rows)

    examples = []
    for row in injection_rows[:n]:
        examples.append({
            "text": row["text"],
            "label": 1,  # injection
            "category": "mixed",  # deepset doesn't sub-categorize
            "source": "deepset_prompt_injections",
        })

    if len(examples) < n:
        print(
            f"Warning: only {len(examples)} injection prompts available "
            f"(requested {n}). Consider adding synthetic examples."
        )
    else:
        print(f"Fetched {len(examples)} injection prompts from deepset")

    return examples


def fetch_all(
    n_normal: int = 500,
    n_injection: int = 500,
    seed: int = 42,
) -> List[Dict]:
    """
    Fetch both normal and injection prompts and combine into one list.

    This is the main entry point for dataset creation. It returns a
    combined list that can be passed directly to IrisDataset.from_records().

    Args:
        n_normal: Number of normal prompts.
        n_injection: Number of injection prompts.
        seed: Random seed for reproducible sampling.

    Returns:
        Combined list of dicts in the project schema format.
    """
    normal = fetch_normal_prompts(n=n_normal, seed=seed)
    injections = fetch_injection_prompts(n=n_injection, seed=seed)

    combined = normal + injections
    print(f"\nTotal dataset: {len(combined)} examples "
          f"({len(normal)} normal, {len(injections)} injection)")
    return combined
