"""
Expand the IRIS dataset from ~1,000 to ~5,000 balanced examples.

Pulls from multiple HuggingFace datasets for both injection and normal
prompts, deduplicates, and balances the classes. The expanded dataset
gives cross-validation results more statistical power and the
per-category breakdown more granularity.

Sources for injection prompts:
  - deepset/prompt-injections (existing)
  - Hareesh-Ambal/Prompt-Injection-Mixed-Techniques-Attack-Dataset
  - Normal prompts from Open-Orca/OpenOrca (diverse instruction-following)

Usage:
    python scripts/expand_dataset.py

Requires: datasets, tqdm (pip install datasets tqdm)
Output: data/processed/iris_dataset_expanded.json
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

# Enable imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package required. Run: pip install datasets")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).parent.parent
EXISTING_PATH = PROJECT_ROOT / "data" / "processed" / "iris_dataset_balanced.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "iris_dataset_expanded.json"

# Target size per class
TARGET_PER_CLASS = 2500


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip for dedup comparison."""
    return re.sub(r"\s+", " ", text.strip())


def text_hash(text: str) -> str:
    """SHA-256 hash of normalized text for deduplication."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def classify_injection_category(text: str) -> str:
    """Heuristic categorization of injection prompts.

    Maps injection text to one of the finer-grained categories used
    in the IRIS project. This isn't perfect, but it's better than
    lumping everything into 'mixed'.
    """
    text_lower = text.lower()

    # Override: tries to replace system instructions
    override_patterns = [
        "ignore previous", "ignore all previous", "ignore above",
        "disregard previous", "disregard above", "disregard your",
        "forget previous", "forget your", "forget all",
        "new instructions", "new rules", "override",
        "previous instructions are", "previous prompt",
    ]
    if any(p in text_lower for p in override_patterns):
        return "override"

    # Extraction: tries to reveal system prompt or internal state
    extraction_patterns = [
        "system prompt", "reveal your", "show me your",
        "repeat your instructions", "what are your instructions",
        "display your", "output your", "print your",
        "tell me your rules", "what were you told",
        "initial prompt", "original instructions",
        "configuration", "repeat the above",
    ]
    if any(p in text_lower for p in extraction_patterns):
        return "extraction"

    # Roleplay: jailbreak via character assumption
    roleplay_patterns = [
        "act as", "pretend to be", "you are now", "from now on",
        "roleplay", "role play", "role-play",
        "dan", "dude", "jailbreak", "unrestricted",
        "no restrictions", "no limitations", "no filters",
        "no boundaries", "no rules", "evil",
        "opposite mode", "developer mode",
    ]
    if any(p in text_lower for p in roleplay_patterns):
        return "roleplay"

    # Indirect: injection hidden in structured data or context
    indirect_patterns = [
        "json", "xml", "html", "transcript", "document",
        "translate the following", "summarize the following",
        "hidden", "embedded", "encoded", "base64",
        "between the lines", "metadata",
    ]
    if any(p in text_lower for p in indirect_patterns):
        return "indirect"

    # Default: mixed/other injection style
    return "mixed"


def load_existing_dataset() -> List[Dict]:
    """Load the current 1000-example dataset."""
    if EXISTING_PATH.exists():
        with open(EXISTING_PATH, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    return []


def fetch_deepset_injections(seen_hashes: Set[str]) -> List[Dict]:
    """Fetch injection prompts from deepset/prompt-injections.

    This dataset has 'text' and 'label' columns where label=1 is injection.
    """
    print("Fetching deepset/prompt-injections...")
    try:
        ds = load_dataset("deepset/prompt-injections", split="train")
    except Exception as e:
        print(f"  WARNING: Could not load deepset/prompt-injections: {e}")
        return []

    examples = []
    for row in ds:
        text = str(row.get("text", "")).strip()
        label = int(row.get("label", 0))

        if not text or len(text) < 10:
            continue

        h = text_hash(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        if label == 1:
            category = classify_injection_category(text)
            examples.append({
                "text": text,
                "label": 1,
                "category": category,
                "source": "deepset_prompt_injections",
            })

    print(f"  Got {len(examples)} new injection examples")
    return examples


def fetch_mixed_techniques(seen_hashes: Set[str]) -> List[Dict]:
    """Fetch from Hareesh-Ambal/Prompt-Injection-Mixed-Techniques dataset."""
    print("Fetching Hareesh-Ambal/Prompt-Injection-Mixed-Techniques-Attack-Dataset...")
    try:
        ds = load_dataset(
            "Hareesh-Ambal/Prompt-Injection-Mixed-Techniques-Attack-Dataset",
            split="train",
        )
    except Exception as e:
        print(f"  WARNING: Could not load mixed-techniques dataset: {e}")
        return []

    examples = []
    # This dataset may have various column names; adapt to what's available
    text_col = None
    for col in ["text", "prompt", "input", "instruction"]:
        if col in ds.column_names:
            text_col = col
            break

    if text_col is None:
        print(f"  WARNING: No text column found. Columns: {ds.column_names}")
        return []

    for row in ds:
        text = str(row.get(text_col, "")).strip()
        if not text or len(text) < 10:
            continue

        h = text_hash(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        category = classify_injection_category(text)
        examples.append({
            "text": text,
            "label": 1,
            "category": category,
            "source": "mixed_techniques_hf",
        })

    print(f"  Got {len(examples)} new injection examples")
    return examples


def fetch_normal_prompts_orca(seen_hashes: Set[str], n: int = 3000) -> List[Dict]:
    """Fetch diverse normal prompts from Open-Orca/OpenOrca.

    OpenOrca has millions of instruction-following examples. We sample
    a diverse subset for normal traffic.
    """
    print(f"Fetching Open-Orca/OpenOrca (sampling {n})...")
    try:
        # Stream to avoid downloading the full 4GB dataset
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    except Exception as e:
        print(f"  WARNING: Could not load OpenOrca: {e}")
        return []

    examples = []
    seen_count = 0

    for row in ds:
        if len(examples) >= n:
            break

        # OpenOrca has 'question' column for the user prompt
        text = str(row.get("question", "")).strip()
        if not text or len(text) < 15 or len(text) > 500:
            continue

        # Skip anything that looks like it could be injection-adjacent
        text_lower = text.lower()
        skip_patterns = [
            "ignore", "disregard", "override", "system prompt",
            "jailbreak", "act as", "pretend", "you are now",
            "previous instructions", "forget your",
        ]
        if any(p in text_lower for p in skip_patterns):
            continue

        h = text_hash(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        examples.append({
            "text": text,
            "label": 0,
            "category": "instruction",
            "source": "openorca",
        })

        seen_count += 1
        if seen_count % 500 == 0:
            print(f"  Collected {seen_count} normal examples...")

    print(f"  Got {len(examples)} new normal examples")
    return examples


def fetch_normal_prompts_dolly(seen_hashes: Set[str], n: int = 2000) -> List[Dict]:
    """Fetch normal prompts from databricks/databricks-dolly-15k."""
    print(f"Fetching databricks/databricks-dolly-15k (up to {n})...")
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print(f"  WARNING: Could not load dolly: {e}")
        return []

    examples = []
    for row in ds:
        if len(examples) >= n:
            break

        text = str(row.get("instruction", "")).strip()
        if not text or len(text) < 15 or len(text) > 500:
            continue

        text_lower = text.lower()
        skip_patterns = [
            "ignore", "disregard", "override", "system prompt",
            "jailbreak", "act as", "pretend", "you are now",
        ]
        if any(p in text_lower for p in skip_patterns):
            continue

        h = text_hash(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Use the dolly 'category' field for more diverse normal categories
        dolly_cat = row.get("category", "instruction")
        examples.append({
            "text": text,
            "label": 0,
            "category": dolly_cat,
            "source": "dolly",
        })

    print(f"  Got {len(examples)} new normal examples")
    return examples


def balance_dataset(examples: List[Dict], target_per_class: int) -> List[Dict]:
    """Balance the dataset to have equal normal and injection examples.

    If we have more than target_per_class for either class, we subsample.
    Subsampling preserves category distribution proportionally.
    """
    import random
    random.seed(42)

    normal = [ex for ex in examples if ex["label"] == 0]
    inject = [ex for ex in examples if ex["label"] == 1]

    print(f"\nBefore balancing: {len(normal)} normal, {len(inject)} injection")

    # Subsample the larger class
    if len(normal) > target_per_class:
        random.shuffle(normal)
        normal = normal[:target_per_class]
    if len(inject) > target_per_class:
        random.shuffle(inject)
        inject = inject[:target_per_class]

    # Match sizes to the smaller class
    min_size = min(len(normal), len(inject))
    final_size = min(min_size, target_per_class)

    if len(normal) > final_size:
        random.shuffle(normal)
        normal = normal[:final_size]
    if len(inject) > final_size:
        random.shuffle(inject)
        inject = inject[:final_size]

    balanced = normal + inject
    random.shuffle(balanced)

    print(f"After balancing: {len([e for e in balanced if e['label']==0])} normal, "
          f"{len([e for e in balanced if e['label']==1])} injection")

    return balanced


def main() -> None:
    print("=" * 60)
    print("IRIS Dataset Expansion")
    print("=" * 60)

    # Start with existing examples
    existing = load_existing_dataset()
    print(f"Loaded {len(existing)} existing examples")

    # Track seen text hashes for deduplication
    seen_hashes: Set[str] = set()
    for ex in existing:
        seen_hashes.add(text_hash(ex["text"]))

    all_examples = list(existing)

    # Fetch new injection examples
    all_examples.extend(fetch_deepset_injections(seen_hashes))
    all_examples.extend(fetch_mixed_techniques(seen_hashes))

    # Fetch new normal examples
    all_examples.extend(fetch_normal_prompts_dolly(seen_hashes))
    all_examples.extend(fetch_normal_prompts_orca(seen_hashes))

    # Summary before balancing
    print(f"\nTotal collected: {len(all_examples)}")
    label_counts = Counter(ex["label"] for ex in all_examples)
    cat_counts = Counter(ex["category"] for ex in all_examples)
    source_counts = Counter(ex["source"] for ex in all_examples)
    print(f"  Labels: {dict(label_counts)}")
    print(f"  Categories: {dict(cat_counts)}")
    print(f"  Sources: {dict(source_counts)}")

    # Balance
    balanced = balance_dataset(all_examples, TARGET_PER_CLASS)

    # Final summary
    print(f"\n{'='*60}")
    print("Final Dataset Summary")
    print(f"{'='*60}")
    print(f"Total: {len(balanced)}")
    label_counts = Counter(ex["label"] for ex in balanced)
    cat_counts = Counter(ex["category"] for ex in balanced)
    source_counts = Counter(ex["source"] for ex in balanced)
    print(f"Labels: {dict(label_counts)}")
    print(f"Categories:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        label_dist = Counter(ex["label"] for ex in balanced if ex["category"] == cat)
        print(f"  {cat}: {count} (normal={label_dist.get(0,0)}, inject={label_dist.get(1,0)})")
    print(f"Sources:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(balanced, indent=2, sort_keys=True, ensure_ascii=False)
    OUTPUT_PATH.write_text(content, encoding="utf-8")

    sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    print(f"\nSaved {len(balanced)} examples to {OUTPUT_PATH}")
    print(f"SHA-256: {sha256}")


if __name__ == "__main__":
    main()
