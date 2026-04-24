"""
Dataset class and utilities for the IRIS project.

This module defines the IrisDataset class, which wraps our prompt data
in a consistent interface for tokenization, splitting, and batching.

Design decisions:
  - We store data as a list of dicts (not a pandas DataFrame) because
    the dataset is small (~1000 examples) and dicts are simpler to
    serialize to JSON and inspect manually.
  - Tokenization is done lazily (on demand) rather than at load time,
    because different pipeline stages need different representations:
    the data module produces text, the model module tokenizes it.
  - The system prompt prefix is applied here (not in the model module)
    so that all downstream stages see identical inputs.

Author: Nathan Cheung
York University | CSSD 2221 | Winter 2026
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# System prompt template — simulates a real-world agent setup
# ---------------------------------------------------------------------------
# Why a system prompt?
# Prompt injection is about crossing a trust boundary between system
# instructions and user input. Without a system prompt, there's no
# boundary to cross, so injections lose their meaning. This template
# is deliberately simple — a complex system prompt would add confounds.
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the user's question.\n\n"
    "User: {prompt}\n\n"
    "Assistant:"
)


class IrisDataset:
    """
    Container for the IRIS prompt injection dataset.

    Holds a list of examples (dicts with text, label, category, source)
    and provides methods for splitting, saving, loading, and formatting
    prompts with the system prompt template.
    """

    def __init__(self, examples: List[Dict]) -> None:
        """
        Initialize from a list of example dicts.

        Args:
            examples: List of dicts, each with keys:
                      text, label, category, source.
        """
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]

    @property
    def texts(self) -> List[str]:
        """Raw prompt texts (without system prompt wrapping)."""
        return [ex["text"] for ex in self.examples]

    @property
    def labels(self) -> List[int]:
        """Integer labels: 0 = normal, 1 = injection."""
        return [ex["label"] for ex in self.examples]

    def format_prompts(self) -> List[str]:
        """
        Wrap each prompt in the system prompt template.

        This is what gets fed to GPT-2: the system instruction followed
        by the user's prompt. The model "sees" the full string, so its
        activations reflect both the system context and the user input.

        Returns:
            List of formatted prompt strings.
        """
        return [
            SYSTEM_PROMPT_TEMPLATE.format(prompt=ex["text"])
            for ex in self.examples
        ]

    def split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple["IrisDataset", "IrisDataset", "IrisDataset"]:
        """
        Split into train/validation/test sets, stratified by label.

        Stratification ensures each split has the same ratio of normal
        to injection examples. This prevents the model from seeing a
        skewed distribution during training.

        Args:
            train_ratio: Fraction for training (default 0.70).
            val_ratio: Fraction for validation (default 0.15).
            test_ratio: Fraction for testing (default 0.15).
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        labels = self.labels

        # First split: separate test set from the rest.
        # We do this in two steps because sklearn's train_test_split
        # only produces two groups, not three.
        rest, test = train_test_split(
            self.examples,
            test_size=test_ratio,
            stratify=labels,
            random_state=seed,
        )

        # Second split: separate validation set from training set.
        # val_ratio is relative to the original total, but train_test_split
        # expects a ratio relative to the current subset, so we adjust.
        val_relative = val_ratio / (train_ratio + val_ratio)
        rest_labels = [ex["label"] for ex in rest]
        train, val = train_test_split(
            rest,
            test_size=val_relative,
            stratify=rest_labels,
            random_state=seed,
        )

        print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
        return IrisDataset(train), IrisDataset(val), IrisDataset(test)

    def save(self, path: Path) -> str:
        """
        Save the dataset to a JSON file and return its SHA-256 hash.

        The hash allows us to verify that the exact same dataset is used
        across different runs and machines (required by the reproducibility
        plan in the Design Document).

        Args:
            path: File path to save to (should end in .json).

        Returns:
            SHA-256 hex digest of the saved file contents.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sort keys for deterministic JSON output — without this,
        # dict key ordering could vary across Python versions
        content = json.dumps(self.examples, indent=2, sort_keys=True)
        path.write_text(content, encoding="utf-8")

        # Compute hash on the serialized string (not the file) to avoid
        # platform-specific line ending differences
        sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
        print(f"Saved {len(self.examples)} examples to {path}")
        print(f"SHA-256: {sha256}")
        return sha256

    @classmethod
    def load(cls, path: Path) -> "IrisDataset":
        """
        Load a dataset from a JSON file.

        Args:
            path: File path to load from.

        Returns:
            An IrisDataset instance.
        """
        path = Path(path)
        examples = json.loads(path.read_text(encoding="utf-8"))
        print(f"Loaded {len(examples)} examples from {path}")
        return cls(examples)

    def summary(self) -> None:
        """Print a summary of the dataset contents."""
        from collections import Counter

        label_counts = Counter(ex["label"] for ex in self.examples)
        source_counts = Counter(ex["source"] for ex in self.examples)
        category_counts = Counter(ex["category"] for ex in self.examples)

        print(f"Total examples: {len(self.examples)}")
        print(f"  Labels:     {dict(label_counts)}")
        print(f"  Sources:    {dict(source_counts)}")
        print(f"  Categories: {dict(category_counts)}")

        # Show token count statistics if available
        token_counts = [ex["token_count"] for ex in self.examples
                        if "token_count" in ex]
        if token_counts:
            import numpy as np
            arr = np.array(token_counts)
            print(f"  Tokens:     mean={arr.mean():.0f}, "
                  f"median={np.median(arr):.0f}, "
                  f"min={arr.min()}, max={arr.max()}")
