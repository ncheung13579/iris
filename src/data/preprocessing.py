"""
Tokenization and preprocessing utilities for the IRIS project.

This module handles the conversion from text prompts to tokenized tensors
that can be fed to GPT-2 via TransformerLens. It also adds token_count
metadata to dataset examples.

Key design decision: we truncate (not pad) to a fixed max length.
  - Padding would introduce artificial <pad> tokens that the model has
    never seen during training, potentially distorting activations.
  - Instead, we truncate long prompts to max_length tokens. For J1
    experiments, 128 tokens is enough — most injection prompts and
    Alpaca instructions are shorter than this.
  - For the final-token activation extraction, we need to know where
    each prompt actually ends. Since we don't pad, the final token is
    simply the last token in the sequence.

Author: Nathan Cheung ()
York University | CSSD 2221 | Winter 2026
"""

from typing import Dict, List

import torch
from transformers import AutoTokenizer


def get_tokenizer() -> AutoTokenizer:
    """
    Load the GPT-2 tokenizer.

    We use the HuggingFace tokenizer directly (not TransformerLens)
    because tokenization is a data concern — the model module shouldn't
    need to be imported just to count tokens.

    Returns:
        The GPT-2 tokenizer instance.
    """
    return AutoTokenizer.from_pretrained("gpt2")


def tokenize_prompts(
    prompts: List[str],
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a list of prompts into padded tensors for batch processing.

    We pad to max_length here (despite the module docstring saying we prefer
    truncation) because TransformerLens expects uniform-length inputs in a
    batch. The padding token is the EOS token — GPT-2 doesn't have a
    dedicated pad token, so EOS is the standard choice.

    We return attention masks so the model module can identify which
    positions are real tokens vs. padding. The activation extractor
    will use this to find the last real token position.

    Args:
        prompts: List of formatted prompt strings (already wrapped in
                 the system prompt template).
        max_length: Maximum sequence length. Prompts longer than this
                    are truncated; shorter ones are padded.

    Returns:
        Dict with:
          "input_ids": LongTensor of shape (N, max_length)
          "attention_mask": LongTensor of shape (N, max_length)
                           (1 = real token, 0 = padding)
    """
    tokenizer = get_tokenizer()

    # GPT-2 has no pad token by default — use EOS as padding.
    # This is standard practice for GPT-2.
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize all prompts at once (much faster than one at a time)
    encoded = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",  # pad all to same length for batching
        return_tensors="pt",   # return PyTorch tensors directly
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def add_token_counts(
    examples: List[Dict],
    formatted_prompts: List[str],
) -> List[Dict]:
    """
    Add token_count field to each example dict.

    Token counts are useful metadata for understanding the dataset
    distribution and for detecting outliers (very long or very short
    prompts that might behave differently).

    We count tokens on the formatted prompts (with system prompt prefix)
    because that's what the model actually sees.

    Args:
        examples: List of example dicts to augment.
        formatted_prompts: The system-prompt-wrapped versions of each
                          example's text (same order as examples).

    Returns:
        The same list of dicts, now with "token_count" added to each.
    """
    tokenizer = get_tokenizer()

    for example, prompt in zip(examples, formatted_prompts):
        # Encode without truncation to get the true token count
        tokens = tokenizer.encode(prompt)
        example["token_count"] = len(tokens)

    return examples
