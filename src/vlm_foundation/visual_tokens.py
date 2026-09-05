"""Visual-token accounting for Qwen3-VL style dynamic resolution."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudget:
    resized_height: int
    resized_width: int
    pre_merge_tokens: int
    llm_visual_tokens: int
    effective_pixels_per_token: int


def visual_token_budget(
    height: int,
    width: int,
    *,
    patch_size: int = 16,
    merge_size: int = 2,
) -> TokenBudget:
    """Count tokens after dimensions are aligned to patch_size * merge_size."""
    factor = patch_size * merge_size
    resized_height = max(factor, math.ceil(height / factor) * factor)
    resized_width = max(factor, math.ceil(width / factor) * factor)
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    pre_merge = grid_h * grid_w
    return TokenBudget(
        resized_height=resized_height,
        resized_width=resized_width,
        pre_merge_tokens=pre_merge,
        llm_visual_tokens=pre_merge // (merge_size**2),
        effective_pixels_per_token=(patch_size * merge_size) ** 2,
    )


def self_attention_elements(text_tokens: int, visual_tokens: int) -> int:
    """Return sequence-length squared, a useful attention-cost proxy."""
    total = text_tokens + visual_tokens
    return total * total


def max_square_side_for_tokens(
    max_visual_tokens: int, *, patch_size: int = 16, merge_size: int = 2
) -> int:
    factor = patch_size * merge_size
    side_in_tokens = int(math.sqrt(max_visual_tokens))
    return side_in_tokens * factor

