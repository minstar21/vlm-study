"""Minimal Vision Transformer shape operations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def patch_grid(height: int, width: int, patch_size: int = 16) -> tuple[int, int]:
    if height % patch_size or width % patch_size:
        raise ValueError("height and width must be divisible by patch_size")
    return height // patch_size, width // patch_size


def patchify(image: ArrayLike, patch_size: int = 16) -> NDArray[np.generic]:
    """Convert one HWC image into [num_patches, patch_size*patch_size*C]."""
    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError("image must have shape [H, W, C]")
    height, width, channels = value.shape
    grid_h, grid_w = patch_grid(height, width, patch_size)
    patches = value.reshape(grid_h, patch_size, grid_w, patch_size, channels)
    return patches.transpose(0, 2, 1, 3, 4).reshape(
        grid_h * grid_w, patch_size * patch_size * channels
    )


def attention_scores(tokens: ArrayLike, query: ArrayLike, key: ArrayLike) -> NDArray[np.float64]:
    """Compute scaled dot-product attention logits for one head."""
    x = np.asarray(tokens, dtype=np.float64)
    q_weight, k_weight = np.asarray(query, dtype=np.float64), np.asarray(key, dtype=np.float64)
    queries, keys = x @ q_weight, x @ k_weight
    return queries @ keys.T / np.sqrt(queries.shape[-1])

