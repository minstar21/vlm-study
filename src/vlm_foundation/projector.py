"""Spatial token merger and multimodal projection components."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gelu(values: ArrayLike) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    coefficient = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(coefficient * (x + 0.044715 * x**3)))


def merge_spatial_tokens(token_grid: ArrayLike, merge_size: int = 2) -> NDArray[np.float64]:
    """Merge each merge_size x merge_size block by concatenating its channels."""
    tokens = np.asarray(token_grid, dtype=np.float64)
    if tokens.ndim != 3:
        raise ValueError("token_grid must have shape [grid_h, grid_w, vision_dim]")
    height, width, dimension = tokens.shape
    if height % merge_size or width % merge_size:
        raise ValueError("token grid must be divisible by merge_size")
    merged = tokens.reshape(
        height // merge_size,
        merge_size,
        width // merge_size,
        merge_size,
        dimension,
    )
    return merged.transpose(0, 2, 1, 3, 4).reshape(
        height // merge_size, width // merge_size, dimension * merge_size**2
    )


def mlp_project(
    tokens: ArrayLike,
    weight_in: ArrayLike,
    bias_in: ArrayLike,
    weight_out: ArrayLike,
    bias_out: ArrayLike,
) -> NDArray[np.float64]:
    """Project merged vision tokens into the language-model hidden dimension."""
    x = np.asarray(tokens, dtype=np.float64)
    hidden = gelu(x @ np.asarray(weight_in) + np.asarray(bias_in))
    return hidden @ np.asarray(weight_out) + np.asarray(bias_out)

