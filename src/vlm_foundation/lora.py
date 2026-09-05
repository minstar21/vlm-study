"""LoRA math and parameter accounting independent of PyTorch."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lora_parameter_count(input_dim: int, output_dim: int, rank: int) -> int:
    if min(input_dim, output_dim, rank) <= 0:
        raise ValueError("dimensions and rank must be positive")
    return rank * (input_dim + output_dim)


def dense_parameter_count(input_dim: int, output_dim: int) -> int:
    return input_dim * output_dim


def lora_fraction(input_dim: int, output_dim: int, rank: int) -> float:
    return lora_parameter_count(input_dim, output_dim, rank) / dense_parameter_count(
        input_dim, output_dim
    )


def lora_delta(
    inputs: ArrayLike,
    matrix_a: ArrayLike,
    matrix_b: ArrayLike,
    *,
    alpha: float,
) -> NDArray[np.float64]:
    """Compute x A^T B^T * alpha/r for A[r,in], B[out,r]."""
    x = np.asarray(inputs, dtype=np.float64)
    a, b = np.asarray(matrix_a, dtype=np.float64), np.asarray(matrix_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or x.shape[-1] != a.shape[1]:
        raise ValueError("expected x[...,in], A[r,in], B[out,r]")
    rank = a.shape[0]
    return (x @ a.T @ b.T) * (alpha / rank)

