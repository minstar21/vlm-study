import json

import numpy as np
import pytest

from vlm_foundation.grounding import grounding_prompt, parse_grounding_json
from vlm_foundation.lora import lora_delta, lora_fraction, lora_parameter_count
from vlm_foundation.projector import merge_spatial_tokens
from vlm_foundation.sft import SftExample, group_split
from vlm_foundation.visual_tokens import self_attention_elements, visual_token_budget
from vlm_foundation.vit import patch_grid, patchify


def test_patchify_shapes() -> None:
    image = np.zeros((32, 48, 3))
    assert patch_grid(32, 48) == (2, 3)
    assert patchify(image).shape == (6, 16 * 16 * 3)


def test_spatial_merge_reduces_four_tokens_to_one() -> None:
    tokens = np.arange(4 * 6 * 3).reshape(4, 6, 3)
    assert merge_spatial_tokens(tokens, 2).shape == (2, 3, 12)


def test_qwen3_visual_token_budget() -> None:
    budget = visual_token_budget(640, 960)
    assert budget.pre_merge_tokens == 40 * 60
    assert budget.llm_visual_tokens == 600
    assert budget.effective_pixels_per_token == 1024
    assert self_attention_elements(100, 600) == 490_000


def test_strict_grounding_json() -> None:
    payload = json.dumps([{"label": "helmet", "bbox_2d": [100, 200, 400, 800]}])
    boxes = parse_grounding_json(payload)
    assert boxes[0].to_pixel(1920, 1080).bbox_2d == (192.0, 216.0, 768.0, 864.0)
    with pytest.raises(ValueError):
        parse_grounding_json(f"```json\n{payload}\n```")
    assert "strict JSON" in grounding_prompt(["helmet"])


def test_lora_parameter_math_and_delta() -> None:
    assert lora_parameter_count(4096, 4096, 8) == 65_536
    assert lora_fraction(4096, 4096, 8) < 0.004
    x = np.array([[1.0, 2.0]])
    a = np.array([[1.0, 0.0]])
    b = np.array([[2.0], [3.0]])
    np.testing.assert_allclose(lora_delta(x, a, b, alpha=1), [[2.0, 3.0]])


def test_group_split_has_no_group_leakage() -> None:
    examples = [
        SftExample(f"{group}-{index}.jpg", "q", "a", group)
        for group in ["a", "b", "c"]
        for index in range(2)
    ]
    train, validation = group_split(examples, validation_fraction=0.34)
    assert {item.group_id for item in train}.isdisjoint(
        {item.group_id for item in validation}
    )

