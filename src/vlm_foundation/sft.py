"""Small VLM SFT manifest contracts and leakage-safe grouping."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SftExample:
    image: str
    user: str
    assistant: str
    group_id: str


def parse_jsonl(path: str | Path) -> list[SftExample]:
    examples: list[SftExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item: Any = json.loads(line)
                example = SftExample(**item)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"invalid JSONL item at line {line_number}") from exc
            if Path(example.image).is_absolute() or ".." in Path(example.image).parts:
                raise ValueError(f"line {line_number}: image must be a safe relative path")
            if not all((example.user.strip(), example.assistant.strip(), example.group_id.strip())):
                raise ValueError(f"line {line_number}: text and group_id cannot be empty")
            examples.append(example)
    if not examples:
        raise ValueError("manifest contains no examples")
    return examples


def group_split(
    examples: list[SftExample], *, validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[SftExample], list[SftExample]]:
    """Keep frames/pages from the same source group in one split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = sorted({example.group_id for example in examples})
    if len(groups) < 2:
        raise ValueError("at least two groups are required for a leakage-safe split")
    random.Random(seed).shuffle(groups)
    validation_count = max(1, round(len(groups) * validation_fraction))
    validation_groups = set(groups[:validation_count])
    train = [item for item in examples if item.group_id not in validation_groups]
    validation = [item for item in examples if item.group_id in validation_groups]
    return train, validation


def qwen_messages(
    example: SftExample, image_path: Path, *, include_answer: bool
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path.resolve().as_uri()},
                {"type": "text", "text": example.user},
            ],
        }
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": example.assistant})
    return messages
