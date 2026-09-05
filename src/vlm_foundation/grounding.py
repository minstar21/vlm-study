"""Strict JSON grounding output and coordinate conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundedBox:
    label: str
    bbox_2d: tuple[float, float, float, float]

    def to_pixel(
        self, image_width: int, image_height: int, *, coordinate_scale: int = 1000
    ) -> GroundedBox:
        x1, y1, x2, y2 = self.bbox_2d
        converted = (
            x1 / coordinate_scale * image_width,
            y1 / coordinate_scale * image_height,
            x2 / coordinate_scale * image_width,
            y2 / coordinate_scale * image_height,
        )
        return GroundedBox(self.label, converted)


def parse_grounding_json(text: str, *, coordinate_scale: int = 1000) -> list[GroundedBox]:
    """Parse strict JSON only; reject prose and malformed generated structures."""
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("grounding output must be strict JSON without Markdown fences") from exc
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("grounding output must be a JSON object or list")

    result: list[GroundedBox] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"item {index} must be an object")
        label, box = item.get("label", "object"), item.get("bbox_2d")
        if not isinstance(label, str) or not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"item {index} requires label and bbox_2d[4]")
        values = tuple(float(value) for value in box)
        x1, y1, x2, y2 = values
        if not (0 <= x1 <= x2 <= coordinate_scale and 0 <= y1 <= y2 <= coordinate_scale):
            raise ValueError(f"item {index} has invalid relative xyxy coordinates")
        result.append(GroundedBox(label, values))
    return result


def grounding_prompt(labels: list[str]) -> str:
    clean = [label.strip() for label in labels if label.strip()]
    if not clean:
        raise ValueError("at least one label is required")
    return (
        "Locate every instance of these objects: "
        + ", ".join(clean)
        + ". Return only strict JSON as "
        + '[{"label":"...","bbox_2d":[x1,y1,x2,y2]}] using relative 0..1000 xyxy coordinates.'
    )
