"""Qwen3-VL inference adapter with cache-only loading by default."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class QwenResult:
    model_id: str
    image_path: str
    prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate(
    image_path: str | Path,
    prompt: str,
    *,
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
    max_new_tokens: int = 256,
    allow_download: bool = False,
) -> QwenResult:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Install the 'models' optional dependencies in an isolated env") from exc

    path = Path(image_path).resolve()
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=not allow_download)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto",
        local_files_only=not allow_download,
    ).eval()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path.as_uri()},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (perf_counter() - started) * 1000
    generated = output_ids[:, input_length:]
    response = processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return QwenResult(
        model_id=model_id,
        image_path=str(path),
        prompt=prompt,
        response=response,
        input_tokens=input_length,
        output_tokens=generated.shape[-1],
        latency_ms=latency_ms,
    )

