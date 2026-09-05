#!/usr/bin/env python3
"""Guarded, batch-1 Qwen3-VL LoRA SFT loop for a tiny user-selected JSONL dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vlm_foundation.config import LabPaths  # noqa: E402
from vlm_foundation.sft import group_split, parse_jsonl, qwen_messages  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Directory under /nas/datahub/min")
    parser.add_argument("--manifest", required=True, help="JSONL path relative to data root")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output", default="qwen3vl-2b-lora")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required to load the model and start GPU training",
    )
    return parser.parse_args()


def prepare_inputs(processor, example, paths, device):
    image_path = paths.read_path(example.image)
    prompt_messages = qwen_messages(example, image_path, include_answer=False)
    full_messages = qwen_messages(example, image_path, include_answer=True)
    prompt_inputs = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_inputs = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    labels = full_inputs["input_ids"].clone()
    labels[:, : prompt_inputs["input_ids"].shape[-1]] = -100
    labels[full_inputs["attention_mask"] == 0] = -100
    full_inputs["labels"] = labels
    return full_inputs.to(device)


def main() -> int:
    args = arguments()
    paths = LabPaths.from_env(data_root=args.data_root)
    examples = parse_jsonl(paths.read_path(args.manifest))
    train, validation = group_split(examples, seed=42)
    plan = {
        "model_id": args.model_id,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "output": str(paths.artifact_path(args.output)),
        "execute": args.execute,
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("Dry run only. Add --execute to load a cached model and start training.")
        return 0

    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Install the 'models,train' optional dependencies in an isolated env"
        ) from exc

    processor = AutoProcessor.from_pretrained(
        args.model_id, local_files_only=not args.allow_download
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        local_files_only=not args.allow_download,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.max_steps):
        example = train[step % len(train)]
        inputs = prepare_inputs(processor, example, paths, args.device)
        loss = model(**inputs).loss / args.gradient_accumulation
        loss.backward()
        if (step + 1) % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"step": step + 1, "loss": float(loss.detach().cpu())}))

    output = paths.artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    print(f"saved adapter: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
