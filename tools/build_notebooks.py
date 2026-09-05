"""Generate the seven VLM course notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


SETUP = '''from pathlib import Path
import sys
import numpy as np

current = Path.cwd().resolve()
PROJECT_ROOT = next(
    (p for p in (current, *current.parents) if (p / "pyproject.toml").is_file()),
    Path("/nas/home/mhlee/vlm-foundation-7days"),
)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
print("project:", PROJECT_ROOT)
print("numpy:", np.__version__)'''


def build(day: int, title: str, paper_ids: list[str], sections: list[tuple[str, str]], snippets: dict[int, str]) -> dict:
    cells = [
        md(
            f'''# Day {day} - {title}

- status: implemented_toy_not_executed_real_model
- stage: VLM_DAY_{day}
- paper_ids: {", ".join(paper_ids)}
- dataset_ids: synthetic_toy, user_selected_nas_data
- seed: 42
- scope: educational implementation; real inference/training is opt-in

이 노트북은 다른 사용자 파일, 공유 환경, checkpoint를 자동으로 변경하지 않는다. 실제 데이터는
`/nas/datahub/min` 아래 사용자가 지정한 경로만 읽는다.''')
    ]
    for number, (heading, body) in enumerate(sections, 1):
        cells.append(md(f"## {number}. {heading}\n\n{body}"))
        if number == 4:
            cells.append(code(SETUP))
        if number in snippets:
            cells.append(code(snippets[number]))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "vlm_foundation": {"schema_version": 1, "day": day, "seed": 42},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


DAY1 = build(
    1,
    "ViT 복습",
    ["vit_2020"],
    [
        ("Learning question", "이미지를 token sequence로 바꾸면 Transformer가 무엇을 보게 되는가? patch embedding, positional information, self-attention의 shape를 손으로 확인한다."),
        ("Background theory", "`image [B,C,H,W] -> non-overlapping patches [B,N,P*P*C] -> linear embedding [B,N,D] -> position -> Transformer`. `N=(H/P)*(W/P)`이며 full self-attention의 score 행렬은 head마다 `[N,N]`이다. 해상도를 2배로 하면 token은 약 4배, attention 요소는 약 16배가 된다."),
        ("Paper connection", "ViT는 CNN의 spatial inductive bias를 줄이고 image patch를 NLP token처럼 처리했다. 여기서는 pretrained accuracy가 아니라 patchification과 attention shape를 재현한다."),
        ("Input/output and shapes", "입력은 HWC toy image, 출력 patch는 `[N,P*P*C]`다. 실제 구현은 patch projection 뒤 `[B,N,D]`가 되며 class token 또는 pooled representation이 global task에 사용된다."),
        ("Minimal implementation", "작은 8x8 image를 4x4 patch 네 개로 분해하고 한 attention head의 scaled dot-product score를 만든다."),
        ("Visualization sanity check", "patch 순서를 좌상단에서 우하단 raster order로 확인한다. patch transpose 순서가 틀리면 내용은 유지돼 보여도 spatial position이 뒤섞인다."),
        ("Experiment", "patch size 2와 4에서 token 수와 attention matrix 크기를 비교한다."),
        ("Metrics", "이날의 metric은 shape invariant와 patch reconstruction 오차다. 모델 accuracy는 다루지 않는다."),
        ("Interpretation", "ViT가 보는 기본 단위는 pixel 하나가 아니라 patch token이다. 작은 물체가 patch보다 작으면 초기 표현부터 정보가 뭉개질 수 있다."),
        ("Failure cases", "채널 순서, 정규화, H/W transpose, divisibility, positional embedding interpolation 오류를 확인한다."),
        ("Real-service implications", "고해상도 OCR과 grounding에서는 patch 수가 곧 memory와 latency로 이어진다. 입력 해상도 정책은 모델 밖의 사소한 전처리가 아니다."),
        ("Review questions", "1. patch size를 절반으로 하면 token 수는 어떻게 변하는가?\n2. position 정보가 없으면 어떤 구분이 어려운가?\n3. `[N,N]` attention이 해상도에 민감한 이유는?\n4. CLS pooling과 dense visual token의 용도 차이는?\n5. VLM에서 global feature 하나만으로 부족한 작업은 무엇인가?"),
    ],
    {
        5: '''from vlm_foundation.vit import attention_scores, patchify

image = np.arange(8 * 8).reshape(8, 8, 1)
patches = patchify(image, patch_size=4)
projection = np.eye(patches.shape[1], 4)
tokens = patches @ projection
scores = attention_scores(tokens, np.eye(4), np.eye(4))
print("patches:", patches.shape)
print("tokens:", tokens.shape)
print("attention scores:", scores.shape)''',
        6: '''print("첫 patch 4x4:\\n", patches[0].reshape(4, 4))
print("두 번째 patch 4x4:\\n", patches[1].reshape(4, 4))''',
        7: '''from vlm_foundation.vit import patch_grid
for patch_size in [2, 4]:
    gh, gw = patch_grid(8, 8, patch_size)
    n = gh * gw
    print(f"patch={patch_size}: tokens={n}, attention elements={n*n}")''',
    },
)


DAY2 = build(
    2,
    "Multimodal projector",
    ["llava_2023", "qwen3_vl_2025"],
    [
        ("Learning question", "vision encoder의 hidden dimension과 LLM hidden dimension이 다를 때 projector는 무엇을 학습하는가? spatial merge와 MLP projection을 분리한다."),
        ("Background theory", "projector는 vision token을 LLM이 소비할 embedding space로 보낸다. 단순 linear, 2-layer MLP, resampler/Q-Former 등 설계가 있다. Qwen3-VL 계열은 spatial merge로 인접 token을 묶은 뒤 merger MLP로 LLM dimension에 맞춘다."),
        ("Paper connection", "LLaVA류는 vision encoder와 LLM 사이 connector의 중요성을 보여준다. Qwen3-VL config는 patch size 16, spatial merge size 2와 vision-to-text output dimension을 명시한다. toy MLP는 교육용 근사다."),
        ("Input/output and shapes", "merge 전 `[Gh,Gw,Dv]`, 2x2 merge 후 `[Gh/2,Gw/2,4*Dv]`, MLP 후 `[N/4,Dllm]`이다. merge는 token 수를 줄이는 대신 한 token이 더 넓은 영역을 대표한다."),
        ("Minimal implementation", "2x2 vision token grid를 channel 방향으로 합친 뒤 두 층 MLP로 projection한다."),
        ("Visualization sanity check", "합쳐진 첫 token이 원래 grid의 어느 2x2 위치에서 왔는지 직접 확인한다."),
        ("Experiment", "merge size 1과 2에서 LLM visual token 수, projector input dimension을 비교한다."),
        ("Metrics", "shape, trainable parameter 수, alignment loss, downstream task quality를 구분한다. projector loss 감소만으로 grounding이 좋아졌다고 말할 수 없다."),
        ("Interpretation", "projector는 단순 dimension adapter이면서 modality alignment의 병목이다. 너무 강한 compression은 작은 글자와 위치 정보를 잃을 수 있다."),
        ("Failure cases", "token order 불일치, vision/LLM dtype mismatch, 잘못된 merge reshape, frozen connector, special image token 개수 불일치를 확인한다."),
        ("Real-service implications", "merge는 context와 KV cache를 줄이지만 detail을 희생한다. 문서 OCR과 장면 요약은 최적 token budget이 다르다."),
        ("Review questions", "1. projector가 필요한 두 dimension은 무엇인가?\n2. 2x2 merge가 token 수를 얼마나 줄이는가?\n3. concatenate와 average merge의 정보 차이는?\n4. projector만 학습하는 phase의 장단점은?\n5. fine detail 작업에서 compression이 만드는 실패는?"),
    ],
    {
        5: '''from vlm_foundation.projector import merge_spatial_tokens, mlp_project

rng = np.random.default_rng(42)
vision_grid = rng.normal(size=(4, 6, 3))
merged = merge_spatial_tokens(vision_grid, merge_size=2)
w1, b1 = rng.normal(size=(12, 8)), np.zeros(8)
w2, b2 = rng.normal(size=(8, 5)), np.zeros(5)
projected = mlp_project(merged, w1, b1, w2, b2)
print("vision grid:", vision_grid.shape)
print("merged:", merged.shape)
print("LLM-space tokens:", projected.reshape(-1, 5).shape)''',
        6: '''print("source top-left 2x2:\\n", vision_grid[:2, :2])
print("merged first token:\\n", merged[0, 0])''',
        7: '''for merge_size in [1, 2]:
    output = merge_spatial_tokens(vision_grid, merge_size)
    print(f"merge={merge_size}: grid={output.shape[:2]}, input_dim={output.shape[-1]}")''',
    },
)


DAY3 = build(
    3,
    "Visual tokens",
    ["qwen2_vl_2024", "qwen3_vl_2025"],
    [
        ("Learning question", "dynamic resolution에서 한 이미지가 몇 visual token을 차지하며, 해상도 선택이 context·memory·작은 물체에 어떤 trade-off를 만드는가?"),
        ("Background theory", "Qwen3-VL은 16x16 patch와 2x2 spatial merge를 사용한다. 정렬된 이미지에서 LLM visual token 하나는 대략 32x32=1024 pixel 영역에 대응한다. 실제 processor는 min/max pixels와 aspect ratio를 고려해 smart resize한다."),
        ("Paper connection", "Qwen-VL 계열의 dynamic resolution은 원본 aspect ratio와 detail을 더 유연하게 유지한다. Qwen3-VL은 Interleaved-MRoPE로 time/height/width position을 다루고, DeepStack으로 여러 ViT layer feature를 LLM에 연결한다."),
        ("Input/output and shapes", "이미지 `[H,W]`는 32의 배수로 맞춰지고 pre-merge grid `[H/16,W/16]`, LLM visual token은 `(H/16)*(W/16)/4`다. 전체 sequence는 text + image/video token이다."),
        ("Minimal implementation", "여러 해상도의 token 수와 attention cost proxy를 계산한다. 이 함수는 token accounting용이며 공식 smart_resize의 대체 구현이 아니다."),
        ("Visualization sanity check", "resized H/W가 32의 배수인지, 극단적 aspect ratio에서 token이 폭증하지 않는지 표로 확인한다."),
        ("Experiment", "같은 text prompt에 224, 640x960, 4K 이미지를 넣을 때 sequence-squared 비용을 비교한다."),
        ("Metrics", "visual token 수, preprocess 시간, peak memory, time-to-first-token, grounding/OCR accuracy를 함께 본다."),
        ("Interpretation", "해상도는 품질 knob이면서 비용 knob이다. max_pixels를 올리면 무조건 좋아지는 것이 아니라 context와 batch capacity를 소모한다."),
        ("Failure cases", "과도한 downscale의 작은 글자 누락, 과도한 upscale, panorama token 폭증, video frame 수와 pixel budget 곱셈을 확인한다."),
        ("Real-service implications", "업무별 resolution policy와 total visual-token budget을 둔다. 요청자가 임의 4K multi-image를 보내 전체 queue를 막지 않도록 제한한다."),
        ("Review questions", "1. 640x960은 몇 LLM visual token인가?\n2. 2x2 merge 전후 token 수 차이는?\n3. image token 증가가 text context에 주는 영향은?\n4. OCR과 captioning의 해상도 정책이 다른 이유는?\n5. video에서는 어떤 두 축이 token을 늘리는가?"),
    ],
    {
        5: '''from vlm_foundation.visual_tokens import self_attention_elements, visual_token_budget

for height, width in [(224, 224), (640, 960), (1080, 1920), (2160, 3840)]:
    budget = visual_token_budget(height, width)
    print((height, width), budget)''',
        6: '''budgets = [visual_token_budget(h, w) for h, w in [(224,224), (640,960), (1080,1920)]]
assert all(b.resized_height % 32 == 0 and b.resized_width % 32 == 0 for b in budgets)
print("alignment check: PASS")''',
        7: '''text_tokens = 512
for height, width in [(224, 224), (640, 960), (2160, 3840)]:
    visual = visual_token_budget(height, width).llm_visual_tokens
    print((height, width), "visual=", visual, "sequence^2=", self_attention_elements(text_tokens, visual))''',
    },
)


DAY4 = build(
    4,
    "Qwen3-VL",
    ["qwen3_vl_2025"],
    [
        ("Learning question", "Qwen3-VL에서 image가 어떤 경로로 language generation에 들어가며 Instruct/Thinking, dense/MoE, 2B/8B 선택은 무엇을 바꾸는가?"),
        ("Background theory", "`dynamic-resolution image -> ViT -> merger/projector -> visual embeddings interleaved with text -> Qwen LLM -> autoregressive tokens`. Qwen3-VL의 주요 갱신은 Interleaved-MRoPE, multi-level ViT feature를 쓰는 DeepStack, video text-timestamp alignment다."),
        ("Paper connection", "Qwen3-VL은 dense와 MoE, Instruct와 Thinking 변형을 제공한다. 이 과정의 기본은 작은 실습에 적합한 `Qwen/Qwen3-VL-2B-Instruct`이며 전체 benchmark 재현을 목표로 하지 않는다."),
        ("Input/output and shapes", "입력은 role/content 형태의 multimodal chat message다. processor가 image token과 text token을 만들고 model.generate가 prompt 뒤 output token을 생성한다. 응답은 free-form text이므로 task별 schema 검증이 필요하다."),
        ("Minimal implementation", "공식 `apply_chat_template`와 동일한 message contract를 구성하되 모델을 로드하지 않는다."),
        ("Visualization sanity check", "message에서 image와 text의 순서, file URI, generation prompt 존재를 확인한다."),
        ("Experiment", "Instruct prompt를 짧은 caption, OCR JSON, grounding JSON으로 바꿔 output contract 차이를 설계한다."),
        ("Metrics", "task별 accuracy 외에 input/output token, TTFT, tokens/sec, peak memory, JSON validity, hallucination rate를 기록한다."),
        ("Interpretation", "VLM은 detector처럼 고정 tensor를 반환하지 않고 language token을 생성한다. 유연성은 높지만 schema failure와 hallucination이 새 오류 축이 된다."),
        ("Failure cases", "chat template 누락, prompt token trim 오류, unsupported transformers version, cache miss, multi-image 순서 혼동, 너무 긴 visual context를 확인한다."),
        ("Real-service implications", "model process는 시작 시 한 번 load하고 요청마다 message/visual token budget을 검증한다. free-form 응답은 JSON parser와 retry/fallback 경계를 거친다."),
        ("Review questions", "1. Interleaved-MRoPE가 표현하는 축은?\n2. DeepStack은 어느 정보를 LLM에 더 제공하는가?\n3. Instruct와 Thinking의 latency/출력 차이는?\n4. free-form generation이 detector보다 만드는 새 위험은?\n5. 2B와 8B를 비교할 때 고정해야 할 조건은?"),
    ],
    {
        5: '''from pathlib import Path

image_path = Path("/nas/datahub/min/<user-selected>/images/example.jpg")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image_path.as_uri()},
        {"type": "text", "text": "Describe only visible facts in one sentence."},
    ],
}]
print(messages)''',
        6: '''assert messages[0]["content"][0]["type"] == "image"
assert messages[0]["content"][1]["type"] == "text"
print("message contract: PASS")''',
        7: '''task_prompts = {
    "caption": "Describe only visible facts in one sentence.",
    "ocr": 'Return only JSON: [{"text":"...","bbox_2d":[x1,y1,x2,y2]}]',
    "grounding": 'Return only JSON: [{"label":"helmet","bbox_2d":[x1,y1,x2,y2]}]',
}
print(task_prompts)''',
    },
)


DAY5 = build(
    5,
    "Grounding inference",
    ["qwen3_vl_2025", "refcoco_2014"],
    [
        ("Learning question", "Qwen3-VL의 language response를 검증 가능한 box로 바꾸려면 prompt, JSON schema, coordinate convention, overlay를 어떻게 고정해야 하는가?"),
        ("Background theory", "Qwen3-VL은 relative coordinate 기반 point/box grounding을 지원한다. 이 과정은 prompt에 `bbox_2d`, `xyxy`, `0..1000`, strict JSON을 명시한다. model output은 신뢰하지 않고 parser에서 type, 길이, 범위, x1<=x2를 검증한다."),
        ("Paper connection", "visual grounding은 표현과 referent region을 연결한다. Qwen3-VL의 autoregressive box 생성은 Grounding DINO의 decoder tensor와 출력 방식이 다르므로 같은 후처리를 적용하지 않는다."),
        ("Input/output and shapes", "입력은 image + object/referring expression이다. 출력 schema는 `[{label: str, bbox_2d: [x1,y1,x2,y2]}]`. relative box를 원본 image W/H 기준 pixel `xyxy`로 변환한다."),
        ("Minimal implementation", "strict JSON을 parse하고 1920x1080 pixel 좌표로 변환한다."),
        ("Visualization sanity check", "pixel box가 화면 범위에 있고 대상 위에 놓이는지 overlay한다. JSON validity만으로 spatial correctness를 보장하지 않는다."),
        ("Experiment", "동일 target에 category, attribute, relation prompt를 사용하고 top-1 IoU와 JSON validity를 비교한다."),
        ("Metrics", "JSON validity, box count, top-1 accuracy@IoU 0.5, mAP, prompt consistency, empty response를 분리한다."),
        ("Interpretation", "생성된 좌표는 언어 token이다. 좌표 문법을 잘 생성하는 능력과 대상 위치를 정확히 이해하는 능력은 구분해 평가해야 한다."),
        ("Failure cases", "Markdown fence, 잘린 JSON, x/y swap, resized/original coordinate 혼동, extreme aspect ratio, duplicate box, 존재하지 않는 객체 hallucination을 기록한다."),
        ("Real-service implications", "schema invalid는 reject/retry하고, low-confidence 또는 안전 관련 결과는 human review로 보낸다. 좌표 convention과 prompt version을 prediction에 함께 저장한다."),
        ("Review questions", "1. strict JSON parser가 필요한 이유는?\n2. relative box를 pixel로 바꾸는 식은?\n3. JSON validity와 IoU가 다른 이유는?\n4. original/resized image 좌표 혼동을 어떻게 탐지하는가?\n5. Grounding DINO와 Qwen3-VL grounding의 출력 방식 차이는?"),
    ],
    {
        5: '''import json
from vlm_foundation.grounding import grounding_prompt, parse_grounding_json

prompt = grounding_prompt(["safety helmet", "forklift"])
raw_response = json.dumps([
    {"label": "safety helmet", "bbox_2d": [100, 120, 360, 420]}
])
boxes = parse_grounding_json(raw_response)
pixel_boxes = [box.to_pixel(1920, 1080) for box in boxes]
print(prompt)
print(pixel_boxes)''',
        6: '''for box in pixel_boxes:
    x1, y1, x2, y2 = box.bbox_2d
    assert 0 <= x1 <= x2 <= 1920 and 0 <= y1 <= y2 <= 1080
print("pixel-boundary check: PASS")''',
        7: '''prompts = [
    "helmet",
    "yellow safety helmet",
    "the helmet worn by the worker on the left",
]
print("Evaluate each prompt on one fixed image/GT:", prompts)''',
    },
)


DAY6 = build(
    6,
    "LoRA",
    ["lora_2021"],
    [
        ("Learning question", "거대한 VLM 전체 weight를 업데이트하지 않고 어떤 low-rank parameter만 학습하며, rank와 target module은 비용과 표현력을 어떻게 바꾸는가?"),
        ("Background theory", "frozen linear weight `W`에 `Delta W = B A * alpha/r`를 더한다. `A[r,in]`, `B[out,r]`만 학습하므로 dense `out*in` 대신 `r*(in+out)` parameter가 필요하다. PEFT 기본 초기화는 B를 0으로 두어 시작 시 base model과 동일하게 만든다."),
        ("Paper connection", "LoRA는 low intrinsic-rank update 가정으로 parameter-efficient fine-tuning을 한다. Qwen3-VL 공식 fine-tuning 코드는 attention의 `q_proj,k_proj,v_proj,o_proj`를 LoRA target으로 제시한다."),
        ("Input/output and shapes", "입력 activation `[B,L,in]`; A를 지나 `[B,L,r]`; B를 지나 `[B,L,out]`; base linear output에 scaled delta를 더한다. adapter file만 저장해도 base checkpoint ID가 반드시 함께 필요하다."),
        ("Minimal implementation", "4096x4096 dense layer와 rank 8 LoRA의 parameter 수를 비교하고 작은 행렬에서 delta를 계산한다."),
        ("Visualization sanity check", "B=0 initialization이면 delta가 정확히 0인지 확인한다. 그렇지 않으면 학습 전부터 base output이 달라진다."),
        ("Experiment", "rank 4/8/16/64의 parameter fraction을 비교한다. rank만 늘리지 말고 validation metric과 overfitting을 함께 본다."),
        ("Metrics", "trainable%, GPU peak memory, step time, train/validation loss, task metric, base capability regression을 기록한다."),
        ("Interpretation", "LoRA는 memory를 줄이지만 activation과 frozen base weight는 여전히 필요하다. trainable parameter가 작다고 전체 학습 memory가 같은 비율로 줄지는 않는다."),
        ("Failure cases", "target module 이름 mismatch, vision/projector가 완전히 frozen돼 domain gap을 못 줄이는 경우, 너무 높은 LR, adapter/base revision 불일치를 확인한다."),
        ("Real-service implications", "여러 작은 adapter를 base model 하나에 교체 적용할 수 있다. adapter registry에 base model revision, prompt format, target modules, rank를 기록한다."),
        ("Review questions", "1. LoRA parameter 수 공식은?\n2. B=0 초기화의 의미는?\n3. alpha/r scaling은 무엇을 조절하는가?\n4. target module 선택이 중요한 이유는?\n5. LoRA가 줄이지 못하는 memory는 무엇인가?"),
    ],
    {
        5: '''from vlm_foundation.lora import lora_delta, lora_fraction, lora_parameter_count

print("trainable params:", lora_parameter_count(4096, 4096, rank=8))
print("dense 대비 fraction:", lora_fraction(4096, 4096, rank=8))
x = np.array([[1.0, 2.0]])
a = np.array([[1.0, 0.0]])
b = np.array([[2.0], [3.0]])
print("delta:", lora_delta(x, a, b, alpha=1))''',
        6: '''zero_b = np.zeros((2, 1))
assert np.allclose(lora_delta(x, a, zero_b, alpha=1), 0)
print("identity-at-initialization check: PASS")''',
        7: '''for rank in [4, 8, 16, 64]:
    print(rank, lora_parameter_count(4096, 4096, rank), f"{100*lora_fraction(4096,4096,rank):.3f}%")''',
    },
)


DAY7 = build(
    7,
    "작은 fine-tuning",
    ["qwen3_vl_2025", "lora_2021"],
    [
        ("Learning question", "작은 VLM dataset으로 overfit·leakage·format memorization을 피하면서 LoRA SFT 한 사이클을 어떻게 설계하고 중단할 것인가?"),
        ("Background theory", "SFT는 image+user prompt를 조건으로 assistant token의 next-token loss를 최소화한다. prompt/image token label은 -100으로 mask하고 assistant answer만 loss에 포함한다. 비슷한 video frame이나 같은 document page는 group 단위로 split한다."),
        ("Paper connection", "Qwen3-VL 공식 training framework는 vision/projector/LLM tune flag, resolution, LoRA rank/alpha를 분리한다. 이 과정은 2B Instruct, batch 1, 20-step smoke run을 기본으로 하며 성능 claim이 아닌 pipeline 검증이다."),
        ("Input/output and shapes", "JSONL 한 줄은 `image,user,assistant,group_id`다. image는 data root 상대 경로다. tokenized full conversation `[1,L]`, labels `[1,L]`, loss는 assistant 구간만 계산한다. output은 LoRA adapter와 processor config다."),
        ("Minimal implementation", "synthetic metadata를 group split해 같은 source가 train/validation 양쪽에 들어가지 않는지 확인한다."),
        ("Visualization sanity check", "학습 전 실제 sample의 image, prompt, answer, token 길이, label mask를 한 건 확인한다. 잘못된 answer format을 대량 학습하기 전에 멈춘다."),
        ("Experiment", "1) base zero-shot, 2) 20-step overfit smoke, 3) 작은 train run을 구분한다. 각 단계에서 validation과 고정 qualitative set을 비교한다."),
        ("Metrics", "train/validation loss, exact JSON validity, task IoU/accuracy, held-out group 성능, base prompt 회귀, peak memory, tokens/sec를 기록한다."),
        ("Interpretation", "train loss 하락만으로 visual grounding을 학습했다고 결론내리지 않는다. image를 바꾸어도 같은 답을 내면 language-format memorization일 수 있다."),
        ("Failure cases", "group leakage, answer-only shortcut, malformed JSON, image path mismatch, assistant mask off-by-one, OOM, catastrophic forgetting, validation prompt tuning을 확인한다."),
        ("Real-service implications", "adapter는 base revision과 함께 배포하고 A/B shadow evaluation을 거친다. training data license, PII, failure examples, rollback artifact를 보존한다."),
        ("Review questions", "1. 왜 prompt token label을 -100으로 mask하는가?\n2. frame random split이 leakage를 만드는 이유는?\n3. 20-step smoke run의 목적은?\n4. format memorization과 visual learning을 어떻게 구분하는가?\n5. 실제 실행 전에 지정할 data root와 manifest는?"),
    ],
    {
        5: '''from vlm_foundation.sft import SftExample, group_split

examples = [
    SftExample(f"images/{group}_{i}.jpg", "Locate the helmet.", "[]", group)
    for group in ["camera_a_clip_1", "camera_a_clip_2", "camera_b_clip_1"]
    for i in range(3)
]
train, validation = group_split(examples, validation_fraction=0.34, seed=42)
print("train groups:", sorted({x.group_id for x in train}))
print("validation groups:", sorted({x.group_id for x in validation}))''',
        6: '''assert {x.group_id for x in train}.isdisjoint({x.group_id for x in validation})
print("group leakage check: PASS")''',
        7: '''print("""Dry-run command:
PYTHONPATH=src python scripts/train_lora.py \\
  --data-root /nas/datahub/min/<user-selected> \\
  --manifest manifests/tiny_sft.jsonl

GPU training starts only when --execute is added.
Missing model files are never downloaded unless --allow-download is also added.
""")''',
    },
)


NOTEBOOKS = {
    "day01_vit_review.ipynb": DAY1,
    "day02_multimodal_projector.ipynb": DAY2,
    "day03_visual_tokens.ipynb": DAY3,
    "day04_qwen3_vl.ipynb": DAY4,
    "day05_grounding_inference.ipynb": DAY5,
    "day06_lora.ipynb": DAY6,
    "day07_small_finetuning.ipynb": DAY7,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in NOTEBOOKS.items():
        path = OUT / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
