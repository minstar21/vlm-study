# Qwen3-VL Architecture Guide

## 1. ViT에서 visual token까지

Qwen3-VL-8B 공식 config 기준 vision encoder의 주요 값은 `patch_size=16`,
`spatial_merge_size=2`, `temporal_patch_size=2`다.

```text
image [H,W,3]
  -> aligned/resized image
  -> 16x16 patch embedding [Gh*Gw,Dv]
  -> ViT blocks
  -> 2x2 spatial merge [Gh*Gw/4,4*Dv]
  -> merger MLP [N_visual,D_llm]
```

정렬된 정적 이미지에서 LLM visual token 하나는 대략 `32x32=1024` pixels에 해당한다.
640x960 이미지는 `(640/16)*(960/16)/4 = 600` visual token이다. 실제 resize는 processor의
`min_pixels`, `max_pixels`, aspect-ratio 정책을 따른다.

## 2. Projector는 무엇을 잇는가

vision encoder와 language model은 hidden dimension과 representation 목적이 다르다. merger/projector는
인접 spatial token을 묶고 LLM hidden dimension으로 옮긴다. 이 과정은 단순 shape 변환이 아니라
두 modality의 feature alignment가 학습되는 지점이다.

## 3. Language model과 위치

visual embedding은 chat template의 image placeholder 위치에 text token과 함께 들어간다.
Qwen3-VL의 Interleaved-MRoPE는 time, height, width frequency를 interleave해 image/video 위치를
다룬다. text는 일반 1D sequence order를 유지한다.

DeepStack은 최종 ViT feature 하나만 쓰지 않고 여러 중간 level feature를 LLM layer에 연결해
fine-grained detail과 alignment를 강화한다. config의 `deepstack_visual_indexes`가 선택된 vision
level을 나타낸다.

## 4. Generation과 grounding

Qwen3-VL은 detector tensor가 아니라 language token을 생성한다. grounding도 다음 JSON 문자열을
생성하도록 instruction을 주는 방식이다.

```json
[{"label": "helmet", "bbox_2d": [100, 120, 360, 420]}]
```

이 과정은 relative `0..1000`, pixel `xyxy`, resized-image coordinate 등 convention을 prompt와
evaluator에서 명시적으로 고정한다. 이 저장소는 `0..1000` relative `xyxy`를 기본 교육 계약으로
사용하지만, 새로운 dataset을 쓸 때 card의 실제 convention을 우선한다.

## 5. LoRA와 SFT

```text
base linear: y = x W^T
LoRA:        y = x W^T + x A^T B^T * alpha/r

A [r,in] and B [out,r] are trainable
W [out,in] stays frozen
```

공식 Qwen3-VL fine-tuning 코드의 기본 attention target은 `q_proj`, `k_proj`, `v_proj`, `o_proj`다.
projector나 vision encoder가 frozen이면 language-side adaptation만 일어나므로 visual domain gap이
큰 작업에는 충분하지 않을 수 있다.

SFT label은 assistant answer token만 남기고 image/user/system/padding 구간을 `-100`으로 mask한다.
같은 video clip, document, product instance는 group 단위로 split해 leakage를 막는다.

