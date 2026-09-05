# VLM Foundation Models in 7 Days

ViT를 복습한 뒤 visual token이 LLM에 들어가는 경로를 직접 구현하고, Qwen3-VL grounding,
LoRA, 작은 supervised fine-tuning까지 연결하는 7일 과정이다.

## 안전 경계

이 프로젝트가 쓰는 source 경로는 하나뿐이다.

```text
/nas/home/mhlee/vlm-foundation-7days
```

데이터는 사용자가 지정한 `/nas/datahub/min/<folder>`만 읽는다. 경로 검증 코드는 data root 밖으로
나가는 `..`와 절대 image path를 거부한다. 다른 사용자 홈, 다른 container, 공유 Python 환경,
서버 설정은 변경하지 않는다. package 설치, checkpoint 다운로드, GPU training도 자동 실행하지 않는다.

## 7일 과정

| Day | 주제 | 핵심 구현 |
|---|---|---|
| 1 | ViT 복습 | patchify, attention shape |
| 2 | Multimodal projector | 2x2 spatial merge, MLP projection |
| 3 | Visual tokens | dynamic-resolution token budget |
| 4 | Qwen3-VL | chat template, Interleaved-MRoPE, DeepStack |
| 5 | Grounding inference | strict JSON, relative box -> pixel box |
| 6 | LoRA | low-rank delta, parameter accounting |
| 7 | 작은 fine-tuning | group split, assistant-only loss, guarded training |

노트북은 `notebooks/`에 Day 순서대로 있다. Day 1~7의 설명과 toy code는 model 없이 읽고
실행할 수 있다.

## 전체 아키텍처

```text
image/video
  -> dynamic resize
  -> ViT patch tokens
  -> spatial merger + multimodal projector
  -> visual embeddings ─┐
                        ├-> Qwen LLM -> autoregressive response tokens
text chat tokens ───────┘

Qwen3-VL additions:
  Interleaved-MRoPE: time/height/width position
  DeepStack: multiple ViT levels injected into the LLM
  text-timestamp alignment: video event time grounding
```

세부 shape는 `docs/ARCHITECTURE.md`, 데이터 계약은 `docs/DATA_FORMAT.md`에서 확인한다.

## Core 실행

```bash
cd /nas/home/mhlee/vlm-foundation-7days
export PYTHONPATH="$PWD/src"
python scripts/check_environment.py
python -m pytest -q
jupyter lab --no-browser
```

VS Code에서는 H 서버에 Remote SSH로 접속하고 `mhleenew` container에 attach한 뒤 위 source
폴더만 연다.

## Qwen3-VL inference

기본 모델은 작은 실습용 `Qwen/Qwen3-VL-2B-Instruct`다. cache에 모델이 없으면 명령이 실패하며,
명시적인 `--allow-download` 없이는 다운로드하지 않는다.

```bash
PYTHONPATH=src python scripts/run_qwen_inference.py \
  --data-root /nas/datahub/min/<사용자가-지정한-폴더> \
  --image images/example.jpg \
  --prompt "Describe only visible facts in one sentence."
```

grounding:

```bash
PYTHONPATH=src python scripts/run_qwen_inference.py \
  --data-root /nas/datahub/min/<사용자가-지정한-폴더> \
  --image images/example.jpg \
  --ground "safety helmet" forklift
```

## 작은 LoRA fine-tuning

먼저 dry-run으로 manifest와 group split, output 경로만 검증한다.

```bash
PYTHONPATH=src python scripts/train_lora.py \
  --data-root /nas/datahub/min/<사용자가-지정한-폴더> \
  --manifest manifests/tiny_sft.jsonl
```

GPU 학습은 `--execute`, cache에 없는 model 다운로드는 `--allow-download`를 각각 추가해야만
시작된다. 기본은 batch 1, 20 step, Qwen3-VL-2B, attention q/k/v/o projection의 rank-8 LoRA다.

## 현재 상태

- 7개 notebook과 core Python 구현: 완료
- 실제 NAS 데이터: 미지정, 미접근
- model checkpoint: 미다운로드
- GPU inference/fine-tuning: 미실행
- package/shared environment 변경: 없음

