# Handoff

## 구현됨

- VLM Day 1~7 notebook
- ViT patchify, multimodal spatial merger/projector, visual-token budget
- strict grounding JSON parser와 coordinate converter
- LoRA parameter/delta math
- leakage-safe SFT JSONL validation/group split
- cache-only Qwen3-VL inference CLI
- dry-run 기본의 tiny batch-1 LoRA training CLI

## 실행하지 않음

- 다른 사용자 파일이나 기존 project 접근/수정
- `/nas/datahub/min` 데이터 읽기
- package 설치/공유 environment 변경
- checkpoint 다운로드
- Qwen3-VL GPU inference 또는 fine-tuning

## 실제 실행 전 필요한 사용자 값

1. `/nas/datahub/min` 아래 사용할 data root
2. inference image 상대 경로 또는 SFT manifest 상대 경로
3. cache에 있는 model ID 또는 다운로드 승인
4. 작은 학습을 실제 시작할지 여부

