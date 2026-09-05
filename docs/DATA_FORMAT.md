# Data Contract

## 읽기 경계

모든 image와 manifest는 사용자가 선택한 `/nas/datahub/min/<folder>` 아래에 있어야 한다.
프로젝트 코드는 해당 data root를 제외한 NAS 경로를 탐색하지 않는다.

```bash
export VLM_DATA_ROOT=/nas/datahub/min/<사용자가-지정한-폴더>
```

## Tiny SFT JSONL

한 줄에 한 example을 둔다.

```json
{"image":"images/camera_a_0001.jpg","user":"Locate the safety helmet. Return strict JSON.","assistant":"[{\"label\":\"safety helmet\",\"bbox_2d\":[100,120,360,420]}]","group_id":"camera_a_clip_001"}
```

- `image`: data root 기준 상대 경로. 절대 경로와 `..` 금지
- `user`: image를 보고 수행할 instruction
- `assistant`: 학습 target. task별 format을 일관되게 유지
- `group_id`: 같은 scene/video/document/product instance를 묶는 leakage 방지 단위

grounding box convention은 dataset 전체에서 하나로 고정한다. 이 예제는 relative `0..1000`
`xyxy`다. 원본 annotation이 pixel 좌표면 manifest 생성 시 변환 규칙과 원본 크기를 기록한다.

## Split

random row split을 사용하지 않는다. 같은 video의 인접 frame, 같은 문서의 page, 같은 물체의 burst
촬영은 `group_id`가 같아야 한다. train/validation/test의 group 집합은 서로 겹치지 않는다.

test는 LoRA rank, learning rate, epoch, prompt template 선택에 사용하지 않는다. 작은 dataset이면
validation 수치와 qualitative failure set을 함께 보고 uncertainty를 명시한다.

## Fine-tuning 전 점검

1. image file 존재와 decode 성공
2. license와 개인정보/민감정보 처리 근거
3. prompt/answer schema 일관성
4. box coordinate 범위와 overlay
5. duplicate와 group leakage
6. train/validation class 및 domain 분포
7. 빈 정답/negative example 포함 여부

