# Environment Contract

## 확인된 실행 환경

- server alias: `daintlabH`
- container: `mhleenew`
- source: `/nas/home/mhlee/vlm-foundation-7days`
- allowed data boundary: `/nas/datahub/min`
- base Python: `/home/mhlee/miniforge3/bin/python` (3.11)
- base Jupyter: `/home/mhlee/miniforge3/bin/jupyter`
- GPU: 8 x NVIDIA A100-SXM4-80GB

현재 환경은 읽기 전용으로 감사하며 공유 environment에 package를 설치하거나 upgrade하지 않는다.

## Real-model 요구사항

Qwen 공식 문서는 Qwen3-VL에 `transformers>=4.57.0`을 요구한다. 공식 예시는
`AutoModelForImageTextToText`와 `AutoProcessor.apply_chat_template`을 사용한다. video/multi-image
utility를 쓸 때 공식 README는 `qwen-vl-utils==0.0.14`를 제시한다.

LoRA 실행에는 PEFT가 필요하다. Qwen3-VL official training path와 generic PEFT는 버전을 함께
검증해야 한다.

## 설치안

아래는 자동 실행하지 않는다. 사용자가 승인한 뒤 본인 전용 경로에 별도 환경을 만든다.

```bash
python -m venv /nas/home/mhlee/venvs/vlm-foundation-7days
source /nas/home/mhlee/venvs/vlm-foundation-7days/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[notebook,models,train,dev]'
python -m ipykernel install --user --name vlm-foundation-7days \
  --display-name 'Python (vlm-foundation-7days)'
```

## 다운로드와 학습 guard

- inference: cache-only가 기본, `--allow-download`가 있어야 missing model download 허용
- fine-tuning: dry-run이 기본, `--execute`가 있어야 model load/GPU training 시작
- output: 선택한 data root 아래 `artifacts/`에만 기록
- 기본 model: `Qwen/Qwen3-VL-2B-Instruct`

## 공식 참고자료

- Qwen3-VL repository: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL Transformers: https://huggingface.co/docs/transformers/model_doc/qwen3_vl
- Qwen3-VL 8B model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Qwen3-VL fine-tuning: https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
- PEFT LoRA: https://huggingface.co/docs/peft/main/conceptual_guides/lora

