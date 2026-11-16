# SAIE RunPod Trainer

IndustSim-Quad(SAIE) 산업 순찰 벤치마크의 14-label 행동 예측을 VLM/VLA 4종으로
supervised fine-tuning하고, 예측과 평가까지 수행하는 RunPod 학습 파이프라인입니다.

| 모델 | Full fine-tuning | LoRA | 비고 |
|---|:---:|:---:|---|
| Gemma 4 12B | ✓ | ✓ | DeepSpeed ZeRO-3, BF16 |
| NaVILA (LLaMA3 8B) | ✓ | ✓ | DeepSpeed ZeRO-2, H100 4장 |
| InternVLA-N1 | ✓ | ✓ | InternNav 기반 |
| InternVLA-M1 | ✓ | ✓ | |
| OmniVLA | – | – | adapter 코드는 있으나 의존성 충돌로 학습하지 못함 |

## 목적

IndustSim-Quad 벤치마크에서 기존 VLA들은 zero-shot으로 과제를 거의 수행하지 못했습니다.
이 학습은 그 원인이 데이터에 있는지 모델에 있는지 가리기 위한 것입니다.

- **벤치마크 데이터에 학습 신호가 있는가.** 모델마다 따로 튜닝하지 않고, 네 모델에 같은
  표준 레시피(Full fine-tuning, LoRA)를 적용해 성능이 오르는지 확인합니다.
- **사전학습의 종류가 결과를 바꾸는가.** 보행 내비게이션 VLA(NaVILA), 내비게이션 VLA
  (InternVLA-N1), 범용 VLA(InternVLA-M1), 로봇 사전학습이 없는 VLM(Gemma 4)을 비교합니다.
- **평가 지표가 실제 능력을 재는가.** 학습된 모델의 예측을 카테고리별로 나눠 기존 지표가
  무엇을 놓치는지 점검합니다.

구성 요소는 다음과 같습니다.

- 모델별 adapter와 freeze/LoRA 규칙: [`models/registry.py`](./models/registry.py), [`configs/models/`](./configs/models/)
- warehouse 단위 split의 SFT 데이터 생성, annotation audit, preflight
- RunPod H100/B200 환경 설치와 smoke, resume, full 실행 스크립트
- 에피소드 단위 분산 평가, 결과 registry(체크포인트·예측 해시 연결), paired bootstrap 비교

## 결과 요약

같은 11개 warehouse, 210개 평가 에피소드에서 유효한 실험 19개를 비교했습니다.
`task-stop`은 이동을 제외한 4개 과제(경보, 게이지, 개수, 상태)의 정확도로, 항상
"이동"만 답하는 모델은 0점을 받습니다.

| 모델 | 방법 | 학습 데이터 | Task completion | task-stop |
|---|---|---|---:|---:|
| InternVLA-M1 | Full FT | balanced | **0.748** | **0.634** |
| InternVLA-N1 | LoRA | balanced | 0.733 | 0.597 |
| Gemma 4 | LoRA | redesigned (seed 0) | 0.687 | 0.515 |
| NaVILA | Full FT | balanced | 0.647 | 0.488 |

- Full fine-tuning이 Gemma 4와 InternVLA-N1에서는 한 가지 답만 내는 상태로 붕괴했지만,
  NaVILA와 InternVLA-M1에서는 붕괴하지 않았습니다. LoRA와 Full FT의 우열은 모델마다 달랐습니다.
- 기존 지표 일부는 변별력이 없었습니다. `Category Acc.`는 `[navigate]` 태그만 맞으면
  정답으로 세어 0.83–0.89에 몰렸고, 같은 프레임의 완전일치 정확도는 0.09–0.34였습니다.
- 전체 19개 결과와 카테고리별 분해는 `scripts/build_table2.py`,
  `scripts/category_breakdown.py`로 다시 만들 수 있습니다.

## 공개 저장소에 포함되지 않는 것

다음은 의도적으로 포함하지 않습니다.

- SAIE raw images, annotations, maps, simulator output
- model weights와 checkpoints
- 실제 59/11 paper split manifest
- 논문 원고, reviewer 원문, 내부 연구 계획
- 개인 정보가 포함된 GPU 결제/청구 자료
- provenance-only legacy prototype

데이터와 model access 권한은 사용자가 별도로 확보해야 합니다. 저장소 공개는 데이터,
논문, model weight의 재배포 권한을 의미하지 않습니다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| standalone unit tests | 지원 |
| config/data/train dry-run | 지원 |
| RunPod H100 preflight | 지원 |
| single-H100 ZeRO-3 smoke/resume scripts | 지원 |
| 실제 H100 end-to-end 실행 | 검증됨 (2026-08) |
| paper-ready corpus | 포함하지 않음 |
| ICRA scientific annotation | schema/gate 지원, 실제 완성 corpus는 포함하지 않음 |
| NaVILA, InternVLA-N1, InternVLA-M1 학습 | 지원 |
| OmniVLA 학습 | 의존성 충돌로 미완 |

현재 개발용 데이터가 `benchmark_ready=false` 또는
`icra_scientific_ready=false`인 것은 launcher 오류가 아닙니다. full run은 의도적으로
준비되지 않은 corpus를 차단합니다.

## Repository 구조

```text
.
├── train.py                       # Gemma 4 multimodal SFT
├── predict_labels.py              # strict 14-label generation
├── evaluate_predictions.py        # classification/balance diagnostics
├── aggregate_saie_scores.py       # warehouse-qualified judge aggregation
├── annotation_audit.py            # non-destructive raw annotation audit
├── data/
│   ├── schema.py                  # canonical labels and prompt
│   ├── build_sft_dataset.py       # raw SAIE -> saie_sft_v2
│   └── dataset.py                 # image loader and target-only collator
├── models/registry.py             # Gemma 4 loading/freeze/LoRA contract
├── configs/
│   ├── base.yaml
│   ├── models/
│   ├── methods/
│   ├── deepspeed/
│   └── splits/
├── scripts/
│   ├── setup_runpod.sh
│   ├── preflight.py
│   ├── launch_runpod.sh
│   ├── evaluate_saie_judge.sh
│   └── make_runpod_bundle.sh
├── tests/
├── ANNOTATION_AUDIT.md
└── 가이드.md                       # 상세 한국어 운영 가이드
```

## 요구 환경

- Linux
- Python 3.12 권장
- CUDA-capable NVIDIA GPU
- full training: H100 80 GB 이상과 충분한 host RAM/disk
- gated model을 읽을 수 있는 Hugging Face token
- raw SAIE warehouse corpus

RunPod dependency는 [`requirements-runpod.txt`](./requirements-runpod.txt)에 exact pin되어
있습니다. 기본 single-H100 recipe는 CPU optimizer offload 때문에 host RAM과 disk를 많이
사용합니다. 실제 요구량과 사전 점검은 [`가이드.md`](./가이드.md)를 따르십시오.

## 데이터 구조

Builder는 다음 raw layout을 기대합니다.

```text
<SAIE_RAW_ROOT>/
└── warehouse_s<seed>/
    ├── annotations/<episode>.json
    ├── master/<episode>.json
    └── frames/<episode>/
        ├── sim_events.json
        └── spot_cam/rgb_XXXX.png
```

paper run에서는 warehouse 단위 split manifest가 필수입니다. 저장소의
`configs/splits/paper_59_11.example.json`은 형식 예시일 뿐 의도적으로 유효하지 않습니다.
실제 manifest를 별도 경로에 만들고 59/11 목록과 hash를 검수해야 합니다.

## 1. Clone 및 RunPod 환경 설치

```bash
git clone https://github.com/juhwan0423/runpod_trainer.git
cd runpod_trainer

bash scripts/setup_runpod.sh
source /workspace/venvs/saie-ft/bin/activate
```

`HF_TOKEN`은 shell 환경변수나 RunPod secret으로만 전달하십시오. `.env`, YAML, shell
script, Git history에 기록하지 마십시오.

```bash
export HF_TOKEN='...'
```

## 2. Annotation audit

```bash
export SAIE_RAW_ROOT=/workspace/raw

python annotation_audit.py \
  --dataset-root "$SAIE_RAW_ROOT" \
  --split-manifest /workspace/manifests/paper_59_11.json \
  --report-json /workspace/data/audit.json \
  --report-md /workspace/data/audit.md \
  --publication-strict
```

원본은 수정하지 않습니다. `--publication-strict` 실패를 repair로 숨기지 말고 exporter와
source data를 먼저 수정하십시오.

## 3. SFT 데이터 생성

```bash
export SAIE_SFT_ROOT=/workspace/data/sft

python data/build_sft_dataset.py \
  --saie-output "$SAIE_RAW_ROOT" \
  --image-root "$SAIE_RAW_ROOT" \
  --out "$SAIE_SFT_ROOT" \
  --split-manifest /workspace/manifests/paper_59_11.json \
  --max-history-frames 1 \
  --strict-frame-paths \
  --duplicate-policy error
```

출력은 `<out>/<ablation>/{train,val,test}.jsonl`, `dataset_manifest.json`,
`repair_log.jsonl`입니다. 기본 학습 입력은 current `spot_cam` image 한 장과 mission이며,
target은 exact canonical label 하나입니다.

개발 smoke corpus는 서로 다른 warehouse가 최소 3개여야 합니다. paper 규모 gate를
의도적으로 끄는 경우 그 결과는 benchmark-ready가 아닙니다.

## 4. Preflight

```bash
export SAIE_DATA_ROOT="$SAIE_SFT_ROOT"
export SAIE_IMAGE_ROOT="$SAIE_RAW_ROOT"
export SAIE_OUTPUT_ROOT=/workspace/saie-runs

python scripts/preflight.py \
  --data-root "$SAIE_DATA_ROOT" \
  --image-root "$SAIE_IMAGE_ROOT" \
  --output-root "$SAIE_OUTPUT_ROOT" \
  --ablation default \
  --require-gpu \
  --require-h100 \
  --check-model-access \
  --check-processor-contract
```

full paper run 전에는 `--require-paper-ready`를 추가합니다. 확장 scientific annotation을
사용하는 실험은 `--require-icra-ready`도 추가합니다.

## 5. H100 smoke와 full training

```bash
bash scripts/launch_runpod.sh smoke 0
bash scripts/launch_runpod.sh smoke-resume 0
bash scripts/launch_runpod.sh full 0
```

기본 조합은 다음과 같습니다.

- model: `configs/models/gemma4.yaml`
- method: `configs/methods/full_ft.yaml`
- 실제 optimization scope: `full_text_backbone`
- frozen: complete vision/audio towers와 input embedders/projectors

즉 `full_ft.yaml`은 문자 그대로 모든 parameter를 학습하는 설정이 아닙니다. vision+text
active parameter까지 학습하려면 smoke 검증 뒤 두 설정을 함께 바꿉니다.

```bash
export SAIE_MODEL_CONFIG=configs/models/gemma4_all_active.yaml
export SAIE_METHOD_CONFIG=configs/methods/full_all_active.yaml
bash scripts/launch_runpod.sh smoke-resume 0
```

LoRA는 개발 비교용입니다.

```bash
export SAIE_METHOD_CONFIG=configs/methods/lora_r16.yaml
bash scripts/launch_runpod.sh smoke 0
```

## 6. Prediction과 label 평가

```bash
python predict_labels.py \
  --config configs/models/gemma4.yaml \
  --checkpoint /workspace/saie-runs/<run>/final \
  --jsonl "$SAIE_SFT_ROOT/default/test.jsonl" \
  --image-root "$SAIE_IMAGE_ROOT" \
  --out /workspace/saie-runs/predictions/test.jsonl \
  --vlm-output-dir /workspace/saie-runs/vlm_output

python evaluate_predictions.py \
  --predictions /workspace/saie-runs/predictions/test.jsonl \
  --reference-jsonl "$SAIE_SFT_ROOT/default/test.jsonl" \
  --train-jsonl "$SAIE_SFT_ROOT/default/train.jsonl" \
  --out /workspace/saie-runs/predictions/metrics.json
```

SAIE의 four-metric judge 실행에는 upstream repository가 별도로 필요합니다.

```bash
git clone https://github.com/SeanJeonghwanLee/SAIE.git /workspace/SAIE-eval
```

warehouse별 judge 실행과 집계는 `scripts/evaluate_saie_judge.sh`와 상세 가이드를
참조하십시오.

## 7. 로컬 검증

```bash
python -m compileall -q .
python -m unittest discover -s tests -v

bash -n scripts/setup_runpod.sh
bash -n scripts/launch_runpod.sh
bash -n scripts/make_runpod_bundle.sh
bash -n scripts/evaluate_saie_judge.sh
```

모델 download, processor contract, CUDA/H100, DeepSpeed runtime 검사는 실제 RunPod
preflight에서 수행합니다.

## 안전한 code bundle 생성

```bash
bash scripts/make_runpod_bundle.sh
cd dist
sha256sum -c saie-finetuning-code_*.tar.gz.sha256
```

bundle script는 공개 실행 코드 allowlist만 archive합니다. raw data, model, checkpoint,
Git metadata, 내부 문서, screenshot은 포함하지 않습니다.

## 중요한 제한사항

- 이 코드는 open-loop 14-label SFT/evaluation harness이며 실제 closed-loop robot safety를
  보증하지 않습니다.
- `admissible_action_labels`, safe-action mask, gaze/localization metadata를 보존할 수 있어도
  현재 기본 collator는 이를 auxiliary loss로 학습하지 않습니다.
- 실제 inspection answer(gauge value, count, state)를 생성하는 head/evaluator는 아직 없습니다.
- full run은 실제 paper split/corpus와 strict annotation gate 없이는 실행하지 마십시오.
- H100 비용이 발생하기 전에 반드시 preflight와 `smoke-resume`을 통과하십시오.

## 문서

- [`가이드.md`](./가이드.md): 전체 한국어 운영·RunPod·평가 가이드
- [`ANNOTATION_AUDIT.md`](./ANNOTATION_AUDIT.md): annotation audit 계약
- [`configs/data_format_decision.md`](./configs/data_format_decision.md): data/prompt/split 결정
- [`configs/README.md`](./configs/README.md): 실행 config map

## License

현재 이 snapshot에는 공개 재사용 라이선스가 포함되어 있지 않습니다. GitHub에 소스가
보인다는 사실만으로 사용·수정·재배포 권한이 부여되지는 않습니다. 프로젝트 소유자가
별도 코드 라이선스와 데이터/논문/third-party notice를 확정해야 합니다.
