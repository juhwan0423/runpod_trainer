# SAIE RunPod Trainer

SAIE/IndustSim-Quad의 14-label industrial patrol policy를 Gemma 4 기반 VLM으로
supervised fine-tuning하고, 예측·진단 평가까지 수행하기 위한 독립 RunPod 학습
harness입니다.

이 저장소의 현재 범위는 다음과 같습니다.

- `google/gemma-4-12B-it` multimodal SFT
- BF16 + DeepSpeed ZeRO-3 optimizer offload
- text-backbone full training, vision+text all-active training, 개발용 LoRA
- warehouse 단위 split과 portable image path를 갖는 `saie_sft_v2` 변환
- strict annotation/data/preflight audit
- exact 14-label prediction과 분류 진단
- upstream SAIE judge용 episode prediction export/집계
- RunPod H100 setup, smoke, resume-smoke, full launch

> **중요:** 이것은 아직 범용 VLA trainer가 아닙니다. NaVILA, OmniVLA,
> InternVLA-M1은 서로 다른 upstream input/action interface가 필요하며 native adapter가
> 이 저장소에 구현되어 있지 않습니다. 현재 직접 학습 가능한 backbone은 Gemma 4입니다.

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
| 실제 H100 end-to-end 2-step 실행 | 아직 이 repository snapshot에서는 미검증 |
| paper-ready corpus | 포함하지 않음 |
| ICRA scientific annotation | schema/gate 지원, 실제 완성 corpus는 포함하지 않음 |
| NaVILA/OmniVLA/InternVLA-M1 training | 미구현 |

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
