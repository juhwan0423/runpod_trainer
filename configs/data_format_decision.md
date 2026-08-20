# IndustSim-Quad 파인튜닝 데이터 형식 결정서

상태: **v2 결정 완료 — 기존 14-label 호환, ICRA 확장 annotation은 별도 계층**

기준일: 2026-08-20
근거 문서:

- IndustSim-Quad paper specification, 특히 §2–4와 Appendix A–E
- 공개 저장소에서 제외된 내부 review/research notes
- upstream [`SAIE/docs/DATA_SCHEMA.md`](https://github.com/SeanJeonghwanLee/SAIE/blob/main/docs/DATA_SCHEMA.md)
- 이 repository의 실행 코드와 config contract

이 문서는 저장 형식, 모델 입력, loss target, split, 검증 규칙을 하나의 계약으로
고정한다. 학습 코드가 편하다는 이유로 이 계약을 바꾸지 않는다. 계약을 바꿀 때는
`schema_version`, `action_space_version`, `prompt_version`을 함께 올린다.

## 0. 최종 결정 요약

1. 기본 학습 단위는 **JSONL 한 줄 = 1 Hz decision frame 하나**다.
2. 기본 입력은 그 decision frame에 정렬된 **현재 `spot_cam` RGB 한 장**과 episode
   mission `q`다. CCTV, pose, hazard 거리, task window, risk flag는 입력하지 않는다.
3. 기본 supervision은 기존 평가기와 문자 단위로 같은 **canonical 14-label 문자열
   하나**다. snake_case로 새 이름을 만들지 않는다.
4. user prompt와 image token은 `-100`으로 mask하고 assistant의 정답 label 및 종료
   token에만 next-token cross-entropy를 적용한다.
5. 논문 primary recipe는 class weight, auxiliary safety loss, oversampling이 없는 표준
   SFT다. 불균형 보정은 명시적인 ablation으로만 둔다.
6. 원 논문의 11개 held-out warehouse는 학습·validation·모델 선택에 절대 사용하지
   않는다. validation은 59개 train-pool warehouse 안에서 warehouse 단위로 고정한다.
7. `risk_adjacent`, task-valid window, safe-action mask, gaze/localization, distractor 및
   hazard taxonomy는 반드시 보강하되 **분석/추가 supervision 계층**으로 둔다. 기본
   14-label target을 깨지 않는다.
8. raw annotation은 수정하지 않는다. 변환 과정의 모든 frame snapping은
   `repair_log.jsonl`에 원래/선택 경로, frame gap, 해결 여부를 전부 기록하고 SHA-256를
   manifest에 넣는다. 제외·중복·결손은 manifest에 별도로 기록한다. publication
   build에서는 repair 0을 요구하고 임의 복구보다 재생성을 우선한다.

## 1. 기존 benchmark와 호환되는 source-of-truth

논문 Appendix A의 원본 warehouse 구조를 canonical source로 삼는다.

```text
warehouse_s<WAREHOUSE_SEED>/
├── warehouse.usd
├── navigable_map.npy
├── floor_metadata.json
├── master/bench_seed<EPISODE_SEED>.json
├── frames/<EPISODE_ID>/
│   ├── spot_cam/rgb_XXXX.png       # 모델 입력 가능
│   ├── cctv_se/rgb_XXXX.png        # audit only
│   ├── cctv_nw/rgb_XXXX.png        # audit only
│   └── sim_events.json
└── annotations/<EPISODE_ID>.json
```

원 논문이 보고한 전체 corpus는 70 warehouses, 1,311 episodes, 80,280 annotation
entries이며 59/11 warehouses로 분리된다. 파인튜닝 산출물은 이 숫자와 split을
manifest에서 재확인해야 한다. 별도의 개발용 local-smoke corpus 규모가 이 숫자와
다르더라도 오류는 아니지만, 이를 논문용 데이터로 오인해서는 안 된다.

### 1.1 Canonical 14-label action vocabulary

다음 문자열이 `action_space_version: saie_14_v1`의 전부다.

| Category | Exact label | Nominal `(delta_x_m, delta_yaw_rad)` |
|---|---|---:|
| navigate | `[navigate] move forward 25 cm` | `(0.25, 0)` |
| navigate | `[navigate] move forward 50 cm` | `(0.50, 0)` |
| navigate | `[navigate] move forward 75 cm` | `(0.75, 0)` |
| navigate | `[navigate] turn left 15 degree` | `(0, +pi/12)` |
| navigate | `[navigate] turn left 30 degree` | `(0, +pi/6)` |
| navigate | `[navigate] turn left 45 degree` | `(0, +pi/4)` |
| navigate | `[navigate] turn right 15 degree` | `(0, -pi/12)` |
| navigate | `[navigate] turn right 30 degree` | `(0, -pi/6)` |
| navigate | `[navigate] turn right 45 degree` | `(0, -pi/4)` |
| navigate | `[navigate] stop` | `(0, 0)` |
| gauge | `[gauge] stop` | `(0, 0)` |
| count | `[count] stop` | `(0, 0)` |
| state | `[state] stop` | `(0, 0)` |
| alert | `[alert] stop` | `(0, 0)` |

`[error] stop`은 wrapper failure sentinel이며 14개 target에 포함하지 않는다. 기존
초안의 `move_forward_fast`, `strafe_left`, `back_up`, `stop_read_gauge` 등은 실제
SAIE/PDF action space에 없으므로 사용하지 않는다.

## 2. 세 계층 데이터 계약

### 2.1 Layer A — immutable raw source

Layer A는 위 warehouse 구조 그대로다. 학습 준비 도구는 이 파일들을 읽기만 한다.
다음 join key를 사용한다.

- warehouse: 디렉터리명 `warehouse_s<seed>`
- episode: `master/<episode>.json`, `annotations/<episode>.json`,
  `frames/<episode>/sim_events.json`
- decision frame: annotation `video_id`의 마지막 `_tNNNN`
- image: `frames/<episode>/spot_cam/rgb_NNNN.png`
- pose/motion/event: 동일하거나 가장 가까운 `render_frame`의 `sim_events` record

파일 배열 순서나 PNG 개수로 timestamp를 추정하지 않는다. `render_frame`, 실제 PNG
파일명, simulator FPS를 사용한다.

### 2.2 Layer B0 — 현재 실행 가능한 normalized SFT JSONL

현재 `data/build_sft_dataset.py`와 `train.py`가 공유하는 실행 계약은
`saie_sft_v2`다. split은 row 안의 문자열이 아니라 `train.jsonl`, `val.jsonl`,
`test.jsonl` 파일과 `dataset_manifest.json`이 결정한다. 다음은 실제 builder가 내는
형태다. 아직 계산할 근거가 없는 값은 `false`나 0으로 꾸미지 않고 `null`로 둔다.

```json
{
  "schema_version": "saie_sft_v2",
  "sample_id": "warehouse_s10491/bench_seed09001_t0390",
  "frame_id": 390,
  "source_frame_id": 390,
  "episode_id": "warehouse_s10491/bench_seed09001",
  "source_video_id": "bench_seed09001_t0390",
  "source_episode_id": "bench_seed09001",
  "warehouse_id": "warehouse_s10491",
  "images": [
    "warehouse_s10491/frames/bench_seed09001/spot_cam/rgb_0390.png"
  ],
  "image": "warehouse_s10491/frames/bench_seed09001/spot_cam/rgb_0390.png",
  "instruction": "Start by walking ... return to your starting position.",
  "action_label": "[gauge] stop",
  "action_group": "gauge",
  "action_category": "gauge",
  "hazard_type": "fallen_worker",
  "hazard_present": true,
  "is_distractor": null,
  "dist_to_hazard_m": null,
  "risk_adjacent": null,
  "in_task_window": null,
  "attention_target": null,
  "target_bbox": null,
  "target_world_position": null,
  "target_bearing_deg": null,
  "view_valid": null,
  "stop_timing": null,
  "admissible_action_labels": null,
  "safe_action_mask": null,
  "candidate_action_safety": null,
  "motion": {},
  "source_annotation": "warehouse_s10491/annotations/bench_seed09001.json",
  "split_tags": ["hazard:fallen_worker"]
}
```

`images[-1] == image`이고 모든 경로는 `--image-root` 기준 상대경로다. repair가 허용된
개발 build에서는 `source_frame_id`가 원 annotation anchor, `frame_id`가 실제 선택 PNG다.
strict publication build에서는 둘이 같아야 한다.

### 2.3 Layer B+ — ICRA 확장 목표 annotation

아래는 simulator/exporter가 장기적으로 제공해야 할 **확장 목표 예시**다. 현재 builder의
모든 nested object를 그대로 생성한다는 뜻이 아니다. 실제 학습 경로로 넘겨야 하는
필드는 우선 Layer B0의 flat pass-through 필드(`hazard_present`, `in_task_window`,
`stop_timing`, `admissible_action_labels`, `safe_action_mask`, gaze 필드)에 기록한다.
`dataset_version`, seed, 정밀 pose·geometry·provenance는 별도 scientific manifest에도
보존한다.

```json
{
  "schema_version": "saie_scientific_annotation_v1",
  "dataset_version": "industsim_quad_icra_v1",
  "action_space_version": "saie_14_v1",
  "sample_id": "warehouse_s10491/bench_seed09001_t0390",
  "warehouse_id": "warehouse_s10491",
  "warehouse_seed": 10491,
  "episode_id": "warehouse_s10491/bench_seed09001",
  "source_episode_id": "bench_seed09001",
  "episode_seed": 9001,
  "split": "train",
  "frame_id": 390,
  "timestamp_s": 13.0,
  "image": "warehouse_s10491/frames/bench_seed09001/spot_cam/rgb_0390.png",
  "source_frames": [
    "warehouse_s10491/frames/bench_seed09001/spot_cam/rgb_0165.png",
    "warehouse_s10491/frames/bench_seed09001/spot_cam/rgb_0390.png"
  ],
  "instruction": "Start by walking ... return to your starting position.",
  "action_label": "[gauge] stop",
  "action_category": "gauge",
  "nominal_motion": {
    "delta_x_m": 0.0,
    "delta_yaw_rad": 0.0,
    "source": "action_lookup"
  },
  "measured_motion": {
    "vx_m_per_s": 0.0,
    "vy_m_per_s": 0.0,
    "omega_rad_per_s": 0.0,
    "delta_x_m": 0.0,
    "delta_yaw_rad": 0.0,
    "duration_s": 1.0,
    "source": "sim_events.motion_summary"
  },
  "robot_pose": {
    "x_m": 1.2,
    "y_m": -3.4,
    "yaw_rad": 1.57,
    "source": "sim_events.robot_pose"
  },
  "task_context": {
    "event_id": "bench_seed09001:gauge:0",
    "event_type": "gauge",
    "target_id": "station_2",
    "is_valid_window": true,
    "target_distance_m": 1.21,
    "target_bearing_deg": 3.7,
    "line_of_sight": true,
    "stop_timing": "valid",
    "validity_rule_version": "saie_task_window_v1"
  },
  "safety_context": {
    "hazard_present": true,
    "hazard_type": "fallen_worker",
    "hazard_instance_id": "bench_seed09001:hazard:0",
    "hazard_phase": "lying",
    "hazard_visible": false,
    "distance_to_hazard_m": 8.4,
    "risk_adjacent": false,
    "required_response": "none",
    "safe_action_mask": [
      true, true, true, true, true, true, true,
      true, true, true, true, true, true, true
    ],
    "safety_rule_version": "saie_swept_v1"
  },
  "attention_target": null,
  "annotation_quality": {
    "episode_valid": true,
    "image_exists": true,
    "frame_alignment": "exact",
    "frame_gap": 0,
    "causal_history": true,
    "required_task_coverage": true
  },
  "provenance": {
    "annotation_file": "warehouse_s10491/annotations/bench_seed09001.json",
    "master_file": "warehouse_s10491/master/bench_seed09001.json",
    "sim_events_file": "warehouse_s10491/frames/bench_seed09001/sim_events.json",
    "converter_version": "saie_sft_builder_v2"
  }
}
```

### 2.4 Layer C — model-rendered conversation

Layer C는 저장하지 않고 processor/chat template로 결정적으로 렌더링한다.

```text
user:
  <current spot_cam image>
  <episode mission>
  Choose exactly one label from the canonical 14-label list: ...
assistant:
  [gauge] stop
```

프롬프트에는 14개 exact string을 매번 같은 순서로 넣는다. `risk_adjacent`, GT task
window, pose, hazard type, CCTV는 프롬프트에 넣지 않는다. 이 원칙은 ECCV reviewer의
“wrapper가 ground-truth event 정보를 쓰는가?”라는 우려를 차단한다.

## 3. 필드별 규칙

### 3.1 필수 학습 필드

| Field | Type | Rule |
|---|---|---|
| `schema_version` | string | 현재 `saie_sft_v2` |
| `sample_id` | string | corpus 전체에서 unique |
| `warehouse_id` | string | split leakage 검사의 단위 |
| `episode_id` | string | `<warehouse>/<source episode>` 형태의 전역 유일 group ID |
| `source_episode_id` | string | 원 annotation/master의 episode ID |
| split file | enum | 현재 실행 파일은 `train`, `val`, `test`; OOD 의미는 별도 manifest/tag로 버전 관리 |
| `source_frame_id` | int | 원 annotation의 decision anchor |
| `frame_id` | int | 실제 선택된 render-frame index; publication에서는 source와 동일 |
| `images` | list[string] | 존재하며 causal한 history; 마지막 원소가 current image |
| `image` | string | `dataset_root` 기준 상대경로, `spot_cam`만 허용 |
| `instruction` | string | raw annotation `q`, 비어 있으면 실패 |
| `action_label` | enum | canonical 14개 중 하나 |
| `action_group` | enum | label bracket의 `navigate/gauge/count/state/alert` |
| `action_category` | enum | `navigate/gauge/count/state/alert` |

`image`는 기본 단일-frame 입력이다. 현재 실행 schema의 `images`는 실제 존재하는 causal
history이며 기본 학습기는 마지막 current frame 한 장만 사용한다. 원 source의 결함 있는
history를 그대로 복사하지 않는다.

### 3.2 Motion

`nominal_motion`과 `measured_motion`을 섞지 않는다.

- `nominal_motion`: canonical label lookup에 따른 평가용 one-step consequence
- `measured_motion`: simulator/odometry가 기록한 실제 1초 motion summary
- angle 저장 단위: radians
- 표와 사람용 로그에서만 degrees를 병기
- source와 duration을 반드시 보존

단일 scripted reference만 정답인 문제를 보완하려면 다음을 추가한다.

- `admissible_action_labels`: 안전하고 task-compatible한 canonical label의 비어 있지 않은
  복수 정답 집합
- `candidate_action_cost`: 14개 label별 planner cost 또는 reference 대비 tolerance
- `safe_action_mask`: §1.1 canonical 순서의 길이 14 boolean list. `true`는 해당 action이
  versioned swept-path rule에서 안전함, `false`는 위반함을 뜻한다.
- `candidate_action_safety`: canonical label 14개를 exact key로 갖는 map. 각 값은
  bare boolean(`true=safe`)이거나 `safe`/`is_safe`/`violates` boolean 하나와
  clearance/rule provenance를 가진 object다. mask와 map을 함께 저장하면 서로
  모순되어서는 안 된다.

단, 이 필드가 없는 현재 데이터에서 값을 추측하지 않는다.

### 3.3 Task-valid window

최종 boolean만 저장하지 않고 판정 원시값과 threshold version을 저장한다.

| Task | Distance | Bearing | Additional condition |
|---|---:|---:|---|
| gauge | 0.8–1.8 m | ±15° | line of sight |
| count | 3.0–5.5 m | ±20° | line of sight |
| alert | 2.0–5.0 m | ±25° | line of sight, fallen state |
| state | pad center ≤0.6 m | n/a | on-pad spatial condition |

권장 `task_context` 필드는 `event_id`, `target_id`, `is_valid_window`, window
start/end, distance, bearing, LOS, `stop_timing`, `time_to_valid_window_s`,
`validity_rule_version`이다. `stop_timing`은 `premature`, `valid`, `late`,
`false_positive`, `not_applicable` 중 하나다.

### 3.4 Safety

논문의 risk-adjacent 정의를 그대로 사용한다. 현재 pose에서 0.25/0.50/0.75 m의
forward candidate 중 하나라도 swept path violation을 만들면 그 frame이
`risk_adjacent=true`다. endpoint만 검사해서는 안 된다.

저장할 값:

- robot pose와 hazard/obstacle identifier
- clearance threshold와 minimum swept clearance
- candidate label별 violation 및 violation source
- `reference_action_violation`
- `risk_adjacent`
- severity와 required response
- evaluator/rule version

현행 기준은 fallen worker 2.35 m, wall은 Spot radius 0.35 m + 0.5 m, box는
robot radius + box half extent + 0.5 m다. task asset은 inspection을 위해 접근해야
하므로 preventive obstacle과 물리 collision을 구별한다. turn은 ground-plane
translation이 없어 risk-adjacent denominator를 정의하는 action에서 제외한다.

### 3.5 Output/wrapper audit

prediction 파일에는 아래를 분리해 저장한다.

- `raw_model_output`
- `native_action`
- `normalized_action_label`
- `parse_status`, `parse_error`
- `prompt_version`, `wrapper_version`
- `normalization_rule`
- `used_ground_truth_context` — strict protocol에서는 반드시 `false`
- decoding config와 checkpoint identifier

strict protocol에서는 `[navigate] stop`을 valid task window라는 이유만으로
`[gauge] stop` 등으로 승격하지 않는다. 기존 zero-shot 결과를 재현하는 legacy
normalization은 별도 metric 이름으로만 남긴다.

## 4. Frame alignment 및 repair 정책

### 4.1 원칙

1. annotation anchor `_tNNNN`과 같은 PNG가 있으면 exact match를 쓴다.
2. 없으면 실제 저장된 `spot_cam/rgb_*.png` 중 nearest frame을 찾고 gap을 기록한다.
3. 동일 거리에서는 미래가 아니라 과거 frame을 선택해 causal 입력을 보장한다.
4. 허용 gap을 넘으면 row를 제외하거나 strict mode에서 실패한다.
5. history는 실제 존재하는 saved frames로 다시 만들고 모두 current anchor 이하인지
   검사한다.
6. CCTV 경로가 하나라도 모델 입력에 들어가면 즉시 실패한다.

nearest snapping은 개발용 복구 수단이다. publication corpus는 원본 exporter를 고쳐
다시 생성하여 `frame_alignment=exact` 비율을 최대화해야 한다.

### 4.2 현재 local smoke에서 확인한 사실

2026-08-20 read-only audit 기준 개발 snapshot은 다음 상태였다. Raw snapshot은 이
standalone repository에 포함되지 않는다.

- 10 warehouses, 200 annotation episodes, 11,794 decision rows
- hazard type은 200/200 모두 `FALL_HUMAN`
- category: navigate 11,060; count 200; gauge 200; state 195; alert 139
- 17 episodes, 997 rows의 `sim_events.invalid=true`; reason은 전부 `robot_fallen`
- source image reference 163,468개와 current anchor 10,931개가 실제 PNG에 없음
- raw history에서 current anchor보다 미래인 image reference 4,130개
- episode 내부 decision frame 역행 4,369건
- warehouse-qualified decision ID 기준 duplicate 253건, 그중 상충 label 22건
- task-stop label인데 auxiliary measured motion이 0이 아닌 row 675개
- task event coverage는 gauge 200/200, count 200/200, state 195/200,
  alert 139/200
- `is_distractor`, `in_task_window`, source `dist_to_hazard_m`, source
  `risk_adjacent`는 명시된 row가 0개
- 예: anchor `t0059`/`rgb_0059.png`는 없고 nearest saved frame은 `rgb_0060.png`

이 숫자는 개발 snapshot이지 논문 통계가 아니다. 특히 invalid episode와 task coverage
결손 때문에 현재 local smoke를 그대로 H100 논문 학습에 사용해서는 안 된다.

비파괴 개발 변환은 invalid episode, exact duplicate, 상충 ID를 제외하고 실제 존재하는
current image로 재정렬하여 10,711 rows를 만들 수 있다. 이것은 dataloader와 checkpoint
경로를 시험하기 위한 **development artifact**일 뿐이다. publication source는 exporter와
simulator를 고쳐 다시 생성해야 하며, 이 10,711-row 파생본을 논문 corpus로 승격하지
않는다.

## 5. Split 계약

### 5.1 기본 split

- `test_id`: 원 논문의 11 held-out warehouses. 모든 학습/early stopping에서 봉인.
- `train`/`val`: 원 59 train-pool warehouses를 warehouse 단위로 고정 분리.
- split은 seed만 저장하지 말고 warehouse ID 목록을 `split_manifest.json`에 저장.
- 한 warehouse가 둘 이상의 split에 나타나면 hard failure.
- 모델 선택은 `val`; 최종 test는 recipe 확정 뒤 한 번 수행.

### 5.2 Hazard generalization split

새 hazard 일반화 가설은 scene holdout과 hazard holdout을 분리해야 검증된다.

- `test_id`: unseen warehouse, seen hazard type
- `test_ood_scene`: unseen layout/template, seen hazard type
- `test_ood_hazard`: unseen warehouse, train에서 한 번도 본 적 없는 hazard type
- `test_negative`: no-hazard 및 hard distractor

현재 corpus는 fallen worker 한 종류뿐이므로 `holdout_hazard`라는 이름만 붙여서 실험을
만들 수 없다. 최소 두 개 이상의 train-seen hazard와 하나 이상의 completely-held-out
hazard가 존재하고 manifest가 이를 확인할 때만 해당 실험을 유효하게 간주한다.

현재 builder가 실제로 쓰는 물리 파일명은 `train.jsonl`, `val.jsonl`, `test.jsonl`뿐이다.
위 `test_id`/`test_ood_*`/`test_negative`는 **실험 partition의 의미 이름**이며 자동으로
네 파일이 생긴다는 뜻이 아니다. 각 partition은 별도 versioned split manifest와
`--ablation` build로 만들고, train/val/test source hash와 held-out taxonomy를 결과에
같이 남긴다.

## 6. 현재 SAIE annotation에서 반드시 보강할 것

### P0 — H100 학습 전에 필수

1. invalid episode 제거 또는 재생성
2. annotation anchor/history를 실제 saved PNG와 causal하게 정렬
3. episode마다 gauge/count/state/alert 네 event coverage 확인
4. warehouse-level train/val/test manifest 고정
5. 14-label/category histogram과 majority floors 생성
6. missing `sim_events`, `motion_summary`, pose join을 검증
7. raw source checksum과 converter report 저장

### P1 — ICRA safety/generalization 주장 전에 필수

1. hazard-free negative episodes
2. 정상 보행자, 앉은 사람, 마네킹, 자재 더미 등 hard distractor
3. 복수 hazard taxonomy: 예를 들어 fallen worker, spill, blocked aisle, smoke/fire
   cue, fallen object. 시각·물리 의미가 명확하고 simulator에서 재현 가능해야 함.
4. hazard lifecycle: absent/standing/walking/falling/lying/recovering
5. false alarm과 premature/late stop annotation
6. warehouse와 hazard type을 독립적으로 hold out한 split
7. 14개 candidate 각각의 safe-action mask 또는 admissible set

### P1 — “이상 지점을 제대로 쳐다보기”를 주장하려면 필수

현재 `[alert] stop`만으로는 정지했는지만 배우며 anomaly를 향해 재지향했는지 평가할 수
없다. 기존 label을 유지하면서 아래 auxiliary layer를 추가한다.

- image-space bbox 또는 segmentation mask
- target world position
- target bearing/elevation
- visible/occluded/truncated 상태
- desired base yaw/view pose
- gaze error와 image-center offset
- response phase: `detect -> stop -> orient -> inspect -> alert -> resume`
- `attention_action` 또는 별도 gaze target

권장 계층은 다음과 같다.

```text
locomotion_action: canonical 14-label       # 기존 결과와 호환
event_decision: normal | task | anomaly
attention_target: image/world localization
response_phase: detect | stop | orient | inspect | alert | resume
inspection_answer: gauge/count/state/report content
```

### P2 — 실제 inspection 능력 확장

현재 task label은 “어디서 멈출지”만 지도하며 gauge 값, cement pack 개수, charging-pad
상태 자체를 답하게 하지 않는다. inspection 결과를 주장하려면 master의 `ground_truth`와
시각 입력을 연결한 `inspection_answer`를 별도 target으로 저장하고, action SFT와
two-stage 또는 multi-task로 평가한다. answer를 user prompt에 넣어 누설해서는 안 된다.

## 7. 학습 token과 loss

기본 RunPod recipe의 정확한 범위는 `full_text_backbone`이다. Gemma 4 Unified의 완전한
visual/audio input stack(`model.vision_tower`, `model.embed_vision`,
`model.audio_tower`, `model.embed_audio`)을 동결하고 전체 language backbone/decoder를
BF16으로 최적화한다. 이는 초기 설계의 “LLM backbone full FT, modality encoder와
입력 projector 고정” 실험이다. vision encoder/projector까지 포함한 literal image+text
active FT는 `gemma4_all_active.yaml` + `full_all_active.yaml`의 별도 ablation이며,
사용하지 않는 audio tower/projector만 동결한다. 두 결과를 모두 단순히 `full FT`라고
합쳐 보고하지 않는다.

개념적으로 다음 mask를 사용한다.

```text
input_ids: [system/user/image/prompt ...][assistant][canonical label][eos]
labels:    [-100 -100 -100 ...        ][-100     ][canonical label][eos]
```

주의사항:

- label 한 개는 tokenizer에서 여러 token이므로 문자열을 한 class처럼 가정하지 않는다.
- truncation 뒤 target token이 하나도 남지 않으면 해당 sample을 hard failure로 처리한다.
- padding token label은 `-100`.
- Gemma 4는 `enable_thinking=false`를 명시하여 label 앞 reasoning 생성을 막는다.
- primary loss는 unweighted next-token CE.
- class-weighted CE, balanced sampler, focal loss는 별도 ablation.
- default는 single-frame. multi-frame은 temporal ablation이며 같은 결과로 섞어 쓰지 않는다.

## 8. 평가 계약

### 8.1 기존 네 지표

- category accuracy
- task completion
- navigation deviation
- safety violation rate

논문 정의의 Safety violation primary는 judge의 risk-adjacent v2 count를 써서 corpus-level
`sum(n_viol) / sum(n_risk)`로 집계한다. episode별 rate 평균이나 모든 frame 분모의 legacy
`safety_violation_rate`로 바꾸지 않는다. Navigation deviation은 실제 이동거리가 아니라
predicted/reference one-step SE(2) 차이다.

### 8.2 리뷰 대응을 위해 함께 보고할 것

- always-stop, always-navigate/majority, uniform-random, oracle
- label accuracy, balanced accuracy, macro-F1
- 14-label 및 5-category confusion matrix
- per-label/category precision, recall, F1
- task별 completion과 stop precision/recall
- premature/false stop rate
- navigation progress/route completion
- NavDev mean뿐 아니라 median/quantile/distribution
- warehouse bootstrap confidence interval 또는 최소 3 seeds
- ID와 OOD-hazard 결과 분리

free generation exact match와 14-label constrained likelihood scoring을 모두 저장한다.
likelihood는 label token log-probability sum과 length-normalized mean을 둘 다 보고하여 짧은
label/긴 label 편향을 드러낸다.

### 8.3 학습 신호를 입증하는 최소 실험

1. 동일 checkpoint의 pre-FT zero-shot 재현
2. standard `full_text_backbone` post-FT
3. 같은 prompt/parser/decoding으로 pre/post 비교
4. 10/25/50/100% train-fraction learning curve
5. label-shuffle 또는 image/query mismatch negative control
6. held-out warehouse ID test
7. held-out hazard-type OOD test
8. 동일 checkpoint의 open-loop와 closed-loop subset 비교

open-loop 결과만 있으면 “deployment safety”가 아니라 diagnostic proxy라고 표현한다.

## 9. Build acceptance gate

H100로 보내는 dataset build는 아래를 모두 만족해야 한다.

- [ ] schema validation 0 error
- [ ] canonical label 외 target 0개
- [ ] missing current image 0개
- [ ] non-causal/future input frame 0개
- [ ] CCTV model input 0개
- [ ] duplicate `sample_id` 0개
- [ ] train/val/test warehouse intersection 0개
- [ ] invalid episode 0개
- [ ] train/val 각 label count > 0
- [ ] publication split의 각 episode에 required four-event coverage
- [ ] test warehouse는 training loader에서 read되지 않음
- [ ] manifest counts가 실제 JSONL counts와 일치
- [ ] dataset checksum과 split manifest 저장
- [ ] `repair_log.jsonl` 전 행과 그 SHA-256, repair count, 제외 사유 공개
- [ ] 기존 PDF 재현 build는 70 warehouses, 59 train-pool/11 test, 1,311 valid episodes,
      80,280 source-valid rows와 정확히 일치

OOD-hazard 실험을 표기하려면 추가로:

- [ ] train hazard types와 OOD hazard types가 disjoint
- [ ] hazard-free/hard-negative coverage 존재
- [ ] split별 hazard histogram 생성

### 9.1 두 종류의 gate를 구별한다

- **benchmark-compatible mechanical gate**: 기존 14-label 데이터에 대해 invalid,
  missing/non-causal image, duplicate ID, warehouse leakage가 0인지 검사한다. 기존
  benchmark의 학습 신호를 확인하는 H100 language-backbone SFT는 이 gate 통과 후 가능하다.
- **ICRA minimum-annotation gate**: 위 조건에 더해 hazard taxonomy, hazard-free/hard-negative,
  gaze/localization, false-stop, admissible-safe-action coverage를 검사한다. novel-hazard
  generalization 또는 “이상 지점을 제대로 쳐다본다”는 주장은 이 gate 전에는 하지 않는다.
  boolean/enum 타입, canonical admissible set, 14-boolean mask 또는 완전한 candidate map,
  finite bbox/world-bearing 구조도 검사한다. 이 gate가 true여도 값의 실제 장면상 기하학적
  정확성, orient controller 학습 성공, held-out 성능을 증명한 것은 아니므로 샘플
  수동검수와 실제 metric이 추가로 필요하다.

따라서 legacy alias인 `paper_ready=true`는 `benchmark_ready=true`와 같은 기계적 의미일
뿐 ICRA scientific claim을 의미하지 않는다. `annotation_audit.py --publication-strict`,
builder의 `icra_scientific_ready`, hazard/gaze coverage manifest, 실제 평가 결과를 함께
보관해야 한다.

## 10. 명시적으로 하지 않는 것

- CCTV를 더 좋은 입력이라는 이유로 학습에 넣지 않는다.
- GT task window/risk flag를 prompt에 넣지 않는다.
- missing boolean을 임의로 `false` 처리하지 않는다.
- test warehouse로 early stopping하거나 prompt를 고르지 않는다.
- generic stop을 GT window 정보로 task stop으로 승격한 결과를 strict 결과라고 부르지 않는다.
- 현재 fallen-worker-only 데이터로 novel-hazard generalization을 주장하지 않는다.
- 14-label을 gaze action으로 몰래 확장하지 않는다. gaze는 versioned auxiliary schema다.
- frame repair가 많이 발생한 개발 snapshot을 publication-ready라고 부르지 않는다.

이 계약의 핵심은 단순하다. 기존 14-label/SE(2) benchmark와의 비교 가능성을 보존하면서,
ICRA 방향에 필요한 safety, negative, localization, gaze, multi-answer 정보를 독립된 감사
가능 계층으로 추가한다. 그러면 “표준 SFT에서도 데이터에 학습 신호가 있다”는 결과와
“새 hazard를 제대로 발견하고 바라보는 능력은 저절로 생기지 않는다”는 결과를 같은
데이터 lineage 안에서 분리해 검증할 수 있다.
