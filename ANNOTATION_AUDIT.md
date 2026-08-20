# SAIE 어노테이션 감사·준비 도구

`annotation_audit.py`는 기존 SAIE 시뮬레이터 산출물을 **수정하지 않고** 다음 세 소스를 교차 검증합니다.

```text
warehouse_s*/master/<episode>.json
warehouse_s*/annotations/<episode>.json
warehouse_s*/frames/<episode>/sim_events.json
warehouse_s*/frames/<episode>/spot_cam/rgb_*.png
```

## 1. 감사만 실행

clone한 `runpod_trainer` 프로젝트 루트에서 실행합니다. Raw corpus 경로는 환경에 맞게
지정합니다.

```bash
python annotation_audit.py \
  --dataset-root /path/to/SAIE/output/local_smoke \
  --split-manifest configs/splits/local_smoke_dev.json \
  --report-json /tmp/saie_annotation_audit.json \
  --report-md /tmp/saie_annotation_audit.md
```

기본 논문 계약은 70 warehouses, 1,311 episodes, **action decision 1 Hz**입니다. PDF Appendix A/Table 4의 `spot_cam 2 Hz`는 이미지 저장 주기이고 annotation decision entry는 1 Hz이므로 둘을 혼동하면 안 됩니다. 작은 smoke subset만 확인할 때 규모·주기 경고를 끄려면 다음처럼 `0`을 명시합니다.

```bash
python annotation_audit.py \
  --dataset-root /path/to/SAIE/output/local_smoke \
  --expected-warehouses 0 \
  --expected-episodes 0 \
  --expected-decision-hz 0
```

종료 코드 정책은 다음과 같습니다.

- 기본: 도구 자체가 정상 실행되면 `0`; 발견 사항은 report에 남김
- `--strict`: 구조 오류, invalid episode, task-stop 누락, 충돌 라벨이 있으면 `2`
- `--publication-strict`: error뿐 아니라 image/history warning과 hazard·distractor·split 등 과학적 coverage gap도 있으면 `2`

`--publication-strict`와 `--prepare-jsonl`을 함께 썼는데 검사가 실패하면 임시 JSONL을 폐기합니다. 기존 목적 파일이 있더라도 교체하지 않습니다.

외부 split manifest를 정답으로 쓰면 master JSON을 수정할 필요가 없습니다. 감사기는
`--split-manifest`의 warehouse 배정 누락·여분·중복과 master에 이미 들어 있는 split의
충돌을 검사합니다. 이 옵션을 생략하면 master 내부 `split`만 검사합니다.

논문 corpus에서는 실제 59/11 ID를 채운 외부 manifest를 반드시 같이 넘깁니다.

```bash
python annotation_audit.py \
  --dataset-root /workspace/data/industsim_quad \
  --split-manifest configs/splits/paper_59_11.json \
  --expected-warehouses 70 \
  --expected-episodes 1311 \
  --publication-strict \
  --report-json /workspace/data/sft/publication_audit.json \
  --report-md /workspace/data/sft/publication_audit.md
```

단, 이 raw 감사의 count gap만으로 최종 규모를 판정하지 않습니다. 실제 학습 입력의
70/59/11, 1,311 episodes, 80,280 source-valid rows는 `data/build_sft_dataset.py`의
`benchmark_ready` gate로 한 번 더 확인합니다.

## 2. 원본 비파괴 감사·복구 intermediate JSONL 준비

```bash
python annotation_audit.py \
  --dataset-root /path/to/SAIE/output/local_smoke \
  --prepare-jsonl /tmp/saie_sft_all.jsonl \
  --num-frames 1 \
  --report-json /tmp/saie_sft_audit.json
```

이 모드는 다음 규칙을 적용합니다.

1. `sim_events.invalid=true` episode는 모두 제외합니다.
2. 같은 `video_id`에 서로 다른 라벨이 있으면 해당 충돌 행을 모두 제외합니다.
3. 같은 `video_id`·같은 라벨의 완전 중복은 첫 행만 남겨 암묵적인 재가중을 막습니다.
4. `video_id` anchor와 정확히 일치하는 PNG가 없으면 가장 가까운 실제 저장 PNG를 선택합니다. 같은 거리면 더 이른 프레임을 선택합니다.
5. `image_history`는 선택된 현재 프레임 이하의 실제 PNG만으로 다시 만들므로 미래 프레임 누출이 없습니다.
6. 원래 anchor, 선택 delta, source 파일과 row index를 `provenance`에 보존합니다.
7. 원본에 없는 `is_distractor`·`in_task_window`를 `false`로 만들지 않고 `null`로 둡니다.

기존 파일 덮어쓰기는 기본적으로 거절합니다. 정말 같은 report/JSONL 경로를 갱신할 때만 `--overwrite`를 추가합니다.

중요: 이 출력은 문제 행을 조사하고 복구 정책을 검증하기 위한 단일
`saie_sft_decision_v1` intermediate입니다. `train.py`에 직접 넣는 학습 split이 아닙니다.
실제 학습기는 `data/build_sft_dataset.py`가 만드는 `saie_sft_v2`의
`train.jsonl`, `val.jsonl`, `test.jsonl`, `dataset_manifest.json`을 요구합니다.

## 3. 준비 JSONL 행 계약

각 행은 다음 필드를 가집니다.

| 필드 | 의미 |
|---|---|
| `schema_version` | `saie_sft_decision_v1` |
| `frame_id` | warehouse·source episode·anchor·원본 row index를 결합한 전역 유일 ID |
| `frame_index` | 실제로 선택한 저장 PNG render frame |
| `source_frame_index` | 원본 `video_id`의 anchor |
| `episode_id` | `<warehouse_id>/<source_episode_id>` 전역 유일 episode ID |
| `source_episode_id` | 원본 master의 episode ID |
| `warehouse_id` | `warehouse_s*` 디렉터리 이름 |
| `image` | `dataset_root` 기준 실제 존재하는 현재 RGB 상대경로 |
| `image_history` | 시간 오름차순이며 `image`에서 끝나는 실제 RGB 상대경로 목록 |
| `instruction` | 원본 `q` |
| `action_label` | 원본 canonical `a` (14-label 중 하나) |
| `action_group` | 원본 bracket tool: `navigate/gauge/count/state/alert` |
| `motion` | 원본 보조 SE(2) motion object |
| `hazard_type` | master hazard를 정규화한 이름 |
| `is_distractor` | 명시된 값 또는 `null` |
| `dist_to_hazard_m` | 가장 가까운 sim `robot_pose`로부터 유도한 거리 또는 `null` |
| `risk_adjacent` | 사람 위험원과 0.75 m forward swept segment만 고려한 유도값 또는 `null` |
| `in_task_window` | 명시된 값 또는 `null` |
| `split`, `split_tags` | master에 명시된 경우만 사용; 없으면 `null`, `[]` |
| `provenance` | source 경로, row index, snap delta, pose delta, risk 범위 |
| `quality_flags` | 스냅 또는 미어노테이션 필드 목록 |

중요: `risk_adjacent` 유도값은 사람 위험원만 검사합니다. 벽·박스까지 포함한 논문 safety-v2 분모를 재현하려면 evaluator의 floor/nav-map geometry를 함께 사용해야 하므로 report에서 범위를 명시합니다.

## 4. 현재 local_smoke에서 확인된 차단 사항

2026-08-20 개발 snapshot을 전수 감사한 기록입니다. 해당 raw snapshot은 이 standalone
repository에 포함되지 않습니다.

- 10 warehouses, 200 episodes, 11,794 decision rows
- `robot_fallen` 등 simulator invalid episode 17개; 해당 episode의 997 rows는 준비 JSONL에서 제외
- 실제 PNG가 없는 source history reference 163,468개
- 현재 anchor 자체가 PNG grid에 없는 rows 10,931개
- 미래 프레임을 포함한 history rows 4,130개
- episode 내부 anchor 역행 4,369건, 중복 `video_id` 253건
- 같은 decision ID에 상충 라벨 22개 ID
- task label episode coverage: gauge 200, count 200, state 195, alert 139
- 원본 row에 `is_distractor`, `in_task_window`, `dist_to_hazard_m`, `risk_adjacent`가 명시된 경우 0개
- hazard는 `fallen_worker` 하나뿐이며 `spill`, `blocked_aisle`, distractor가 없음
- raw `episode_id`가 warehouse마다 반복되므로 warehouse-qualified ID와 warehouse-level split이 필수
- motion window 중앙값은 1.0초로 PDF의 1 Hz decision-entry 계약과 일치함 (`spot_cam` 저장은 별도로 2 Hz)

따라서 현재 산출물은 개발용 smoke/preflight에는 쓸 수 있지만, `--publication-strict`를 통과하는 논문용 학습 corpus는 아닙니다. 특히 누락 task-stop을 단순 음성 라벨로 간주하거나 `is_distractor=false`를 일괄 채우면 안 됩니다.

## 5. 테스트

```bash
python -m unittest discover -s tests -v
```

테스트는 frame snapping, causal history, 상충 중복 제외, strict atomic block, hazard 거리/risk 유도를 검증합니다.
