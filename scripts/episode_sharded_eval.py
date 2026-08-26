#!/usr/bin/env python3
"""Split and merge fail-closed, episode-disjoint SAIE inference shards.

The standard layout is::

    ROOT/
      split_manifest.json
      shards/shard0/input.jsonl
      ...

After each shard is evaluated, its directory must additionally contain::

    test_predictions.jsonl
    test_predictions.jsonl.manifest.json
    vlm_output/prediction_export_manifest.json
    vlm_output/<warehouse_id>/<source_episode_id>.json

``merge`` validates the complete source/sample/episode contract before it
atomically publishes a fresh global prediction JSONL and fresh manifests.
It never reuses a shard manifest as the global manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_SPLIT = "saie_episode_shard_split_v1"
SCHEMA_PREDICTION = "saie_prediction_manifest_v1"
SCHEMA_VLM = "saie_vlm_export_v1"
SHARD_COUNT = 4
PREDICTION_NAME = "test_predictions.jsonl"
PREDICTION_MANIFEST_NAME = f"{PREDICTION_NAME}.manifest.json"
VLM_DIR_NAME = "vlm_output"
VLM_MANIFEST_NAME = "prediction_export_manifest.json"


class ContractError(RuntimeError):
    """Raised when an input violates the fail-closed sharding contract."""


@dataclass(frozen=True)
class JsonlRow:
    value: dict[str, Any]
    raw: str
    source_index: int

    @property
    def sample_id(self) -> str:
        return self.value["sample_id"]

    @property
    def episode_key(self) -> tuple[str, str]:
        return (self.value["warehouse_id"], self.value["source_episode_id"])


@dataclass(frozen=True)
class SourceContract:
    path: Path
    sha256: str
    rows: tuple[JsonlRow, ...]
    episode_keys: frozenset[tuple[str, str]]
    warehouses: frozenset[str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_file_tree_sha256(
    files: Iterable[Path], *, relative_to: Path
) -> str:
    """Hash file names and bytes so a validated output tree is reproducible."""
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.relative_to(relative_to).as_posix()):
        relative = path.relative_to(relative_to).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def require_regular_file(path: Path, description: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"{description} is missing or is not a regular file: {path}")


def load_json(path: Path, description: str) -> dict[str, Any]:
    require_regular_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{description} must contain a JSON object: {path}")
    return value


def read_jsonl(path: Path, description: str) -> tuple[JsonlRow, ...]:
    require_regular_file(path, description)
    result: list[JsonlRow] = []
    seen_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ContractError(
                        f"invalid JSON in {description} {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ContractError(
                        f"{description} row must be an object: {path}:{line_number}"
                    )
                sample_id = value.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ContractError(
                        f"{description} has an empty/non-string sample_id: {path}:{line_number}"
                    )
                if sample_id in seen_ids:
                    raise ContractError(
                        f"duplicate sample_id in {description}: {sample_id!r} ({path}:{line_number})"
                    )
                seen_ids.add(sample_id)
                warehouse_id = value.get("warehouse_id")
                episode_id = value.get("source_episode_id")
                if not isinstance(warehouse_id, str) or not warehouse_id:
                    raise ContractError(
                        f"{description} has an empty/non-string warehouse_id: "
                        f"{path}:{line_number}"
                    )
                if not isinstance(episode_id, str) or not episode_id:
                    raise ContractError(
                        f"{description} has an empty/non-string source_episode_id: "
                        f"{path}:{line_number}"
                    )
                result.append(JsonlRow(value=value, raw=raw, source_index=len(result)))
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read {description} {path}: {exc}") from exc
    if not result:
        raise ContractError(f"{description} is empty: {path}")
    return tuple(result)


def validate_source(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_episodes: int,
    expected_warehouses: int,
) -> SourceContract:
    path = path.resolve()
    rows = read_jsonl(path, "source JSONL")
    digest = file_sha256(path)
    episode_keys = frozenset(row.episode_key for row in rows)
    warehouses = frozenset(key[0] for key in episode_keys)
    expected_digest = expected_sha256.lower()
    if digest != expected_digest:
        raise ContractError(
            f"source SHA-256 mismatch: expected {expected_digest}, observed {digest} ({path})"
        )
    observed = {
        "rows": len(rows),
        "episodes": len(episode_keys),
        "warehouses": len(warehouses),
    }
    expected = {
        "rows": expected_rows,
        "episodes": expected_episodes,
        "warehouses": expected_warehouses,
    }
    mismatches = [
        f"{name}: expected {expected[name]}, observed {observed[name]}"
        for name in expected
        if expected[name] != observed[name]
    ]
    if mismatches:
        raise ContractError("source cardinality mismatch: " + "; ".join(mismatches))
    return SourceContract(
        path=path,
        sha256=digest,
        rows=rows,
        episode_keys=episode_keys,
        warehouses=warehouses,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl_raw(path: Path, rows: Iterable[JsonlRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.raw)
            handle.write("\n")


def write_jsonl_values(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def temporary_directory_for(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )


def publish_directory(stage: Path, destination: Path) -> None:
    if path_lexists(destination):
        raise ContractError(f"destination already exists: {destination}")
    try:
        os.rename(stage, destination)
    except OSError as exc:
        raise ContractError(f"cannot atomically publish {destination}: {exc}") from exc


def episode_key_json(key: tuple[str, str], rows: int) -> dict[str, Any]:
    return {"warehouse_id": key[0], "source_episode_id": key[1], "rows": rows}


def deterministic_episode_assignment(
    rows: Iterable[JsonlRow],
) -> tuple[dict[tuple[str, str], int], list[int]]:
    grouped: dict[tuple[str, str], list[JsonlRow]] = defaultdict(list)
    for row in rows:
        grouped[row.episode_key].append(row)
    ordered_groups = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    loads = [0] * SHARD_COUNT
    assignment: dict[tuple[str, str], int] = {}
    for key, group_rows in ordered_groups:
        shard_index = min(range(SHARD_COUNT), key=lambda index: (loads[index], index))
        assignment[key] = shard_index
        loads[shard_index] += len(group_rows)
    return assignment, loads


def split_command(args: argparse.Namespace) -> None:
    contract = validate_source(
        args.source_jsonl,
        expected_sha256=args.expected_sha256,
        expected_rows=args.expected_rows,
        expected_episodes=args.expected_episodes,
        expected_warehouses=args.expected_warehouses,
    )
    destination = args.output_dir.resolve()
    if path_lexists(destination):
        raise ContractError(f"destination already exists: {destination}")

    # Largest-processing-time greedy bin packing. The full episode key is the
    # deterministic secondary key, and the lowest shard index breaks load ties.
    episode_assignment, loads = deterministic_episode_assignment(contract.rows)

    shard_rows: list[list[JsonlRow]] = [[] for _ in range(SHARD_COUNT)]
    for row in contract.rows:
        shard_rows[episode_assignment[row.episode_key]].append(row)

    stage = temporary_directory_for(destination)
    try:
        shard_summaries: list[dict[str, Any]] = []
        for shard_index, rows in enumerate(shard_rows):
            relative_path = Path("shards") / f"shard{shard_index}" / "input.jsonl"
            staged_path = stage / relative_path
            write_jsonl_raw(staged_path, rows)
            keys = sorted({row.episode_key for row in rows})
            group_counts = Counter(row.episode_key for row in rows)
            shard_summaries.append(
                {
                    "index": shard_index,
                    "input_jsonl": relative_path.as_posix(),
                    "input_jsonl_sha256": file_sha256(staged_path),
                    "rows": len(rows),
                    "episodes": len(keys),
                    "warehouses": len({key[0] for key in keys}),
                    "episode_groups": [
                        episode_key_json(key, group_counts[key]) for key in keys
                    ],
                }
            )

        manifest = {
            "schema_version": SCHEMA_SPLIT,
            "assignment_algorithm": "largest_episode_first_greedy_rows_v1",
            "group_key": ["warehouse_id", "source_episode_id"],
            "preserves_source_order_within_shard": True,
            "source_jsonl": str(contract.path),
            "source_jsonl_sha256": contract.sha256,
            "rows": len(contract.rows),
            "episodes": len(contract.episode_keys),
            "warehouses": len(contract.warehouses),
            "shard_count": SHARD_COUNT,
            "shards": shard_summaries,
        }
        write_json(stage / "split_manifest.json", manifest)
        publish_directory(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(destination),
                "source_sha256": contract.sha256,
                "rows": len(contract.rows),
                "episodes": len(contract.episode_keys),
                "shard_rows": loads,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def require_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        raise ContractError(f"{description}: expected {expected!r}, observed {actual!r}")


def require_common(
    manifests: Sequence[dict[str, Any]], field: str, description: str, *, nonempty: bool = False
) -> Any:
    if not manifests:
        raise ContractError(f"no manifests supplied for {description}")
    if field not in manifests[0]:
        raise ContractError(f"{description} is missing required field {field!r}")
    reference = manifests[0][field]
    if nonempty and (not isinstance(reference, str) or not reference):
        raise ContractError(f"{description} has empty/non-string field {field!r}")
    for index, manifest in enumerate(manifests[1:], start=1):
        if field not in manifest:
            raise ContractError(f"{description} {index} is missing required field {field!r}")
        if manifest[field] != reference:
            raise ContractError(
                f"{description} field {field!r} differs between shards: "
                f"{reference!r} != {manifest[field]!r}"
            )
    return reference


def validate_split_manifest(
    root: Path, contract: SourceContract
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = load_json(root / "split_manifest.json", "split manifest")
    require_equal(manifest.get("schema_version"), SCHEMA_SPLIT, "split manifest schema")
    require_equal(manifest.get("source_jsonl_sha256"), contract.sha256, "split source SHA-256")
    require_equal(manifest.get("rows"), len(contract.rows), "split source rows")
    require_equal(manifest.get("episodes"), len(contract.episode_keys), "split source episodes")
    require_equal(manifest.get("warehouses"), len(contract.warehouses), "split warehouses")
    require_equal(manifest.get("shard_count"), SHARD_COUNT, "split shard count")
    require_equal(
        manifest.get("assignment_algorithm"),
        "largest_episode_first_greedy_rows_v1",
        "split assignment algorithm",
    )
    require_equal(
        manifest.get("group_key"),
        ["warehouse_id", "source_episode_id"],
        "split group key",
    )
    require_equal(
        manifest.get("preserves_source_order_within_shard"),
        True,
        "split source-order guarantee",
    )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise ContractError(f"split manifest must describe exactly {SHARD_COUNT} shards")
    by_index: dict[int, dict[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, dict):
            raise ContractError("every split manifest shard entry must be an object")
        index = shard.get("index")
        if not isinstance(index, int) or index < 0 or index >= SHARD_COUNT:
            raise ContractError(f"invalid split shard index: {index!r}")
        if index in by_index:
            raise ContractError(f"duplicate split shard index: {index}")
        by_index[index] = shard
    if set(by_index) != set(range(SHARD_COUNT)):
        raise ContractError("split manifest shard indices are not exactly 0,1,2,3")
    return manifest, [by_index[index] for index in range(SHARD_COUNT)]


def validate_shard_inputs(
    root: Path,
    shard_entries: Sequence[dict[str, Any]],
    contract: SourceContract,
) -> list[tuple[JsonlRow, ...]]:
    source_by_id = {row.sample_id: row for row in contract.rows}
    observed_ids: set[str] = set()
    observed_episodes: set[tuple[str, str]] = set()
    all_rows: list[tuple[JsonlRow, ...]] = []
    expected_assignment, _ = deterministic_episode_assignment(contract.rows)
    for index, entry in enumerate(shard_entries):
        expected_relative = Path("shards") / f"shard{index}" / "input.jsonl"
        require_equal(entry.get("input_jsonl"), expected_relative.as_posix(), f"shard {index} path")
        path = root / expected_relative
        rows = read_jsonl(path, f"shard {index} input")
        require_equal(file_sha256(path), entry.get("input_jsonl_sha256"), f"shard {index} SHA-256")
        require_equal(len(rows), entry.get("rows"), f"shard {index} row count")
        keys = {row.episode_key for row in rows}
        expected_keys = {
            key for key, assigned_index in expected_assignment.items() if assigned_index == index
        }
        require_equal(keys, expected_keys, f"shard {index} deterministic episode assignment")
        require_equal(len(keys), entry.get("episodes"), f"shard {index} episode count")
        require_equal(
            len({key[0] for key in keys}), entry.get("warehouses"), f"shard {index} warehouse count"
        )
        group_counts = Counter(row.episode_key for row in rows)
        described_groups = entry.get("episode_groups")
        expected_groups = [episode_key_json(key, group_counts[key]) for key in sorted(keys)]
        require_equal(described_groups, expected_groups, f"shard {index} episode groups")
        overlap = observed_episodes.intersection(keys)
        if overlap:
            raise ContractError(
                f"episode appears in multiple shards: {sorted(overlap)[0]!r}"
            )
        observed_episodes.update(keys)
        for row in rows:
            if row.sample_id in observed_ids:
                raise ContractError(f"sample appears in multiple shards: {row.sample_id!r}")
            observed_ids.add(row.sample_id)
            source_row = source_by_id.get(row.sample_id)
            if source_row is None:
                raise ContractError(f"shard sample is absent from source: {row.sample_id!r}")
            if row.value != source_row.value:
                raise ContractError(f"shard input row differs from source: {row.sample_id!r}")
        # The source indices must be strictly increasing inside every shard.
        indices = [source_by_id[row.sample_id].source_index for row in rows]
        if indices != sorted(indices):
            raise ContractError(f"shard {index} does not preserve source row order")
        all_rows.append(rows)
    require_equal(observed_ids, set(source_by_id), "shard input sample union")
    require_equal(observed_episodes, set(contract.episode_keys), "shard input episode union")
    return all_rows


def validate_prediction_record(
    row: JsonlRow,
    manifest: dict[str, Any],
    source_row: JsonlRow,
    *,
    shard_index: int,
) -> None:
    for field in (
        "warehouse_id",
        "source_episode_id",
        "episode_id",
        "frame_id",
        "source_frame_id",
    ):
        require_equal(
            row.value.get(field),
            source_row.value.get(field),
            f"shard {shard_index} prediction {row.sample_id!r} source identity {field}",
        )
    require_equal(
        row.value.get("gold"),
        source_row.value.get("action_label"),
        f"shard {shard_index} prediction {row.sample_id!r} gold label",
    )
    parse_status = row.value.get("parse_status")
    if not isinstance(parse_status, str) or not parse_status:
        raise ContractError(
            f"shard {shard_index} prediction {row.sample_id!r} has invalid parse_status"
        )
    require_equal(
        row.value.get("used_ground_truth_context"),
        False,
        f"shard {shard_index} prediction {row.sample_id!r} ground-truth-context policy",
    )
    for field in (
        "checkpoint",
        "checkpoint_sha256",
        "wrapper_version",
        "normalization_rule",
        "prompt_version",
    ):
        require_equal(
            row.value.get(field),
            manifest.get(field),
            f"shard {shard_index} prediction {row.sample_id!r} field {field}",
        )
    require_equal(
        row.value.get("decoding_config"),
        manifest.get("decoding_config"),
        f"shard {shard_index} prediction {row.sample_id!r} decoding config",
    )


def load_episode_file(path: Path, description: str) -> list[dict[str, Any]]:
    require_regular_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ContractError(f"{description} must contain a JSON array of objects: {path}")
    return value


def scan_vlm_episode_files(
    directory: Path, shard_index: int
) -> dict[tuple[str, str], tuple[Path, list[dict[str, Any]]]]:
    if not directory.is_dir() or directory.is_symlink():
        raise ContractError(f"shard {shard_index} VLM output directory is missing: {directory}")
    result: dict[tuple[str, str], tuple[Path, list[dict[str, Any]]]] = {}
    for path in sorted(directory.rglob("*.json")):
        relative = path.relative_to(directory)
        if relative == Path(VLM_MANIFEST_NAME):
            continue
        if len(relative.parts) != 2:
            raise ContractError(
                f"unexpected VLM JSON path in shard {shard_index}; expected warehouse/episode.json: "
                f"{relative}"
            )
        warehouse_id = relative.parts[0]
        episode_filename = relative.parts[1]
        episode_id = episode_filename[:-5]
        if not warehouse_id or not episode_id:
            raise ContractError(f"invalid VLM episode path in shard {shard_index}: {relative}")
        key = (warehouse_id, episode_id)
        if key in result:
            raise ContractError(f"duplicate VLM episode file in shard {shard_index}: {key!r}")
        result[key] = (
            path,
            load_episode_file(path, f"shard {shard_index} VLM episode {key!r}"),
        )
    return result


def validate_manifest_schemas(
    prediction_manifests: Sequence[dict[str, Any]],
    vlm_manifests: Sequence[dict[str, Any]],
) -> None:
    for index, manifest in enumerate(prediction_manifests):
        require_equal(
            manifest.get("schema_version"), SCHEMA_PREDICTION, f"shard {index} prediction schema"
        )
    for index, manifest in enumerate(vlm_manifests):
        require_equal(manifest.get("schema_version"), SCHEMA_VLM, f"shard {index} VLM schema")

    # These fields define a single comparable inference run. Mixing any of
    # them would make the merged score scientifically invalid.
    for field in (
        "checkpoint",
        "checkpoint_sha256",
        "wrapper_version",
        "normalization_rule",
        "prompt_version",
    ):
        prediction_value = require_common(
            prediction_manifests, field, "prediction manifests", nonempty=True
        )
        vlm_value = require_common(vlm_manifests, field, "VLM manifests", nonempty=True)
        require_equal(vlm_value, prediction_value, f"prediction/VLM common field {field}")
    for field in ("allow_extracted_label", "image_root", "num_frames"):
        prediction_value = require_common(prediction_manifests, field, "prediction manifests")
        vlm_value = require_common(vlm_manifests, field, "VLM manifests")
        require_equal(vlm_value, prediction_value, f"prediction/VLM common field {field}")
    require_common(prediction_manifests, "decoding_config", "prediction manifests")
    require_common(prediction_manifests, "model_requested_revision", "prediction manifests")
    require_common(prediction_manifests, "model_resolved_commit", "prediction manifests")
    used_ground_truth = require_common(
        vlm_manifests, "used_ground_truth_context", "VLM manifests"
    )
    require_equal(used_ground_truth, False, "VLM ground-truth-context policy")


def source_frame_counter(rows: Iterable[JsonlRow], key: tuple[str, str]) -> Counter[int]:
    frames: Counter[int] = Counter()
    for row in rows:
        if row.episode_key != key:
            continue
        frame = row.value.get("source_frame_id")
        if not isinstance(frame, int):
            frame = row.value.get("source_frame_index")
        if not isinstance(frame, int):
            raise ContractError(
                f"source row has no integer source frame anchor: {row.sample_id!r}"
            )
        frames[frame] += 1
    return frames


def merge_command(args: argparse.Namespace) -> None:
    contract = validate_source(
        args.source_jsonl,
        expected_sha256=args.expected_sha256,
        expected_rows=args.expected_rows,
        expected_episodes=args.expected_episodes,
        expected_warehouses=args.expected_warehouses,
    )
    shards_root = args.shards_root.resolve()
    if not shards_root.is_dir() or shards_root.is_symlink():
        raise ContractError(f"shards root is missing or invalid: {shards_root}")
    destination = args.output_dir.resolve()
    if path_lexists(destination):
        raise ContractError(f"destination already exists: {destination}")

    _, shard_entries = validate_split_manifest(shards_root, contract)
    shard_inputs = validate_shard_inputs(shards_root, shard_entries, contract)
    source_by_id = {row.sample_id: row for row in contract.rows}

    prediction_manifests: list[dict[str, Any]] = []
    vlm_manifests: list[dict[str, Any]] = []
    predictions_by_id: dict[str, dict[str, Any]] = {}
    all_vlm_files: dict[tuple[str, str], tuple[Path, list[dict[str, Any]]]] = {}
    shard_provenance_entries: list[dict[str, Any]] = []

    for shard_index, input_rows in enumerate(shard_inputs):
        shard_dir = shards_root / "shards" / f"shard{shard_index}"
        input_path = shard_dir / "input.jsonl"
        prediction_path = shard_dir / PREDICTION_NAME
        prediction_rows = read_jsonl(prediction_path, f"shard {shard_index} predictions")
        prediction_manifest = load_json(
            shard_dir / PREDICTION_MANIFEST_NAME,
            f"shard {shard_index} prediction manifest",
        )
        vlm_dir = shard_dir / VLM_DIR_NAME
        vlm_manifest = load_json(
            vlm_dir / VLM_MANIFEST_NAME,
            f"shard {shard_index} VLM manifest",
        )
        prediction_manifests.append(prediction_manifest)
        vlm_manifests.append(vlm_manifest)

        input_ids = {row.sample_id for row in input_rows}
        prediction_ids = {row.sample_id for row in prediction_rows}
        require_equal(
            prediction_ids, input_ids, f"shard {shard_index} prediction/input sample union"
        )
        require_equal(
            prediction_manifest.get("input_jsonl_sha256"),
            file_sha256(input_path),
            f"shard {shard_index} prediction input SHA-256",
        )
        require_equal(
            prediction_manifest.get("rows"),
            len(prediction_rows),
            f"shard {shard_index} prediction manifest rows",
        )
        parse_counts = dict(
            sorted(Counter(row.value.get("parse_status") for row in prediction_rows).items())
        )
        require_equal(
            prediction_manifest.get("parse_status_counts"),
            parse_counts,
            f"shard {shard_index} parse-status counts",
        )

        episode_keys = {row.episode_key for row in input_rows}
        require_equal(
            vlm_manifest.get("source_jsonl_sha256"),
            file_sha256(input_path),
            f"shard {shard_index} VLM source SHA-256",
        )
        require_equal(
            vlm_manifest.get("rows"), len(input_rows), f"shard {shard_index} VLM rows"
        )
        require_equal(
            vlm_manifest.get("episodes"),
            len(episode_keys),
            f"shard {shard_index} VLM episodes",
        )

        for row in prediction_rows:
            validate_prediction_record(
                row,
                prediction_manifest,
                source_by_id[row.sample_id],
                shard_index=shard_index,
            )
            if row.sample_id in predictions_by_id:
                raise ContractError(f"prediction sample collision across shards: {row.sample_id!r}")
            predictions_by_id[row.sample_id] = row.value

        episode_files = scan_vlm_episode_files(vlm_dir, shard_index)
        require_equal(
            set(episode_files), episode_keys, f"shard {shard_index} VLM/input episode set"
        )
        episode_row_total = 0
        for key, (path, episode_rows) in episode_files.items():
            if key in all_vlm_files:
                raise ContractError(f"VLM episode collision across shards: {key!r}")
            expected_frames = source_frame_counter(input_rows, key)
            observed_frames: Counter[int] = Counter()
            observed_frame_order: list[int] = []
            for row_index, episode_row in enumerate(episode_rows):
                frame_id = episode_row.get("frame_id")
                if not isinstance(frame_id, int):
                    raise ContractError(
                        f"VLM episode {key!r} has non-integer frame_id at row {row_index}: {path}"
                    )
                require_equal(
                    episode_row.get("normalization_rule"),
                    vlm_manifest.get("normalization_rule"),
                    f"VLM episode {key!r} row {row_index} normalization rule",
                )
                require_equal(
                    episode_row.get("used_ground_truth_context"),
                    False,
                    f"VLM episode {key!r} row {row_index} ground-truth-context policy",
                )
                observed_frames[frame_id] += 1
                observed_frame_order.append(frame_id)
            if observed_frame_order != sorted(observed_frame_order):
                raise ContractError(f"VLM episode rows are not frame-sorted: {key!r} ({path})")
            require_equal(observed_frames, expected_frames, f"VLM episode frame anchors {key!r}")
            episode_row_total += len(episode_rows)
            all_vlm_files[key] = (path, episode_rows)
        require_equal(
            episode_row_total,
            len(input_rows),
            f"shard {shard_index} VLM episode-file row total",
        )
        shard_provenance_entries.append(
            {
                "index": shard_index,
                "input_jsonl": input_path.relative_to(shards_root).as_posix(),
                "input_jsonl_sha256": file_sha256(input_path),
                "prediction_jsonl": prediction_path.relative_to(shards_root).as_posix(),
                "prediction_jsonl_sha256": file_sha256(prediction_path),
                "prediction_manifest": (
                    shard_dir / PREDICTION_MANIFEST_NAME
                ).relative_to(shards_root).as_posix(),
                "prediction_manifest_sha256": file_sha256(
                    shard_dir / PREDICTION_MANIFEST_NAME
                ),
                "vlm_export_manifest": (
                    vlm_dir / VLM_MANIFEST_NAME
                ).relative_to(shards_root).as_posix(),
                "vlm_export_manifest_sha256": file_sha256(vlm_dir / VLM_MANIFEST_NAME),
                "vlm_episode_files": len(episode_files),
                "vlm_episode_tree_sha256": relative_file_tree_sha256(
                    (path for path, _ in episode_files.values()), relative_to=vlm_dir
                ),
                "rows": len(input_rows),
                "episodes": len(episode_keys),
            }
        )

    validate_manifest_schemas(prediction_manifests, vlm_manifests)
    require_equal(set(predictions_by_id), set(source_by_id), "global prediction/source sample union")
    require_equal(set(all_vlm_files), set(contract.episode_keys), "global VLM/source episode set")
    require_equal(
        sum(len(rows) for _, rows in all_vlm_files.values()),
        len(contract.rows),
        "global VLM/source row total",
    )

    ordered_predictions = [predictions_by_id[row.sample_id] for row in contract.rows]
    first_prediction_manifest = prediction_manifests[0]
    first_vlm_manifest = vlm_manifests[0]
    parse_counts = dict(
        sorted(Counter(row.get("parse_status") for row in ordered_predictions).items())
    )
    final_prediction_path = destination / PREDICTION_NAME
    final_vlm_dir = destination / VLM_DIR_NAME
    shard_provenance = {
        "schema_version": "saie_shard_merge_provenance_v1",
        "paths_relative_to": str(shards_root),
        "split_manifest": "split_manifest.json",
        "split_manifest_sha256": file_sha256(shards_root / "split_manifest.json"),
        "shards": shard_provenance_entries,
    }

    # Construct both global manifests from validated common values. Do not copy
    # a per-shard manifest: its paths, hashes, and cardinalities are shard-local.
    global_prediction_manifest = {
        "schema_version": SCHEMA_PREDICTION,
        "wrapper_version": first_prediction_manifest["wrapper_version"],
        "normalization_rule": first_prediction_manifest["normalization_rule"],
        "allow_extracted_label": first_prediction_manifest["allow_extracted_label"],
        "prompt_version": first_prediction_manifest["prompt_version"],
        "checkpoint": first_prediction_manifest["checkpoint"],
        "checkpoint_sha256": first_prediction_manifest["checkpoint_sha256"],
        "model_requested_revision": first_prediction_manifest["model_requested_revision"],
        "model_resolved_commit": first_prediction_manifest["model_resolved_commit"],
        "input_jsonl": str(contract.path),
        "input_jsonl_sha256": contract.sha256,
        "image_root": first_prediction_manifest["image_root"],
        "num_frames": first_prediction_manifest["num_frames"],
        "output_jsonl": str(final_prediction_path),
        "rows": len(ordered_predictions),
        "parse_status_counts": parse_counts,
        "decoding_config": first_prediction_manifest["decoding_config"],
        "shard_provenance": shard_provenance,
    }
    global_vlm_manifest = {
        "schema_version": SCHEMA_VLM,
        "wrapper_version": first_vlm_manifest["wrapper_version"],
        "normalization_rule": first_vlm_manifest["normalization_rule"],
        "allow_extracted_label": first_vlm_manifest["allow_extracted_label"],
        "used_ground_truth_context": first_vlm_manifest["used_ground_truth_context"],
        "prompt_version": first_vlm_manifest["prompt_version"],
        "checkpoint": first_vlm_manifest["checkpoint"],
        "checkpoint_sha256": first_vlm_manifest["checkpoint_sha256"],
        "source_jsonl": str(contract.path),
        "source_jsonl_sha256": contract.sha256,
        "image_root": first_vlm_manifest["image_root"],
        "num_frames": first_vlm_manifest["num_frames"],
        "prediction_jsonl": str(final_prediction_path),
        "episodes": len(all_vlm_files),
        "rows": len(ordered_predictions),
        "shard_provenance": shard_provenance,
    }

    stage = temporary_directory_for(destination)
    try:
        write_jsonl_values(stage / PREDICTION_NAME, ordered_predictions)
        write_json(stage / PREDICTION_MANIFEST_NAME, global_prediction_manifest)
        for key in sorted(all_vlm_files):
            source_path, _ = all_vlm_files[key]
            relative = Path(key[0]) / f"{key[1]}.json"
            target = stage / VLM_DIR_NAME / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
        write_json(stage / VLM_DIR_NAME / VLM_MANIFEST_NAME, global_vlm_manifest)
        publish_directory(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(destination),
                "prediction_jsonl": str(final_prediction_path),
                "rows": len(ordered_predictions),
                "episodes": len(all_vlm_files),
                "parse_status_counts": parse_counts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def add_source_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-jsonl", type=Path, required=True, help="canonical full test JSONL")
    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="required SHA-256 of the canonical source JSONL",
    )
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-warehouses", type=int, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser(
        "split", help="create four deterministic, episode-disjoint inference shards"
    )
    add_source_contract_arguments(split_parser)
    split_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory that will contain split_manifest.json and shards/",
    )
    split_parser.set_defaults(handler=split_command)

    merge_parser = subparsers.add_parser(
        "merge", help="validate and merge four completed inference shards"
    )
    add_source_contract_arguments(merge_parser)
    merge_parser.add_argument(
        "--shards-root",
        type=Path,
        required=True,
        help="split output directory containing split_manifest.json and shards/",
    )
    merge_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for the merged prediction JSONL and VLM outputs",
    )
    merge_parser.set_defaults(handler=merge_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("expected_rows", "expected_episodes", "expected_warehouses"):
        if getattr(args, name) <= 0:
            raise ContractError(f"--{name.replace('_', '-')} must be positive")
    expected_sha256 = args.expected_sha256.lower()
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ContractError("--expected-sha256 must be exactly 64 hexadecimal characters")
    args.expected_sha256 = expected_sha256
    args.handler(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
