#!/usr/bin/env python3
"""Strictly audit an active/fixed maze-8 training-dataset pair.

The two datasets are allowed to differ only in the camera treatment:

* the top-level ``active_camera`` flag;
* ``dataset.camera_pitch_range_deg``; and
* ``dataset.camera_yaw_range_deg``.

Everything structural is required to be paired exactly.  Depth images are
expected to differ because they are rendered from different camera poses, so
their names and encoding are checked while their content hashes are reported
without requiring equality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


EXPECTED_SAMPLE_COLUMNS = (
    "sample_id",
    "pose_id",
    "dir_idx",
    "px",
    "py",
    "pz",
    "qw",
    "qx",
    "qy",
    "qz",
    "vdes_bx",
    "vdes_by",
    "vdes_bz",
    "goal_wx",
    "goal_wy",
    "goal_wz",
    "guide_offset",
    "guide_len",
    "guide_mask",
    "guide_cost",
    "selected_topology",
    "camera_pitch",
    "camera_yaw",
    "camera_target_pitch",
    "camera_target_yaw",
)
STRUCTURAL_COLUMN_COUNT = 21
CAMERA_COLUMNS = EXPECTED_SAMPLE_COLUMNS[STRUCTURAL_COLUMN_COUNT:]
ALLOWED_METADATA_DIFFERENCES = {
    "active_camera",
    "dataset.camera_pitch_range_deg",
    "dataset.camera_yaw_range_deg",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class InputError(RuntimeError):
    """The verifier could not read its inputs, as opposed to finding a mismatch."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InputError(f"Cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"Cannot read metadata {path}: {exc}") from exc

    value: Any
    try:
        try:
            from ruamel.yaml import YAML  # type: ignore

            value = YAML(typ="safe").load(text)
        except ImportError:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise InputError(
                    "A YAML reader is required (install ruamel.yaml or PyYAML)"
                ) from exc
            value = yaml.safe_load(text)
    except InputError:
        raise
    except Exception as exc:
        raise InputError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(value, Mapping):
        raise InputError(f"Metadata root must be a mapping: {path}")
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def without_allowed_metadata(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-compatible deep copy with treatment fields removed."""

    copied = json.loads(canonical_json(value).decode("utf-8"))
    copied.pop("active_camera", None)
    dataset = copied.get("dataset")
    if isinstance(dataset, MutableMapping):
        dataset.pop("camera_pitch_range_deg", None)
        dataset.pop("camera_yaw_range_deg", None)
    return copied


def flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    if isinstance(value, Mapping):
        if not value:
            output[prefix] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten(value[key], path))
    elif isinstance(value, list):
        if not value:
            output[prefix] = []
        for index, item in enumerate(value):
            output.update(flatten(item, f"{prefix}[{index}]"))
    else:
        output[prefix] = value
    return output


def metadata_differences(
    active: Mapping[str, Any], fixed: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    active_flat = flatten(without_allowed_metadata(active))
    fixed_flat = flatten(without_allowed_metadata(fixed))
    differences = []
    missing = "<missing>"
    for path in sorted(set(active_flat) | set(fixed_flat)):
        active_value = active_flat.get(path, missing)
        fixed_value = fixed_flat.get(path, missing)
        if active_value != fixed_value:
            differences.append(
                {"path": path, "active": active_value, "fixed": fixed_value}
            )
    return differences


def relative_file_set(root: Path) -> List[str]:
    try:
        return sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
    except OSError as exc:
        raise InputError(f"Cannot enumerate dataset {root}: {exc}") from exc


def bounded_list(values: Iterable[str], limit: int = 20) -> List[str]:
    items = sorted(values)
    if len(items) <= limit:
        return items
    return items[:limit] + [f"... ({len(items) - limit} more)"]


def parse_positive_int(metadata: Mapping[str, Any], key: str, errors: List[str]) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"metadata.{key} must be a positive integer, got {value!r}")
        return 0
    return value


def check_camera_range(value: Any, *, expect_zero: bool) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return False
    if expect_zero:
        return all(float(item) == 0.0 for item in value)
    return min(float(item) for item in value) < 0.0 < max(float(item) for item in value)


def validate_metadata_contract(
    label: str, metadata: Mapping[str, Any], expected_active: bool, errors: List[str]
) -> None:
    if metadata.get("active_camera") is not expected_active:
        errors.append(
            f"{label} metadata.active_camera must be {expected_active}, "
            f"got {metadata.get('active_camera')!r}"
        )
    if metadata.get("maze_type") != 8:
        errors.append(f"{label} metadata.maze_type must be 8")
    if metadata.get("sensor_model") != "looper_insight_9":
        errors.append(f"{label} metadata.sensor_model must be looper_insight_9")
    if metadata.get("image_width") != 160 or metadata.get("image_height") != 192:
        errors.append(f"{label} metadata image dimensions must be 160x192 (width x height)")

    schema = metadata.get("schema")
    if not isinstance(schema, Mapping) or tuple(schema.get("sample_columns", ())) != EXPECTED_SAMPLE_COLUMNS:
        errors.append(f"{label} metadata.schema.sample_columns does not match the 25-column contract")

    sensor = metadata.get("sensor")
    if not isinstance(sensor, Mapping) or (
        sensor.get("network_width"), sensor.get("network_height")
    ) != (160, 192):
        errors.append(f"{label} sensor network dimensions must be 160x192 (width x height)")

    dataset = metadata.get("dataset")
    if not isinstance(dataset, Mapping):
        errors.append(f"{label} metadata.dataset must be a mapping")
        return
    for name in ("camera_pitch_range_deg", "camera_yaw_range_deg"):
        value = dataset.get(name)
        if not check_camera_range(value, expect_zero=not expected_active):
            expected = "[0, 0]" if not expected_active else "a range spanning zero"
            errors.append(f"{label} dataset.{name} must be {expected}, got {value!r}")


def png_contract(path: Path) -> Tuple[int, int, int, int]:
    """Read a PNG IHDR and return width, height, bit depth and color type."""

    try:
        with path.open("rb") as stream:
            header = stream.read(33)
    except OSError as exc:
        raise InputError(f"Cannot read PNG {path}: {exc}") from exc
    if len(header) < 33 or header[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG or truncated before IHDR")
    length = struct.unpack(">I", header[8:12])[0]
    if length != 13 or header[12:16] != b"IHDR":
        raise ValueError("first PNG chunk is not a 13-byte IHDR")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
    return width, height, bit_depth, color_type


def hash_record(digest: "hashlib._Hash", path: str, value: bytes) -> None:
    path_bytes = path.encode("utf-8")
    digest.update(struct.pack(">I", len(path_bytes)))
    digest.update(path_bytes)
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def hash_csv_prefix(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for row_index, row in enumerate((header[:STRUCTURAL_COLUMN_COUNT],) + tuple(
        tuple(item[:STRUCTURAL_COLUMN_COUNT]) for item in rows
    )):
        for column_index, value in enumerate(row):
            hash_record(digest, f"{row_index}:{column_index}", value.encode("utf-8"))
    return digest.hexdigest()


def read_samples(path: Path) -> Tuple[List[str], List[List[str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration:
                return [], []
            return header, [row for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise InputError(f"Cannot parse samples CSV {path}: {exc}") from exc


def decimal_is_zero(value: str) -> bool:
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return False
    return parsed.is_finite() and parsed == Decimal(0)


def decimal_is_nonzero(value: str) -> bool:
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return False
    return parsed.is_finite() and parsed != Decimal(0)


def decimal_is_finite(value: str) -> bool:
    try:
        return Decimal(value).is_finite()
    except InvalidOperation:
        return False


def combined_component_hash(components: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for path, value in sorted(components.items()):
        hash_record(digest, path, value.encode("ascii"))
    return digest.hexdigest()


def verify_pair(active_dir: Path, fixed_dir: Path) -> Dict[str, Any]:
    active_dir = active_dir.resolve()
    fixed_dir = fixed_dir.resolve()
    for label, root in (("active", active_dir), ("fixed", fixed_dir)):
        if not root.is_dir():
            raise InputError(f"{label} dataset directory does not exist: {root}")

    active_meta = load_yaml(active_dir / "dataset_metadata.yaml")
    fixed_meta = load_yaml(fixed_dir / "dataset_metadata.yaml")
    errors: List[str] = []
    validate_metadata_contract("active", active_meta, True, errors)
    validate_metadata_contract("fixed", fixed_meta, False, errors)

    differences = metadata_differences(active_meta, fixed_meta)
    for difference in differences:
        errors.append(
            "Disallowed metadata difference at {path}: active={active!r}, "
            "fixed={fixed!r}".format(**difference)
        )

    active_files = relative_file_set(active_dir)
    fixed_files = relative_file_set(fixed_dir)
    active_file_set = set(active_files)
    fixed_file_set = set(fixed_files)
    if active_file_set != fixed_file_set:
        only_active = bounded_list(active_file_set - fixed_file_set)
        only_fixed = bounded_list(fixed_file_set - active_file_set)
        errors.append(
            f"Relative file sets differ; only_active={only_active}, only_fixed={only_fixed}"
        )

    env_num = parse_positive_int(active_meta, "env_num", errors)
    image_num = parse_positive_int(active_meta, "image_num", errors)
    direction_num = parse_positive_int(active_meta, "direction_num", errors)
    active_dataset_meta = active_meta.get("dataset")
    if isinstance(active_dataset_meta, Mapping):
        nested_counts = (
            active_dataset_meta.get("environment_count"),
            active_dataset_meta.get("images_per_environment"),
            active_dataset_meta.get("directions_per_image"),
        )
        if nested_counts != (env_num, image_num, direction_num):
            errors.append(
                "metadata dataset counts do not match env_num/image_num/direction_num: "
                f"nested={nested_counts}, top-level={(env_num, image_num, direction_num)}"
            )
    seed = active_meta.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        errors.append(f"metadata.seed must be an integer, got {seed!r}")
        train_seeds: List[int] = []
    else:
        train_seeds = list(range(seed, seed + env_num))

    expected_structural_paths: List[str] = ["_SUCCESS"]
    expected_sample_paths: List[str] = []
    expected_image_paths: List[str] = []
    for env_index in range(env_num):
        expected_structural_paths.extend(
            [
                f"pointcloud-{env_index}.ply",
                f"pose-{env_index}.csv",
                f"guides-{env_index}.csv",
            ]
        )
        expected_sample_paths.append(f"samples-{env_index}.csv")
        expected_image_paths.extend(
            f"{env_index}/img_{pose_index}_depth.png"
            for pose_index in range(image_num)
        )

    expected_files = {
        "dataset_metadata.yaml",
        *expected_structural_paths,
        *expected_sample_paths,
        *expected_image_paths,
    }
    for label, files in (("active", active_file_set), ("fixed", fixed_file_set)):
        missing = bounded_list(expected_files - files)
        unexpected = bounded_list(files - expected_files)
        if missing:
            errors.append(f"{label} dataset is missing expected files: {missing}")
        if unexpected:
            errors.append(f"{label} dataset has unexpected files: {unexpected}")

    structural_components: Dict[str, Dict[str, str]] = {"active": {}, "fixed": {}}
    paired_file_hashes: Dict[str, str] = {}
    for relative in expected_structural_paths:
        active_path = active_dir / relative
        fixed_path = fixed_dir / relative
        if not active_path.is_file() or not fixed_path.is_file():
            continue
        active_hash = sha256_file(active_path)
        fixed_hash = sha256_file(fixed_path)
        structural_components["active"][relative] = active_hash
        structural_components["fixed"][relative] = fixed_hash
        if active_hash != fixed_hash:
            errors.append(
                f"Structural file differs byte-for-byte: {relative} "
                f"(active={active_hash}, fixed={fixed_hash})"
            )
        else:
            paired_file_hashes[relative] = active_hash

    camera_nonzero_counts = {column: 0 for column in CAMERA_COLUMNS}
    sample_row_counts = {"active": 0, "fixed": 0}
    sample_prefix_hashes: Dict[str, Dict[str, str]] = {"active": {}, "fixed": {}}
    for relative in expected_sample_paths:
        active_path = active_dir / relative
        fixed_path = fixed_dir / relative
        if not active_path.is_file() or not fixed_path.is_file():
            continue
        active_header, active_rows = read_samples(active_path)
        fixed_header, fixed_rows = read_samples(fixed_path)
        for label, header in (("active", active_header), ("fixed", fixed_header)):
            if tuple(header) != EXPECTED_SAMPLE_COLUMNS:
                errors.append(
                    f"{label} {relative} has unexpected sample columns: {header!r}"
                )
        if active_header != fixed_header:
            errors.append(f"Sample headers differ: {relative}")
        if len(active_rows) != len(fixed_rows):
            errors.append(
                f"Sample row counts differ in {relative}: "
                f"active={len(active_rows)}, fixed={len(fixed_rows)}"
            )
        sample_row_counts["active"] += len(active_rows)
        sample_row_counts["fixed"] += len(fixed_rows)
        for row_index in range(max(len(active_rows), len(fixed_rows))):
            if row_index >= len(active_rows) or row_index >= len(fixed_rows):
                continue
            active_row = active_rows[row_index]
            fixed_row = fixed_rows[row_index]
            if len(active_row) != len(EXPECTED_SAMPLE_COLUMNS):
                errors.append(
                    f"active {relative}:{row_index + 2} has {len(active_row)} fields, expected 25"
                )
                continue
            if len(fixed_row) != len(EXPECTED_SAMPLE_COLUMNS):
                errors.append(
                    f"fixed {relative}:{row_index + 2} has {len(fixed_row)} fields, expected 25"
                )
                continue
            for column_index in range(STRUCTURAL_COLUMN_COUNT):
                if active_row[column_index] != fixed_row[column_index]:
                    errors.append(
                        f"Structural sample field differs at {relative}:{row_index + 2} "
                        f"column {EXPECTED_SAMPLE_COLUMNS[column_index]}: "
                        f"active={active_row[column_index]!r}, fixed={fixed_row[column_index]!r}"
                    )
            for camera_index, column in enumerate(CAMERA_COLUMNS, STRUCTURAL_COLUMN_COUNT):
                if not decimal_is_zero(fixed_row[camera_index]):
                    errors.append(
                        f"Fixed camera label is not exact numeric zero at "
                        f"{relative}:{row_index + 2} column {column}: "
                        f"{fixed_row[camera_index]!r}"
                    )
                if not decimal_is_finite(active_row[camera_index]):
                    errors.append(
                        f"Active camera label is not a finite number at "
                        f"{relative}:{row_index + 2} column {column}: "
                        f"{active_row[camera_index]!r}"
                    )
                if decimal_is_nonzero(active_row[camera_index]):
                    camera_nonzero_counts[column] += 1

        if tuple(active_header) == EXPECTED_SAMPLE_COLUMNS:
            sample_prefix_hashes["active"][relative] = hash_csv_prefix(
                active_header, active_rows
            )
        if tuple(fixed_header) == EXPECTED_SAMPLE_COLUMNS:
            sample_prefix_hashes["fixed"][relative] = hash_csv_prefix(
                fixed_header, fixed_rows
            )
        active_prefix = sample_prefix_hashes["active"].get(relative)
        fixed_prefix = sample_prefix_hashes["fixed"].get(relative)
        if active_prefix is not None and fixed_prefix is not None and active_prefix != fixed_prefix:
            errors.append(
                f"Structural sample-prefix hash differs: {relative} "
                f"(active={active_prefix}, fixed={fixed_prefix})"
            )

    expected_rows = env_num * image_num * direction_num
    for label, row_count in sample_row_counts.items():
        if row_count != expected_rows:
            errors.append(
                f"{label.capitalize()} sample count is {row_count}, expected {expected_rows} "
                f"from env_num*image_num*direction_num"
            )
    for column, count in camera_nonzero_counts.items():
        if count == 0:
            errors.append(f"Active camera column {column} has no nonzero value")

    image_components: Dict[str, Dict[str, str]] = {"active": {}, "fixed": {}}
    png_contract_counts: Dict[str, Dict[str, int]] = {"active": {}, "fixed": {}}
    for label, root in (("active", active_dir), ("fixed", fixed_dir)):
        for relative in expected_image_paths:
            path = root / relative
            if not path.is_file():
                continue
            image_components[label][relative] = sha256_file(path)
            try:
                width, height, bit_depth, color_type = png_contract(path)
            except ValueError as exc:
                errors.append(f"{label} {relative}: {exc}")
                continue
            contract_key = f"{width}x{height}:bit{bit_depth}:color_type{color_type}"
            png_contract_counts[label][contract_key] = (
                png_contract_counts[label].get(contract_key, 0) + 1
            )
            if (width, height, bit_depth, color_type) != (160, 192, 16, 0):
                errors.append(
                    f"{label} {relative} must be 160x192 uint16 single-channel PNG, "
                    f"got {contract_key}"
                )

    canonical_metadata_hash = hashlib.sha256(
        canonical_json(without_allowed_metadata(active_meta))
    ).hexdigest()
    for label in ("active", "fixed"):
        structural_components[label]["@metadata_without_treatment"] = (
            canonical_metadata_hash
            if label == "active"
            else hashlib.sha256(
                canonical_json(without_allowed_metadata(fixed_meta))
            ).hexdigest()
        )
        structural_components[label].update(
            {f"@sample_prefix/{key}": value for key, value in sample_prefix_hashes[label].items()}
        )
    structural_hashes = {
        label: combined_component_hash(components)
        for label, components in structural_components.items()
    }
    expected_structural_components = len(expected_structural_paths) + len(
        expected_sample_paths
    ) + 1  # treatment-stripped metadata
    structural_complete = all(
        len(structural_components[label]) == expected_structural_components
        for label in ("active", "fixed")
    )
    structural_pair_hash = None
    if structural_complete and structural_hashes["active"] == structural_hashes["fixed"]:
        structural_pair_hash = structural_hashes["active"]
    image_group_hashes = {
        label: combined_component_hash(components)
        for label, components in image_components.items()
    }

    return {
        "ok": not errors,
        "errors": errors,
        "contract": {
            "maze_type": 8,
            "sensor_model": "looper_insight_9",
            "image_width": 160,
            "image_height": 192,
            "png_dtype": "uint16",
            "png_channels": 1,
            "structural_sample_columns": STRUCTURAL_COLUMN_COUNT,
            "camera_sample_columns": list(CAMERA_COLUMNS),
            "allowed_metadata_differences": sorted(ALLOWED_METADATA_DIFFERENCES),
        },
        "datasets": {"active": str(active_dir), "fixed": str(fixed_dir)},
        "training_seed_set": train_seeds,
        "counts": {
            "environments": env_num,
            "images_per_environment": image_num,
            "directions_per_image": direction_num,
            "sample_rows": sample_row_counts["active"],
            "sample_rows_per_arm": sample_row_counts,
            "depth_images_per_arm": len(expected_image_paths),
            "observed_depth_images_per_arm": {
                label: len(components) for label, components in image_components.items()
            },
            "relative_files_per_arm": {
                "active": len(active_files), "fixed": len(fixed_files)
            },
            "structural_files_per_arm": len(expected_structural_paths),
        },
        "dimensions": {
            "width": 160,
            "height": 192,
            "bit_depth": 16,
            "channels": 1,
            "observed_png_contracts": png_contract_counts,
        },
        "structural_pair_hash": structural_pair_hash,
        "structural_hashes": structural_hashes,
        "paired_structural_file_sha256": paired_file_hashes,
        "sample_prefix_group_hashes": {
            label: combined_component_hash(components)
            for label, components in sample_prefix_hashes.items()
        },
        "active_camera_nonzero_counts": camera_nonzero_counts,
        "image_group_sha256": image_group_hashes,
        "image_hash_equality_required": False,
        "file_sets_equal": active_file_set == fixed_file_set,
        "image_filename_sets_equal": {
            relative for relative in active_file_set if relative.endswith("_depth.png")
        } == {
            relative for relative in fixed_file_set if relative.endswith("_depth.png")
        },
        "metadata_without_treatment_equal": not differences,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly verify paired active/fixed maze-8 training datasets."
    )
    parser.add_argument("active_dataset", type=Path, help="Active-camera dataset directory")
    parser.add_argument("fixed_dataset", type=Path, help="Fixed-camera dataset directory")
    parser.add_argument(
        "--output-json", type=Path, help="Also write the complete audit report to this path"
    )
    return parser


def emit_report(report: Mapping[str, Any], output_json: Optional[Path]) -> None:
    serialized = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output_json is not None:
        try:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            raise InputError(f"Cannot write report {output_json}: {exc}") from exc
    sys.stdout.write(serialized)


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        report = verify_pair(args.active_dataset, args.fixed_dataset)
        emit_report(report, args.output_json)
    except InputError as exc:
        report = {"ok": False, "errors": [str(exc)], "failure_kind": "input_error"}
        try:
            emit_report(report, args.output_json)
        except InputError as output_exc:
            sys.stderr.write(f"verify_paired_datasets.py: {output_exc}\n")
        return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
