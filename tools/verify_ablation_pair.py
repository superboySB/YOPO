#!/usr/bin/env python3
"""Audit an active/fixed camera benchmark for controlled-variable fairness."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from benchmark_active_camera import (
    ARM_ACTIVE,
    ARM_FIXED,
    SCHEMA_VERSION,
    aggregate_results,
    canonical_json,
    content_hash,
    paired_results,
    validate_manifest,
)


DEFAULT_TRAINING_ALLOWED = {
    "treatment.active_camera",
    "treatment.camera_mode",
    "dataset.path",
    "dataset.metadata_sha256",
    "dataset.metadata.active_camera",
    "dataset.metadata.dataset.camera_pitch_range_deg",
    "dataset.metadata.dataset.camera_yaw_range_deg",
    "config.active_camera",
    "config.dataset_path",
    "artifacts.output_dir",
    "artifacts.resolved_config",
    "artifacts.resolved_config_sha256",
    "artifacts.metrics_sha256",
    "artifacts.checkpoints",
    "created_utc",
}


def load_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON {}: {}".format(path, exc)) from exc


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, item in sorted(value.items()):
            path = "{}.{}".format(prefix, key) if prefix else str(key)
            output.update(flatten(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = "{}[{}]".format(prefix, index)
            output.update(flatten(item, path))
    else:
        output[prefix] = value
    return output


def allowed_path(path, allowed):
    return any(
        path == item or path.startswith(item + ".") or path.startswith(item + "[")
        for item in allowed
    )


def compare_training_manifests(active, fixed, allowed):
    active_flat = flatten(active)
    fixed_flat = flatten(fixed)
    differences = []
    for path in sorted(set(active_flat) | set(fixed_flat)):
        if allowed_path(path, allowed):
            continue
        active_value = active_flat.get(path, "<missing>")
        fixed_value = fixed_flat.get(path, "<missing>")
        if active_value != fixed_value:
            differences.append(
                {"path": path, "active": active_value, "fixed": fixed_value}
            )
    return differences


def verify_training_checkpoint_binding(
    manifest, arm, training_manifest, supplied_manifest_binding
):
    """Close the offline chain: supplied manifest -> record -> benchmark checkpoint."""
    errors = []
    runs = [run for run in manifest.get("runs", []) if run.get("arm") == arm]
    if not runs:
        return ["{} arm has no runs in pair manifest".format(arm)], None
    if not isinstance(supplied_manifest_binding, dict) or not supplied_manifest_binding.get(
        "sha256"
    ):
        return [
            "{} supplied training manifest has no computed SHA binding".format(arm)
        ], None

    run_bindings = []
    for run in runs:
        run_id = run.get("run_id")
        treatment = run.get("treatment", {})
        artifact = treatment.get("checkpoint_artifact")
        if not isinstance(artifact, dict):
            errors.append("{} has no checkpoint artifact binding".format(run_id))
            continue
        checkpoint_path = treatment.get("checkpoint")
        checkpoint_name = Path(str(checkpoint_path)).name if checkpoint_path else None
        checkpoint_sha = treatment.get("checkpoint_sha256")
        epoch = artifact.get("epoch")
        manifest_sha = artifact.get("training_manifest_sha256")
        if artifact.get("checkpoint_sha256") != checkpoint_sha:
            errors.append(
                "{} checkpoint artifact SHA differs from treatment checkpoint SHA".format(run_id)
            )
        if not checkpoint_name or not checkpoint_sha or not isinstance(epoch, int) or epoch <= 0:
            errors.append("{} has an incomplete checkpoint artifact binding".format(run_id))
        if not manifest_sha:
            errors.append("{} checkpoint artifact has no training manifest SHA".format(run_id))
        run_bindings.append(
            {
                "checkpoint_name": checkpoint_name,
                "checkpoint_sha256": checkpoint_sha,
                "epoch": epoch,
                "training_manifest_sha256": manifest_sha,
            }
        )

    unique_bindings = {canonical_json(binding) for binding in run_bindings}
    if len(unique_bindings) != 1:
        errors.append("{} runs do not share one checkpoint artifact binding".format(arm))
        return errors, None
    if not run_bindings:
        return errors, None
    binding = run_bindings[0]
    supplied_sha = supplied_manifest_binding.get("sha256")
    if supplied_sha != binding["training_manifest_sha256"]:
        errors.append(
            "{} supplied training manifest SHA does not match pair manifest: {} != {}".format(
                arm, supplied_sha, binding["training_manifest_sha256"]
            )
        )

    checkpoint_records = training_manifest.get("artifacts", {}).get("checkpoints", {})
    record = checkpoint_records.get(binding["checkpoint_name"])
    if not isinstance(record, dict):
        errors.append(
            "{} training manifest has no record for {}".format(
                arm, binding["checkpoint_name"]
            )
        )
    else:
        if record.get("path") != binding["checkpoint_name"]:
            errors.append("{} training checkpoint record path is not bound".format(arm))
        if record.get("sha256") != binding["checkpoint_sha256"]:
            errors.append("{} training checkpoint record SHA is not bound".format(arm))
        if record.get("epoch") != binding["epoch"]:
            errors.append("{} training checkpoint record epoch is not bound".format(arm))

    completed_epochs = training_manifest.get("optimization", {}).get("completed_epochs")
    if not isinstance(completed_epochs, int) or completed_epochs < binding["epoch"]:
        errors.append("{} manifest predates its bound checkpoint epoch".format(arm))
    return errors, binding


def almost_equal(left, right, tolerance=1e-12):
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            almost_equal(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            almost_equal(a, b, tolerance) for a, b in zip(left, right)
        )
    return left == right


def verify(
    manifest, summary=None, training_pair=None, training_allowed=None,
    training_seeds=None, allow_incomplete=False, expected_training_epochs=None,
    training_manifest_bindings=None,
):
    errors = list(validate_manifest(manifest))
    warnings = []
    if manifest.get("schema") != SCHEMA_VERSION:
        errors.append("Unexpected manifest schema: {}".format(manifest.get("schema")))

    runs_by_id = {run.get("run_id"): run for run in manifest.get("runs", [])}
    seeds = sorted({run.get("seed") for run in manifest.get("runs", [])})
    if any(seed in set(training_seeds or []) for seed in seeds):
        errors.append("Evaluation seeds overlap declared training seeds")
    if any(isinstance(seed, int) and seed < 100 for seed in seeds):
        warnings.append(
            "At least one evaluation seed is below 100; defaults 101..105 are intended as holdout seeds"
        )

    pair_hashes = {}
    for run in manifest.get("runs", []):
        pair_hashes.setdefault(run.get("pair_id"), set()).add(run.get("controlled_hash"))
        controlled = run.get("controlled_variables", {})
        if controlled.get("sensor", {}).get("model") != "looper_insight_9":
            errors.append("{} does not use Insight 9".format(run.get("run_id")))
        if controlled.get("seed") != run.get("seed"):
            errors.append("{} seed differs between run and controls".format(run.get("run_id")))
    for pair_id, hashes in pair_hashes.items():
        if len(hashes) != 1:
            errors.append("{} has unequal controlled-variable hashes".format(pair_id))

    training_differences = []
    verified_training_bindings = {}
    if training_pair is not None:
        active_training, fixed_training = training_pair
        for label, training, expected_active in (
            ("active", active_training, True), ("fixed", fixed_training, False)
        ):
            treatment = training.get("treatment", {})
            metadata = training.get("dataset", {}).get("metadata", {})
            optimization = training.get("optimization", {})
            contract = training.get("model_contract", {})
            if treatment.get("active_camera") is not expected_active:
                errors.append(f"{label} training manifest has wrong camera treatment")
            if metadata.get("active_camera") is not expected_active:
                errors.append(f"{label} dataset metadata has wrong camera treatment")
            if contract.get("sensor_model") != "looper_insight_9":
                errors.append(f"{label} training does not use Looper Insight 9")
            if (contract.get("image_width"), contract.get("image_height")) != (160, 192):
                errors.append(f"{label} training has wrong network depth dimensions")
            if float(contract.get("camera_loss_weight") or 0.0) <= 0.0:
                errors.append(f"{label} training disabled the shared camera loss")
            target_epochs = optimization.get("train_epoch")
            if optimization.get("completed_epochs") != target_epochs:
                errors.append(f"{label} training did not complete all requested epochs")
            if expected_training_epochs is not None and target_epochs != expected_training_epochs:
                errors.append(
                    f"{label} training epoch count is {target_epochs}, expected {expected_training_epochs}"
                )
            base_seed = metadata.get("seed")
            env_num = metadata.get("env_num")
            if isinstance(base_seed, int) and isinstance(env_num, int):
                overlap = set(range(base_seed, base_seed + env_num)) & set(seeds)
                if overlap:
                    errors.append(f"{label} training/evaluation map seeds overlap: {sorted(overlap)}")
            binding_errors, verified_binding = verify_training_checkpoint_binding(
                manifest,
                label,
                training,
                (training_manifest_bindings or {}).get(label),
            )
            errors.extend(binding_errors)
            verified_training_bindings[label] = verified_binding
        allowed = set(training_allowed or DEFAULT_TRAINING_ALLOWED)
        training_differences = compare_training_manifests(
            training_pair[0], training_pair[1], allowed
        )
        for difference in training_differences:
            errors.append(
                "Training manifests differ at {path}: active={active!r}, fixed={fixed!r}".format(
                    **difference
                )
            )

    result_checks = {
        "runs_in_summary": 0,
        "complete_pairs": 0,
        "fixed_camera_contract_checked": 0,
        "active_camera_contract_checked": 0,
        "camera_sync_contract_checked": 0,
    }
    if summary is not None:
        if summary.get("manifest_hash") != content_hash(manifest):
            errors.append("Summary is not bound to this pair manifest")
        summary_runs = summary.get("runs", [])
        result_checks["runs_in_summary"] = len(summary_runs)
        seen = set()
        for result in summary_runs:
            run_id = result.get("run_id")
            if run_id in seen:
                errors.append("Duplicate result for {}".format(run_id))
                continue
            seen.add(run_id)
            spec = runs_by_id.get(run_id)
            if spec is None:
                errors.append("Summary contains unknown run {}".format(run_id))
                continue
            if result.get("arm") != spec.get("arm") or result.get("seed") != spec.get("seed"):
                errors.append("{} result arm/seed does not match manifest".format(run_id))
            if result.get("controlled_hash") != spec.get("controlled_hash"):
                errors.append("{} result controlled_hash does not match manifest".format(run_id))
            if result.get("run_spec_hash") != spec.get("run_spec_hash"):
                errors.append("{} result run_spec_hash does not match manifest".format(run_id))
            if result.get("checkpoint_sha256") != spec.get("treatment", {}).get(
                "checkpoint_sha256"
            ):
                errors.append("{} checkpoint SHA does not match manifest".format(run_id))

            if result.get("status") == "planned":
                if not allow_incomplete:
                    errors.append("{} is still planned".format(run_id))
                continue
            tolerance = float(
                spec.get("controlled_variables", {})
                .get("camera_contract", {})
                .get("zero_tolerance_rad", 1e-4)
            )
            active_threshold = float(
                spec.get("controlled_variables", {})
                .get("camera_contract", {})
                .get("active_motion_threshold_rad", 0.01)
            )
            command_max = float(result.get("camera_command_max_abs_rad") or 0.0)
            state_max = float(result.get("camera_state_max_abs_rad") or 0.0)
            if result.get("arm") == ARM_FIXED:
                result_checks["fixed_camera_contract_checked"] += 1
                if int(result.get("camera_command_samples") or 0) <= 0 or int(
                    result.get("camera_state_samples") or 0
                ) <= 0:
                    errors.append("{} fixed camera topics were not both observed".format(run_id))
                if command_max > tolerance or state_max > tolerance:
                    errors.append(
                        "{} fixed camera moved: command={}, state={}, tolerance={}".format(
                            run_id, command_max, state_max, tolerance
                        )
                    )
            elif result.get("arm") == ARM_ACTIVE:
                result_checks["active_camera_contract_checked"] += 1
                require_motion = bool(
                    spec.get("controlled_variables", {})
                    .get("camera_contract", {})
                    .get("require_active_motion", True)
                )
                if int(result.get("camera_state_samples") or 0) <= 0:
                    errors.append("{} active camera state topic was not observed".format(run_id))
                if require_motion and state_max < active_threshold:
                    errors.append(
                        "{} active camera state never exceeded {} rad".format(run_id, active_threshold)
                    )
            if result.get("status") == "camera_contract_violation":
                errors.append("{} has a camera contract violation".format(run_id))
            if result.get("collision_detected") is None and not allow_incomplete:
                errors.append("{} has no evaluable collision outcome".format(run_id))
            expected_depth = {"width": 160, "height": 192, "encoding": "32FC1", "step": 640}
            if result.get("depth_contract") != expected_depth:
                errors.append("{} runtime depth contract is not Insight 9 160x192 simulation".format(run_id))
            depth_fps = result.get("depth_observed_fps")
            if depth_fps is None or not (10.0 <= float(depth_fps) <= 20.0):
                errors.append("{} observed depth FPS is outside [10,20]".format(run_id))

            sync_contract = (
                spec.get("controlled_variables", {}).get("camera_contract", {})
            )
            sync_slop = float(sync_contract.get("depth_state_sync_slop_s", 0.03))
            minimum_match_rate = float(
                sync_contract.get("minimum_depth_state_match_rate", 0.98)
            )
            stamped_topic = sync_contract.get(
                "stamped_state_topic", "/yopo/camera/orientation_stamped"
            )
            result_checks["camera_sync_contract_checked"] += 1
            stamped_samples = int(result.get("camera_state_stamped_samples") or 0)
            matched_samples = int(result.get("depth_state_matched_samples") or 0)
            unmatched_samples = int(result.get("depth_state_unmatched_samples") or 0)
            match_rate = result.get("depth_state_match_rate")
            max_stamp_error = result.get("depth_state_max_stamp_error_s")
            invalid_depth_stamps = int(result.get("depth_invalid_stamp_samples") or 0)
            invalid_state_stamps = int(
                result.get("camera_state_invalid_stamp_samples") or 0
            )
            if result.get("camera_state_stamped_topic") != stamped_topic:
                errors.append("{} used the wrong stamped camera-state topic".format(run_id))
            if stamped_samples <= 0 or matched_samples <= 0:
                errors.append("{} has no stamped depth/camera matches".format(run_id))
            if matched_samples + unmatched_samples != int(result.get("depth_samples") or 0):
                errors.append("{} depth sync accounting does not match depth_samples".format(run_id))
            try:
                match_rate_ok = (
                    match_rate is not None
                    and math.isfinite(float(match_rate))
                    and float(match_rate) >= minimum_match_rate
                )
            except (TypeError, ValueError):
                match_rate_ok = False
            if not match_rate_ok:
                errors.append(
                    "{} depth/camera match rate {} is below {}".format(
                        run_id, match_rate, minimum_match_rate
                    )
                )
            try:
                stamp_error_ok = (
                    max_stamp_error is not None
                    and math.isfinite(float(max_stamp_error))
                    and 0.0 <= float(max_stamp_error) <= sync_slop
                )
            except (TypeError, ValueError):
                stamp_error_ok = False
            if not stamp_error_ok:
                errors.append(
                    "{} max depth/camera stamp error {} exceeds {}s".format(
                        run_id, max_stamp_error, sync_slop
                    )
                )
            if invalid_depth_stamps or invalid_state_stamps:
                errors.append(
                    "{} has invalid sync stamps: depth={}, camera={}".format(
                        run_id, invalid_depth_stamps, invalid_state_stamps
                    )
                )
            if result.get("camera_sync_contract_ok") is not True:
                errors.append("{} failed the runtime camera sync contract".format(run_id))
            if result.get("status") == "camera_sync_violation":
                errors.append("{} has a camera sync violation".format(run_id))

        missing = set(runs_by_id) - seen
        if missing:
            message = "Summary is incomplete; missing runs: {}".format(", ".join(sorted(missing)))
            (warnings if allow_incomplete else errors).append(message)
        recomputed_aggregate = aggregate_results(summary_runs)
        recomputed_paired = paired_results(summary_runs)
        by_pair = {}
        for result in summary_runs:
            by_pair.setdefault(result.get("pair_id"), []).append(result)
        for pair_id, pair_runs in by_pair.items():
            if len(pair_runs) != 2:
                continue
            map_hashes = {result.get("map_sha256") for result in pair_runs}
            map_bounds = [result.get("map_bounds_xyz") for result in pair_runs]
            map_counts = {result.get("map_point_count") for result in pair_runs}
            if None in map_hashes or len(map_hashes) != 1:
                errors.append("{} active/fixed runtime maps do not have the same SHA".format(pair_id))
            if not almost_equal(map_bounds[0], map_bounds[1]):
                errors.append("{} active/fixed runtime map bounds differ".format(pair_id))
            if None in map_counts or len(map_counts) != 1:
                errors.append("{} active/fixed runtime map point counts differ".format(pair_id))
        result_checks["complete_pairs"] = recomputed_paired["complete_pairs"]
        if not allow_incomplete and recomputed_paired["complete_pairs"] != len(seeds):
            errors.append(
                "Expected {} complete valid seed pairs, got {}".format(
                    len(seeds), recomputed_paired["complete_pairs"]
                )
            )
        if "aggregate" in summary and not almost_equal(summary["aggregate"], recomputed_aggregate):
            errors.append("Summary aggregate does not match raw runs")
        if "paired" in summary and not almost_equal(summary["paired"], recomputed_paired):
            errors.append("Summary paired metrics do not match raw runs")

    return {
        "ok": not errors,
        "schema": SCHEMA_VERSION,
        "manifest_hash": content_hash(manifest),
        "evaluation_seeds": seeds,
        "pair_count": len(pair_hashes),
        "errors": errors,
        "warnings": warnings,
        "result_checks": result_checks,
        "training_manifest_differences": training_differences,
        "training_checkpoint_bindings": verified_training_bindings,
    }


def parse_seed_list(text):
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def make_parser():
    parser = argparse.ArgumentParser(
        description="Verify that an active/fixed-camera A/B benchmark is fairly controlled."
    )
    parser.add_argument("--manifest", required=True, help="pair_manifest.json")
    parser.add_argument("--summary", default=None, help="Optional summary.json with raw runs.")
    parser.add_argument("--active-training-manifest", default=None)
    parser.add_argument("--fixed-training-manifest", default=None)
    parser.add_argument(
        "--allow-training-difference",
        action="append",
        default=[],
        help="Additional dotted path or leaf key allowed to differ.",
    )
    parser.add_argument(
        "--training-seeds",
        type=parse_seed_list,
        default=[],
        help="Comma-separated training seeds; overlap with evaluation is an error.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--expected-training-epochs", type=int, default=None)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        manifest = load_json(args.manifest)
        summary = load_json(args.summary) if args.summary else None
        if bool(args.active_training_manifest) != bool(args.fixed_training_manifest):
            raise ValueError("Provide both active and fixed training manifests, or neither")
        training_pair = None
        training_manifest_bindings = None
        if args.active_training_manifest:
            active_manifest_path = Path(args.active_training_manifest).resolve()
            fixed_manifest_path = Path(args.fixed_training_manifest).resolve()
            training_pair = (
                load_json(active_manifest_path),
                load_json(fixed_manifest_path),
            )
            training_manifest_bindings = {
                "active": {
                    "path": str(active_manifest_path),
                    "sha256": file_sha256(active_manifest_path),
                },
                "fixed": {
                    "path": str(fixed_manifest_path),
                    "sha256": file_sha256(fixed_manifest_path),
                },
            }
        allowed = DEFAULT_TRAINING_ALLOWED | set(args.allow_training_difference)
        report = verify(
            manifest,
            summary=summary,
            training_pair=training_pair,
            training_allowed=allowed,
            training_seeds=args.training_seeds,
            allow_incomplete=args.allow_incomplete,
            expected_training_epochs=args.expected_training_epochs,
            training_manifest_bindings=training_manifest_bindings,
        )
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output_json:
            Path(args.output_json).write_text(text + "\n")
        print(text)
        return 0 if report["ok"] else 1
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
