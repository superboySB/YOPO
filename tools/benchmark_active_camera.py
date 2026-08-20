#!/usr/bin/env python3
"""Reproducible active-camera versus fixed-camera ROS benchmark.

The orchestrator deliberately restarts roscore, controller, sensor, planner and
monitor for every arm/seed.  A pair manifest records the variables that must be
identical and the two intended treatment differences: checkpoint and camera
actuation.  The same file also contains commands that can be copied verbatim
when a failed run needs to be reproduced.

The monitor mode lives in this file so it can share the result schema without
making the orchestration process import ROS.
"""

import argparse
import csv
import datetime as _datetime
import gzip
import hashlib
import json
import math
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "yopo.active-camera-ablation.v1"
ARM_ACTIVE = "active"
ARM_FIXED = "fixed"
ARMS = (ARM_ACTIVE, ARM_FIXED)


class StampPairAudit:
    """Bounded, one-to-one approximate timestamp matcher for runtime auditing."""

    def __init__(self, slop_s, max_pending=200):
        self.slop_s = float(slop_s)
        self.max_pending = int(max_pending)
        if not math.isfinite(self.slop_s) or self.slop_s <= 0.0:
            raise ValueError("sync slop must be finite and positive")
        if self.max_pending < 2:
            raise ValueError("max_pending must be at least 2")
        self.reset()

    def reset(self):
        self.depth_stamps = []
        self.state_stamps = []
        self.depth_samples = 0
        self.state_samples = 0
        self.invalid_depth_stamps = 0
        self.invalid_state_stamps = 0
        self.matched_samples = 0
        self.dropped_depth_samples = 0
        self.dropped_state_samples = 0
        self.max_stamp_error_s = None

    def add_depth(self, stamp_s):
        self._add("depth", stamp_s)

    def add_state(self, stamp_s):
        self._add("state", stamp_s)

    def _add(self, kind, stamp_s):
        try:
            stamp_s = float(stamp_s)
        except (TypeError, ValueError):
            stamp_s = float("nan")
        if kind == "depth":
            self.depth_samples += 1
            queue = self.depth_stamps
            invalid_name = "invalid_depth_stamps"
            dropped_name = "dropped_depth_samples"
        elif kind == "state":
            self.state_samples += 1
            queue = self.state_stamps
            invalid_name = "invalid_state_stamps"
            dropped_name = "dropped_state_samples"
        else:
            raise ValueError("Unknown stamp stream: {}".format(kind))
        if not math.isfinite(stamp_s) or stamp_s <= 0.0:
            setattr(self, invalid_name, getattr(self, invalid_name) + 1)
            return
        queue.append(stamp_s)
        queue.sort()
        while len(queue) > self.max_pending:
            queue.pop(0)
            setattr(self, dropped_name, getattr(self, dropped_name) + 1)
        self._match_available()

    def _match_available(self):
        while self.depth_stamps and self.state_stamps:
            best = min(
                (
                    (abs(depth - state), depth_index, state_index)
                    for depth_index, depth in enumerate(self.depth_stamps)
                    for state_index, state in enumerate(self.state_stamps)
                ),
                key=lambda item: item[0],
            )
            if best[0] <= self.slop_s:
                error, depth_index, state_index = best
                self.depth_stamps.pop(depth_index)
                self.state_stamps.pop(state_index)
                self.matched_samples += 1
                self.max_stamp_error_s = (
                    error if self.max_stamp_error_s is None
                    else max(self.max_stamp_error_s, error)
                )
                continue
            if self.depth_stamps[0] < self.state_stamps[-1] - self.slop_s:
                self.depth_stamps.pop(0)
                self.dropped_depth_samples += 1
                continue
            if self.state_stamps[0] < self.depth_stamps[-1] - self.slop_s:
                self.state_stamps.pop(0)
                self.dropped_state_samples += 1
                continue
            break

    def snapshot(self):
        unmatched_depth = self.depth_samples - self.matched_samples
        return {
            "depth_samples": self.depth_samples,
            "camera_state_stamped_samples": self.state_samples,
            "depth_state_matched_samples": self.matched_samples,
            "depth_state_unmatched_samples": unmatched_depth,
            "depth_state_match_rate": (
                self.matched_samples / float(self.depth_samples)
                if self.depth_samples else None
            ),
            "depth_state_max_stamp_error_s": self.max_stamp_error_s,
            "depth_invalid_stamp_samples": self.invalid_depth_stamps,
            "camera_state_invalid_stamp_samples": self.invalid_state_stamps,
            "depth_state_dropped_depth_samples": self.dropped_depth_samples,
            "depth_state_dropped_camera_samples": self.dropped_state_samples,
            "depth_state_pending_depth_samples": len(self.depth_stamps),
            "depth_state_pending_camera_samples": len(self.state_stamps),
        }


def parse_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def bool_text(value):
    return "true" if value else "false"


def parse_vec3(text):
    try:
        values = [float(item) for item in str(text).split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected x,y,z") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return values


def parse_seeds(text):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            try:
                values.append(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError("Seeds must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one seed is required")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("Seeds must not contain duplicates")
    return values


def utc_now():
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_artifact(path, expected_active, expected_epoch):
    checkpoint = Path(path).resolve()
    sidecar_path = checkpoint.with_suffix(".manifest.json")
    if not sidecar_path.is_file():
        raise ValueError("Checkpoint lacks required hash-binding sidecar: {}".format(sidecar_path))
    sidecar = json.loads(sidecar_path.read_text())
    record = sidecar.get("checkpoint", {})
    actual_hash = file_sha256(checkpoint)
    if record.get("sha256") != actual_hash:
        raise ValueError("Checkpoint SHA mismatch: {}".format(checkpoint))
    if bool(sidecar.get("active_camera")) != bool(expected_active):
        raise ValueError("Checkpoint camera treatment mismatch: {}".format(checkpoint))
    if expected_epoch is not None and record.get("epoch") != expected_epoch:
        raise ValueError(
            "Checkpoint {} is epoch {}, expected {}".format(
                checkpoint, record.get("epoch"), expected_epoch
            )
        )
    training_manifest = checkpoint.parent / sidecar.get("training_manifest", "")
    resolved_config = checkpoint.parent / sidecar.get("resolved_config", "")
    if file_sha256(training_manifest) != sidecar.get("training_manifest_sha256"):
        raise ValueError("Checkpoint training-manifest binding failed: {}".format(checkpoint))
    if file_sha256(resolved_config) != sidecar.get("resolved_config_sha256"):
        raise ValueError("Checkpoint resolved-config binding failed: {}".format(checkpoint))
    training_record = json.loads(training_manifest.read_text())
    optimization = training_record.get("optimization", {})
    if optimization.get("completed_epochs", 0) < record.get("epoch", 0):
        raise ValueError("Checkpoint was saved before its epoch completed: {}".format(checkpoint))
    if expected_epoch is not None and optimization.get("train_epoch") != expected_epoch:
        raise ValueError("Training manifest requested a different epoch count: {}".format(checkpoint))
    return {
        "checkpoint_sha256": actual_hash,
        "checkpoint_sidecar": str(sidecar_path),
        "checkpoint_sidecar_sha256": file_sha256(sidecar_path),
        "training_manifest": str(training_manifest),
        "training_manifest_sha256": file_sha256(training_manifest),
        "epoch": record.get("epoch"),
    }


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(str(temporary), str(path))


def git_metadata():
    def run(*args):
        try:
            return subprocess.check_output(
                ["git"] + list(args), cwd=str(ROOT), stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    tracked_sources = (
        "Simulator/src/config/config.yaml",
        "Simulator/src/src/maps.cpp",
        "Simulator/src/src/test_simulator_cuda.cpp",
        "YOPO/config/traj_opt.yaml",
        "YOPO/policy/yopo_network.py",
        "YOPO/policy/yopo_dataset.py",
        "YOPO/policy/yopo_trainer.py",
        "YOPO/test_yopo_ros.py",
        "tools/benchmark_active_camera.py",
        "tools/verify_ablation_pair.py",
        "tools/plot_active_camera_ablation.py",
    )
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
        "source_files_sha256": {
            relative: file_sha256(ROOT / relative) for relative in tracked_sources
        },
    }


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def sourced_command(setups, command, master_uri):
    prefix = ["source /opt/ros/noetic/setup.bash"]
    for setup in setups:
        prefix.append("source {}".format(shlex.quote(str(setup))))
    prefix.append("export ROS_MASTER_URI={}".format(shlex.quote(master_uri)))
    return " && ".join(prefix + [command])


def yopo_pythonpath():
    return ":".join(
        [
            "/opt/ros/noetic/lib/python3/dist-packages",
            str(ROOT / "Controller/devel/lib/python3/dist-packages"),
            str(ROOT / "Simulator/devel/lib/python3/dist-packages"),
            str(ROOT / "YOPO"),
        ]
    )


def write_controller_launch(path, start):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<launch>
  <node pkg="so3_quadrotor_simulator" type="quadrotor_simulator_so3"
        name="quadrotor_simulator_so3" output="screen">
    <param name="rate/odom" value="100.0"/>
    <param name="simulator/init_state_x" value="{x}"/>
    <param name="simulator/init_state_y" value="{y}"/>
    <param name="simulator/init_state_z" value="{z}"/>
    <remap from="~odom" to="/sim/odom"/>
    <remap from="~imu" to="/sim/imu"/>
    <remap from="~cmd" to="so3_cmd"/>
    <remap from="~force_disturbance" to="force_disturbance"/>
    <remap from="~moment_disturbance" to="moment_disturbance"/>
  </node>

  <node pkg="so3_control" type="network_control_node"
        name="network_controller_node" output="screen">
    <param name="is_simulation" value="true"/>
    <param name="use_disturbance_observer" value="true"/>
    <param name="hover_thrust" value="0.375"/>
    <param name="record_log" value="false"/>
    <remap from="~odom" to="/sim/odom"/>
    <remap from="~imu" to="/sim/imu"/>
    <remap from="~position_cmd" to="/so3_control/pos_cmd"/>
    <remap from="~so3_cmd" to="so3_cmd"/>
  </node>
</launch>
""".format(x=start[0], y=start[1], z=start[2])
    )
    return path


def controlled_variables(args, seed):
    """Return the object that must be byte-identical within an A/B pair."""
    return {
        "seed": int(seed),
        "sensor": {
            "model": "looper_insight_9",
            "maze_type": int(args.maze_type),
            "map_viz_resolution_m": float(args.map_viz_resolution),
            "depth_topic": args.depth_topic,
            "odom_topic": args.odom_topic,
            "map_topic": args.map_topic,
            "collision_counter_topic": args.collision_counter_topic,
            "max_depth_m": float(args.max_depth),
        },
        "flight": {
            "start": list(args.start),
            "end": list(args.end),
            "repeats": int(args.repeats),
            "velocity_mps": float(args.velocity),
            "arrive_distance_m": float(args.arrive_dist),
            "segment_timeout_s": float(args.segment_timeout),
            "fixed_body_yaw": bool(args.fixed_yaw),
        },
        "planner": {
            "sgm_time_s": float(args.sgm_time),
            "radius_min_m": args.radius_min,
            "radius_max_m": args.radius_max,
            "visualize": False,
            "control_topic": args.control_topic,
        },
        "collision_metric": {
            "vehicle_radius_m": float(args.collision_radius),
            "episode_gap_s": float(args.collision_episode_gap),
            "map_wait_s": float(args.map_wait),
            "require_map": bool(args.require_map),
        },
        "camera_contract": {
            "command_topic": args.camera_command_topic,
            "state_topic": args.camera_state_topic,
            "stamped_state_topic": args.camera_state_stamped_topic,
            "zero_tolerance_rad": float(args.camera_zero_tolerance),
            "active_motion_threshold_rad": float(args.active_motion_threshold),
            "require_active_motion": bool(args.require_active_motion),
            "depth_state_sync_slop_s": float(args.camera_sync_slop),
            "minimum_depth_state_match_rate": float(args.min_depth_state_match_rate),
        },
        "controller": {
            "odom_rate_hz": 100.0,
            "hover_thrust": 0.375,
            "disturbance_observer": True,
        },
    }


def weight_for_arm(args, arm):
    return str(Path(args.active_weight if arm == ARM_ACTIVE else args.fixed_weight).resolve())


def run_order(seeds, policy, order_seed):
    ordered = []
    rng = random.Random(order_seed)
    for index, seed in enumerate(seeds):
        if policy == "active-first":
            arms = [ARM_ACTIVE, ARM_FIXED]
        elif policy == "fixed-first":
            arms = [ARM_FIXED, ARM_ACTIVE]
        elif policy == "random":
            arms = [ARM_ACTIVE, ARM_FIXED]
            rng.shuffle(arms)
        else:  # balanced
            arms = [ARM_ACTIVE, ARM_FIXED] if index % 2 == 0 else [ARM_FIXED, ARM_ACTIVE]
        ordered.extend((seed, arm) for arm in arms)
    return ordered


def make_commands(args, seed, arm, run_dir, controller_launch):
    enabled = arm == ARM_ACTIVE
    master_uri = "http://127.0.0.1:{}".format(args.ros_master_port)
    controller_setup = ROOT / "Controller/devel/setup.bash"
    simulator_setup = ROOT / "Simulator/devel/setup.bash"

    controller_inner = "roslaunch {}".format(shlex.quote(str(controller_launch)))
    controller = sourced_command([controller_setup], controller_inner, master_uri)

    sensor_parts = [
        "rosrun",
        "sensor_simulator",
        "sensor_simulator_cuda",
        "__name:=sensor_simulator_node",
        "_active_camera:={}".format(bool_text(enabled)),
        "_maze_type:={}".format(args.maze_type),
        "_seed:={}".format(seed),
        "_map_viz_resolution:={}".format(args.map_viz_resolution),
        "_uav_collision_radius:={}".format(args.collision_radius),
    ]
    sensor_inner = "cd {} && {}".format(
        shlex.quote(str(ROOT / "Simulator")), shell_join(sensor_parts)
    )
    sensor = sourced_command([simulator_setup], sensor_inner, master_uri)

    planner_parts = [
        args.python,
        "YOPO/test_yopo_ros.py",
        "--weight",
        weight_for_arm(args, arm),
        "--velocity",
        args.velocity,
        "--max-depth",
        args.max_depth,
        "--arrive-dist",
        args.arrive_dist,
        "--sgm-time",
        args.sgm_time,
        "--visualize",
        0,
        "--active-camera",
        bool_text(enabled),
        "--odom-topic",
        args.odom_topic,
        "--depth-topic",
        args.depth_topic,
        "--camera-state-stamped-topic",
        args.camera_state_stamped_topic,
        "--camera-sync-queue-size",
        args.camera_sync_queue_size,
        "--camera-sync-slop",
        args.camera_sync_slop,
        "--ctrl-topic",
        args.control_topic,
    ]
    if args.radius_min is not None:
        planner_parts.extend(["--radius-min", args.radius_min])
    if args.radius_max is not None:
        planner_parts.extend(["--radius-max", args.radius_max])
    if args.fixed_yaw:
        planner_parts.append("--fixed-yaw")
    planner_inner = "cd {} && export PYTHONPATH={} && export CUDA_VISIBLE_DEVICES={} && {}".format(
        shlex.quote(str(ROOT)),
        shlex.quote(yopo_pythonpath()),
        shlex.quote(str(args.gpu)),
        shell_join(planner_parts),
    )
    planner = sourced_command([controller_setup, simulator_setup], planner_inner, master_uri)

    monitor_parts = [
        args.python,
        str(Path(__file__).resolve()),
        "--monitor",
        "--run-dir",
        run_dir,
        "--arm",
        arm,
        "--seed",
        seed,
        "--start={}".format(",".join(map(str, args.start))),
        "--end={}".format(",".join(map(str, args.end))),
        "--repeats",
        args.repeats,
        "--arrive-dist",
        args.arrive_dist,
        "--segment-timeout",
        args.segment_timeout,
        "--collision-radius",
        args.collision_radius,
        "--collision-episode-gap",
        args.collision_episode_gap,
        "--map-wait",
        args.map_wait,
        "--map-topic",
        args.map_topic,
        "--odom-topic",
        args.odom_topic,
        "--camera-command-topic",
        args.camera_command_topic,
        "--camera-state-topic",
        args.camera_state_topic,
        "--camera-state-stamped-topic",
        args.camera_state_stamped_topic,
        "--camera-sync-slop",
        args.camera_sync_slop,
        "--min-depth-state-match-rate",
        args.min_depth_state_match_rate,
        "--collision-counter-topic",
        args.collision_counter_topic,
        "--camera-zero-tolerance",
        args.camera_zero_tolerance,
        "--active-motion-threshold",
        args.active_motion_threshold,
        "--plot-map-max-points",
        args.plot_map_max_points,
        "--require-map",
        bool_text(args.require_map),
        "--require-active-motion",
        bool_text(args.require_active_motion),
    ]
    monitor_inner = "cd {} && export PYTHONPATH={} && {}".format(
        shlex.quote(str(ROOT)), shlex.quote(yopo_pythonpath()), shell_join(monitor_parts)
    )
    monitor = sourced_command([controller_setup, simulator_setup], monitor_inner, master_uri)
    return {
        "roscore": sourced_command(
            [], "roscore -p {}".format(args.ros_master_port), master_uri
        ),
        "controller": controller,
        "sensor": sensor,
        "planner": planner,
        "monitor": monitor,
    }


def build_manifest(args, output_dir):
    run_specs = []
    checkpoint_hashes = {
        arm: file_sha256(weight_for_arm(args, arm)) for arm in ARMS
    }
    for sequence, (seed, arm) in enumerate(
        run_order(args.seeds, args.order, args.order_seed), start=1
    ):
        pair_id = "seed_{:06d}".format(seed)
        run_id = "{}_{}".format(pair_id, arm)
        run_dir = output_dir / "runs" / run_id
        controller_launch = run_dir / "controller.launch"
        controlled = controlled_variables(args, seed)
        commands = make_commands(args, seed, arm, run_dir, controller_launch)
        run_spec = {
                "sequence": sequence,
                "pair_id": pair_id,
                "run_id": run_id,
                "seed": seed,
                "arm": arm,
                "treatment": {
                    "camera_enabled": arm == ARM_ACTIVE,
                    "checkpoint": weight_for_arm(args, arm),
                    "checkpoint_sha256": checkpoint_hashes[arm],
                    "checkpoint_artifact": getattr(args, "checkpoint_artifacts", {}).get(arm),
                },
                "controlled_variables": controlled,
                "controlled_hash": content_hash(controlled),
                "run_dir": str(run_dir),
                "commands": commands,
            }
        run_spec["run_spec_hash"] = content_hash(
            {
                "seed": run_spec["seed"],
                "arm": run_spec["arm"],
                "treatment": run_spec["treatment"],
                "controlled_variables": run_spec["controlled_variables"],
                "commands": run_spec["commands"],
            }
        )
        run_specs.append(run_spec)
    return {
        "schema": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "repository": git_metadata(),
        "order_policy": args.order,
        "order_seed": args.order_seed,
        "expected_treatment_differences": [
            "camera_enabled",
            "checkpoint",
            "checkpoint_sha256",
        ],
        "runs": run_specs,
    }


def validate_manifest(manifest):
    errors = []
    pairs = {}
    for run in manifest.get("runs", []):
        pairs.setdefault(run.get("pair_id"), []).append(run)
        if run.get("controlled_hash") != content_hash(run.get("controlled_variables")):
            errors.append("{} has an invalid controlled_hash".format(run.get("run_id")))
        expected_run_hash = content_hash(
            {
                "seed": run.get("seed"),
                "arm": run.get("arm"),
                "treatment": run.get("treatment"),
                "controlled_variables": run.get("controlled_variables"),
                "commands": run.get("commands"),
            }
        )
        if run.get("run_spec_hash") != expected_run_hash:
            errors.append("{} has an invalid run_spec_hash".format(run.get("run_id")))
        expected_enabled = run.get("arm") == ARM_ACTIVE
        if run.get("treatment", {}).get("camera_enabled") is not expected_enabled:
            errors.append("{} camera treatment does not match arm".format(run.get("run_id")))
        mode = "--active-camera {}".format(bool_text(expected_enabled))
        if mode not in run.get("commands", {}).get("planner", ""):
            errors.append("{} planner command lacks {}".format(run.get("run_id"), mode))
        stamped_topic = (
            run.get("controlled_variables", {})
            .get("camera_contract", {})
            .get("stamped_state_topic")
        )
        stamped_mode = "--camera-state-stamped-topic {}".format(stamped_topic)
        if stamped_mode not in run.get("commands", {}).get("planner", ""):
            errors.append("{} planner command lacks stamped camera state".format(run.get("run_id")))
        sensor_mode = "_active_camera:={}".format(bool_text(expected_enabled))
        if sensor_mode not in run.get("commands", {}).get("sensor", ""):
            errors.append("{} sensor command lacks {}".format(run.get("run_id"), sensor_mode))

    for pair_id, runs in pairs.items():
        arms = {run.get("arm") for run in runs}
        if len(runs) != 2 or arms != set(ARMS):
            errors.append("{} must contain exactly active and fixed runs".format(pair_id))
            continue
        if len({run.get("controlled_hash") for run in runs}) != 1:
            errors.append("{} changes controlled variables between arms".format(pair_id))
    return errors


class ManagedProcess:
    def __init__(self, name, command, log_path):
        self.name = name
        self.command = command
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w")
        self.process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(ROOT),
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    def assert_running(self):
        code = self.process.poll()
        if code is not None:
            raise RuntimeError(
                "{} exited early with code {}; see {}".format(self.name, code, self.log_path)
            )

    def wait(self, timeout):
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stop(self, grace=4.0):
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
            deadline = time.time() + grace
            while time.time() < deadline and self.process.poll() is None:
                time.sleep(0.1)
            if self.process.poll() is None:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.log_file.close()


def wait_for_master(master_uri, timeout):
    deadline = time.time() + timeout
    command = (
        "source /opt/ros/noetic/setup.bash && export ROS_MASTER_URI={} && rosparam list"
    ).format(shlex.quote(master_uri))
    while time.time() < deadline:
        result = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return
        time.sleep(0.25)
    raise RuntimeError("ROS master did not become ready at {}".format(master_uri))


def sleep_and_check(process, delay):
    deadline = time.time() + delay
    while time.time() < deadline:
        process.assert_running()
        time.sleep(min(0.25, max(0.0, deadline - time.time())))
    process.assert_running()


def failure_result(run_spec, status, message):
    return {
        "schema": SCHEMA_VERSION,
        "run_id": run_spec["run_id"],
        "pair_id": run_spec["pair_id"],
        "seed": run_spec["seed"],
        "arm": run_spec["arm"],
        "camera_enabled": run_spec["treatment"]["camera_enabled"],
        "checkpoint": run_spec["treatment"]["checkpoint"],
        "checkpoint_sha256": run_spec["treatment"].get("checkpoint_sha256"),
        "sequence": run_spec.get("sequence"),
        "status": status,
        "success": False,
        "collision_free_success": False,
        "collision_detected": None,
        "error": message,
        "controlled_hash": run_spec["controlled_hash"],
        "run_spec_hash": run_spec.get("run_spec_hash"),
    }


def execute_run(args, run_spec):
    run_dir = Path(run_spec["run_dir"])
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    old_manifest_path = run_dir / "run_manifest.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError(
            "Refusing to mix a new run with stale result: {}".format(result_path)
        )
    if args.resume and old_manifest_path.exists():
        previous_spec = json.loads(old_manifest_path.read_text())
        if previous_spec.get("run_spec_hash") != run_spec.get("run_spec_hash"):
            raise ValueError("Resume run specification changed for {}".format(run_spec["run_id"]))
    controller_launch = write_controller_launch(run_dir / "controller.launch", args.start)
    if str(controller_launch) not in run_spec["commands"]["controller"]:
        raise RuntimeError("Controller launch path differs from manifest")
    atomic_write_json(run_dir / "run_manifest.json", run_spec)
    if args.dry_run:
        result = failure_result(run_spec, "planned", "dry-run: processes were not launched")
        atomic_write_json(run_dir / "result.json", result)
        return result

    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text())
        if (
            previous.get("run_spec_hash") == run_spec.get("run_spec_hash")
            and previous.get("checkpoint_sha256")
            == run_spec["treatment"].get("checkpoint_sha256")
            and previous.get("status") in ("ok", "collision", "incomplete")
        ):
            return previous

    processes = []
    master_uri = "http://127.0.0.1:{}".format(args.ros_master_port)
    try:
        roscore = ManagedProcess("roscore", run_spec["commands"]["roscore"], logs_dir / "roscore.log")
        processes.append(roscore)
        wait_for_master(master_uri, args.master_timeout)
        roscore.assert_running()

        controller = ManagedProcess(
            "controller", run_spec["commands"]["controller"], logs_dir / "controller.log"
        )
        processes.append(controller)
        sleep_and_check(controller, args.controller_delay)

        sensor = ManagedProcess("sensor", run_spec["commands"]["sensor"], logs_dir / "sensor.log")
        processes.append(sensor)
        sleep_and_check(sensor, args.sensor_delay)

        planner = ManagedProcess("planner", run_spec["commands"]["planner"], logs_dir / "planner.log")
        processes.append(planner)
        sleep_and_check(planner, args.planner_delay)

        monitor = ManagedProcess("monitor", run_spec["commands"]["monitor"], logs_dir / "monitor.log")
        processes.append(monitor)
        timeout = args.run_timeout
        if timeout is None:
            timeout = args.repeats * 2 * args.segment_timeout + args.map_wait + 45.0
        return_code = monitor.wait(timeout)
        if return_code is None:
            result = failure_result(run_spec, "orchestrator_timeout", "monitor exceeded {:.1f}s".format(timeout))
        elif result_path.exists():
            result = json.loads(result_path.read_text())
            if return_code != 0 and result.get("status") == "ok":
                result["status"] = "monitor_exit_{}".format(return_code)
        else:
            result = failure_result(
                run_spec,
                "monitor_exit_{}".format(return_code),
                "monitor did not write result.json",
            )
    except Exception as exc:
        result = failure_result(run_spec, "launch_error", str(exc))
    finally:
        for process in reversed(processes):
            process.stop()
        if not args.dry_run:
            time.sleep(args.cooldown)

    result.update(
        {
            "checkpoint": run_spec["treatment"]["checkpoint"],
            "checkpoint_sha256": run_spec["treatment"].get("checkpoint_sha256"),
            "controlled_hash": run_spec["controlled_hash"],
            "run_spec_hash": run_spec["run_spec_hash"],
            "sequence": run_spec["sequence"],
        }
    )
    atomic_write_json(result_path, result)
    return result


def finite_values(results, key):
    output = []
    for result in results:
        value = result.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                output.append(value)
    return output


def valid_evaluation_run(result):
    return (
        result.get("status") in ("ok", "collision", "incomplete")
        and result.get("collision_detected") is not None
        and result.get("camera_contract_ok") is True
        and result.get("camera_sync_contract_ok") is True
    )


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return [None, None]
    p = successes / float(total)
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def aggregate_results(results):
    aggregate = {}
    for arm in ARMS:
        arm_results = [result for result in results if result.get("arm") == arm]
        # Infrastructure failures are retained in raw runs but must not be
        # silently scored as collision-free failures.
        evaluated = [result for result in arm_results if valid_evaluation_run(result)]
        successes = sum(bool(result.get("success")) for result in evaluated)
        collision_free_successes = sum(
            bool(result.get("collision_free_success")) for result in evaluated
        )
        collisions = sum(bool(result.get("collision_detected")) for result in evaluated)
        clearances = finite_values(evaluated, "min_clearance_m")
        successful_times = finite_values(
            [result for result in evaluated if result.get("success")], "duration_s"
        )
        aggregate[arm] = {
            "runs": len(arm_results),
            "evaluated_runs": len(evaluated),
            "unevaluated_runs": len(arm_results) - len(evaluated),
            "successes": successes,
            "success_rate": successes / len(evaluated) if evaluated else None,
            "success_rate_wilson95": wilson_interval(successes, len(evaluated)),
            "collision_free_successes": collision_free_successes,
            "collision_free_success_rate": (
                collision_free_successes / len(evaluated) if evaluated else None
            ),
            "collision_free_success_rate_wilson95": wilson_interval(
                collision_free_successes, len(evaluated)
            ),
            "collision_runs": collisions,
            "collision_rate": collisions / len(evaluated) if evaluated else None,
            "collision_rate_wilson95": wilson_interval(collisions, len(evaluated)),
            "collision_episodes": sum(int(result.get("collision_episode_count") or 0) for result in evaluated),
            "mean_min_clearance_m": sum(clearances) / len(clearances) if clearances else None,
            "mean_success_duration_s": (
                sum(successful_times) / len(successful_times) if successful_times else None
            ),
        }
    return aggregate


def paired_results(results):
    """Keep seed-level paired outcomes; do not replace them with pooled rates."""
    grouped = {}
    for result in results:
        if not valid_evaluation_run(result):
            continue
        grouped.setdefault(result.get("seed"), {})[result.get("arm")] = result
    pairs = []
    for seed in sorted(grouped, key=lambda value: (value is None, value)):
        arms = grouped[seed]
        if ARM_ACTIVE not in arms or ARM_FIXED not in arms:
            continue
        active = arms[ARM_ACTIVE]
        fixed = arms[ARM_FIXED]
        active_collision = int(bool(active.get("collision_detected")))
        fixed_collision = int(bool(fixed.get("collision_detected")))
        active_success = int(bool(active.get("success")))
        fixed_success = int(bool(fixed.get("success")))
        active_collision_free_success = int(bool(active.get("collision_free_success")))
        fixed_collision_free_success = int(bool(fixed.get("collision_free_success")))
        active_clearance = active.get("min_clearance_m")
        fixed_clearance = fixed.get("min_clearance_m")
        clearance_delta = None
        if active_clearance is not None and fixed_clearance is not None:
            clearance_delta = float(active_clearance) - float(fixed_clearance)
        pairs.append(
            {
                "seed": seed,
                "active_run_id": active.get("run_id"),
                "fixed_run_id": fixed.get("run_id"),
                "collision_indicator_active": active_collision,
                "collision_indicator_fixed": fixed_collision,
                "collision_active_minus_fixed": active_collision - fixed_collision,
                "success_indicator_active": active_success,
                "success_indicator_fixed": fixed_success,
                "success_active_minus_fixed": active_success - fixed_success,
                "collision_free_success_indicator_active": active_collision_free_success,
                "collision_free_success_indicator_fixed": fixed_collision_free_success,
                "collision_free_success_active_minus_fixed": (
                    active_collision_free_success - fixed_collision_free_success
                ),
                "min_clearance_active_m": active_clearance,
                "min_clearance_fixed_m": fixed_clearance,
                "min_clearance_active_minus_fixed_m": clearance_delta,
            }
        )
    collision_deltas = [item["collision_active_minus_fixed"] for item in pairs]
    success_deltas = [item["success_active_minus_fixed"] for item in pairs]
    collision_free_success_deltas = [
        item["collision_free_success_active_minus_fixed"] for item in pairs
    ]
    clearance_deltas = [
        item["min_clearance_active_minus_fixed_m"]
        for item in pairs
        if item["min_clearance_active_minus_fixed_m"] is not None
    ]
    return {
        "complete_pairs": len(pairs),
        "per_seed": pairs,
        "mean_collision_indicator_delta_active_minus_fixed": (
            sum(collision_deltas) / len(collision_deltas) if collision_deltas else None
        ),
        "mean_success_indicator_delta_active_minus_fixed": (
            sum(success_deltas) / len(success_deltas) if success_deltas else None
        ),
        "mean_collision_free_success_delta_active_minus_fixed": (
            sum(collision_free_success_deltas) / len(collision_free_success_deltas)
            if collision_free_success_deltas else None
        ),
        "mean_min_clearance_delta_active_minus_fixed_m": (
            sum(clearance_deltas) / len(clearance_deltas) if clearance_deltas else None
        ),
    }


SUMMARY_FIELDS = [
    "sequence",
    "pair_id",
    "run_id",
    "seed",
    "arm",
    "camera_enabled",
    "checkpoint",
    "checkpoint_sha256",
    "status",
    "success",
    "collision_free_success",
    "collision_detected",
    "collision_episode_count",
    "sim_collision_counter_delta",
    "duration_s",
    "path_length_m",
    "min_center_obstacle_distance_m",
    "min_clearance_m",
    "completed_segments",
    "camera_contract_ok",
    "camera_sync_contract_ok",
    "camera_command_max_abs_rad",
    "camera_state_max_abs_rad",
    "camera_state_stamped_samples",
    "depth_state_matched_samples",
    "depth_state_unmatched_samples",
    "depth_state_match_rate",
    "depth_state_max_stamp_error_s",
    "map_topic_used",
    "map_point_count",
    "trajectory_csv",
    "camera_csv",
    "collision_csv",
    "map_csv_gz",
    "controlled_hash",
    "error",
]


def write_summary(output_dir, manifest, results):
    output_dir = Path(output_dir)
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in sorted(results, key=lambda item: item.get("sequence", 10 ** 9)):
            writer.writerow(result)
    summary = {
        "schema": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "manifest": str(output_dir / "pair_manifest.json"),
        "manifest_hash": content_hash(manifest),
        "aggregate": aggregate_results(results),
        "paired": paired_results(results),
        "runs": results,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def default_output_dir():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return ROOT / "reports" / "active_camera_ablation" / stamp


def run_orchestrator(args):
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            "Output directory is not empty; choose a new directory or use --resume: {}".format(
                output_dir
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_weight_check:
        args.checkpoint_artifacts = {}
        for arm in ARMS:
            path = Path(weight_for_arm(args, arm))
            if not path.is_file():
                raise FileNotFoundError("{} checkpoint not found: {}".format(arm, path))
            args.checkpoint_artifacts[arm] = validate_checkpoint_artifact(
                path, arm == ARM_ACTIVE, args.expected_checkpoint_epoch
            )

    manifest = build_manifest(args, output_dir)
    existing_manifest_path = output_dir / "pair_manifest.json"
    if args.resume and existing_manifest_path.is_file():
        previous_manifest = json.loads(existing_manifest_path.read_text())
        if previous_manifest.get("repository") != manifest.get("repository"):
            raise ValueError("Cannot resume after source/repository state changed")
        previous_hashes = {
            run.get("run_id"): run.get("run_spec_hash")
            for run in previous_manifest.get("runs", [])
        }
        current_hashes = {
            run.get("run_id"): run.get("run_spec_hash") for run in manifest.get("runs", [])
        }
        if previous_hashes != current_hashes:
            raise ValueError("Cannot resume after experiment specification changed")
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Invalid A/B manifest:\n- " + "\n- ".join(errors))
    atomic_write_json(output_dir / "pair_manifest.json", manifest)

    results = []
    for run_spec in manifest["runs"]:
        print(
            "[{}/{}] seed={} arm={} camera={} starting".format(
                run_spec["sequence"],
                len(manifest["runs"]),
                run_spec["seed"],
                run_spec["arm"],
                bool_text(run_spec["treatment"]["camera_enabled"]),
            ),
            flush=True,
        )
        result = execute_run(args, run_spec)
        results.append(result)
        write_summary(output_dir, manifest, results)
        print(
            "  status={} success={} collision={} clearance={}".format(
                result.get("status"),
                result.get("success"),
                result.get("collision_detected"),
                result.get("min_clearance_m"),
            ),
            flush=True,
        )

    summary = write_summary(output_dir, manifest, results)
    if args.plot and not args.dry_run:
        try:
            from plot_active_camera_ablation import generate_plots

            generated = generate_plots(output_dir / "summary.json", output_dir / "plots")
            summary["plots"] = [str(path) for path in generated]
            atomic_write_json(output_dir / "summary.json", summary)
        except Exception as exc:
            print("warning: plots were not generated: {}".format(exc), file=sys.stderr)
    print("A/B output: {}".format(output_dir), flush=True)
    return 0 if all(valid_evaluation_run(result) for result in results) else 1


def run_monitor(args):
    # ROS/scientific imports intentionally stay out of dry-run and verification.
    import threading

    import numpy as np
    import rospy
    from geometry_msgs.msg import PoseStamped, Vector3, Vector3Stamped
    from nav_msgs.msg import Odometry
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import Image, PointCloud2
    from std_msgs.msg import Int32

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / "odom_trajectory.csv"
    camera_path = run_dir / "camera.csv"
    collision_path = run_dir / "collision_episodes.csv"
    map_path = run_dir / "map_points.csv.gz"
    result_path = run_dir / "result.json"
    lock = threading.RLock()
    sync_audit = StampPairAudit(args.camera_sync_slop, max_pending=500)

    class NumpyTree:
        def __init__(self, points):
            self.points = points

        def query(self, point, k=1):
            del k
            distances = np.linalg.norm(self.points - point[None, :], axis=1)
            index = int(np.argmin(distances))
            return float(distances[index]), index

    state = {
        "recording": False,
        "recording_start": None,
        "pos": None,
        "last_pos": None,
        "path_length": 0.0,
        "map_tree": None,
        "map_points": None,
        "map_topic_used": None,
        "map_sha256": None,
        "map_bounds_xyz": None,
        "min_center_distance": float("inf"),
        "collision_active": False,
        "collision_current": None,
        "collision_episodes": [],
        "counter_initial": None,
        "counter_last": None,
        "counter_active_until": 0.0,
        "camera_command_max": 0.0,
        "camera_state_max": 0.0,
        "camera_command_nonzero": 0,
        "camera_state_nonzero": 0,
        "odom_samples": 0,
        "camera_samples": 0,
        "camera_command_samples": 0,
        "camera_state_samples": 0,
        "camera_command_seen": False,
        "camera_state_seen": False,
        "camera_state_stamped_seen": False,
        "depth_samples": 0,
        "depth_first_wall_time": None,
        "depth_last_wall_time": None,
        "depth_contract": None,
        "max_altitude_m": float("-inf"),
        "current_goal": None,
        "camera_command": [0.0, 0.0],
        "camera_state": [0.0, 0.0],
    }

    trajectory_stream = trajectory_path.open("w", newline="")
    camera_stream = camera_path.open("w", newline="")
    trajectory_fields = [
        "t_s",
        "wall_time_s",
        "ros_time_s",
        "x",
        "y",
        "z",
        "body_yaw_rad",
        "camera_pitch_rad",
        "camera_yaw_rad",
        "target_pitch_rad",
        "target_yaw_rad",
        "qx",
        "qy",
        "qz",
        "qw",
        "vx",
        "vy",
        "vz",
        "path_length_m",
        "goal_x",
        "goal_y",
        "goal_z",
        "goal_distance_m",
        "center_obstacle_distance_m",
        "clearance_m",
        "collision",
        "collision_active",
        "sim_collision_counter",
    ]
    camera_fields = ["wall_time_s", "ros_time_s", "kind", "pitch_rad", "yaw_rad", "z"]
    trajectory_writer = csv.DictWriter(trajectory_stream, fieldnames=trajectory_fields)
    camera_writer = csv.DictWriter(camera_stream, fieldnames=camera_fields)
    trajectory_writer.writeheader()
    camera_writer.writeheader()

    def header_stamp(msg):
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is not None and stamp.to_sec() > 0:
            return stamp.to_sec()
        return None

    def ros_stamp(msg):
        stamp = header_stamp(msg)
        if stamp is not None:
            return stamp
        return rospy.Time.now().to_sec()

    def start_collision(now, pos, center_distance, sources):
        state["collision_current"] = {
            "index": len(state["collision_episodes"]),
            "start_wall_time_s": now,
            "end_wall_time_s": None,
            "duration_s": None,
            "start_x": None if pos is None else float(pos[0]),
            "start_y": None if pos is None else float(pos[1]),
            "start_z": None if pos is None else float(pos[2]),
            "end_x": None,
            "end_y": None,
            "end_z": None,
            "min_center_obstacle_distance_m": center_distance,
            "min_clearance_m": (
                None if center_distance is None else center_distance - args.collision_radius
            ),
            "sources": sorted(sources),
            "sim_counter_start": state["counter_last"],
            "sim_counter_end": None,
            "samples": 0,
        }
        state["collision_active"] = True

    def finish_collision(now, pos):
        episode = state["collision_current"]
        if episode is None:
            return
        episode["end_wall_time_s"] = now
        episode["duration_s"] = max(0.0, now - episode["start_wall_time_s"])
        if pos is not None:
            episode["end_x"], episode["end_y"], episode["end_z"] = map(float, pos)
        episode["sim_counter_end"] = state["counter_last"]
        episode["sources"] = "+".join(sorted(set(episode["sources"])))
        state["collision_episodes"].append(episode)
        state["collision_current"] = None
        state["collision_active"] = False

    def update_collision(now, pos, center_distance):
        map_collision = center_distance is not None and center_distance <= args.collision_radius
        counter_collision = now <= state["counter_active_until"]
        sources = set()
        if map_collision:
            sources.add("map_kdtree")
        if counter_collision:
            sources.add("sim_counter")
        colliding = bool(sources)
        if colliding and not state["collision_active"]:
            start_collision(now, pos, center_distance, sources)
        elif not colliding and state["collision_active"]:
            finish_collision(now, pos)
        if state["collision_current"] is not None:
            episode = state["collision_current"]
            episode["samples"] += 1
            episode["sources"] = sorted(set(episode["sources"]) | sources)
            if center_distance is not None:
                previous = episode["min_center_obstacle_distance_m"]
                if previous is None or center_distance < previous:
                    episode["min_center_obstacle_distance_m"] = center_distance
                    episode["min_clearance_m"] = center_distance - args.collision_radius

    def odom_cb(msg):
        now = time.time()
        pos = np.asarray(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=np.float64,
        )
        with lock:
            state["pos"] = pos
            if not state["recording"]:
                state["last_pos"] = pos
                return
            if state["last_pos"] is not None:
                step = float(np.linalg.norm(pos - state["last_pos"]))
                if step < 2.0:  # reject only resets/teleports, not fast flight
                    state["path_length"] += step
            state["last_pos"] = pos
            state["max_altitude_m"] = max(state["max_altitude_m"], float(pos[2]))
            center_distance = None
            if state["map_tree"] is not None:
                center_distance = float(state["map_tree"].query(pos, k=1)[0])
                state["min_center_distance"] = min(state["min_center_distance"], center_distance)
            update_collision(now, pos, center_distance)
            goal = state["current_goal"]
            goal_distance = None if goal is None else float(np.linalg.norm(goal - pos))
            qx = float(msg.pose.pose.orientation.x)
            qy = float(msg.pose.pose.orientation.y)
            qz = float(msg.pose.pose.orientation.z)
            qw = float(msg.pose.pose.orientation.w)
            body_yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            trajectory_writer.writerow(
                {
                    "t_s": "{:.9f}".format(now - state["recording_start"]),
                    "wall_time_s": "{:.9f}".format(now),
                    "ros_time_s": "{:.9f}".format(ros_stamp(msg)),
                    "x": pos[0],
                    "y": pos[1],
                    "z": pos[2],
                    "body_yaw_rad": body_yaw,
                    "camera_pitch_rad": state["camera_state"][0],
                    "camera_yaw_rad": state["camera_state"][1],
                    "target_pitch_rad": state["camera_command"][0],
                    "target_yaw_rad": state["camera_command"][1],
                    "qx": qx,
                    "qy": qy,
                    "qz": qz,
                    "qw": qw,
                    "vx": msg.twist.twist.linear.x,
                    "vy": msg.twist.twist.linear.y,
                    "vz": msg.twist.twist.linear.z,
                    "path_length_m": state["path_length"],
                    "goal_x": None if goal is None else goal[0],
                    "goal_y": None if goal is None else goal[1],
                    "goal_z": None if goal is None else goal[2],
                    "goal_distance_m": goal_distance,
                    "center_obstacle_distance_m": center_distance,
                    "clearance_m": (
                        None if center_distance is None else center_distance - args.collision_radius
                    ),
                    "collision": int(state["collision_active"]),
                    "collision_active": int(state["collision_active"]),
                    "sim_collision_counter": state["counter_last"],
                }
            )
            state["odom_samples"] += 1
            if state["odom_samples"] % 100 == 0:
                trajectory_stream.flush()

    def camera_cb(kind):
        def callback(msg):
            now = time.time()
            magnitude = max(abs(float(msg.x)), abs(float(msg.y)))
            with lock:
                if kind == "command":
                    state["camera_command"] = [float(msg.x), float(msg.y)]
                    state["camera_command_seen"] = True
                    if not state["recording"]:
                        return
                    state["camera_command_samples"] += 1
                    state["camera_command_max"] = max(state["camera_command_max"], magnitude)
                    if magnitude > args.active_motion_threshold:
                        state["camera_command_nonzero"] += 1
                else:
                    state["camera_state"] = [float(msg.x), float(msg.y)]
                    state["camera_state_seen"] = True
                    if not state["recording"]:
                        return
                    state["camera_state_samples"] += 1
                    state["camera_state_max"] = max(state["camera_state_max"], magnitude)
                    if magnitude > args.active_motion_threshold:
                        state["camera_state_nonzero"] += 1
                camera_writer.writerow(
                    {
                        "wall_time_s": "{:.9f}".format(now),
                        "ros_time_s": "{:.9f}".format(rospy.Time.now().to_sec()),
                        "kind": kind,
                        "pitch_rad": msg.x,
                        "yaw_rad": msg.y,
                        "z": msg.z,
                    }
                )
                state["camera_samples"] += 1
                if state["camera_samples"] % 30 == 0:
                    camera_stream.flush()

        return callback

    def camera_stamped_cb(msg):
        now = time.time()
        vector = msg.vector
        stamp_s = header_stamp(msg)
        with lock:
            state["camera_state_stamped_seen"] = True
            if not state["recording"]:
                return
            sync_audit.add_state(stamp_s)
            camera_writer.writerow(
                {
                    "wall_time_s": "{:.9f}".format(now),
                    "ros_time_s": "" if stamp_s is None else "{:.9f}".format(stamp_s),
                    "kind": "state_stamped",
                    "pitch_rad": vector.x,
                    "yaw_rad": vector.y,
                    "z": vector.z,
                }
            )
            state["camera_samples"] += 1
            if state["camera_samples"] % 30 == 0:
                camera_stream.flush()

    def counter_cb(msg):
        now = time.time()
        value = int(msg.data)
        with lock:
            if state["counter_initial"] is None:
                state["counter_initial"] = value
            previous = state["counter_last"]
            state["counter_last"] = value
            if not state["recording"]:
                state["counter_initial"] = value
                return
            if previous is not None and value > previous:
                state["counter_active_until"] = now + args.collision_episode_gap

    def depth_cb(msg):
        now = time.time()
        contract = {
            "width": int(msg.width),
            "height": int(msg.height),
            "encoding": str(msg.encoding),
            "step": int(msg.step),
        }
        with lock:
            if state["depth_contract"] is None:
                state["depth_contract"] = contract
            elif state["depth_contract"] != contract:
                state["depth_contract"] = {"inconsistent": True, "latest": contract}
            if not state["recording"]:
                return
            sync_audit.add_depth(header_stamp(msg))
            state["depth_samples"] += 1
            if state["depth_first_wall_time"] is None:
                state["depth_first_wall_time"] = now
            state["depth_last_wall_time"] = now

    def save_plot_map(points):
        if args.plot_map_max_points <= 0:
            return
        count = min(points.shape[0], args.plot_map_max_points)
        if count < points.shape[0]:
            indices = np.linspace(0, points.shape[0] - 1, count, dtype=np.int64)
            selected = points[indices]
        else:
            selected = points
        with gzip.open(str(map_path), "wt", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x", "y", "z"])
            writer.writerows(selected.tolist())

    def map_cb(msg):
        with lock:
            if state["map_tree"] is not None:
                return
        points = np.asarray(
            [
                (float(point[0]), float(point[1]), float(point[2]))
                for point in point_cloud2.read_points(
                    msg, field_names=("x", "y", "z"), skip_nans=True
                )
            ],
            dtype=np.float64,
        )
        if points.size == 0:
            return
        tree = cKDTree(points) if cKDTree is not None else NumpyTree(points)
        canonical_points = np.asarray(points, dtype="<f4")
        map_sha256 = hashlib.sha256(canonical_points.tobytes(order="C")).hexdigest()
        bounds = {
            "min": canonical_points.min(axis=0).astype(float).tolist(),
            "max": canonical_points.max(axis=0).astype(float).tolist(),
        }
        with lock:
            if state["map_tree"] is not None:
                return
            state["map_tree"] = tree
            state["map_points"] = points
            state["map_topic_used"] = args.map_topic
            state["map_sha256"] = map_sha256
            state["map_bounds_xyz"] = bounds
        save_plot_map(points)
        rospy.loginfo("A/B monitor loaded %d map points from %s", points.shape[0], args.map_topic)

    monitor_wall_start = time.time()
    rospy.init_node("active_camera_ablation_{}_{}".format(args.arm, args.seed), anonymous=True)
    goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
    subscribers = [
        rospy.Subscriber(args.odom_topic, Odometry, odom_cb, queue_size=100, tcp_nodelay=True),
        rospy.Subscriber(args.camera_command_topic, Vector3, camera_cb("command"), queue_size=100),
        rospy.Subscriber(args.camera_state_topic, Vector3, camera_cb("state"), queue_size=100),
        rospy.Subscriber(
            args.camera_state_stamped_topic, Vector3Stamped, camera_stamped_cb,
            queue_size=100, tcp_nodelay=True,
        ),
        rospy.Subscriber(args.collision_counter_topic, Int32, counter_cb, queue_size=100),
        rospy.Subscriber(args.map_topic, PointCloud2, map_cb, queue_size=1),
        rospy.Subscriber(args.depth_topic, Image, depth_cb, queue_size=10, tcp_nodelay=True),
    ]

    rate = rospy.Rate(20)
    wait_start = time.time()
    while not rospy.is_shutdown() and state["pos"] is None and time.time() - wait_start < 30.0:
        rate.sleep()
    if state["pos"] is None:
        trajectory_stream.close()
        camera_stream.close()
        atomic_write_json(
            result_path,
            {
                "schema": SCHEMA_VERSION,
                "run_id": "seed_{:06d}_{}".format(args.seed, args.arm),
                "pair_id": "seed_{:06d}".format(args.seed),
                "seed": args.seed,
                "arm": args.arm,
                "camera_enabled": args.arm == ARM_ACTIVE,
                "status": "no_odometry",
                "success": False,
                "error": "No odometry received within 30 seconds",
            },
        )
        return 2

    map_start = time.time()
    while (
        not rospy.is_shutdown()
        and state["map_tree"] is None
        and time.time() - map_start < args.map_wait
    ):
        rate.sleep()
    if args.require_map and state["map_tree"] is None:
        trajectory_stream.close()
        camera_stream.close()
        atomic_write_json(
            result_path,
            {
                "schema": SCHEMA_VERSION,
                "run_id": "seed_{:06d}_{}".format(args.seed, args.arm),
                "pair_id": "seed_{:06d}".format(args.seed),
                "seed": args.seed,
                "arm": args.arm,
                "camera_enabled": args.arm == ARM_ACTIVE,
                "status": "no_map",
                "success": False,
                "error": "No map received from {}".format(args.map_topic),
            },
        )
        return 3

    readiness_start = time.time()
    while not rospy.is_shutdown() and time.time() - readiness_start < 30.0:
        with lock:
            sensor_ready = (
                state["depth_contract"] is not None
                and state["camera_state_seen"]
                and state["camera_state_stamped_seen"]
            )
        planner_ready = goal_pub.get_num_connections() > 0
        if sensor_ready and planner_ready:
            break
        rate.sleep()
    with lock:
        sensor_ready = (
            state["depth_contract"] is not None
            and state["camera_state_seen"]
            and state["camera_state_stamped_seen"]
        )
        depth_contract = state["depth_contract"]
    if not sensor_ready or goal_pub.get_num_connections() <= 0:
        for subscriber in subscribers:
            subscriber.unregister()
        trajectory_stream.close()
        camera_stream.close()
        atomic_write_json(
            result_path,
            {
                "schema": SCHEMA_VERSION,
                "run_id": "seed_{:06d}_{}".format(args.seed, args.arm),
                "pair_id": "seed_{:06d}".format(args.seed),
                "seed": args.seed,
                "arm": args.arm,
                "status": "not_ready",
                "success": False,
                "collision_detected": None,
                "depth_contract": depth_contract,
                "error": "Depth/legacy+stamped camera state or planner goal subscriber did not become ready",
            },
        )
        return 5

    expected_depth_contract = {"width": 160, "height": 192, "encoding": "32FC1", "step": 640}
    if depth_contract != expected_depth_contract:
        for subscriber in subscribers:
            subscriber.unregister()
        trajectory_stream.close()
        camera_stream.close()
        atomic_write_json(
            result_path,
            {
                "schema": SCHEMA_VERSION,
                "run_id": "seed_{:06d}_{}".format(args.seed, args.arm),
                "pair_id": "seed_{:06d}".format(args.seed),
                "seed": args.seed,
                "arm": args.arm,
                "status": "depth_contract_violation",
                "success": False,
                "collision_detected": None,
                "depth_contract": depth_contract,
                "expected_depth_contract": expected_depth_contract,
            },
        )
        return 6

    start = np.asarray(args.start, dtype=np.float64)
    end = np.asarray(args.end, dtype=np.float64)
    # repeats=1 is one outbound run. Additional repeats append a return and
    # another outbound leg, keeping one continuous simulator process.
    goals = [end]
    for _ in range(args.repeats - 1):
        goals.extend([start, end])

    def publish_goal(goal):
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "world"
        message.pose.position.x = float(goal[0])
        message.pose.position.y = float(goal[1])
        message.pose.position.z = float(goal[2])
        message.pose.orientation.w = 1.0
        goal_pub.publish(message)

    with lock:
        benchmark_start = time.time()
        state["recording_start"] = benchmark_start
        state["recording"] = True
        state["last_pos"] = None if state["pos"] is None else state["pos"].copy()
        state["path_length"] = 0.0
        state["min_center_distance"] = float("inf")
        state["collision_active"] = False
        state["collision_current"] = None
        state["collision_episodes"] = []
        state["counter_initial"] = state["counter_last"]
        state["counter_active_until"] = 0.0
        state["camera_command_max"] = 0.0
        state["camera_state_max"] = 0.0
        state["camera_command_nonzero"] = 0
        state["camera_state_nonzero"] = 0
        state["odom_samples"] = 0
        state["camera_samples"] = 0
        state["camera_command_samples"] = 0
        state["camera_state_samples"] = 0
        state["depth_samples"] = 0
        state["depth_first_wall_time"] = None
        state["depth_last_wall_time"] = None
        sync_audit.reset()
        state["max_altitude_m"] = float("-inf")
        benchmark_start_path = 0.0
    segments = []
    completed = True
    for index, goal in enumerate(goals):
        with lock:
            state["current_goal"] = goal
            segment_start_path = state["path_length"]
            current_pos = state["pos"].copy()
        segment_start = time.time()
        straight_distance = float(np.linalg.norm(goal - current_pos))
        last_publish = 0.0
        reached = False
        final_distance = None
        current_path = segment_start_path
        while not rospy.is_shutdown():
            now = time.time()
            if now - last_publish >= 0.5:
                publish_goal(goal)
                last_publish = now
            with lock:
                current_pos = None if state["pos"] is None else state["pos"].copy()
                current_path = state["path_length"]
            if current_pos is not None:
                final_distance = float(np.linalg.norm(current_pos - goal))
                if final_distance <= args.arrive_dist:
                    reached = True
                    break
            if now - segment_start > args.segment_timeout:
                completed = False
                break
            rate.sleep()
        duration = time.time() - segment_start
        segments.append(
            {
                "index": index,
                "goal": goal.tolist(),
                "reached": reached,
                "duration_s": duration,
                "path_length_m": max(0.0, current_path - segment_start_path),
                "straight_distance_m": straight_distance,
                "final_distance_m": final_distance,
            }
        )
        if not reached:
            break
        time.sleep(0.25)

    benchmark_end = time.time()
    with lock:
        finish_collision(benchmark_end, state["pos"])
        state["recording"] = False
        collision_episodes = list(state["collision_episodes"])
        counter_initial = state["counter_initial"]
        counter_last = state["counter_last"]
        min_center = state["min_center_distance"]
        command_max = state["camera_command_max"]
        camera_state_max = state["camera_state_max"]
        command_nonzero = state["camera_command_nonzero"]
        state_nonzero = state["camera_state_nonzero"]
        path_length = max(0.0, state["path_length"] - benchmark_start_path)
        map_points = state["map_points"]
        map_topic_used = state["map_topic_used"]
        odom_samples = state["odom_samples"]
        camera_samples = state["camera_samples"]
        depth_samples = state["depth_samples"]
        depth_first = state["depth_first_wall_time"]
        depth_last = state["depth_last_wall_time"]
        depth_contract = state["depth_contract"]
        map_sha256 = state["map_sha256"]
        map_bounds_xyz = state["map_bounds_xyz"]
        max_altitude = state["max_altitude_m"]
        sync_metrics = sync_audit.snapshot()

    for subscriber in subscribers:
        subscriber.unregister()

    trajectory_stream.flush()
    camera_stream.flush()
    trajectory_stream.close()
    camera_stream.close()

    collision_fields = [
        "index",
        "start_wall_time_s",
        "end_wall_time_s",
        "duration_s",
        "start_x",
        "start_y",
        "start_z",
        "end_x",
        "end_y",
        "end_z",
        "min_center_obstacle_distance_m",
        "min_clearance_m",
        "sources",
        "sim_counter_start",
        "sim_counter_end",
        "samples",
    ]
    with collision_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=collision_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(collision_episodes)

    counter_delta = None
    if counter_initial is not None and counter_last is not None:
        counter_delta = max(0, counter_last - counter_initial)
    collision_detected = bool(collision_episodes) or bool(counter_delta)
    depth_observed_fps = None
    if depth_samples > 1 and depth_first is not None and depth_last is not None and depth_last > depth_first:
        depth_observed_fps = (depth_samples - 1) / (depth_last - depth_first)
    success = completed and len(segments) == len(goals) and all(item["reached"] for item in segments)
    if args.arm == ARM_FIXED:
        camera_contract_ok = (
            state["camera_command_samples"] > 0
            and state["camera_state_samples"] > 0
            and command_max <= args.camera_zero_tolerance
            and camera_state_max <= args.camera_zero_tolerance
        )
    else:
        moved = (
            state["camera_state_samples"] > 0
            and camera_state_max >= args.active_motion_threshold
        )
        camera_contract_ok = moved or not args.require_active_motion

    sync_match_rate = sync_metrics["depth_state_match_rate"]
    sync_max_error = sync_metrics["depth_state_max_stamp_error_s"]
    camera_sync_contract_ok = (
        sync_metrics["camera_state_stamped_samples"] > 0
        and sync_metrics["depth_state_matched_samples"] > 0
        and sync_metrics["depth_invalid_stamp_samples"] == 0
        and sync_metrics["camera_state_invalid_stamp_samples"] == 0
        and sync_match_rate is not None
        and sync_match_rate >= args.min_depth_state_match_rate
        and sync_max_error is not None
        and sync_max_error <= args.camera_sync_slop
    )

    if not camera_contract_ok:
        status = "camera_contract_violation"
    elif not camera_sync_contract_ok:
        status = "camera_sync_violation"
    elif success and collision_detected:
        status = "collision"
    elif success:
        status = "ok"
    else:
        status = "incomplete"

    result = {
        "schema": SCHEMA_VERSION,
        "run_id": "seed_{:06d}_{}".format(args.seed, args.arm),
        "pair_id": "seed_{:06d}".format(args.seed),
        "seed": args.seed,
        "arm": args.arm,
        "camera_enabled": args.arm == ARM_ACTIVE,
        "status": status,
        "success": success,
        "collision_free_success": success and not collision_detected,
        "collision_detected": collision_detected,
        "collision_episode_count": len(collision_episodes),
        "collision_episodes": collision_episodes,
        "sim_collision_counter_initial": counter_initial,
        "sim_collision_counter_final": counter_last,
        "sim_collision_counter_delta": counter_delta,
        "duration_s": benchmark_end - benchmark_start,
        "path_length_m": path_length,
        "min_center_obstacle_distance_m": None if math.isinf(min_center) else min_center,
        "min_clearance_m": None if math.isinf(min_center) else min_center - args.collision_radius,
        "completed_segments": sum(item["reached"] for item in segments),
        "expected_segments": len(goals),
        "segments": segments,
        "camera_contract_ok": camera_contract_ok,
        "camera_sync_contract_ok": camera_sync_contract_ok,
        "camera_command_max_abs_rad": command_max,
        "camera_state_max_abs_rad": camera_state_max,
        "camera_command_nonzero_samples": command_nonzero,
        "camera_state_nonzero_samples": state_nonzero,
        "odom_samples": odom_samples,
        "camera_samples": camera_samples,
        "camera_command_samples": state["camera_command_samples"],
        "camera_state_samples": state["camera_state_samples"],
        "depth_contract": depth_contract,
        "depth_samples": depth_samples,
        "depth_observed_fps": depth_observed_fps,
        "camera_state_stamped_topic": args.camera_state_stamped_topic,
        "camera_sync_slop_s": args.camera_sync_slop,
        "minimum_depth_state_match_rate": args.min_depth_state_match_rate,
        **sync_metrics,
        "map_topic_used": map_topic_used,
        "map_point_count": None if map_points is None else int(map_points.shape[0]),
        "map_sha256": map_sha256,
        "map_bounds_xyz": map_bounds_xyz,
        "max_altitude_m": None if not math.isfinite(max_altitude) else max_altitude,
        "trajectory_csv": str(trajectory_path.relative_to(run_dir.parent.parent)),
        "camera_csv": str(camera_path.relative_to(run_dir.parent.parent)),
        "collision_csv": str(collision_path.relative_to(run_dir.parent.parent)),
        "map_csv_gz": (
            str(map_path.relative_to(run_dir.parent.parent)) if map_path.exists() else None
        ),
    }
    atomic_write_json(result_path, result)
    return 0 if status in ("ok", "collision") else 4


def make_parser():
    parser = argparse.ArgumentParser(
        description="Run a controlled Insight 9 active/fixed-camera ROS ablation."
    )
    parser.add_argument("--monitor", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--active-weight", default="", help="Active-camera checkpoint.")
    parser.add_argument("--fixed-weight", default="", help="Fixed-camera checkpoint.")
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("101,102,103,104,105"),
        help="Holdout map seeds (default: 101..105; do not reuse training seeds).",
    )
    parser.add_argument("--seed", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, default=ARM_ACTIVE, help=argparse.SUPPRESS)
    parser.add_argument("--maze-type", type=int, default=8)
    parser.add_argument("--map-viz-resolution", type=float, default=0.1)
    parser.add_argument("--velocity", type=float, default=3.0)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--sgm-time", type=float, default=1.4)
    parser.add_argument("--radius-min", type=float, default=None)
    parser.add_argument("--radius-max", type=float, default=None)
    parser.add_argument("--start", type=parse_vec3, default=parse_vec3("-20,0,2"))
    parser.add_argument("--end", type=parse_vec3, default=parse_vec3("20,0,2"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--arrive-dist", type=float, default=1.0)
    parser.add_argument("--segment-timeout", type=float, default=70.0)
    parser.add_argument("--run-timeout", type=float, default=None)
    parser.add_argument("--collision-radius", type=float, default=0.45)
    parser.add_argument("--collision-episode-gap", type=float, default=0.35)
    parser.add_argument("--camera-zero-tolerance", type=float, default=1e-4)
    parser.add_argument("--active-motion-threshold", type=float, default=0.01)
    parser.add_argument("--camera-sync-queue-size", type=int, default=30)
    parser.add_argument("--camera-sync-slop", type=float, default=0.03)
    parser.add_argument("--min-depth-state-match-rate", type=float, default=0.98)
    parser.add_argument("--require-active-motion", type=parse_bool, default=True)
    parser.add_argument("--require-map", type=parse_bool, default=True)
    parser.add_argument("--map-wait", type=float, default=15.0)
    parser.add_argument("--plot-map-max-points", type=int, default=50000)
    parser.add_argument("--fixed-yaw", action="store_true", help="Diagnostic only; applied to both arms.")
    parser.add_argument("--order", choices=("balanced", "active-first", "fixed-first", "random"), default="balanced")
    parser.add_argument("--order-seed", type=int, default=20260820)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--ros-master-port", type=int, default=11311)
    parser.add_argument("--master-timeout", type=float, default=15.0)
    parser.add_argument("--controller-delay", type=float, default=2.0)
    parser.add_argument("--sensor-delay", type=float, default=5.0)
    parser.add_argument("--planner-delay", type=float, default=5.0)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-dir", default="/tmp/yopo_active_camera_monitor", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-weight-check", action="store_true")
    parser.add_argument("--expected-checkpoint-epoch", type=int, default=50)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    parser.set_defaults(plot=True)
    parser.add_argument("--odom-topic", default="/sim/odom")
    parser.add_argument("--depth-topic", default="/depth_image")
    parser.add_argument("--map-topic", default="/mock_map")
    parser.add_argument("--control-topic", default="/so3_control/pos_cmd")
    parser.add_argument("--camera-command-topic", default="/yopo/camera/command")
    parser.add_argument("--camera-state-topic", default="/yopo/camera/orientation")
    parser.add_argument(
        "--camera-state-stamped-topic", default="/yopo/camera/orientation_stamped"
    )
    parser.add_argument("--collision-counter-topic", default="/yopo/collision_counter_total")
    return parser


def validate_args(args):
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.maze_type <= 0:
        raise ValueError("--maze-type must be positive")
    if args.map_viz_resolution <= 0:
        raise ValueError("--map-viz-resolution must be positive")
    if args.collision_radius <= 0:
        raise ValueError("--collision-radius must be positive")
    positive = {
        "--velocity": args.velocity,
        "--max-depth": args.max_depth,
        "--arrive-dist": args.arrive_dist,
        "--segment-timeout": args.segment_timeout,
        "--map-wait": args.map_wait,
        "--master-timeout": args.master_timeout,
        "--active-motion-threshold": args.active_motion_threshold,
        "--camera-sync-slop": args.camera_sync_slop,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
    if args.camera_sync_queue_size < 2:
        raise ValueError("--camera-sync-queue-size must be at least 2")
    if not (0.0 < args.min_depth_state_match_rate <= 1.0):
        raise ValueError("--min-depth-state-match-rate must be in (0,1]")
    if not (1024 <= args.ros_master_port <= 65535):
        raise ValueError("--ros-master-port must be in [1024,65535]")
    if not all(math.isfinite(float(value)) for value in args.start + args.end):
        raise ValueError("--start and --end must contain only finite values")
    if math.dist(args.start, args.end) <= 1e-6:
        raise ValueError("--start and --end must differ")
    if args.maze_type == 8 and (
        not math.isclose(args.start[2], 2.0, abs_tol=1e-6)
        or not math.isclose(args.end[2], 2.0, abs_tol=1e-6)
    ):
        raise ValueError("maze_type=8 benchmark is calibrated for start/end z=2 m")
    if not args.monitor and (not args.active_weight or not args.fixed_weight):
        raise ValueError("--active-weight and --fixed-weight are required")
    if not args.monitor:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", args.ros_master_port))
            except OSError as exc:
                raise ValueError(
                    "ROS master port {} is already in use; choose an isolated --ros-master-port".format(
                        args.ros_master_port
                    )
                ) from exc


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        validate_args(args)
        if args.monitor:
            return run_monitor(args)
        return run_orchestrator(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
