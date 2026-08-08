#!/usr/bin/env python3
"""Closed-loop acceptance test for the joystick depth/plan watchdog.

The flight stack must already be running in joystick mode with its joystick
device set to ``--fifo``.  This test owns both deterministic clear-depth input
and native Linux ``js_event`` injection.  It proves normal full-up flight,
stops only the four depth streams while continuing to inject full-up at 50 Hz,
checks the zero-command watchdog response, resumes depth, checks recovery, and
finally centers the stick and verifies that the vehicle stops.

The program deliberately refuses to run beside any other publisher on the
four depth topics.  It never launches, stops, or reconfigures the flight stack.
Every abnormal exit sends a best-effort high-rate centered-stick burst before
writing the JSON report.
"""

import argparse
import datetime as _datetime
import json
import math
import os
import stat
import struct
import sys
import threading
import time
from pathlib import Path

import rosgraph
import rospy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs.msg import Image


JS_EVENT = struct.Struct("IhBB")
JS_EVENT_AXIS = 0x02
TRAJECTORY_STATUS_EMPTY = 0

TOPIC_VDES = "/yopo/vdes_body"
TOPIC_CMD = "/so3_control/pos_cmd"
TOPIC_ODOM = "/sim/odom"
DEPTH_TOPICS = (
    "/depth_image_front",
    "/depth_image_left",
    "/depth_image_right",
    "/depth_image_back",
)
EXPECTED_TOPIC_TYPES = {
    TOPIC_VDES: "geometry_msgs/Vector3Stamped",
    TOPIC_CMD: "quadrotor_msgs/PositionCommand",
    TOPIC_ODOM: "nav_msgs/Odometry",
}


class SetupError(RuntimeError):
    """The existing stack or local fixture is not suitable for this test."""


class GateFailure(RuntimeError):
    """A measured closed-loop acceptance gate failed."""


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_float(value):
    return float(value) if _finite(value) else None


def _vector(values):
    result = [_safe_float(value) for value in values]
    return result if all(value is not None for value in result) else None


def _norm(values):
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _dot(lhs, rhs):
    return sum(float(a) * float(b) for a, b in zip(lhs, rhs))


def _subtract(lhs, rhs):
    return [float(a) - float(b) for a, b in zip(lhs, rhs)]


def _distance(lhs, rhs):
    return _norm(_subtract(lhs, rhs))


def _median(values):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def _percentile(values, percentile):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    alpha = position - lower
    return values[lower] * (1.0 - alpha) + values[upper] * alpha


def _sample_rate(samples):
    if len(samples) < 2:
        return 0.0
    elapsed = float(samples[-1]["t"]) - float(samples[0]["t"])
    return float(len(samples) - 1) / elapsed if elapsed > 0.0 else 0.0


def _rotation_body_to_world(q_xyzw):
    x, y, z, w = (float(value) for value in q_xyzw)
    magnitude = math.sqrt(x * x + y * y + z * z + w * w)
    if magnitude <= 1e-12 or not math.isfinite(magnitude):
        return None
    x, y, z, w = x / magnitude, y / magnitude, z / magnitude, w / magnitude
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _body_to_heading(vector_body, q_xyzw):
    """Undo full attitude, then undo yaw, yielding the operator heading frame."""
    rotation = _rotation_body_to_world(q_xyzw)
    if rotation is None:
        return None
    world = [
        sum(rotation[row][column] * vector_body[column] for column in range(3))
        for row in range(3)
    ]
    yaw = math.atan2(rotation[1][0], rotation[0][0])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return [
        cos_yaw * world[0] + sin_yaw * world[1],
        -sin_yaw * world[0] + cos_yaw * world[1],
        world[2],
    ]


def _world_to_heading(vector_world, q_xyzw):
    rotation = _rotation_body_to_world(q_xyzw)
    if rotation is None:
        return None
    yaw = math.atan2(rotation[1][0], rotation[0][0])
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return [
        cos_yaw * vector_world[0] + sin_yaw * vector_world[1],
        -sin_yaw * vector_world[0] + cos_yaw * vector_world[1],
        vector_world[2],
    ]


def _expected_mapping(raw_horizontal, raw_vertical, args):
    horizontal = max(-1.0, min(1.0, float(raw_horizontal) / args.axis_max))
    vertical = max(-1.0, min(1.0, float(raw_vertical) / args.axis_max))
    if args.invert_x:
        horizontal = -horizontal
    if args.invert_y:
        vertical = -vertical
    if args.swap_xy:
        body_x, body_y = vertical, horizontal
    else:
        body_x, body_y = horizontal, vertical
    magnitude = math.hypot(body_x, body_y)
    if magnitude <= args.deadzone:
        return [0.0, 0.0, 0.0]
    command = (min(magnitude, 1.0) - args.deadzone) / (1.0 - args.deadzone)
    scale = args.max_speed * command / magnitude
    return [body_x * scale, body_y * scale, 0.0]


def _open_fifo(path_text):
    path = Path(path_text).expanduser()
    created = False
    if path.exists():
        if not stat.S_ISFIFO(path.stat().st_mode):
            raise SetupError("Joystick path exists but is not a FIFO: %s" % path)
    else:
        if not path.parent.exists():
            raise SetupError("FIFO parent does not exist: %s" % path.parent)
        os.mkfifo(str(path), 0o600)
        created = True
    try:
        descriptor = os.open(str(path), os.O_RDWR | os.O_NONBLOCK)
    except BaseException:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return path, descriptor, created


def _raw_center_burst(fifo_fd, axis_x, axis_y, duration, frequency_hz):
    """Best-effort emergency center usable before the acceptance object exists."""
    started = time.monotonic()
    deadline = started + max(0.0, float(duration))
    period = 1.0 / max(1.0, float(frequency_hz))
    result = {
        "reason": "fallback center before acceptance initialization",
        "attempted": 0,
        "sent": 0,
        "first_latency_s": None,
        "elapsed_s": None,
        "completed": False,
        "error": None,
    }
    first = True
    while first or time.monotonic() < deadline:
        first = False
        result["attempted"] += 1
        event_time = int(time.monotonic() * 1000.0) & 0xFFFFFFFF
        payload = (
            JS_EVENT.pack(event_time, 0, JS_EVENT_AXIS, axis_x)
            + JS_EVENT.pack(event_time, 0, JS_EVENT_AXIS, axis_y)
        )
        try:
            written = os.write(fifo_fd, payload)
            if written != len(payload):
                raise OSError("short FIFO write: %d/%d" % (written, len(payload)))
        except BaseException as exc:
            result["error"] = "%s: %s" % (type(exc).__name__, exc)
            break
        result["sent"] += 1
        if result["first_latency_s"] is None:
            result["first_latency_s"] = time.monotonic() - started
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        try:
            time.sleep(min(period, remaining))
        except BaseException as exc:
            result["error"] = "%s: %s" % (type(exc).__name__, exc)
            break
    result["elapsed_s"] = time.monotonic() - started
    result["completed"] = result["sent"] > 0 and result["error"] is None
    return result


def _find_zero_command_run(samples, start_time, count, tolerance, max_gap):
    """Return the first contiguous zero-command run at/after ``start_time``."""
    run = []
    for sample in samples:
        if sample["t"] < start_time:
            continue
        is_zero = (
            sample["valid"]
            and sample["trajectory_flag"] == TRAJECTORY_STATUS_EMPTY
            and sample["velocity_norm"] <= tolerance
            and sample["acceleration_norm"] <= tolerance
        )
        if not is_zero or (run and sample["t"] - run[-1]["t"] > max_gap):
            run = []
        if is_zero:
            run.append(sample)
            if len(run) >= count:
                return {
                    "first_t": run[0]["t"],
                    "confirmed_t": run[-1]["t"],
                    "sample_count": len(run),
                    "max_velocity_norm_mps": max(item["velocity_norm"] for item in run),
                    "max_acceleration_norm_mps2": max(item["acceleration_norm"] for item in run),
                    "max_inter_sample_gap_s": max(
                        [run[index]["t"] - run[index - 1]["t"] for index in range(1, len(run))]
                        or [0.0]
                    ),
                }
    return None


def _positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return value


def _nonnegative_float(text):
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return value


class DepthWatchdogAcceptance:
    def __init__(self, args, fifo_path, fifo_fd, fifo_created):
        self.args = args
        self.fifo_path = fifo_path
        self.fifo_fd = fifo_fd
        self.fifo_created = fifo_created
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.samples = {TOPIC_VDES: [], TOPIC_CMD: [], TOPIC_ODOM: []}
        self.latest_quaternion = None
        self.latest_odom = None
        self.safety_origin = None
        self.message_totals = {TOPIC_VDES: 0, TOPIC_CMD: 0, TOPIC_ODOM: 0}
        self.observed_topic_types = {}
        self.checks = []
        self.phase_results = {}
        self.runtime_error = None
        self.failed_gate = None
        self.interrupted = False
        self.emergency_center_attempts = []
        self.normal_center_result = None
        self.depth_publishers = []
        self.depth_timer = None
        self.depth_payload = None
        self.depth_enabled = False
        self.depth_publish_cycles = 0
        self.depth_publish_messages = 0
        self.depth_publish_times = []
        self.depth_pause_t = None
        self.depth_resume_t = None
        self.depth_conflicts_before_start = []
        self.depth_runtime_conflicts = []
        self._last_ownership_check = 0.0
        self._last_injected_axes = [None, None]
        self._injected_event_pairs = 0
        self.injection_history = []
        self.expected_full = _expected_mapping(0, self.args.full_up_raw, self.args)
        expected_norm = _norm(self.expected_full)
        if expected_norm <= 0.0:
            raise SetupError("Configured full-up raw input maps to zero velocity")
        self.expected_unit = [value / expected_norm for value in self.expected_full]

        self.subscribers = [
            rospy.Subscriber(TOPIC_VDES, Vector3Stamped, self._vdes_cb, queue_size=200, tcp_nodelay=True),
            rospy.Subscriber(TOPIC_CMD, PositionCommand, self._cmd_cb, queue_size=300, tcp_nodelay=True),
            rospy.Subscriber(TOPIC_ODOM, Odometry, self._odom_cb, queue_size=400, tcp_nodelay=True),
        ]

    def _relative_time(self, absolute=None):
        return float((time.monotonic() if absolute is None else absolute) - self.started)

    def _append(self, topic, sample):
        sample["t"] = time.monotonic()
        with self.lock:
            self.samples[topic].append(sample)
            self.message_totals[topic] += 1

    def _vdes_cb(self, message):
        vector_body = _vector((message.vector.x, message.vector.y, message.vector.z))
        with self.lock:
            quaternion = None if self.latest_quaternion is None else list(self.latest_quaternion)
        vector_heading = None
        if vector_body is not None and quaternion is not None:
            vector_heading = _body_to_heading(vector_body, quaternion)
        self._append(
            TOPIC_VDES,
            {
                "valid": vector_body is not None and vector_heading is not None,
                "vector_body": vector_body,
                "vector_heading": vector_heading,
                "frame_id": message.header.frame_id,
            },
        )

    def _cmd_cb(self, message):
        velocity = _vector((message.velocity.x, message.velocity.y, message.velocity.z))
        acceleration = _vector((message.acceleration.x, message.acceleration.y, message.acceleration.z))
        position = _vector((message.position.x, message.position.y, message.position.z))
        with self.lock:
            quaternion = None if self.latest_quaternion is None else list(self.latest_quaternion)
        velocity_heading = None
        if velocity is not None and quaternion is not None:
            velocity_heading = _world_to_heading(velocity, quaternion)
        self._append(
            TOPIC_CMD,
            {
                "valid": velocity is not None and acceleration is not None and position is not None,
                "velocity_world": velocity,
                "velocity_heading": velocity_heading,
                "velocity_norm": _norm(velocity) if velocity is not None else None,
                "acceleration_norm": _norm(acceleration) if acceleration is not None else None,
                "trajectory_flag": int(message.trajectory_flag),
            },
        )

    def _odom_cb(self, message):
        position = _vector((message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z))
        velocity = _vector((message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z))
        quaternion = _vector(
            (
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
                message.pose.pose.orientation.w,
            )
        )
        velocity_heading = None
        valid = position is not None and velocity is not None and quaternion is not None
        if valid:
            valid = _rotation_body_to_world(quaternion) is not None
        if valid:
            velocity_heading = _world_to_heading(velocity, quaternion)
            valid = velocity_heading is not None
        now = time.monotonic()
        with self.lock:
            if valid:
                self.latest_quaternion = list(quaternion)
                if self.safety_origin is None:
                    self.safety_origin = list(position)
            self.latest_odom = {
                "received_t": now,
                "valid": bool(valid),
                "position": position,
                "velocity": velocity,
            }
        sample = {
            "valid": bool(valid),
            "position_world": position,
            "velocity_world": velocity,
            "velocity_heading": velocity_heading,
            "speed": _norm(velocity) if velocity is not None else None,
        }
        sample["t"] = now
        with self.lock:
            self.samples[TOPIC_ODOM].append(sample)
            self.message_totals[TOPIC_ODOM] += 1

    def _check_record(self, name, passed, measured, requirement):
        record = {
            "name": str(name),
            "passed": bool(passed),
            "measured": measured,
            "requirement": str(requirement),
        }
        self.checks.append(record)
        return record

    def _require(self, name, passed, measured, requirement):
        record = self._check_record(name, passed, measured, requirement)
        if not passed:
            self.failed_gate = record
            raise GateFailure(
                "%s failed: measured=%r requirement=%s" % (name, measured, requirement)
            )

    def _published_depth_nodes(self):
        try:
            publishers = rosgraph.Master(rospy.get_name()).getSystemState()[0]
        except Exception as exc:
            raise SetupError("Could not inspect ROS publisher ownership: %s" % exc) from exc
        state = dict(publishers)
        return {topic: list(state.get(topic, [])) for topic in DEPTH_TOPICS}

    def _check_no_depth_publishers_before_start(self):
        ownership = self._published_depth_nodes()
        conflicts = [
            {"topic": topic, "nodes": nodes}
            for topic, nodes in ownership.items()
            if nodes
        ]
        self.depth_conflicts_before_start = conflicts
        if conflicts:
            raise SetupError(
                "Refusing to mix deterministic depth with existing publishers: %s" % conflicts
            )

    def _check_depth_ownership_runtime(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_ownership_check < self.args.ownership_check_period:
            return
        self._last_ownership_check = now
        own_name = rospy.get_name()
        ownership = self._published_depth_nodes()
        conflicts = []
        for topic, nodes in ownership.items():
            others = [node for node in nodes if node != own_name]
            if others:
                conflicts.append({"topic": topic, "nodes": others})
        if conflicts:
            self.depth_runtime_conflicts.extend(conflicts)
            raise SetupError("Another depth publisher appeared during the test: %s" % conflicts)

    def start_depth(self):
        self._check_no_depth_publishers_before_start()
        pixel_count = self.args.depth_width * self.args.depth_height
        self.depth_payload = struct.pack(
            "<%df" % pixel_count,
            *([self.args.depth_value] * pixel_count),
        )
        self.depth_publishers = [rospy.Publisher(topic, Image, queue_size=1) for topic in DEPTH_TOPICS]
        with self.lock:
            self.depth_enabled = True
        self.depth_timer = rospy.Timer(rospy.Duration(1.0 / self.args.depth_hz), self._depth_timer_cb)
        deadline = time.monotonic() + self.args.publisher_registration_timeout
        while time.monotonic() < deadline:
            ownership = self._published_depth_nodes()
            if all(rospy.get_name() in ownership[topic] for topic in DEPTH_TOPICS):
                self._check_depth_ownership_runtime(force=True)
                return
            time.sleep(0.02)
        raise SetupError("Timed out waiting for all four owned depth publishers to register")

    def _depth_timer_cb(self, _event):
        try:
            with self.lock:
                if not self.depth_enabled:
                    return
                now = time.monotonic()
                stamp = rospy.Time.now()
                for topic, publisher in zip(DEPTH_TOPICS, self.depth_publishers):
                    message = Image()
                    message.header.stamp = stamp
                    message.header.frame_id = topic.rsplit("_", 1)[-1] + "_depth"
                    message.height = self.args.depth_height
                    message.width = self.args.depth_width
                    message.encoding = "32FC1"
                    message.is_bigendian = 0
                    message.step = self.args.depth_width * 4
                    message.data = self.depth_payload
                    publisher.publish(message)
                self.depth_publish_cycles += 1
                self.depth_publish_messages += len(DEPTH_TOPICS)
                self.depth_publish_times.append(now)
        except BaseException as exc:
            # The main loop turns this into a synchronous failure through the
            # absence of expected cycles; logging here must not kill Timer.
            rospy.logerr_throttle(1.0, "Depth fixture publication failed: %s", exc)

    def pause_depth(self):
        with self.lock:
            self.depth_enabled = False
            self.depth_pause_t = time.monotonic()
            return self.depth_pause_t

    def resume_depth(self):
        with self.lock:
            self.depth_resume_t = time.monotonic()
            self.depth_enabled = True
            return self.depth_resume_t

    def stop_depth(self):
        if self.depth_timer is not None:
            self.depth_timer.shutdown()
            self.depth_timer = None
        with self.lock:
            self.depth_enabled = False
        for publisher in self.depth_publishers:
            publisher.unregister()
        self.depth_publishers = []

    def inject_axes(self, raw_horizontal, raw_vertical):
        event_time = int(time.monotonic() * 1000.0) & 0xFFFFFFFF
        payload = (
            JS_EVENT.pack(event_time, int(raw_horizontal), JS_EVENT_AXIS, self.args.axis_x)
            + JS_EVENT.pack(event_time, int(raw_vertical), JS_EVENT_AXIS, self.args.axis_y)
        )
        offset = 0
        deadline = time.monotonic() + self.args.fifo_write_timeout
        while offset < len(payload):
            try:
                written = os.write(self.fifo_fd, payload[offset:])
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SetupError("FIFO stayed full while injecting joystick events")
                time.sleep(0.002)
                continue
            except OSError as exc:
                raise SetupError("Failed to write joystick FIFO: %s" % exc) from exc
            if written <= 0:
                raise SetupError("FIFO write returned zero bytes")
            offset += written
        self._last_injected_axes = [int(raw_horizontal), int(raw_vertical)]
        self._injected_event_pairs += 1
        self.injection_history.append(
            {
                "t": time.monotonic(),
                "axes": [int(raw_horizontal), int(raw_vertical)],
            }
        )

    def _check_safety(self):
        now = time.monotonic()
        with self.lock:
            latest = None if self.latest_odom is None else dict(self.latest_odom)
            origin = None if self.safety_origin is None else list(self.safety_origin)
        if latest is None:
            raise SetupError("No odometry is available during an active test phase")
        if now - latest["received_t"] > self.args.odom_timeout:
            raise GateFailure("Odometry became stale for %.3fs" % (now - latest["received_t"]))
        if not latest["valid"]:
            raise GateFailure("Odometry became non-finite")
        position, velocity = latest["position"], latest["velocity"]
        altitude = position[2]
        speed = _norm(velocity)
        distance = _distance(position, origin)
        if not self.args.safety_min_altitude <= altitude <= self.args.safety_max_altitude:
            raise GateFailure("Altitude %.3fm left the configured safety range" % altitude)
        if speed > self.args.safety_max_speed:
            raise GateFailure("Speed %.3fm/s exceeded the safety maximum" % speed)
        if distance > self.args.safety_max_distance:
            raise GateFailure("Position moved %.3fm outside the safety radius" % distance)

    def _drive(self, raw_horizontal, raw_vertical, duration, predicate=None):
        """Inject at the configured rate while servicing safety and ownership gates."""
        started = time.monotonic()
        deadline = started + float(duration)
        period = 1.0 / self.args.inject_hz
        next_injection = started
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._check_safety()
            self._check_depth_ownership_runtime()
            if predicate is not None:
                result = predicate()
                if result is not None:
                    return result
            now = time.monotonic()
            if now >= next_injection:
                self.inject_axes(raw_horizontal, raw_vertical)
                next_injection += period
                if next_injection < now - period:
                    next_injection = now + period
            time.sleep(min(0.005, max(0.0005, next_injection - time.monotonic())))
        if rospy.is_shutdown():
            raise SetupError("ROS shut down during the watchdog test")
        return predicate() if predicate is not None else None

    def _send_center_burst(self, duration, frequency_hz, reason):
        started = time.monotonic()
        deadline = started + max(0.0, float(duration))
        period = 1.0 / max(1.0, float(frequency_hz))
        result = {
            "reason": str(reason),
            "requested_duration_s": float(duration),
            "requested_frequency_hz": float(frequency_hz),
            "attempted": 0,
            "sent": 0,
            "first_latency_s": None,
            "elapsed_s": None,
            "completed": False,
            "error": None,
        }
        first = True
        while first or time.monotonic() < deadline:
            first = False
            result["attempted"] += 1
            try:
                self.inject_axes(0, 0)
            except BaseException as exc:  # never replace the original error
                result["error"] = "%s: %s" % (type(exc).__name__, exc)
                break
            result["sent"] += 1
            if result["first_latency_s"] is None:
                result["first_latency_s"] = time.monotonic() - started
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                time.sleep(min(period, remaining))
            except BaseException as exc:
                result["error"] = "%s: %s" % (type(exc).__name__, exc)
                break
        result["elapsed_s"] = time.monotonic() - started
        result["completed"] = result["sent"] > 0 and result["error"] is None
        return result

    def emergency_center(self, reason):
        result = self._send_center_burst(
            self.args.emergency_center_duration,
            self.args.inject_hz,
            reason,
        )
        self.emergency_center_attempts.append(result)
        return result

    def _samples_between(self, topic, start, end=None):
        with self.lock:
            source = list(self.samples[topic])
        return [sample for sample in source if sample["t"] >= start and (end is None or sample["t"] <= end)]

    def _vdes_metrics(self, start, end=None):
        samples = self._samples_between(TOPIC_VDES, start, end)
        valid = [sample for sample in samples if sample["valid"] and sample["vector_heading"] is not None]
        errors, cosines, magnitudes = [], [], []
        expected_norm = _norm(self.expected_full)
        for sample in valid:
            vector = sample["vector_heading"]
            magnitude = _norm(vector)
            magnitudes.append(magnitude)
            errors.append(_norm(_subtract(vector, self.expected_full)))
            cosines.append(_dot(vector, self.expected_unit) / magnitude if magnitude > 1e-9 else -1.0)
        return {
            "sample_count": len(samples),
            "valid_sample_count": len(valid),
            "rate_hz": _sample_rate(samples),
            "expected_heading_vector_mps": list(self.expected_full),
            "expected_magnitude_mps": expected_norm,
            "median_magnitude_mps": _median(magnitudes),
            "min_direction_cosine": min(cosines) if cosines else None,
            "median_error_mps": _median(errors),
            "p95_error_mps": _percentile(errors, 95.0),
        }

    def _zero_signal_metrics(self, start, end):
        vdes = self._samples_between(TOPIC_VDES, start, end)
        commands = self._samples_between(TOPIC_CMD, start, end)
        valid_vdes = [sample for sample in vdes if sample["valid"] and sample["vector_heading"] is not None]
        valid_cmd = [sample for sample in commands if sample["valid"]]
        vdes_norms = [_norm(sample["vector_heading"]) for sample in valid_vdes]
        zero_cmd = [
            sample
            for sample in valid_cmd
            if sample["trajectory_flag"] == TRAJECTORY_STATUS_EMPTY
            and sample["velocity_norm"] <= self.args.command_zero_tolerance
            and sample["acceleration_norm"] <= self.args.command_zero_tolerance
        ]
        return {
            "vdes_sample_count": len(vdes),
            "vdes_valid_sample_count": len(valid_vdes),
            "vdes_p99_norm_mps": _percentile(vdes_norms, 99.0),
            "command_sample_count": len(commands),
            "command_valid_sample_count": len(valid_cmd),
            "zero_empty_command_fraction": (
                float(len(zero_cmd)) / len(valid_cmd) if valid_cmd else None
            ),
            "command_velocity_p99_mps": _percentile(
                [sample["velocity_norm"] for sample in valid_cmd], 99.0
            ),
            "command_acceleration_p99_mps2": _percentile(
                [sample["acceleration_norm"] for sample in valid_cmd], 99.0
            ),
        }

    def _stopping_metrics(self, start, end):
        samples = self._samples_between(TOPIC_ODOM, start, end)
        valid = [sample for sample in samples if sample["valid"]]
        speeds = [sample["speed"] for sample in valid]
        drift = None
        if len(valid) >= 2:
            drift = _distance(valid[0]["position_world"], valid[-1]["position_world"])
        return {
            "sample_count": len(samples),
            "valid_sample_count": len(valid),
            "rate_hz": _sample_rate(samples),
            "median_speed_mps": _median(speeds),
            "p95_speed_mps": _percentile(speeds, 95.0),
            "drift_m": drift,
        }

    def wait_for_stack(self):
        deadline = time.monotonic() + self.args.topic_timeout
        last_topics = {}
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                last_topics = dict(rospy.get_published_topics(namespace=""))
            except rospy.ROSException:
                last_topics = {}
            wrong = {
                topic: {"observed": last_topics.get(topic), "expected": expected}
                for topic, expected in EXPECTED_TOPIC_TYPES.items()
                if topic in last_topics and last_topics[topic] != expected
            }
            if wrong:
                raise SetupError("ROS topic type mismatch: %s" % wrong)
            with self.lock:
                totals = dict(self.message_totals)
            if (
                all(topic in last_topics for topic in EXPECTED_TOPIC_TYPES)
                and totals[TOPIC_VDES] > 0
                and totals[TOPIC_CMD] > 0
                and totals[TOPIC_ODOM] > 0
            ):
                self.observed_topic_types = {
                    topic: last_topics[topic] for topic in EXPECTED_TOPIC_TYPES
                }
                return
            time.sleep(0.05)
        raise SetupError(
            "Timed out waiting %.1fs for existing stack: topics=%s samples=%s"
            % (self.args.topic_timeout, last_topics, self.message_totals)
        )

    def _recent_active_condition(self, active_start):
        now = time.monotonic()
        window_start = max(active_start, now - self.args.active_stable_window)
        vdes = self._vdes_metrics(window_start, now)
        commands = [sample for sample in self._samples_between(TOPIC_CMD, window_start, now) if sample["valid"]]
        odometry = [sample for sample in self._samples_between(TOPIC_ODOM, window_start, now) if sample["valid"]]
        cmd_projections = [
            _dot(sample["velocity_heading"], self.expected_unit)
            for sample in commands
            if sample["velocity_heading"] is not None
        ]
        odom_projections = [
            _dot(sample["velocity_heading"], self.expected_unit)
            for sample in odometry
            if sample["velocity_heading"] is not None
        ]
        ready = (
            vdes["valid_sample_count"] >= self.args.min_signal_samples
            and vdes["p95_error_mps"] is not None
            and vdes["p95_error_mps"] <= self.args.vdes_tolerance
            and len(cmd_projections) >= self.args.min_signal_samples
            and _median(cmd_projections) >= self.args.min_active_command_projection
            and len(odom_projections) >= self.args.min_signal_samples
            and _median(odom_projections) >= self.args.min_active_odom_projection
        )
        return (
            {
                "window_s": now - window_start,
                "vdes": vdes,
                "command_projection_median_mps": _median(cmd_projections),
                "command_sample_count": len(cmd_projections),
                "odom_projection_median_mps": _median(odom_projections),
                "odom_sample_count": len(odom_projections),
            }
            if ready
            else None
        )

    def execute(self):
        self.start_depth()
        self.wait_for_stack()

        unlock_start = time.monotonic()
        self._drive(0, 0, self.args.unlock_duration)
        unlock_end = time.monotonic()
        self.phase_results["unlock_center"] = {
            "duration_s": unlock_end - unlock_start,
            "injected_axes": [0, 0],
        }

        baseline_start = time.monotonic()
        self._drive(0, 0, self.args.baseline_duration)
        baseline_end = time.monotonic()
        baseline = self._zero_signal_metrics(
            max(baseline_start, baseline_end - self.args.center_window), baseline_end
        )
        self.phase_results["baseline_center"] = baseline
        self._require(
            "baseline_vdes_zero",
            baseline["vdes_p99_norm_mps"] is not None
            and baseline["vdes_p99_norm_mps"] <= self.args.vdes_zero_tolerance,
            baseline["vdes_p99_norm_mps"],
            "<= %.3f m/s" % self.args.vdes_zero_tolerance,
        )
        self._require(
            "baseline_command_zero",
            baseline["zero_empty_command_fraction"] is not None
            and baseline["zero_empty_command_fraction"] >= self.args.zero_command_fraction,
            baseline["zero_empty_command_fraction"],
            ">= %.3f EMPTY commands with near-zero velocity/acceleration"
            % self.args.zero_command_fraction,
        )

        active_start = time.monotonic()
        active = self._drive(
            0,
            self.args.full_up_raw,
            self.args.active_timeout,
            lambda: self._recent_active_condition(active_start),
        )
        self._require(
            "pre_pause_nonzero_flight",
            active is not None,
            active,
            "full-up vdes, nonzero PositionCommand, and odometry projection within %.3fs"
            % self.args.active_timeout,
        )
        active["latency_s"] = time.monotonic() - active_start
        self.phase_results["pre_pause_full_up"] = active

        pause_start = self.pause_depth()

        def stopped_predicate():
            with self.lock:
                commands = list(self.samples[TOPIC_CMD])
            return _find_zero_command_run(
                commands,
                pause_start,
                self.args.zero_command_samples,
                self.args.command_zero_tolerance,
                self.args.max_command_gap,
            )

        zero_run = self._drive(
            0,
            self.args.full_up_raw,
            self.args.stop_latency_max,
            stopped_predicate,
        )
        self._require(
            "watchdog_zero_command_observed",
            zero_run is not None,
            None,
            "%d consecutive EMPTY zero commands by %.3fs after depth pause"
            % (self.args.zero_command_samples, self.args.stop_latency_max),
        )
        zero_run["first_latency_s"] = zero_run["first_t"] - pause_start
        zero_run["confirmed_latency_s"] = zero_run["confirmed_t"] - pause_start
        zero_run["first_t"] = self._relative_time(zero_run["first_t"])
        zero_run["confirmed_t"] = self._relative_time(zero_run["confirmed_t"])
        self._require(
            "watchdog_not_premature",
            zero_run["first_latency_s"]
            >= self.args.expected_watchdog_timeout - self.args.watchdog_early_tolerance,
            zero_run["first_latency_s"],
            ">= expected timeout %.3fs - early tolerance %.3fs"
            % (self.args.expected_watchdog_timeout, self.args.watchdog_early_tolerance),
        )
        self._require(
            "watchdog_stop_latency",
            zero_run["confirmed_latency_s"] <= self.args.stop_latency_max,
            zero_run["confirmed_latency_s"],
            "<= %.3fs from depth pause to confirmed zero-command run"
            % self.args.stop_latency_max,
        )

        self._drive(0, self.args.full_up_raw, self.args.pause_hold_duration)
        resume_start = time.monotonic()
        pause_commands = [
            sample for sample in self._samples_between(TOPIC_CMD, pause_start, resume_start) if sample["valid"]
        ]
        commands_after_zero = [
            sample
            for sample in pause_commands
            if sample["t"] >= self.started + zero_run["first_t"]
        ]
        pause_zero_fraction = (
            float(
                sum(
                    sample["trajectory_flag"] == TRAJECTORY_STATUS_EMPTY
                    and sample["velocity_norm"] <= self.args.command_zero_tolerance
                    and sample["acceleration_norm"] <= self.args.command_zero_tolerance
                    for sample in commands_after_zero
                )
            )
            / len(commands_after_zero)
            if commands_after_zero
            else None
        )
        pause_vdes = self._vdes_metrics(pause_start, resume_start)
        pause_injections = [
            sample
            for sample in self.injection_history
            if pause_start <= sample["t"] <= resume_start
        ]
        full_up_injections = [
            sample
            for sample in pause_injections
            if sample["axes"] == [0, self.args.full_up_raw]
        ]
        injection_gaps = [
            full_up_injections[index]["t"] - full_up_injections[index - 1]["t"]
            for index in range(1, len(full_up_injections))
        ]
        if full_up_injections:
            injection_gaps.extend(
                [
                    full_up_injections[0]["t"] - pause_start,
                    resume_start - full_up_injections[-1]["t"],
                ]
            )
        pause_injection_rate = (
            float(len(full_up_injections) - 1)
            / (full_up_injections[-1]["t"] - full_up_injections[0]["t"])
            if len(full_up_injections) >= 2
            and full_up_injections[-1]["t"] > full_up_injections[0]["t"]
            else None
        )
        pause_injection_metrics = {
            "sample_count": len(pause_injections),
            "full_up_sample_count": len(full_up_injections),
            "wrong_axis_sample_count": len(pause_injections) - len(full_up_injections),
            "measured_rate_hz": pause_injection_rate,
            "max_coverage_gap_s": max(injection_gaps) if injection_gaps else None,
            "configured_rate_hz": self.args.inject_hz,
        }
        self._require(
            "full_up_injection_continues_while_depth_paused",
            len(full_up_injections) >= self.args.min_pause_injection_samples
            and len(pause_injections) == len(full_up_injections)
            and pause_injection_rate is not None
            and pause_injection_rate >= self.args.min_measured_inject_hz
            and max(injection_gaps) <= self.args.max_injection_gap,
            pause_injection_metrics,
            ">=%d full-up pairs, >=%.1fHz, max coverage gap<=%.3fs, no center/other axes"
            % (
                self.args.min_pause_injection_samples,
                self.args.min_measured_inject_hz,
                self.args.max_injection_gap,
            ),
        )
        self._require(
            "full_up_vdes_continues_while_depth_paused",
            pause_vdes["valid_sample_count"] >= self.args.min_pause_vdes_samples
            and pause_vdes["p95_error_mps"] is not None
            and pause_vdes["p95_error_mps"] <= self.args.vdes_tolerance
            and pause_vdes["min_direction_cosine"] is not None
            and pause_vdes["min_direction_cosine"] >= self.args.vdes_direction_cosine,
            pause_vdes,
            "full-up vdes remains within %.3fm/s and cosine >= %.3f"
            % (self.args.vdes_tolerance, self.args.vdes_direction_cosine),
        )
        self._require(
            "zero_command_held_until_depth_resume",
            pause_zero_fraction is not None
            and pause_zero_fraction >= self.args.zero_command_fraction,
            pause_zero_fraction,
            ">= %.3f after the first watchdog zero command"
            % self.args.zero_command_fraction,
        )
        with self.lock:
            depth_times_before_resume = list(self.depth_publish_times)
        during_pause_cycles = sum(
            pause_start < sample_time < resume_start for sample_time in depth_times_before_resume
        )
        self._require(
            "only_depth_stream_was_paused",
            during_pause_cycles == 0,
            during_pause_cycles,
            "0 clear-depth publication cycles while full-up injection and vdes continue",
        )
        self.phase_results["depth_paused_full_up"] = {
            "duration_s": resume_start - pause_start,
            "zero_command_run": zero_run,
            "zero_command_fraction_after_trigger": pause_zero_fraction,
            "vdes": pause_vdes,
            "joystick_injection": pause_injection_metrics,
            "depth_publish_cycles_during_pause": during_pause_cycles,
        }

        actual_resume = self.resume_depth()

        def recovery_predicate():
            commands = [
                sample
                for sample in self._samples_between(TOPIC_CMD, actual_resume)
                if sample["valid"] and sample["velocity_heading"] is not None
            ]
            recovering = [
                sample
                for sample in commands
                if _dot(sample["velocity_heading"], self.expected_unit)
                >= self.args.min_recovery_command_projection
            ]
            if not recovering:
                return None
            first_command = recovering[0]
            odometry = [
                sample
                for sample in self._samples_between(TOPIC_ODOM, first_command["t"])
                if sample["valid"] and sample["velocity_heading"] is not None
                and _dot(sample["velocity_heading"], self.expected_unit)
                >= self.args.min_recovery_odom_projection
            ]
            if not odometry:
                return None
            return {
                "first_nonzero_command_latency_s": first_command["t"] - actual_resume,
                "first_nonzero_flight_latency_s": odometry[0]["t"] - actual_resume,
                "command_projection_mps": _dot(first_command["velocity_heading"], self.expected_unit),
                "odom_projection_mps": _dot(odometry[0]["velocity_heading"], self.expected_unit),
            }

        recovery = self._drive(
            0,
            self.args.full_up_raw,
            self.args.recovery_latency_max,
            recovery_predicate,
        )
        self._require(
            "depth_resume_recovers_nonzero_plan_and_flight",
            recovery is not None,
            recovery,
            "nonzero command and odometry projection within %.3fs"
            % self.args.recovery_latency_max,
        )
        self._drive(0, self.args.full_up_raw, self.args.recovery_hold_duration)
        with self.lock:
            first_depth_after_resume = next(
                (sample_time for sample_time in self.depth_publish_times if sample_time >= actual_resume),
                None,
            )
        recovery["first_depth_cycle_latency_s"] = (
            first_depth_after_resume - actual_resume if first_depth_after_resume is not None else None
        )
        recovery["duration_after_detection_s"] = self.args.recovery_hold_duration
        self.phase_results["depth_resumed_full_up"] = recovery

        center_start = time.monotonic()
        self._drive(0, 0, self.args.center_duration)
        center_end = time.monotonic()
        center_window_start = max(center_start, center_end - self.args.center_window)
        zero_metrics = self._zero_signal_metrics(center_window_start, center_end)
        stopping = self._stopping_metrics(center_window_start, center_end)
        center = {"zero_signals": zero_metrics, "stopping": stopping}
        self.phase_results["final_center"] = center
        self._require(
            "final_center_vdes_zero",
            zero_metrics["vdes_p99_norm_mps"] is not None
            and zero_metrics["vdes_p99_norm_mps"] <= self.args.vdes_zero_tolerance,
            zero_metrics["vdes_p99_norm_mps"],
            "<= %.3f m/s" % self.args.vdes_zero_tolerance,
        )
        self._require(
            "final_center_command_zero",
            zero_metrics["zero_empty_command_fraction"] is not None
            and zero_metrics["zero_empty_command_fraction"] >= self.args.zero_command_fraction,
            zero_metrics["zero_empty_command_fraction"],
            ">= %.3f EMPTY zero commands" % self.args.zero_command_fraction,
        )
        self._require(
            "final_center_vehicle_stopped",
            stopping["median_speed_mps"] is not None
            and stopping["median_speed_mps"] <= self.args.stop_speed_median
            and stopping["p95_speed_mps"] <= self.args.stop_speed_p95
            and stopping["drift_m"] is not None
            and stopping["drift_m"] <= self.args.stop_drift,
            stopping,
            "median<=%.3fm/s p95<=%.3fm/s drift<=%.3fm"
            % (self.args.stop_speed_median, self.args.stop_speed_p95, self.args.stop_drift),
        )

    def build_report(self):
        checks_pass = bool(self.checks) and all(check["passed"] for check in self.checks)
        passed = checks_pass and self.runtime_error is None and self.failed_gate is None and not self.interrupted
        with self.lock:
            totals = dict(self.message_totals)
            depth_times = list(self.depth_publish_times)
        last_before_pause = None
        first_after_resume = None
        if self.depth_pause_t is not None:
            last_before_pause = next(
                (value for value in reversed(depth_times) if value <= self.depth_pause_t), None
            )
        if self.depth_resume_t is not None:
            first_after_resume = next(
                (value for value in depth_times if value >= self.depth_resume_t), None
            )
        status = "passed" if passed else ("error" if self.runtime_error else "failed")
        if self.interrupted:
            status = "interrupted"
        return {
            "schema_version": 1,
            "test": "YOPO joystick active-depth watchdog pause/recovery acceptance",
            "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "status": status,
            "passed": passed,
            "duration_s": self._relative_time(),
            "runtime_error": self.runtime_error,
            "failed_gate": self.failed_gate,
            "configuration": {
                "fifo": str(self.fifo_path),
                "axis_x": self.args.axis_x,
                "axis_y": self.args.axis_y,
                "full_up_raw": self.args.full_up_raw,
                "inject_hz": self.args.inject_hz,
                "expected_full_up_heading_mps": list(self.expected_full),
                "depth_topics": list(DEPTH_TOPICS),
                "depth_shape": [self.args.depth_height, self.args.depth_width],
                "depth_encoding": "32FC1",
                "depth_value_m": self.args.depth_value,
                "depth_hz": self.args.depth_hz,
                "expected_watchdog_timeout_s": self.args.expected_watchdog_timeout,
                "watchdog_early_tolerance_s": self.args.watchdog_early_tolerance,
                "stop_latency_max_s": self.args.stop_latency_max,
                "recovery_latency_max_s": self.args.recovery_latency_max,
                "zero_command_samples": self.args.zero_command_samples,
                "command_zero_tolerance": self.args.command_zero_tolerance,
            },
            "thresholds": {
                "min_measured_inject_hz": self.args.min_measured_inject_hz,
                "max_injection_gap_s": self.args.max_injection_gap,
                "min_pause_injection_samples": self.args.min_pause_injection_samples,
                "min_signal_samples": self.args.min_signal_samples,
                "min_pause_vdes_samples": self.args.min_pause_vdes_samples,
                "vdes_tolerance_mps": self.args.vdes_tolerance,
                "vdes_zero_tolerance_mps": self.args.vdes_zero_tolerance,
                "vdes_direction_cosine": self.args.vdes_direction_cosine,
                "min_active_command_projection_mps": self.args.min_active_command_projection,
                "min_active_odom_projection_mps": self.args.min_active_odom_projection,
                "min_recovery_command_projection_mps": self.args.min_recovery_command_projection,
                "min_recovery_odom_projection_mps": self.args.min_recovery_odom_projection,
                "zero_command_fraction": self.args.zero_command_fraction,
                "stop_speed_median_mps": self.args.stop_speed_median,
                "stop_speed_p95_mps": self.args.stop_speed_p95,
                "stop_drift_m": self.args.stop_drift,
                "odom_timeout_s": self.args.odom_timeout,
                "safety_altitude_range_m": [
                    self.args.safety_min_altitude,
                    self.args.safety_max_altitude,
                ],
                "safety_max_speed_mps": self.args.safety_max_speed,
                "safety_max_distance_m": self.args.safety_max_distance,
            },
            "fifo": {
                "path": str(self.fifo_path),
                "created_by_test": self.fifo_created,
                "kept_after_test": not (self.fifo_created and self.args.remove_created_fifo),
                "opened_flags": ["O_RDWR", "O_NONBLOCK"],
                "event_pairs_injected": self._injected_event_pairs,
                "last_injected_axes": list(self._last_injected_axes),
                "configured_rate_hz": self.args.inject_hz,
            },
            "topics": {
                "expected_types": dict(EXPECTED_TOPIC_TYPES),
                "observed_types": dict(self.observed_topic_types),
                "message_totals": totals,
            },
            "depth_fixture": {
                "conflicts_before_start": list(self.depth_conflicts_before_start),
                "runtime_conflicts": list(self.depth_runtime_conflicts),
                "publish_cycles": self.depth_publish_cycles,
                "published_messages": self.depth_publish_messages,
                "pause_t_from_test_start_s": (
                    self._relative_time(self.depth_pause_t) if self.depth_pause_t is not None else None
                ),
                "resume_t_from_test_start_s": (
                    self._relative_time(self.depth_resume_t) if self.depth_resume_t is not None else None
                ),
                "last_cycle_to_pause_s": (
                    self.depth_pause_t - last_before_pause
                    if self.depth_pause_t is not None and last_before_pause is not None
                    else None
                ),
                "resume_to_first_cycle_s": (
                    first_after_resume - self.depth_resume_t
                    if self.depth_resume_t is not None and first_after_resume is not None
                    else None
                ),
            },
            "phases": self.phase_results,
            "checks": list(self.checks),
            "emergency_center": {
                "attempted": bool(self.emergency_center_attempts),
                "attempts": list(self.emergency_center_attempts),
            },
            "final_safe_center": self.normal_center_result,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example (the planner must use the exact same FIFO):
  source /opt/ros/noetic/setup.bash
  source /workspace/YOPO/Controller/devel/setup.bash
  python3 tools/depth_watchdog_e2e_test.py \\
    --fifo /tmp/yopo-depth-watchdog.fifo \\
    --output /tmp/depth_watchdog_e2e.json

Do not run sensor_simulator_cuda or joystick_e2e_test.py at the same time.
This tool refuses any pre-existing or concurrently appearing depth publisher.
""",
    )
    parser.add_argument("--self-test", action="store_true", help="Run pure helper checks without ROS I/O.")
    parser.add_argument("--fifo", default="/tmp/yopo-depth-watchdog.fifo")
    parser.add_argument("--output", default="", help="Default: /tmp/depth_watchdog_e2e_<UTC>.json")
    parser.add_argument("--remove-created-fifo", action="store_true")

    parser.add_argument("--axis-x", type=int, default=0)
    parser.add_argument("--axis-y", type=int, default=1)
    parser.add_argument("--axis-max", type=_positive_float, default=32767.0)
    parser.add_argument("--full-up-raw", type=int, default=32767)
    parser.add_argument("--deadzone", type=_nonnegative_float, default=0.08)
    parser.add_argument("--invert-x", type=int, choices=(0, 1), default=1)
    parser.add_argument("--invert-y", type=int, choices=(0, 1), default=0)
    parser.add_argument("--swap-xy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-speed", type=_positive_float, default=6.0)
    parser.add_argument("--inject-hz", type=_positive_float, default=50.0)
    parser.add_argument("--min-measured-inject-hz", type=_positive_float, default=40.0)
    parser.add_argument("--max-injection-gap", type=_positive_float, default=0.06)
    parser.add_argument("--min-pause-injection-samples", type=int, default=12)
    parser.add_argument("--fifo-write-timeout", type=_positive_float, default=0.25)

    parser.add_argument("--depth-width", type=int, default=8)
    parser.add_argument("--depth-height", type=int, default=8)
    parser.add_argument("--depth-value", type=_positive_float, default=4.0)
    parser.add_argument("--depth-hz", type=_positive_float, default=15.0)
    parser.add_argument("--ownership-check-period", type=_positive_float, default=0.5)
    parser.add_argument("--publisher-registration-timeout", type=_positive_float, default=3.0)

    parser.add_argument("--topic-timeout", type=_positive_float, default=45.0)
    parser.add_argument("--unlock-duration", type=_positive_float, default=1.0)
    parser.add_argument("--baseline-duration", type=_positive_float, default=1.5)
    parser.add_argument("--active-timeout", type=_positive_float, default=6.0)
    parser.add_argument("--active-stable-window", type=_positive_float, default=0.4)
    parser.add_argument("--pause-hold-duration", type=_positive_float, default=0.25)
    parser.add_argument("--recovery-hold-duration", type=_positive_float, default=0.30)
    parser.add_argument("--center-duration", type=_positive_float, default=3.0)
    parser.add_argument("--center-window", type=_positive_float, default=1.0)
    parser.add_argument("--emergency-center-duration", type=_positive_float, default=0.35)

    parser.add_argument("--expected-watchdog-timeout", type=_positive_float, default=0.20)
    parser.add_argument("--watchdog-early-tolerance", type=_nonnegative_float, default=0.08)
    parser.add_argument("--stop-latency-max", type=_positive_float, default=0.35)
    parser.add_argument("--recovery-latency-max", type=_positive_float, default=0.60)
    parser.add_argument("--zero-command-samples", type=int, default=5)
    parser.add_argument("--max-command-gap", type=_positive_float, default=0.08)
    parser.add_argument("--command-zero-tolerance", type=_nonnegative_float, default=0.05)
    parser.add_argument("--zero-command-fraction", type=float, default=0.99)

    parser.add_argument("--min-signal-samples", type=int, default=5)
    parser.add_argument("--min-pause-vdes-samples", type=int, default=8)
    parser.add_argument("--vdes-tolerance", type=_nonnegative_float, default=0.12)
    parser.add_argument("--vdes-zero-tolerance", type=_nonnegative_float, default=0.05)
    parser.add_argument("--vdes-direction-cosine", type=float, default=0.98)
    parser.add_argument("--min-active-command-projection", type=_positive_float, default=0.50)
    parser.add_argument("--min-active-odom-projection", type=_positive_float, default=1.00)
    parser.add_argument("--min-recovery-command-projection", type=_positive_float, default=0.20)
    parser.add_argument("--min-recovery-odom-projection", type=_positive_float, default=0.10)
    parser.add_argument("--stop-speed-median", type=_nonnegative_float, default=0.35)
    parser.add_argument("--stop-speed-p95", type=_nonnegative_float, default=0.50)
    parser.add_argument("--stop-drift", type=_nonnegative_float, default=0.35)

    parser.add_argument("--odom-timeout", type=_positive_float, default=0.5)
    parser.add_argument("--safety-min-altitude", type=float, default=0.5)
    parser.add_argument("--safety-max-altitude", type=float, default=5.0)
    parser.add_argument("--safety-max-speed", type=_positive_float, default=8.0)
    parser.add_argument("--safety-max-distance", type=_positive_float, default=30.0)

    args = parser.parse_args(argv)
    args.invert_x = bool(args.invert_x)
    args.invert_y = bool(args.invert_y)
    args.swap_xy = bool(args.swap_xy)
    if args.self_test:
        return args
    if args.axis_x == args.axis_y or min(args.axis_x, args.axis_y) < 0 or max(args.axis_x, args.axis_y) > 255:
        parser.error("--axis-x and --axis-y must be distinct values in [0,255]")
    if args.axis_max > 32767.0:
        parser.error("--axis-max must be <=32767")
    if abs(args.full_up_raw) > args.axis_max:
        parser.error("absolute --full-up-raw must be <= --axis-max")
    if args.depth_width <= 0 or args.depth_height <= 0:
        parser.error("depth width/height must be positive")
    if args.zero_command_samples < 2:
        parser.error("--zero-command-samples must be >=2")
    if args.min_signal_samples < 1 or args.min_pause_vdes_samples < 1:
        parser.error("minimum sample counts must be positive")
    if args.min_pause_injection_samples < 2:
        parser.error("--min-pause-injection-samples must be >=2")
    if args.min_measured_inject_hz > args.inject_hz:
        parser.error("--min-measured-inject-hz must be <= --inject-hz")
    if not 0.0 <= args.deadzone < 1.0:
        parser.error("--deadzone must be in [0,1)")
    if not 0.0 <= args.zero_command_fraction <= 1.0:
        parser.error("--zero-command-fraction must be in [0,1]")
    if not -1.0 <= args.vdes_direction_cosine <= 1.0:
        parser.error("--vdes-direction-cosine must be in [-1,1]")
    if args.watchdog_early_tolerance >= args.expected_watchdog_timeout:
        parser.error("--watchdog-early-tolerance must be less than the expected timeout")
    if args.stop_latency_max <= args.expected_watchdog_timeout:
        parser.error("--stop-latency-max must be greater than expected watchdog timeout")
    if args.stop_speed_p95 < args.stop_speed_median:
        parser.error("--stop-speed-p95 must be >= --stop-speed-median")
    if not (_finite(args.safety_min_altitude) and _finite(args.safety_max_altitude)):
        parser.error("safety altitudes must be finite")
    if args.safety_max_altitude <= args.safety_min_altitude:
        parser.error("safety maximum altitude must exceed minimum altitude")
    return args


def _default_output_path():
    stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp/depth_watchdog_e2e_%s.json" % stamp)


def _write_report(report, output_path):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path


def _run_self_test(args):
    mapped = _expected_mapping(0, 32767, args)
    assert all(abs(a - b) < 1e-9 for a, b in zip(mapped, [args.max_speed, 0.0, 0.0]))
    samples = [
        {
            "t": 0.19,
            "valid": True,
            "trajectory_flag": 0,
            "velocity_norm": 1.0,
            "acceleration_norm": 0.0,
        }
    ] + [
        {
            "t": 0.21 + 0.02 * index,
            "valid": True,
            "trajectory_flag": 0,
            "velocity_norm": 0.0,
            "acceleration_norm": 0.0,
        }
        for index in range(5)
    ]
    run = _find_zero_command_run(samples, 0.0, 5, 0.05, 0.08)
    assert run is not None and abs(run["first_t"] - 0.21) < 1e-9
    broken = list(samples)
    broken[3] = dict(broken[3], t=0.50)
    assert _find_zero_command_run(broken, 0.0, 5, 0.05, 0.08) is None
    quaternion = [0.0, 0.0, 0.0, 1.0]
    assert _body_to_heading([2.0, 0.0, 0.0], quaternion) == [2.0, 0.0, 0.0]
    print("depth_watchdog_e2e_self_test=passed")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return _run_self_test(args)
    output_path = Path(args.output).expanduser() if args.output else _default_output_path()
    fifo_path = None
    fifo_fd = None
    fifo_created = False
    acceptance = None
    report = None
    fallback_center = None
    exit_code = 2
    try:
        fifo_path, fifo_fd, fifo_created = _open_fifo(args.fifo)
        rospy.init_node("yopo_depth_watchdog_e2e_test", anonymous=True)
        acceptance = DepthWatchdogAcceptance(args, fifo_path, fifo_fd, fifo_created)
        acceptance.execute()
        exit_code = 0
    except KeyboardInterrupt:
        if acceptance is not None:
            acceptance.interrupted = True
            acceptance.runtime_error = "Interrupted by user"
            acceptance.emergency_center(acceptance.runtime_error)
        elif fifo_fd is not None:
            fallback_center = _raw_center_burst(
                fifo_fd,
                args.axis_x,
                args.axis_y,
                args.emergency_center_duration,
                args.inject_hz,
            )
            report = {
                "schema_version": 1,
                "test": "YOPO joystick active-depth watchdog pause/recovery acceptance",
                "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
                "status": "interrupted",
                "passed": False,
                "runtime_error": "Interrupted by user",
                "emergency_center": {
                    "attempted": True,
                    "attempts": [fallback_center],
                },
            }
        exit_code = 130
    except BaseException as exc:
        message = "%s: %s" % (type(exc).__name__, exc)
        if acceptance is not None:
            if isinstance(exc, GateFailure) and acceptance.failed_gate is None:
                acceptance.failed_gate = {
                    "name": "asynchronous_runtime_gate",
                    "passed": False,
                    "measured": message,
                    "requirement": "no safety/runtime gate failure",
                }
            elif not isinstance(exc, GateFailure):
                acceptance.runtime_error = message
            acceptance.emergency_center(message)
        else:
            if fifo_fd is not None:
                fallback_center = _raw_center_burst(
                    fifo_fd,
                    args.axis_x,
                    args.axis_y,
                    args.emergency_center_duration,
                    args.inject_hz,
                )
            report = {
                "schema_version": 1,
                "test": "YOPO joystick active-depth watchdog pause/recovery acceptance",
                "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
                "status": "error",
                "passed": False,
                "runtime_error": message,
                "emergency_center": {
                    "attempted": fallback_center is not None,
                    "attempts": [fallback_center] if fallback_center is not None else [],
                },
            }
        print("depth_watchdog_e2e_test: %s" % message, file=sys.stderr)
        exit_code = 1 if isinstance(exc, GateFailure) else 2
    finally:
        if acceptance is not None and fifo_fd is not None:
            acceptance.normal_center_result = acceptance._send_center_burst(
                0.5, args.inject_hz, "final cleanup"
            )
            acceptance.stop_depth()
            report = acceptance.build_report()
            if report["passed"]:
                exit_code = 0
        if fifo_fd is not None:
            try:
                os.close(fifo_fd)
            except OSError:
                pass
        if fifo_created and args.remove_created_fifo and fifo_path is not None:
            try:
                fifo_path.unlink()
            except OSError as exc:
                print("warning: could not remove FIFO %s: %s" % (fifo_path, exc), file=sys.stderr)

    if report is None:
        report = {
            "schema_version": 1,
            "test": "YOPO joystick active-depth watchdog pause/recovery acceptance",
            "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "status": "error",
            "passed": False,
            "runtime_error": "No report was produced",
        }
    try:
        written = _write_report(report, output_path)
    except Exception as exc:
        print("Failed to write report %s: %s" % (output_path, exc), file=sys.stderr)
        return 2
    print("depth_watchdog_e2e_report=%s" % written)
    print("depth_watchdog_e2e_status=%s" % report.get("status"))
    if report.get("failed_gate"):
        print("depth_watchdog_e2e_failed_gate=%s" % report["failed_gate"].get("name"))
    if report.get("runtime_error"):
        print("depth_watchdog_e2e_error=%s" % report["runtime_error"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
