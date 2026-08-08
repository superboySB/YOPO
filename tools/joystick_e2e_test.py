#!/usr/bin/env python3
"""Deterministic FIFO-driven end-to-end acceptance test for YOPO joystick mode.

This program deliberately does not launch or stop roscore, the simulator, the
sensor, the controller, or the YOPO planner.  The existing planner must be
started in joystick mode with ``--joystick-device`` pointing at the same FIFO.

The test keeps the FIFO open with O_RDWR|O_NONBLOCK, writes native Linux
``js_event`` axis records, subscribes to the command and feedback topics, and
writes a machine-readable JSON report.  It always sends a short burst of
centered-stick events before closing the FIFO.
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

import rospy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs.msg import Image
from std_msgs.msg import Int32


JS_EVENT = struct.Struct("IhBB")
JS_EVENT_AXIS = 0x02

TOPIC_VDES = "/yopo/vdes_body"
TOPIC_CMD = "/so3_control/pos_cmd"
TOPIC_ODOM = "/sim/odom"
DEFAULT_COLLISION_TOPIC = "/yopo/collision_counter_total"
COLLISION_TOPIC_TYPE = "std_msgs/Int32"
CLEAR_DEPTH_TOPICS = (
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
TRAJECTORY_STATUS_EMPTY = 0


class AcceptanceSetupError(RuntimeError):
    pass


class AcceptanceSafetyAbort(RuntimeError):
    def __init__(self, message, emergency_center_attempted=False):
        super().__init__(message)
        self.emergency_center_attempted = bool(emergency_center_attempted)


def _finite(value):
    return math.isfinite(float(value))


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _safe_vector(values):
    return [_safe_float(value) for value in values]


def _valid_vector(values):
    return values is not None and all(value is not None for value in values)


def _norm(values):
    return math.sqrt(sum(float(value) * float(value) for value in values))


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


def _component_median(vectors):
    vectors = [vector for vector in vectors if _valid_vector(vector)]
    if not vectors:
        return None
    return [_median(vector[index] for vector in vectors) for index in range(3)]


def _sample_rate(samples):
    if len(samples) < 2:
        return 0.0
    elapsed = float(samples[-1]["t"]) - float(samples[0]["t"])
    if elapsed <= 0.0:
        return 0.0
    return float(len(samples) - 1) / elapsed


def _linear_slope(timed_values):
    """Return the least-squares value/time slope for finite ``(time, value)`` pairs."""
    pairs = [
        (float(sample_time), float(value))
        for sample_time, value in timed_values
        if _finite(sample_time) and _finite(value)
    ]
    if len(pairs) < 2:
        return None
    mean_time = sum(sample_time for sample_time, _value in pairs) / float(len(pairs))
    mean_value = sum(value for _sample_time, value in pairs) / float(len(pairs))
    denominator = sum((sample_time - mean_time) ** 2 for sample_time, _value in pairs)
    if denominator <= 1e-12:
        return None
    return (
        sum(
            (sample_time - mean_time) * (value - mean_value)
            for sample_time, value in pairs
        )
        / denominator
    )


def _stamp_seconds(message):
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    seconds = float(stamp.to_sec())
    return seconds if seconds > 0.0 and math.isfinite(seconds) else None


def _quaternion_to_rotation(q_xyzw):
    """Return the body-to-world rotation for an xyzw quaternion."""
    x, y, z, w = (float(value) for value in q_xyzw)
    magnitude = math.sqrt(x * x + y * y + z * z + w * w)
    if magnitude <= 1e-12 or not math.isfinite(magnitude):
        return None
    x /= magnitude
    y /= magnitude
    z /= magnitude
    w /= magnitude
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _wrap_angle(angle):
    """Wrap a finite angle to [-pi, pi)."""
    angle = float(angle)
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quaternion(q_xyzw):
    rotation = _quaternion_to_rotation(q_xyzw)
    if rotation is None:
        return None
    yaw = math.atan2(rotation[1][0], rotation[0][0])
    return yaw if math.isfinite(yaw) else None


def _world_to_body(vector_world, q_xyzw):
    rotation = _quaternion_to_rotation(q_xyzw)
    if rotation is None or not all(_finite(value) for value in vector_world):
        return None
    # v_b = R_wb.T * v_w
    return [
        rotation[0][column] * vector_world[0]
        + rotation[1][column] * vector_world[1]
        + rotation[2][column] * vector_world[2]
        for column in range(3)
    ]


def _heading_to_world(vector_heading, q_xyzw):
    """Rotate a planar heading-frame command into world without roll/pitch."""
    rotation = _quaternion_to_rotation(q_xyzw)
    if rotation is None or not all(_finite(value) for value in vector_heading):
        return None
    yaw = math.atan2(rotation[1][0], rotation[0][0])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return [
        cos_yaw * vector_heading[0] - sin_yaw * vector_heading[1],
        sin_yaw * vector_heading[0] + cos_yaw * vector_heading[1],
        vector_heading[2],
    ]


def _heading_to_body(vector_heading, q_xyzw):
    vector_world = _heading_to_world(vector_heading, q_xyzw)
    if vector_world is None:
        return None
    return _world_to_body(vector_world, q_xyzw)


def _expected_stick_mapping(raw_horizontal, raw_vertical, args):
    """Independent copy of the documented radial joystick mapping."""
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

    raw_magnitude = math.hypot(body_x, body_y)
    if raw_magnitude <= args.deadzone:
        return [0.0, 0.0, 0.0]
    command_magnitude = (min(raw_magnitude, 1.0) - args.deadzone) / (1.0 - args.deadzone)
    scale = args.max_speed * command_magnitude / raw_magnitude
    return [body_x * scale, body_y * scale, 0.0]


def _open_fifo(path_text):
    path = Path(path_text).expanduser()
    created = False
    if path.exists():
        if not stat.S_ISFIFO(path.stat().st_mode):
            raise AcceptanceSetupError("Joystick test path exists but is not a FIFO: %s" % path)
    else:
        if not path.parent.exists():
            raise AcceptanceSetupError("FIFO parent directory does not exist: %s" % path.parent)
        os.mkfifo(str(path), 0o600)
        created = True
    try:
        descriptor = os.open(str(path), os.O_RDWR | os.O_NONBLOCK)
    except Exception:
        if created:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    return path, descriptor, created


def _check(name, passed, measured=None, requirement=None):
    return {
        "name": name,
        "passed": bool(passed),
        "measured": measured,
        "requirement": requirement,
    }


class JoystickAcceptance:
    def __init__(self, args, fifo_path, fifo_fd, fifo_created):
        self.args = args
        self.fifo_path = fifo_path
        self.fifo_fd = fifo_fd
        self.fifo_created = fifo_created
        self.lock = threading.Lock()
        self.current_phase = None
        self.phases = []
        self.latest_odom_quaternion = None
        self.latest_odom_safety = None
        self.reference_yaw = None
        self.yaw_observations = {"odom": [], "cmd": []}
        self.safety_origin_position = None
        self.safety_check_count = 0
        self.safety_violations = []
        self.emergency_center_results = []
        self.message_totals = {TOPIC_VDES: 0, TOPIC_CMD: 0, TOPIC_ODOM: 0}
        self.observed_topic_types = {}
        self.collision_topic = rospy.resolve_name(self.args.collision_topic)
        self.collision_observed_topic_type = None
        self.collision_baseline_count = None
        self.collision_final_count = None
        self.collision_max_count = None
        self.collision_sample_count = 0
        self.collision_invalid_sample_count = 0
        self.phase_results = []
        self.comparisons = []
        self.aborted_reason = None
        self.runtime_error = None
        self.clear_depth_publish_count = 0
        self.clear_depth_publishers = []
        self.clear_depth_timer = None
        self.clear_depth_payload = None

        self.subscribers = [
            rospy.Subscriber(TOPIC_VDES, Vector3Stamped, self._vdes_callback, queue_size=100, tcp_nodelay=True),
            rospy.Subscriber(TOPIC_CMD, PositionCommand, self._cmd_callback, queue_size=200, tcp_nodelay=True),
            rospy.Subscriber(TOPIC_ODOM, Odometry, self._odom_callback, queue_size=300, tcp_nodelay=True),
            rospy.Subscriber(
                self.collision_topic,
                Int32,
                self._collision_callback,
                queue_size=100,
                tcp_nodelay=True,
            ),
        ]
        if self.args.publish_clear_depth:
            self._start_clear_depth_publisher()

    def _start_clear_depth_publisher(self):
        try:
            topics_before = dict(rospy.get_published_topics(namespace=""))
        except rospy.ROSException:
            topics_before = {}
        conflicts = [topic for topic in CLEAR_DEPTH_TOPICS if topic in topics_before]
        if conflicts:
            raise AcceptanceSetupError(
                "--publish-clear-depth refuses to mix with existing depth publishers: %s. "
                "Stop the real sensor or omit --publish-clear-depth." % conflicts
            )

        pixel_count = self.args.depth_width * self.args.depth_height
        self.clear_depth_payload = struct.pack(
            "<%df" % pixel_count,
            *([self.args.clear_depth_value] * pixel_count),
        )
        self.clear_depth_publishers = [
            rospy.Publisher(topic, Image, queue_size=1) for topic in CLEAR_DEPTH_TOPICS
        ]
        self.clear_depth_timer = rospy.Timer(
            rospy.Duration(1.0 / self.args.depth_hz),
            self._publish_clear_depth,
        )
        rospy.loginfo(
            "Publishing deterministic clear depth on four topics: %dx%d 32FC1, %.3fm at %.2fHz",
            self.args.depth_width,
            self.args.depth_height,
            self.args.clear_depth_value,
            self.args.depth_hz,
        )

    def _publish_clear_depth(self, timer_event):
        del timer_event
        stamp = rospy.Time.now()
        try:
            for topic, publisher in zip(CLEAR_DEPTH_TOPICS, self.clear_depth_publishers):
                message = Image()
                message.header.stamp = stamp
                message.header.frame_id = topic.rsplit("_", 1)[-1] + "_depth"
                message.height = self.args.depth_height
                message.width = self.args.depth_width
                message.encoding = "32FC1"
                message.is_bigendian = 0
                message.step = self.args.depth_width * 4
                message.data = self.clear_depth_payload
                publisher.publish(message)
            self.clear_depth_publish_count += 1
        except Exception as exc:
            rospy.logerr_throttle(1.0, "Failed to publish clear-depth fixture: %s", exc)

    def stop_auxiliary_publishers(self):
        if self.clear_depth_timer is not None:
            self.clear_depth_timer.shutdown()
            self.clear_depth_timer = None
        for publisher in self.clear_depth_publishers:
            publisher.unregister()
        self.clear_depth_publishers = []

    def _append_sample(self, topic, sample):
        with self.lock:
            self.message_totals[topic] += 1
            phase = self.current_phase
            if phase is None:
                return
            sample["t"] = time.monotonic() - phase["_start_monotonic"]
            sample["ros_stamp"] = sample.get("ros_stamp")
            phase["samples"][topic].append(sample)

    def _vdes_callback(self, message):
        vector = _safe_vector((message.vector.x, message.vector.y, message.vector.z))
        with self.lock:
            quaternion = None if self.latest_odom_quaternion is None else list(self.latest_odom_quaternion)
        self._append_sample(
            TOPIC_VDES,
            {
                "vector_body": vector,
                "frame_id": message.header.frame_id,
                "quaternion_xyzw": quaternion,
                "valid": _valid_vector(vector),
                "ros_stamp": _stamp_seconds(message),
            },
        )

    def _odom_callback(self, message):
        q = _safe_vector(
            (
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
            )
        ) + [_safe_float(message.pose.pose.orientation.w)]
        position = _safe_vector(
            (message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z)
        )
        velocity_world = _safe_vector(
            (message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z)
        )
        velocity_body = None
        quaternion_valid = all(value is not None for value in q)
        yaw = _yaw_from_quaternion(q) if quaternion_valid else None
        if quaternion_valid and _valid_vector(velocity_world):
            velocity_body = _world_to_body(velocity_world, q)
            if velocity_body is not None:
                velocity_body = _safe_vector(velocity_body)
        valid = (
            _valid_vector(position)
            and _valid_vector(velocity_world)
            and _valid_vector(velocity_body)
            and quaternion_valid
            and yaw is not None
        )
        received_monotonic = time.monotonic()
        with self.lock:
            if quaternion_valid and yaw is not None:
                self.latest_odom_quaternion = list(q)
                if self.reference_yaw is None:
                    self.reference_yaw = float(yaw)
            self.yaw_observations["odom"].append(
                {
                    "yaw": _safe_float(yaw) if yaw is not None else None,
                    "valid": yaw is not None,
                    "received_monotonic": received_monotonic,
                }
            )
            self.latest_odom_safety = {
                "received_monotonic": received_monotonic,
                "position_world": list(position) if position is not None else None,
                "velocity_world": list(velocity_world) if velocity_world is not None else None,
                "valid": valid,
            }
            if valid and self.safety_origin_position is None:
                self.safety_origin_position = list(position)
        self._append_sample(
            TOPIC_ODOM,
            {
                "position_world": position,
                "velocity_world": velocity_world,
                "velocity_body": velocity_body,
                "quaternion_xyzw": q,
                "yaw": _safe_float(yaw) if yaw is not None else None,
                "valid": valid,
                "ros_stamp": _stamp_seconds(message),
            },
        )

    def _cmd_callback(self, message):
        velocity_world = _safe_vector((message.velocity.x, message.velocity.y, message.velocity.z))
        acceleration_world = _safe_vector(
            (message.acceleration.x, message.acceleration.y, message.acceleration.z)
        )
        position_world = _safe_vector((message.position.x, message.position.y, message.position.z))
        with self.lock:
            quaternion = None if self.latest_odom_quaternion is None else list(self.latest_odom_quaternion)
        velocity_body = None
        if quaternion is not None and _valid_vector(velocity_world):
            velocity_body = _world_to_body(velocity_world, quaternion)
            if velocity_body is not None:
                velocity_body = _safe_vector(velocity_body)
        yaw = _safe_float(message.yaw)
        yaw_dot = _safe_float(message.yaw_dot)
        with self.lock:
            self.yaw_observations["cmd"].append(
                {
                    "yaw": yaw,
                    "yaw_dot": yaw_dot,
                    "valid": yaw is not None and yaw_dot is not None,
                    "received_monotonic": time.monotonic(),
                }
            )
        valid = (
            _valid_vector(velocity_world)
            and _valid_vector(velocity_body)
            and _valid_vector(acceleration_world)
            and _valid_vector(position_world)
            and yaw is not None
            and yaw_dot is not None
        )
        self._append_sample(
            TOPIC_CMD,
            {
                "position_world": position_world,
                "velocity_world": velocity_world,
                "velocity_body": velocity_body,
                "quaternion_xyzw": quaternion,
                "acceleration_world": acceleration_world,
                "yaw": yaw,
                "yaw_dot": yaw_dot,
                "trajectory_flag": int(message.trajectory_flag),
                "valid": valid,
                "ros_stamp": _stamp_seconds(message),
            },
        )

    def _collision_callback(self, message):
        """Record the cumulative simulator collision counter without resetting its baseline."""
        value = getattr(message, "data", None)
        valid = isinstance(value, int) and not isinstance(value, bool) and _finite(value)
        with self.lock:
            self.collision_sample_count += 1
            if not valid:
                self.collision_invalid_sample_count += 1
                return
            value = int(value)
            if self.collision_baseline_count is None:
                self.collision_baseline_count = value
            self.collision_final_count = value
            self.collision_max_count = (
                value
                if self.collision_max_count is None
                else max(self.collision_max_count, value)
            )
            # A successfully deserialized callback also proves the negotiated
            # ROS type, including when an optional topic appeared after stack
            # discovery completed.
            self.collision_observed_topic_type = COLLISION_TOPIC_TYPE

    def wait_for_stack(self):
        deadline = time.monotonic() + self.args.topic_timeout
        last_types = {}
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                last_types = dict(rospy.get_published_topics(namespace=""))
            except rospy.ROSException:
                last_types = {}
            wrong = {
                topic: (last_types.get(topic), expected)
                for topic, expected in EXPECTED_TOPIC_TYPES.items()
                if topic in last_types and last_types[topic] != expected
            }
            if wrong:
                raise AcceptanceSetupError("ROS topic type mismatch: %s" % wrong)
            collision_type = last_types.get(self.collision_topic)
            if (
                self.args.require_collision_topic
                and collision_type is not None
                and collision_type != COLLISION_TOPIC_TYPE
            ):
                raise AcceptanceSetupError(
                    "ROS collision topic type mismatch: %s publishes %s, expected %s"
                    % (self.collision_topic, collision_type, COLLISION_TOPIC_TYPE)
                )
            with self.lock:
                received = dict(self.message_totals)
                if collision_type is not None:
                    self.collision_observed_topic_type = collision_type
                collision_baseline = self.collision_baseline_count
            topics_present = all(topic in last_types for topic in EXPECTED_TOPIC_TYPES)
            data_present = received[TOPIC_VDES] > 0 and received[TOPIC_ODOM] > 0
            collision_ready = (
                not self.args.require_collision_topic
                or (
                    collision_type == COLLISION_TOPIC_TYPE
                    and isinstance(collision_baseline, int)
                    and not isinstance(collision_baseline, bool)
                    and _finite(collision_baseline)
                )
            )
            if topics_present and data_present and collision_ready:
                self.observed_topic_types = {topic: last_types[topic] for topic in EXPECTED_TOPIC_TYPES}
                return
            time.sleep(0.1)
        if rospy.is_shutdown():
            raise AcceptanceSetupError("ROS shut down while waiting for the flight stack")
        missing = [topic for topic in EXPECTED_TOPIC_TYPES if topic not in last_types]
        with self.lock:
            received = dict(self.message_totals)
            collision_baseline = self.collision_baseline_count
        collision_wait = None
        if self.args.require_collision_topic:
            collision_wait = {
                "topic": self.collision_topic,
                "observed_type": last_types.get(self.collision_topic),
                "expected_type": COLLISION_TOPIC_TYPE,
                "baseline": collision_baseline,
            }
        raise AcceptanceSetupError(
            "Timed out waiting %.1fs for existing ROS stack; missing_topics=%s received=%s collision=%s"
            % (self.args.topic_timeout, missing, received, collision_wait)
        )

    def inject_axes(self, raw_horizontal, raw_vertical):
        event_time = int(time.monotonic() * 1000.0) & 0xFFFFFFFF
        payload = (
            JS_EVENT.pack(event_time, int(raw_horizontal), JS_EVENT_AXIS, self.args.axis_x)
            + JS_EVENT.pack(event_time, int(raw_vertical), JS_EVENT_AXIS, self.args.axis_y)
        )
        offset = 0
        deadline = time.monotonic() + 0.25
        while offset < len(payload):
            try:
                written = os.write(self.fifo_fd, payload[offset:])
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise AcceptanceSetupError("FIFO stayed full while injecting joystick events")
                time.sleep(0.005)
                continue
            except OSError as exc:
                raise AcceptanceSetupError("Failed to write joystick FIFO: %s" % exc) from exc
            if written <= 0:
                raise AcceptanceSetupError("FIFO write returned zero bytes")
            offset += written

    def run_phase(self, name, kind, raw_horizontal, raw_vertical, duration, direction=None, level=None, source=None):
        phase = {
            "name": name,
            "kind": kind,
            "direction": direction,
            "level": level,
            "source": source,
            "raw_horizontal": int(raw_horizontal),
            "raw_vertical": int(raw_vertical),
            "expected_vdes_heading": _expected_stick_mapping(raw_horizontal, raw_vertical, self.args),
            "duration_requested_s": float(duration),
            "start_wall_time": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "start_ros_time": float(rospy.Time.now().to_sec()),
            "_start_monotonic": time.monotonic(),
            "injected_event_pairs": 0,
            "samples": {TOPIC_VDES: [], TOPIC_CMD: [], TOPIC_ODOM: []},
        }
        with self.lock:
            if self.current_phase is not None:
                raise AcceptanceSetupError("Internal error: overlapping test phases")
            self.current_phase = phase
            self.phases.append(phase)

        period = 1.0 / self.args.inject_hz
        next_write = phase["_start_monotonic"]
        deadline = phase["_start_monotonic"] + duration
        try:
            # A recovery/center phase inherits the preceding active kernel-axis
            # state until its first js_event is consumed.  Put that first zero
            # pair on the FIFO before checking a boundary which may already
            # have been crossed; otherwise the safety exception itself can
            # prevent the command which makes the vehicle stop.
            if kind in ("center", "unlock") and raw_horizontal == 0 and raw_vertical == 0:
                self.inject_axes(0, 0)
                phase["injected_event_pairs"] += 1
                next_write += period
            while time.monotonic() < deadline:
                if rospy.is_shutdown():
                    raise AcceptanceSetupError("ROS shut down during phase %s" % name)
                self._check_flight_safety(phase)
                now = time.monotonic()
                if now >= next_write:
                    self.inject_axes(raw_horizontal, raw_vertical)
                    phase["injected_event_pairs"] += 1
                    next_write += period
                    if next_write < now - period:
                        next_write = now + period
                time.sleep(min(0.01, max(0.001, next_write - time.monotonic())))
            self._check_flight_safety(phase)
        finally:
            phase["end_ros_time"] = float(rospy.Time.now().to_sec())
            phase["duration_actual_s"] = time.monotonic() - phase["_start_monotonic"]
            with self.lock:
                if self.current_phase is phase:
                    self.current_phase = None
        return phase

    def _check_flight_safety(self, phase):
        """Abort a phase immediately when the latest odometry leaves the safe envelope."""
        now = time.monotonic()
        with self.lock:
            sample = None if self.latest_odom_safety is None else dict(self.latest_odom_safety)
            origin = None if self.safety_origin_position is None else list(self.safety_origin_position)
            collision_baseline = self.collision_baseline_count
            collision_max = self.collision_max_count
            collision_final = self.collision_final_count
            collision_sample_count = self.collision_sample_count
            self.safety_check_count += 1

        if self.args.require_collision_topic:
            collision_increment = (
                collision_max - collision_baseline
                if collision_baseline is not None and collision_max is not None
                else None
            )
            collision_violation = None
            if collision_baseline is None:
                collision_violation = "missing_collision_baseline"
            elif collision_increment is not None and collision_increment > 0:
                collision_violation = "collision_counter_increased"
            if collision_violation is not None:
                record = {
                    "phase": phase["name"],
                    "phase_elapsed_s": now - phase["_start_monotonic"],
                    "violation": collision_violation,
                    "measured": collision_increment,
                    "requirement": "cumulative collision increment from baseline must equal 0",
                    "collision_topic": self.collision_topic,
                    "collision_baseline_count": collision_baseline,
                    "collision_final_count": collision_final,
                    "collision_max_count": collision_max,
                    "collision_sample_count": collision_sample_count,
                }
                with self.lock:
                    self.safety_violations.append(record)
                message = (
                    "Collision safety gate violated during %s: baseline=%s final=%s max=%s increment=%s"
                    % (
                        phase["name"],
                        collision_baseline,
                        collision_final,
                        collision_max,
                        collision_increment,
                    )
                )
                self.aborted_reason = message
                # Collision is an immediate physical-safety signal.  Center
                # the stick before unwinding the phase and reporting failure.
                self.send_emergency_center(message)
                raise AcceptanceSafetyAbort(message, emergency_center_attempted=True)

        violation = None
        measured = None
        requirement = None
        if sample is None:
            violation = "missing_odometry"
            requirement = "a finite odometry sample must be available during every phase"
        else:
            odom_age = now - float(sample["received_monotonic"])
            position = sample.get("position_world")
            velocity = sample.get("velocity_world")
            if odom_age > self.args.safety_odom_timeout:
                violation = "stale_odometry"
                measured = odom_age
                requirement = "<= %.6f s" % self.args.safety_odom_timeout
            elif not sample.get("valid") or not _valid_vector(position) or not _valid_vector(velocity):
                violation = "nonfinite_odometry"
                requirement = "finite position, velocity, and attitude"
            else:
                altitude = float(position[2])
                speed = _norm(velocity)
                position_distance = _distance(position, origin) if origin is not None else None
                if altitude < self.args.safety_min_altitude:
                    violation = "altitude_below_minimum"
                    measured = altitude
                    requirement = ">= %.6f m" % self.args.safety_min_altitude
                elif altitude > self.args.safety_max_altitude:
                    violation = "altitude_above_maximum"
                    measured = altitude
                    requirement = "<= %.6f m" % self.args.safety_max_altitude
                elif speed > self.args.safety_max_speed:
                    violation = "speed_above_maximum"
                    measured = speed
                    requirement = "<= %.6f m/s" % self.args.safety_max_speed
                elif position_distance is None:
                    violation = "missing_safety_origin"
                    requirement = "a finite initial odometry position"
                elif position_distance > self.args.safety_max_position_distance:
                    violation = "position_outside_boundary"
                    measured = position_distance
                    requirement = "<= %.6f m from initial position" % self.args.safety_max_position_distance

        if violation is None:
            return

        record = {
            "phase": phase["name"],
            "phase_elapsed_s": now - phase["_start_monotonic"],
            "violation": violation,
            "measured": _safe_float(measured) if measured is not None else None,
            "requirement": requirement,
            "position_world": (
                [
                    value if value is None else _safe_float(value)
                    for value in sample.get("position_world", [])
                ]
                if sample is not None and sample.get("position_world") is not None
                else None
            ),
            "velocity_world": (
                [
                    value if value is None else _safe_float(value)
                    for value in sample.get("velocity_world", [])
                ]
                if sample is not None and sample.get("velocity_world") is not None
                else None
            ),
        }
        with self.lock:
            self.safety_violations.append(record)
        message = "Flight safety boundary violated during %s: %s measured=%s required=%s" % (
            phase["name"],
            violation,
            record["measured"],
            requirement,
        )
        self.aborted_reason = message
        raise AcceptanceSafetyAbort(message)

    def collision_metrics(self):
        """Return collision observations and an optional strict acceptance gate."""
        with self.lock:
            observed_type = self.collision_observed_topic_type
            baseline = self.collision_baseline_count
            final = self.collision_final_count
            maximum = self.collision_max_count
            sample_count = self.collision_sample_count
            invalid_sample_count = self.collision_invalid_sample_count

        increment = (
            maximum - baseline
            if baseline is not None and maximum is not None
            else None
        )
        final_increment = (
            final - baseline
            if baseline is not None and final is not None
            else None
        )
        required = bool(self.args.require_collision_topic)

        def gated_check(name, condition, measured, requirement):
            check = _check(
                name,
                bool(condition) if required else True,
                measured,
                requirement if required else "optional observation only; not enforced",
            )
            check["condition_met"] = bool(condition)
            check["enforced"] = required
            return check

        baseline_valid = (
            isinstance(baseline, int)
            and not isinstance(baseline, bool)
            and _finite(baseline)
        )
        final_valid = (
            isinstance(final, int)
            and not isinstance(final, bool)
            and _finite(final)
        )
        maximum_valid = (
            isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and _finite(maximum)
        )
        checks = [
            gated_check(
                "collision_topic_present",
                observed_type is not None,
                observed_type,
                "topic must be published",
            ),
            gated_check(
                "collision_topic_type",
                observed_type == COLLISION_TOPIC_TYPE,
                observed_type,
                COLLISION_TOPIC_TYPE,
            ),
            gated_check(
                "collision_baseline_received",
                baseline_valid,
                baseline,
                "a finite integer baseline",
            ),
            gated_check(
                "collision_samples_present",
                sample_count > 0,
                sample_count,
                "> 0",
            ),
            gated_check(
                "collision_samples_valid",
                invalid_sample_count == 0,
                invalid_sample_count,
                "0 invalid samples",
            ),
            gated_check(
                "collision_final_count_valid",
                final_valid,
                final,
                "a finite integer",
            ),
            gated_check(
                "collision_max_count_valid",
                maximum_valid,
                maximum,
                "a finite integer",
            ),
            gated_check(
                "collision_count_unchanged",
                increment is not None and increment == 0,
                increment,
                "0 cumulative collisions after baseline",
            ),
        ]
        return {
            "topic": self.collision_topic,
            "expected_topic_type": COLLISION_TOPIC_TYPE,
            "observed_topic_type": observed_type,
            "required": required,
            "baseline_count": baseline,
            "final_count": final,
            "max_count": maximum,
            "increment_from_baseline": increment,
            "final_increment_from_baseline": final_increment,
            "sample_count": sample_count,
            "invalid_sample_count": invalid_sample_count,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }

    def _send_center_burst(self, duration, frequency_hz):
        """Best-effort zero-axis burst which never propagates an exception."""
        started = time.monotonic()
        duration = max(0.0, float(duration))
        frequency_hz = max(1.0, float(frequency_hz))
        deadline = started + duration
        period = 1.0 / frequency_hz
        result = {
            "requested_duration_s": duration,
            "requested_frequency_hz": frequency_hz,
            "event_pairs_attempted": 0,
            "event_pairs_sent": 0,
            "first_center_latency_s": None,
            "elapsed_s": 0.0,
            "completed": False,
            "error": None,
        }

        # Always make one immediate attempt, including for a zero-duration
        # caller.  Catch BaseException intentionally: this routine is used
        # while preserving an already-active KeyboardInterrupt or flight-safety
        # exception and must never replace it with a second exception.
        first_attempt = True
        while first_attempt or time.monotonic() < deadline:
            first_attempt = False
            result["event_pairs_attempted"] += 1
            try:
                self.inject_axes(0, 0)
            except BaseException as exc:  # noqa: B036 - preserve the original abort
                result["error"] = "%s: %s" % (type(exc).__name__, exc)
                break
            result["event_pairs_sent"] += 1
            if result["first_center_latency_s"] is None:
                result["first_center_latency_s"] = time.monotonic() - started

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            try:
                time.sleep(min(period, remaining))
            except BaseException as exc:  # preserve an already-active abort
                result["error"] = "%s: %s" % (type(exc).__name__, exc)
                break

        result["elapsed_s"] = time.monotonic() - started
        result["completed"] = result["event_pairs_sent"] > 0 and result["error"] is None
        return result

    def send_emergency_center(self, reason, duration=0.30, frequency_hz=50.0):
        """Send and record a high-rate center burst before abort analysis."""
        try:
            result = self._send_center_burst(duration, frequency_hz)
        except BaseException as exc:  # defensive: never mask the original abort
            result = {
                "requested_duration_s": float(duration),
                "requested_frequency_hz": float(frequency_hz),
                "event_pairs_attempted": 0,
                "event_pairs_sent": 0,
                "first_center_latency_s": None,
                "elapsed_s": 0.0,
                "completed": False,
                "error": "%s: %s" % (type(exc).__name__, exc),
            }
        result["reason"] = str(reason)
        result["recorded_at"] = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
        with self.lock:
            self.emergency_center_results.append(result)
        try:
            if result["completed"]:
                rospy.logwarn(
                    "Emergency center sent before abort analysis: %d event pairs, first latency %.6fs",
                    result["event_pairs_sent"],
                    result["first_center_latency_s"],
                )
            else:
                rospy.logerr("Emergency center burst was incomplete: %s", result["error"])
        except BaseException:
            pass
        return result

    def send_safe_center(self, duration=0.5):
        result = self._send_center_burst(duration, 20.0)
        if not result["completed"]:
            try:
                rospy.logwarn("Could not complete final centered-stick burst: %s", result["error"])
            except BaseException:
                pass
        return result

    @staticmethod
    def _window_samples(phase, topic, requested_window):
        duration = float(phase.get("duration_actual_s", phase["duration_requested_s"]))
        # Always discard at least the first half when a deliberately shortened
        # test duration is smaller than the requested steady-state window.
        window = min(float(requested_window), max(0.1, 0.5 * duration))
        start = max(0.0, duration - window)
        return [sample for sample in phase["samples"][topic] if float(sample["t"]) >= start], window

    def yaw_metrics(self, phase=None):
        """Check fixed-yaw behavior against the first finite odometry yaw."""
        with self.lock:
            reference_yaw = self.reference_yaw
            if phase is None:
                odom_samples = [dict(item) for item in self.yaw_observations["odom"]]
                cmd_samples = [dict(item) for item in self.yaw_observations["cmd"]]
            else:
                odom_samples = list(phase["samples"][TOPIC_ODOM])
                cmd_samples = list(phase["samples"][TOPIC_CMD])

        reference_valid = reference_yaw is not None and math.isfinite(float(reference_yaw))
        valid_odom = [
            sample
            for sample in odom_samples
            if sample.get("yaw") is not None and math.isfinite(float(sample["yaw"]))
        ]
        valid_cmd = [
            sample
            for sample in cmd_samples
            if sample.get("yaw") is not None
            and sample.get("yaw_dot") is not None
            and math.isfinite(float(sample["yaw"]))
            and math.isfinite(float(sample["yaw_dot"]))
        ]
        invalid_odom = len(odom_samples) - len(valid_odom)
        invalid_cmd = len(cmd_samples) - len(valid_cmd)

        actual_errors = (
            [abs(_wrap_angle(sample["yaw"] - reference_yaw)) for sample in valid_odom]
            if reference_valid
            else []
        )
        command_errors = (
            [abs(_wrap_angle(sample["yaw"] - reference_yaw)) for sample in valid_cmd]
            if reference_valid
            else []
        )
        command_yaw_dot = [abs(float(sample["yaw_dot"])) for sample in valid_cmd]
        max_actual_error = max(actual_errors) if actual_errors else None
        max_command_error = max(command_errors) if command_errors else None
        command_yaw_dot_p99 = _percentile(command_yaw_dot, 99.0)

        checks = [
            _check(
                "yaw_reference_available",
                reference_valid,
                _safe_float(reference_yaw) if reference_valid else None,
                "first finite odometry yaw",
            ),
            _check("actual_yaw_samples_present", bool(valid_odom), len(valid_odom), "> 0"),
            _check("actual_yaw_finite", invalid_odom == 0, invalid_odom, "0 invalid samples"),
            _check(
                "actual_yaw_drift",
                max_actual_error is not None
                and max_actual_error <= self.args.max_actual_yaw_drift,
                max_actual_error,
                "<= %.6f rad" % self.args.max_actual_yaw_drift,
            ),
            _check("command_yaw_samples_present", bool(valid_cmd), len(valid_cmd), "> 0"),
            _check("command_yaw_finite", invalid_cmd == 0, invalid_cmd, "0 invalid samples"),
            _check(
                "command_yaw_error",
                max_command_error is not None
                and max_command_error <= self.args.max_command_yaw_error,
                max_command_error,
                "<= %.6f rad" % self.args.max_command_yaw_error,
            ),
            _check(
                "command_yaw_dot_p99",
                command_yaw_dot_p99 is not None
                and command_yaw_dot_p99 <= self.args.max_command_yaw_dot_p99,
                command_yaw_dot_p99,
                "<= %.6f rad/s" % self.args.max_command_yaw_dot_p99,
            ),
        ]
        return {
            "scope": "overall" if phase is None else phase["name"],
            "reference_yaw_rad": _safe_float(reference_yaw) if reference_valid else None,
            "actual_sample_count": len(odom_samples),
            "actual_valid_sample_count": len(valid_odom),
            "actual_invalid_sample_count": invalid_odom,
            "actual_max_abs_drift_rad": max_actual_error,
            "command_sample_count": len(cmd_samples),
            "command_valid_sample_count": len(valid_cmd),
            "command_invalid_sample_count": invalid_cmd,
            "command_max_abs_yaw_error_rad": max_command_error,
            "command_abs_yaw_dot_p99_radps": command_yaw_dot_p99,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }

    def mapping_metrics(self, phase, window):
        samples, effective_window = self._window_samples(phase, TOPIC_VDES, window)
        valid = [sample for sample in samples if sample["valid"]]
        invalid_count = sum(1 for sample in phase["samples"][TOPIC_VDES] if not sample["valid"])
        expected_heading = phase["expected_vdes_heading"]
        expected_magnitude = _norm(expected_heading)
        paired = []
        missing_attitude_count = 0
        attitude_ages = []
        stamped_odom = [
            sample
            for sample in phase["samples"][TOPIC_ODOM]
            if sample["valid"] and sample.get("ros_stamp") is not None
        ]
        for sample in valid:
            if expected_magnitude <= 1e-9:
                expected_body = [0.0, 0.0, 0.0]
            else:
                quaternion = sample.get("quaternion_xyzw")
                stamp = sample.get("ros_stamp")
                if stamp is not None and stamped_odom:
                    prior = [odom for odom in stamped_odom if odom["ros_stamp"] <= stamp]
                    if prior:
                        odom = max(prior, key=lambda item: item["ros_stamp"])
                    else:
                        odom = min(stamped_odom, key=lambda item: abs(item["ros_stamp"] - stamp))
                    quaternion = odom["quaternion_xyzw"]
                    attitude_ages.append(abs(stamp - odom["ros_stamp"]))
                expected_body = (
                    _heading_to_body(expected_heading, quaternion) if quaternion is not None else None
                )
            if expected_body is None or not all(_finite(value) for value in expected_body):
                missing_attitude_count += 1
                continue
            paired.append((sample["vector_body"], expected_body))

        vectors = [measured for measured, _expected in paired]
        expected_vectors = [expected for _measured, expected in paired]
        median_vector = _component_median(vectors)
        median_expected_body = _component_median(expected_vectors)
        errors = [_distance(measured, expected) for measured, expected in paired]
        direction_cosines = [
            _dot(measured, expected) / (_norm(measured) * _norm(expected))
            for measured, expected in paired
            if _norm(measured) > 1e-9 and _norm(expected) > 1e-9
        ]
        direction_cosine = _median(direction_cosines)

        rate = _sample_rate(samples)
        median_error = _median(errors)
        p95_error = _percentile(errors, 95.0)
        checks = [
            _check("vdes_samples_present", bool(valid), len(valid), "> 0"),
            _check("vdes_finite", invalid_count == 0, invalid_count, "0 invalid samples"),
            _check(
                "vdes_attitude_available",
                missing_attitude_count == 0,
                missing_attitude_count,
                "0 samples missing attitude for heading-to-body conversion",
            ),
            _check("vdes_rate", rate >= self.args.min_vdes_hz, rate, ">= %.3f Hz" % self.args.min_vdes_hz),
            _check(
                "vdes_median_error",
                median_error is not None and median_error <= self.args.vdes_tolerance,
                median_error,
                "<= %.6f m/s" % self.args.vdes_tolerance,
            ),
            _check(
                "vdes_p95_error",
                p95_error is not None and p95_error <= self.args.vdes_tolerance,
                p95_error,
                "<= %.6f m/s" % self.args.vdes_tolerance,
            ),
        ]
        if expected_magnitude > 1e-9:
            checks.append(
                _check(
                    "vdes_direction",
                    direction_cosine is not None and direction_cosine >= self.args.vdes_direction_cos,
                    direction_cosine,
                    ">= %.6f cosine" % self.args.vdes_direction_cos,
                )
            )
        return {
            "window_s": effective_window,
            "sample_count": len(samples),
            "valid_sample_count": len(paired),
            "invalid_sample_count": invalid_count,
            "missing_attitude_count": missing_attitude_count,
            "attitude_age_median_s": _median(attitude_ages),
            "attitude_age_p95_s": _percentile(attitude_ages, 95.0),
            "rate_hz": rate,
            "expected_vector_heading": expected_heading,
            "median_expected_vector_body": median_expected_body,
            "median_vector_body": median_vector,
            "median_error_mps": median_error,
            "p95_error_mps": p95_error,
            "direction_cosine": direction_cosine,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }

    def altitude_metrics(self, phase=None):
        """Measure actual and commanded world-z against the first test frame.

        A per-phase result uses that phase's first finite odometry sample.  The
        overall result starts at the first active phase, which is exactly when
        the planner freezes its joystick altitude; startup/takeoff tracking is
        intentionally outside that locked-altitude interval.
        """
        if phase is None:
            first_active = next(
                (index for index, item in enumerate(self.phases) if item["kind"] == "active"),
                None,
            )
            selected = self.phases[first_active:] if first_active is not None else []
            odom_samples = [sample for item in selected for sample in item["samples"][TOPIC_ODOM]]
            cmd_samples = [sample for item in selected for sample in item["samples"][TOPIC_CMD]]
            scope = "overall_from_first_active"
        else:
            odom_samples = list(phase["samples"][TOPIC_ODOM])
            cmd_samples = list(phase["samples"][TOPIC_CMD])
            scope = phase["name"]

        actual_altitudes = [
            float(sample["position_world"][2])
            for sample in odom_samples
            if sample.get("valid") and _valid_vector(sample.get("position_world"))
        ]
        command_altitudes = [
            float(sample["position_world"][2])
            for sample in cmd_samples
            if sample.get("valid") and _valid_vector(sample.get("position_world"))
        ]
        reference_altitude = actual_altitudes[0] if actual_altitudes else None
        actual_errors = (
            [abs(altitude - reference_altitude) for altitude in actual_altitudes]
            if reference_altitude is not None
            else []
        )
        command_errors = (
            [abs(altitude - reference_altitude) for altitude in command_altitudes]
            if reference_altitude is not None
            else []
        )
        max_actual_drift = max(actual_errors) if actual_errors else None
        max_command_drift = max(command_errors) if command_errors else None
        requirement = "<= %.6f m from first finite odometry altitude" % self.args.max_altitude_drift
        checks = [
            _check(
                "altitude_reference_available",
                reference_altitude is not None,
                reference_altitude,
                "first finite odometry altitude",
            ),
            _check("altitude_odom_samples_present", bool(actual_altitudes), len(actual_altitudes), "> 0"),
            _check(
                "actual_altitude_drift",
                max_actual_drift is not None and max_actual_drift <= self.args.max_altitude_drift,
                max_actual_drift,
                requirement,
            ),
            _check("altitude_pos_cmd_samples_present", bool(command_altitudes), len(command_altitudes), "> 0"),
            _check(
                "command_altitude_drift",
                max_command_drift is not None and max_command_drift <= self.args.max_altitude_drift,
                max_command_drift,
                requirement,
            ),
        ]
        return {
            "scope": scope,
            "reference_altitude_m": reference_altitude,
            "actual_sample_count": len(actual_altitudes),
            "actual_min_altitude_m": min(actual_altitudes) if actual_altitudes else None,
            "actual_max_altitude_m": max(actual_altitudes) if actual_altitudes else None,
            "actual_max_abs_drift_m": max_actual_drift,
            "command_sample_count": len(command_altitudes),
            "command_min_altitude_m": min(command_altitudes) if command_altitudes else None,
            "command_max_altitude_m": max(command_altitudes) if command_altitudes else None,
            "command_max_abs_drift_m": max_command_drift,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }

    def active_motion_metrics(self, phase):
        odom_samples, odom_window = self._window_samples(phase, TOPIC_ODOM, self.args.stable_window)
        cmd_samples, cmd_window = self._window_samples(phase, TOPIC_CMD, self.args.stable_window)
        valid_odom = [sample for sample in odom_samples if sample["valid"]]
        valid_cmd = [sample for sample in cmd_samples if sample["valid"]]
        odom_invalid = sum(1 for sample in phase["samples"][TOPIC_ODOM] if not sample["valid"])
        cmd_invalid = sum(1 for sample in phase["samples"][TOPIC_CMD] if not sample["valid"])

        expected_heading = phase["expected_vdes_heading"]

        actual_pairs = []
        for sample in valid_odom:
            expected_world = _heading_to_world(expected_heading, sample["quaternion_xyzw"])
            if expected_world is None or _norm(expected_world) <= 1e-9:
                continue
            direction_world = [value / _norm(expected_world) for value in expected_world]
            actual_pairs.append((sample, direction_world))
        actual_vectors_world = [sample["velocity_world"] for sample, _direction in actual_pairs]
        actual_vectors_body = [sample["velocity_body"] for sample, _direction in actual_pairs]
        actual_speeds = [_norm(vector) for vector in actual_vectors_world]
        actual_projections = [
            _dot(sample["velocity_world"], direction) for sample, direction in actual_pairs
        ]
        actual_projection_slope = _linear_slope(
            (sample["t"], _dot(sample["velocity_world"], direction))
            for sample, direction in actual_pairs
        )
        actual_cosines = [
            _dot(sample["velocity_world"], direction) / _norm(sample["velocity_world"])
            for sample, direction in actual_pairs
            if _norm(sample["velocity_world"]) >= self.args.min_direction_speed
        ]
        actual_median_vector_body = _component_median(actual_vectors_body)
        actual_median_vector_world = _component_median(actual_vectors_world)
        actual_median_projection = _median(actual_projections)
        actual_projection_ratio = (
            actual_median_projection / _norm(expected_heading)
            if actual_median_projection is not None and _norm(expected_heading) > 1e-9
            else None
        )
        actual_positive_fraction = (
            sum(projection > 0.0 for projection in actual_projections) / float(len(actual_projections))
            if actual_projections
            else None
        )

        cmd_pairs = []
        for sample in valid_cmd:
            expected_world = _heading_to_world(expected_heading, sample["quaternion_xyzw"])
            if expected_world is None or _norm(expected_world) <= 1e-9:
                continue
            direction_world = [value / _norm(expected_world) for value in expected_world]
            cmd_pairs.append((sample, direction_world))
        cmd_vectors_world = [sample["velocity_world"] for sample, _direction in cmd_pairs]
        cmd_vectors_body = [sample["velocity_body"] for sample, _direction in cmd_pairs]
        cmd_speeds = [_norm(vector) for vector in cmd_vectors_world]
        cmd_projections = [
            _dot(sample["velocity_world"], direction) for sample, direction in cmd_pairs
        ]
        cmd_cosines = [
            _dot(sample["velocity_world"], direction) / _norm(sample["velocity_world"])
            for sample, direction in cmd_pairs
            if _norm(sample["velocity_world"]) >= self.args.min_direction_speed
        ]

        actual_rate = _sample_rate(odom_samples)
        cmd_rate = _sample_rate(cmd_samples)
        actual_direction_cosine = _median(actual_cosines)
        cmd_direction_cosine = _median(cmd_cosines)
        cmd_median_projection = _median(cmd_projections)
        cmd_odom_projection_error = (
            abs(cmd_median_projection - actual_median_projection)
            if cmd_median_projection is not None and actual_median_projection is not None
            else None
        )
        cmd_empty_fraction = (
            sum(sample["trajectory_flag"] == TRAJECTORY_STATUS_EMPTY for sample in valid_cmd)
            / float(len(valid_cmd))
            if valid_cmd
            else None
        )
        altitude = self.altitude_metrics(phase)
        checks = [
            _check("odom_samples_present", bool(valid_odom), len(valid_odom), "> 0"),
            _check("odom_finite", odom_invalid == 0, odom_invalid, "0 invalid samples"),
            _check("odom_rate", actual_rate >= self.args.min_odom_hz, actual_rate, ">= %.3f Hz" % self.args.min_odom_hz),
            _check(
                "actual_projection",
                actual_median_projection is not None
                and actual_median_projection >= self.args.min_actual_projection,
                actual_median_projection,
                ">= %.6f m/s" % self.args.min_actual_projection,
            ),
            _check(
                "actual_projection_ratio",
                actual_projection_ratio is not None
                and actual_projection_ratio >= self.args.min_actual_projection_ratio,
                actual_projection_ratio,
                ">= %.6f of expected stick-command magnitude"
                % self.args.min_actual_projection_ratio,
            ),
            _check(
                "actual_abs_projection_stable_slope",
                actual_projection_slope is not None
                and abs(actual_projection_slope) <= self.args.max_actual_projection_slope,
                abs(actual_projection_slope) if actual_projection_slope is not None else None,
                "<= %.6f m/s^2 over the final steady-state window"
                % self.args.max_actual_projection_slope,
            ),
            _check(
                "actual_direction",
                actual_direction_cosine is not None
                and actual_direction_cosine >= self.args.actual_direction_cos,
                actual_direction_cosine,
                ">= %.6f cosine" % self.args.actual_direction_cos,
            ),
            _check(
                "actual_positive_fraction",
                actual_positive_fraction is not None
                and actual_positive_fraction >= self.args.actual_positive_fraction,
                actual_positive_fraction,
                ">= %.6f" % self.args.actual_positive_fraction,
            ),
            _check("pos_cmd_samples_present", bool(valid_cmd), len(valid_cmd), "> 0"),
            _check("pos_cmd_finite", cmd_invalid == 0, cmd_invalid, "0 invalid samples"),
            _check("pos_cmd_rate", cmd_rate >= self.args.min_cmd_hz, cmd_rate, ">= %.3f Hz" % self.args.min_cmd_hz),
            _check(
                "pos_cmd_empty_fraction",
                cmd_empty_fraction is not None
                and cmd_empty_fraction >= self.args.min_empty_command_fraction,
                cmd_empty_fraction,
                ">= %.6f TRAJECTORY_STATUS_EMPTY" % self.args.min_empty_command_fraction,
            ),
        ] + list(altitude["checks"])
        if not self.args.skip_cmd_direction_check:
            checks.extend(
                [
                    _check(
                        "pos_cmd_projection",
                        cmd_median_projection is not None
                        and cmd_median_projection >= self.args.min_cmd_projection,
                        cmd_median_projection,
                        ">= %.6f m/s" % self.args.min_cmd_projection,
                    ),
                    _check(
                        "pos_cmd_direction",
                        cmd_direction_cosine is not None
                        and cmd_direction_cosine >= self.args.cmd_direction_cos,
                        cmd_direction_cosine,
                        ">= %.6f cosine" % self.args.cmd_direction_cos,
                    ),
                ]
            )
        if not self.args.skip_cmd_odom_projection_check:
            checks.append(
                _check(
                    "pos_cmd_odom_projection_error",
                    cmd_odom_projection_error is not None
                    and cmd_odom_projection_error <= self.args.max_cmd_odom_projection_error,
                    cmd_odom_projection_error,
                    "<= %.6f m/s absolute median projection difference"
                    % self.args.max_cmd_odom_projection_error,
                )
            )
        return {
            "odom_window_s": odom_window,
            "odom_sample_count": len(odom_samples),
            "odom_valid_sample_count": len(valid_odom),
            "odom_invalid_sample_count": odom_invalid,
            "odom_rate_hz": actual_rate,
            "expected_vector_heading": expected_heading,
            "actual_median_velocity_body": actual_median_vector_body,
            "actual_median_velocity_world": actual_median_vector_world,
            "actual_median_speed_mps": _median(actual_speeds),
            "actual_p95_speed_mps": _percentile(actual_speeds, 95.0),
            "actual_median_projection_mps": actual_median_projection,
            "actual_projection_ratio": actual_projection_ratio,
            "actual_projection_slope_mps2": actual_projection_slope,
            "actual_abs_projection_slope_mps2": (
                abs(actual_projection_slope) if actual_projection_slope is not None else None
            ),
            "actual_direction_cosine": actual_direction_cosine,
            "actual_positive_fraction": actual_positive_fraction,
            "pos_cmd_window_s": cmd_window,
            "pos_cmd_sample_count": len(cmd_samples),
            "pos_cmd_valid_sample_count": len(valid_cmd),
            "pos_cmd_invalid_sample_count": cmd_invalid,
            "pos_cmd_rate_hz": cmd_rate,
            "pos_cmd_median_velocity_body": _component_median(cmd_vectors_body),
            "pos_cmd_median_velocity_world": _component_median(cmd_vectors_world),
            "pos_cmd_median_speed_mps": _median(cmd_speeds),
            "pos_cmd_median_projection_mps": cmd_median_projection,
            "pos_cmd_odom_projection_error_mps": cmd_odom_projection_error,
            "pos_cmd_direction_cosine": cmd_direction_cosine,
            "pos_cmd_empty_fraction": cmd_empty_fraction,
            "altitude": altitude,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }

    def centered_motion_metrics(self, phase):
        samples, effective_window = self._window_samples(phase, TOPIC_ODOM, self.args.center_window)
        cmd_samples, cmd_window = self._window_samples(phase, TOPIC_CMD, self.args.center_window)
        valid = [sample for sample in samples if sample["valid"]]
        valid_cmd = [sample for sample in cmd_samples if sample["valid"]]
        invalid_count = sum(1 for sample in phase["samples"][TOPIC_ODOM] if not sample["valid"])
        cmd_invalid_count = sum(1 for sample in phase["samples"][TOPIC_CMD] if not sample["valid"])
        speeds = [_norm(sample["velocity_body"]) for sample in valid]
        positions = [sample["position_world"] for sample in valid]
        cmd_velocity_norms = [_norm(sample["velocity_world"]) for sample in valid_cmd]
        cmd_acceleration_norms = [_norm(sample["acceleration_world"]) for sample in valid_cmd]
        cmd_empty_fraction = (
            sum(sample["trajectory_flag"] == TRAJECTORY_STATUS_EMPTY for sample in valid_cmd)
            / float(len(valid_cmd))
            if valid_cmd
            else None
        )
        drift = _distance(positions[-1], positions[0]) if len(positions) >= 2 else None
        median_speed = _median(speeds)
        p95_speed = _percentile(speeds, 95.0)
        rate = _sample_rate(samples)
        cmd_rate = _sample_rate(cmd_samples)
        cmd_velocity_p99 = _percentile(cmd_velocity_norms, 99.0)
        cmd_acceleration_p99 = _percentile(cmd_acceleration_norms, 99.0)
        altitude = self.altitude_metrics(phase)
        checks = [
            _check("odom_samples_present", bool(valid), len(valid), "> 0"),
            _check("odom_finite", invalid_count == 0, invalid_count, "0 invalid samples"),
            _check("odom_rate", rate >= self.args.min_odom_hz, rate, ">= %.3f Hz" % self.args.min_odom_hz),
            _check(
                "center_median_speed",
                median_speed is not None and median_speed <= self.args.stop_speed,
                median_speed,
                "<= %.6f m/s" % self.args.stop_speed,
            ),
            _check(
                "center_p95_speed",
                p95_speed is not None and p95_speed <= self.args.stop_speed_p95,
                p95_speed,
                "<= %.6f m/s" % self.args.stop_speed_p95,
            ),
            _check(
                "center_drift",
                drift is not None and drift <= self.args.stop_drift,
                drift,
                "<= %.6f m" % self.args.stop_drift,
            ),
            _check("center_pos_cmd_samples_present", bool(valid_cmd), len(valid_cmd), "> 0"),
            _check(
                "center_pos_cmd_finite",
                cmd_invalid_count == 0,
                cmd_invalid_count,
                "0 invalid samples",
            ),
            _check(
                "center_pos_cmd_rate",
                cmd_rate >= self.args.min_cmd_hz,
                cmd_rate,
                ">= %.3f Hz" % self.args.min_cmd_hz,
            ),
            _check(
                "center_pos_cmd_empty_fraction",
                cmd_empty_fraction is not None
                and cmd_empty_fraction >= self.args.min_empty_command_fraction,
                cmd_empty_fraction,
                ">= %.6f TRAJECTORY_STATUS_EMPTY" % self.args.min_empty_command_fraction,
            ),
            _check(
                "center_pos_cmd_velocity_p99",
                cmd_velocity_p99 is not None
                and cmd_velocity_p99 <= self.args.center_command_tolerance,
                cmd_velocity_p99,
                "<= %.6f m/s" % self.args.center_command_tolerance,
            ),
            _check(
                "center_pos_cmd_acceleration_p99",
                cmd_acceleration_p99 is not None
                and cmd_acceleration_p99 <= self.args.center_command_tolerance,
                cmd_acceleration_p99,
                "<= %.6f m/s^2" % self.args.center_command_tolerance,
            ),
        ] + list(altitude["checks"])
        return {
            "window_s": effective_window,
            "sample_count": len(samples),
            "valid_sample_count": len(valid),
            "invalid_sample_count": invalid_count,
            "rate_hz": rate,
            "median_speed_mps": median_speed,
            "p95_speed_mps": p95_speed,
            "drift_m": drift,
            "pos_cmd_window_s": cmd_window,
            "pos_cmd_sample_count": len(cmd_samples),
            "pos_cmd_valid_sample_count": len(valid_cmd),
            "pos_cmd_invalid_sample_count": cmd_invalid_count,
            "pos_cmd_rate_hz": cmd_rate,
            "pos_cmd_empty_fraction": cmd_empty_fraction,
            "pos_cmd_velocity_p99_mps": cmd_velocity_p99,
            "pos_cmd_acceleration_p99_mps2": cmd_acceleration_p99,
            "altitude": altitude,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }

    def analyze_phases(self):
        results = []
        for phase in self.phases:
            mapping_window = self.args.center_window if phase["kind"] in ("center", "unlock") else self.args.stable_window
            mapping = self.mapping_metrics(phase, mapping_window)
            result = {
                "name": phase["name"],
                "kind": phase["kind"],
                "direction": phase["direction"],
                "level": phase["level"],
                "source": phase["source"],
                "raw_horizontal": phase["raw_horizontal"],
                "raw_vertical": phase["raw_vertical"],
                "expected_vdes_heading": phase["expected_vdes_heading"],
                "duration_requested_s": phase["duration_requested_s"],
                "duration_actual_s": phase.get("duration_actual_s"),
                "injected_event_pairs": phase["injected_event_pairs"],
                "sample_counts_total": {
                    topic: len(phase["samples"][topic]) for topic in (TOPIC_VDES, TOPIC_CMD, TOPIC_ODOM)
                },
                "mapping": mapping,
            }
            invalid_by_topic = {
                topic: sum(1 for sample in phase["samples"][topic] if not sample["valid"])
                for topic in (TOPIC_VDES, TOPIC_CMD, TOPIC_ODOM)
            }
            result["invalid_sample_counts_total"] = invalid_by_topic
            finite_checks = [
                _check(
                    "%s_all_finite" % topic.strip("/").replace("/", "_"),
                    invalid_by_topic[topic] == 0,
                    invalid_by_topic[topic],
                    "0 invalid samples during phase",
                )
                for topic in (TOPIC_VDES, TOPIC_CMD, TOPIC_ODOM)
            ]
            yaw = self.yaw_metrics(phase)
            result["yaw"] = yaw
            checks = finite_checks + list(mapping["checks"]) + list(yaw["checks"])
            if phase["kind"] == "active":
                motion = self.active_motion_metrics(phase)
                result["motion"] = motion
                checks.extend(motion["checks"])
            elif phase["kind"] == "center":
                stopped = self.centered_motion_metrics(phase)
                result["stopped"] = stopped
                checks.extend(stopped["checks"])
            result["checks"] = checks
            result["passed"] = all(check["passed"] for check in checks)
            if not self.args.summary_only:
                result["samples"] = phase["samples"]
            results.append(result)
        self.phase_results = results
        return results

    def _full_vs_half_comparison(self, direction, half, full):
        half_projection = (
            half.get("motion", {}).get("actual_median_projection_mps")
            if half is not None
            else None
        )
        full_projection = (
            full.get("motion", {}).get("actual_median_projection_mps")
            if full is not None
            else None
        )
        half_expected = (
            _norm(half.get("expected_vdes_heading", [])) if half is not None else None
        )
        full_expected = (
            _norm(full.get("expected_vdes_heading", [])) if full is not None else None
        )
        ratio = None
        if half_projection is not None and full_projection is not None and half_projection > 1e-9:
            ratio = full_projection / half_projection
        expected_ratio = None
        if half_expected is not None and full_expected is not None and half_expected > 1e-9:
            expected_ratio = full_expected / half_expected
        ratio_relative_error = (
            abs(ratio / expected_ratio - 1.0)
            if ratio is not None and expected_ratio is not None and expected_ratio > 1e-9
            else None
        )
        difference = (
            full_projection - half_projection
            if half_projection is not None and full_projection is not None
            else None
        )
        checks = [
            _check("half_and_full_present", half is not None and full is not None, None, "both phases present"),
            _check(
                "full_over_half_ratio",
                ratio is not None and ratio >= self.args.full_half_ratio,
                ratio,
                ">= %.6f" % self.args.full_half_ratio,
            ),
            _check(
                "full_over_half_margin",
                difference is not None and difference >= self.args.full_half_margin,
                difference,
                ">= %.6f m/s" % self.args.full_half_margin,
            ),
            _check(
                "full_over_half_expected_ratio_relative_error",
                ratio_relative_error is not None
                and ratio_relative_error <= self.args.max_full_half_ratio_relative_error,
                ratio_relative_error,
                "<= %.6f relative error from the deadzone-adjusted expected input ratio"
                % self.args.max_full_half_ratio_relative_error,
            ),
        ]
        return {
            "direction": direction,
            "half_expected_magnitude_mps": half_expected,
            "full_expected_magnitude_mps": full_expected,
            "expected_ratio": expected_ratio,
            "half_projection_mps": half_projection,
            "full_projection_mps": full_projection,
            "ratio": ratio,
            "ratio_relative_error": ratio_relative_error,
            "difference_mps": difference,
            "checks": checks,
            "passed": all(check["passed"] for check in checks),
        }

    def analyze_full_vs_half(self):
        by_key = {
            (result["direction"], result["level"]): result
            for result in self.phase_results
            if result["kind"] == "active"
        }
        comparisons = []
        for direction in self.args.directions:
            half = by_key.get((direction, "half"))
            full = by_key.get((direction, "full"))
            comparisons.append(self._full_vs_half_comparison(direction, half, full))
        self.comparisons = comparisons
        return comparisons

    def execute(self):
        self.wait_for_stack()

        unlock = self.run_phase("unlock_center", "unlock", 0, 0, self.args.unlock_duration)
        unlock_mapping = self.mapping_metrics(unlock, self.args.center_window)
        unlock_yaw = self.yaw_metrics(unlock)
        if not unlock_mapping["passed"] or not unlock_yaw["passed"]:
            self.aborted_reason = (
                "Centered-stick handshake or fixed-yaw gate failed.  The planner may not be using this FIFO "
                "in joystick mode; active flight phases were skipped for safety."
            )
            raise AcceptanceSafetyAbort(self.aborted_reason)

        baseline = self.run_phase("baseline_center", "center", 0, 0, self.args.baseline_duration)
        baseline_mapping = self.mapping_metrics(baseline, self.args.center_window)
        baseline_stopped = self.centered_motion_metrics(baseline)
        baseline_yaw = self.yaw_metrics(baseline)
        if not baseline_mapping["passed"] or not baseline_stopped["passed"] or not baseline_yaw["passed"]:
            self.aborted_reason = (
                "Baseline centered-stick phase did not produce zero vdes, fixed yaw, and a stopped vehicle; "
                "active flight phases were skipped for safety."
            )
            raise AcceptanceSafetyAbort(self.aborted_reason)

        raw_half = int(round(self.args.axis_max * self.args.half_fraction))
        raw_full = int(round(self.args.axis_max))
        direction_axes = {
            "up": (0, +1),
            "down": (0, -1),
            "left": (-1, 0),
            "right": (+1, 0),
        }
        active_specs = [
            (direction,) + direction_axes[direction]
            for direction in self.args.directions
        ]
        half_results = {}
        for direction, horizontal_sign, vertical_sign in active_specs:
            for level, magnitude in (("half", raw_half), ("full", raw_full)):
                phase_name = "%s_%s" % (direction, level)
                active_phase = self.run_phase(
                    phase_name,
                    "active",
                    horizontal_sign * magnitude,
                    vertical_sign * magnitude,
                    self.args.active_duration,
                    direction=direction,
                    level=level,
                )
                # Evaluate the complete active gate before issuing any further
                # nonzero input.  A failed active gate gets only the mandatory
                # centered recovery phase below, never another flight phase.
                active_mapping = self.mapping_metrics(active_phase, self.args.stable_window)
                active_motion = self.active_motion_metrics(active_phase)
                active_yaw = self.yaw_metrics(active_phase)
                active_snapshot = {
                    "expected_vdes_heading": active_phase["expected_vdes_heading"],
                    "motion": active_motion,
                }
                pair_comparison = None
                if level == "half":
                    half_results[direction] = active_snapshot
                else:
                    pair_comparison = self._full_vs_half_comparison(
                        direction,
                        half_results.get(direction),
                        active_snapshot,
                    )
                active_passed = (
                    active_mapping["passed"]
                    and active_motion["passed"]
                    and active_yaw["passed"]
                    and (pair_comparison is None or pair_comparison["passed"])
                )
                if not active_passed:
                    rospy.logerr(
                        "Active acceptance gate failed during %s; issuing centered recovery before abort.",
                        phase_name,
                    )

                center_phase = self.run_phase(
                    "center_after_%s" % phase_name,
                    "center",
                    0,
                    0,
                    self.args.center_duration,
                    source=phase_name,
                )
                center_mapping = self.mapping_metrics(center_phase, self.args.center_window)
                center_stopped = self.centered_motion_metrics(center_phase)
                center_yaw = self.yaw_metrics(center_phase)
                center_passed = center_mapping["passed"] and center_stopped["passed"] and center_yaw["passed"]

                if not active_passed or not center_passed:
                    failures = []
                    if not active_mapping["passed"]:
                        failures.append("active mapping")
                    if not active_motion["passed"]:
                        failures.append("active motion/command")
                    if not active_yaw["passed"]:
                        failures.append("active fixed yaw")
                    if pair_comparison is not None and not pair_comparison["passed"]:
                        failures.append("full/half proportional response")
                    if not center_mapping["passed"]:
                        failures.append("center mapping")
                    if not center_stopped["passed"]:
                        failures.append("center motion/command")
                    if not center_yaw["passed"]:
                        failures.append("center fixed yaw")
                    self.aborted_reason = (
                        "Immediate phase gate failed after %s (%s); remaining active phases were skipped for safety."
                        % (phase_name, ", ".join(failures))
                    )
                    raise AcceptanceSafetyAbort(self.aborted_reason)

    def build_report(self):
        self.analyze_phases()
        self.analyze_full_vs_half()
        overall_yaw = self.yaw_metrics()
        overall_altitude = self.altitude_metrics()
        collision_counter = self.collision_metrics()
        phases_passed = bool(self.phase_results) and all(result["passed"] for result in self.phase_results)
        comparisons_passed = bool(self.comparisons) and all(item["passed"] for item in self.comparisons)
        passed = (
            self.runtime_error is None
            and self.aborted_reason is None
            and phases_passed
            and comparisons_passed
            and overall_yaw["passed"]
            and overall_altitude["passed"]
            and collision_counter["passed"]
        )
        with self.lock:
            totals = dict(self.message_totals)
            safety_origin = (
                None if self.safety_origin_position is None else list(self.safety_origin_position)
            )
            safety_violations = [dict(item) for item in self.safety_violations]
            safety_check_count = self.safety_check_count
            emergency_center_results = [dict(item) for item in self.emergency_center_results]
        if safety_violations:
            passed = False
        return {
            "schema_version": 1,
            "test": "YOPO joystick FIFO ROS closed-loop acceptance",
            "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "status": "passed" if passed else ("error" if self.runtime_error else "failed"),
            "passed": passed,
            "aborted_reason": self.aborted_reason,
            "runtime_error": self.runtime_error,
            "assumptions": [
                "The existing YOPO planner was launched in joystick mode with this exact FIFO path.",
                "The planner uses axis/deadzone/inversion/speed settings matching this report.",
                "Stick x/y denotes a yaw/heading-frame horizontal command; /yopo/vdes_body is checked after attitude conversion.",
                "/sim/odom linear velocity is world-frame as emitted by this repository's simulator.",
                "/so3_control/pos_cmd vectors are world-frame and joystick mode keeps yaw fixed.",
                "Fixed yaw is measured against the first finite /sim/odom yaw observed by this tool.",
                "Locked altitude is measured from the first finite odometry frame of each phase and, overall, from the first active phase.",
                "The position safety boundary is measured from the first finite odometry position seen by this tool.",
                "This tool does not launch or stop any ROS flight-stack process.",
            ],
            "fifo": {
                "path": str(self.fifo_path),
                "created_by_test": self.fifo_created,
                "kept_after_test": not (self.fifo_created and self.args.remove_created_fifo),
            },
            "configuration": {
                "axis_x": self.args.axis_x,
                "axis_y": self.args.axis_y,
                "axis_max": self.args.axis_max,
                "half_fraction": self.args.half_fraction,
                "deadzone": self.args.deadzone,
                "invert_x": self.args.invert_x,
                "invert_y": self.args.invert_y,
                "swap_xy": self.args.swap_xy,
                "max_speed_mps": self.args.max_speed,
                "directions": list(self.args.directions),
                "unlock_duration_s": self.args.unlock_duration,
                "baseline_duration_s": self.args.baseline_duration,
                "active_duration_s": self.args.active_duration,
                "center_duration_s": self.args.center_duration,
                "stable_window_s": self.args.stable_window,
                "center_window_s": self.args.center_window,
                "inject_hz": self.args.inject_hz,
                "publish_clear_depth": self.args.publish_clear_depth,
                "clear_depth_width": self.args.depth_width if self.args.publish_clear_depth else None,
                "clear_depth_height": self.args.depth_height if self.args.publish_clear_depth else None,
                "clear_depth_value_m": self.args.clear_depth_value if self.args.publish_clear_depth else None,
                "clear_depth_hz": self.args.depth_hz if self.args.publish_clear_depth else None,
                "clear_depth_frame_count": self.clear_depth_publish_count,
                "collision_topic": self.collision_topic,
                "require_collision_topic": self.args.require_collision_topic,
            },
            "thresholds": {
                "vdes_tolerance_mps": self.args.vdes_tolerance,
                "vdes_direction_cos": self.args.vdes_direction_cos,
                "actual_direction_cos": self.args.actual_direction_cos,
                "actual_positive_fraction": self.args.actual_positive_fraction,
                "min_actual_projection_mps": self.args.min_actual_projection,
                "min_actual_projection_ratio": self.args.min_actual_projection_ratio,
                "max_actual_abs_projection_slope_mps2": self.args.max_actual_projection_slope,
                "cmd_direction_cos": self.args.cmd_direction_cos,
                "min_cmd_projection_mps": self.args.min_cmd_projection,
                "max_cmd_odom_projection_error_mps": self.args.max_cmd_odom_projection_error,
                "skip_cmd_odom_projection_check": self.args.skip_cmd_odom_projection_check,
                "min_empty_command_fraction": self.args.min_empty_command_fraction,
                "center_command_tolerance": self.args.center_command_tolerance,
                "full_half_ratio": self.args.full_half_ratio,
                "full_half_margin_mps": self.args.full_half_margin,
                "max_full_half_ratio_relative_error": self.args.max_full_half_ratio_relative_error,
                "stop_speed_mps": self.args.stop_speed,
                "stop_speed_p95_mps": self.args.stop_speed_p95,
                "stop_drift_m": self.args.stop_drift,
                "min_vdes_hz": self.args.min_vdes_hz,
                "min_cmd_hz": self.args.min_cmd_hz,
                "min_odom_hz": self.args.min_odom_hz,
                "safety_min_altitude_m": self.args.safety_min_altitude,
                "safety_max_altitude_m": self.args.safety_max_altitude,
                "safety_max_speed_mps": self.args.safety_max_speed,
                "safety_max_position_distance_m": self.args.safety_max_position_distance,
                "safety_odom_timeout_s": self.args.safety_odom_timeout,
                "max_actual_yaw_drift_rad": self.args.max_actual_yaw_drift,
                "max_command_yaw_error_rad": self.args.max_command_yaw_error,
                "max_command_abs_yaw_dot_p99_radps": self.args.max_command_yaw_dot_p99,
                "max_altitude_drift_m": self.args.max_altitude_drift,
            },
            "safety": {
                "origin_position_world": safety_origin,
                "check_count": safety_check_count,
                "violations": safety_violations,
                "passed": not safety_violations,
            },
            "emergency_center": {
                "attempted": bool(emergency_center_results),
                "attempts": emergency_center_results,
                "all_completed": bool(emergency_center_results)
                and all(item.get("completed", False) for item in emergency_center_results),
            },
            "fixed_yaw": overall_yaw,
            "fixed_altitude": overall_altitude,
            "collision_counter": collision_counter,
            "topics": {
                "expected_types": EXPECTED_TOPIC_TYPES,
                "observed_types": self.observed_topic_types,
                "message_totals": totals,
            },
            "phase_results": self.phase_results,
            "full_vs_half": self.comparisons,
        }


def _positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive number")
    return value


def _nonnegative_float(text):
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative number")
    return value


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("expected a finite number")
    return value


def _direction_list(text):
    directions = tuple(part.strip().lower() for part in str(text).split(",") if part.strip())
    allowed = {"up", "down", "left", "right"}
    if not directions:
        raise argparse.ArgumentTypeError("expected at least one of up,down,left,right")
    invalid = [direction for direction in directions if direction not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError("unknown direction(s): %s" % ",".join(invalid))
    if len(set(directions)) != len(directions):
        raise argparse.ArgumentTypeError("directions must not be repeated")
    return directions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example (source ROS and Controller messages first):
  source /opt/ros/noetic/setup.bash
  source /workspace/YOPO/Controller/devel/setup.bash
  cd /workspace/YOPO

Create the FIFO before launching the existing stack:
  mkfifo /tmp/yopo-js-test
  bash tools/launch_sim.sh --no-rviz \\
    --weight /workspace/YOPO/YOPO/saved/YOPO_0/epoch200.pth \\
    --joystick-device /tmp/yopo-js-test

  python3 tools/joystick_e2e_test.py --fifo /tmp/yopo-js-test

For a deterministic obstacle-free test, start only roscore, the quadrotor
controller/simulator, and the planner, then add --publish-clear-depth.  Do not
run the real sensor at the same time.  This tool never starts or stops the stack.
""",
    )
    parser.add_argument("--fifo", default="/tmp/yopo-js-test", help="FIFO passed to planner --joystick-device.")
    parser.add_argument("--output", default="", help="JSON report path. Default: /tmp/joystick_e2e_<UTC>.json")
    parser.add_argument("--summary-only", action="store_true", help="Omit raw per-topic samples from JSON.")
    parser.add_argument(
        "--remove-created-fifo",
        action="store_true",
        help="Remove the FIFO on exit only if this invocation created it.",
    )

    parser.add_argument("--axis-x", type=int, default=0, help="Horizontal right-stick Linux axis number.")
    parser.add_argument("--axis-y", type=int, default=1, help="Vertical right-stick Linux axis number.")
    parser.add_argument("--axis-max", type=float, default=32767.0)
    parser.add_argument("--half-fraction", type=float, default=0.5, help="Raw travel used for half-stick phases.")
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--invert-x", type=int, choices=(0, 1), default=1)
    parser.add_argument("--invert-y", type=int, choices=(0, 1), default=0)
    parser.add_argument("--swap-xy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-speed", type=_positive_float, default=6.0, help="Planner --velocity value.")
    parser.add_argument(
        "--directions",
        type=_direction_list,
        default=("up", "down", "left", "right"),
        help="Comma-separated active directions and order. Default: up,down,left,right.",
    )

    parser.add_argument("--topic-timeout", type=_positive_float, default=45.0)
    parser.add_argument("--unlock-duration", type=_positive_float, default=1.0)
    parser.add_argument("--baseline-duration", type=_positive_float, default=3.0)
    parser.add_argument("--active-duration", type=_positive_float, default=4.0)
    parser.add_argument("--center-duration", type=_positive_float, default=3.0)
    parser.add_argument("--stable-window", type=_positive_float, default=1.5)
    parser.add_argument("--center-window", type=_positive_float, default=1.0)
    parser.add_argument("--inject-hz", type=_positive_float, default=20.0)
    parser.add_argument(
        "--safety-min-altitude",
        type=_finite_float,
        default=0.5,
        help="Immediate-abort minimum world-frame altitude in meters.",
    )
    parser.add_argument(
        "--safety-max-altitude",
        type=_finite_float,
        default=5.0,
        help="Immediate-abort maximum world-frame altitude in meters.",
    )
    parser.add_argument(
        "--safety-max-speed",
        type=_positive_float,
        default=8.0,
        help="Immediate-abort maximum 3-D odometry speed in m/s.",
    )
    parser.add_argument(
        "--safety-max-position-distance",
        type=_positive_float,
        default=45.0,
        help="Immediate-abort maximum 3-D distance from the initial odometry position in meters.",
    )
    parser.add_argument(
        "--safety-odom-timeout",
        type=_positive_float,
        default=0.5,
        help="Immediate-abort maximum age of the latest odometry sample in seconds.",
    )
    parser.add_argument(
        "--max-actual-yaw-drift",
        type=_nonnegative_float,
        default=0.15,
        help="Maximum wrapped odometry yaw drift from the first finite odometry yaw, in radians.",
    )
    parser.add_argument(
        "--max-command-yaw-error",
        type=_nonnegative_float,
        default=0.05,
        help="Maximum wrapped PositionCommand.yaw error from the yaw reference, in radians.",
    )
    parser.add_argument(
        "--max-command-yaw-dot-p99",
        type=_nonnegative_float,
        default=0.05,
        help="Maximum p99 absolute PositionCommand.yaw_dot, in radians per second.",
    )
    parser.add_argument(
        "--max-altitude-drift",
        type=_nonnegative_float,
        default=0.20,
        help=(
            "Maximum actual and commanded world-z drift from the first finite odometry frame "
            "of each active/center phase and the complete locked-altitude interval, in meters."
        ),
    )
    parser.add_argument(
        "--publish-clear-depth",
        action="store_true",
        help="Publish four synchronized clear 32FC1 depth images; default off.",
    )
    parser.add_argument(
        "--collision-topic",
        default=DEFAULT_COLLISION_TOPIC,
        help="Cumulative std_msgs/Int32 collision counter topic.",
    )
    parser.add_argument(
        "--require-collision-topic",
        action="store_true",
        help=(
            "Require a valid collision-counter baseline and abort on any increase; "
            "intended for real sensor_simulator acceptance."
        ),
    )
    parser.add_argument("--depth-width", type=int, default=8, help="Clear-depth image width.")
    parser.add_argument("--depth-height", type=int, default=8, help="Clear-depth image height.")
    parser.add_argument("--clear-depth-value", type=_positive_float, default=4.0, help="Clear depth in meters.")
    parser.add_argument("--depth-hz", type=_positive_float, default=15.0, help="Clear-depth publication rate.")

    parser.add_argument("--vdes-tolerance", type=_nonnegative_float, default=0.08)
    parser.add_argument("--vdes-direction-cos", type=float, default=0.995)
    parser.add_argument("--actual-direction-cos", type=float, default=0.60)
    parser.add_argument("--actual-positive-fraction", type=float, default=0.75)
    parser.add_argument("--min-actual-projection", type=_nonnegative_float, default=0.25)
    parser.add_argument(
        "--min-actual-projection-ratio",
        type=_nonnegative_float,
        default=0.75,
        help=(
            "Minimum steady-state odometry projection divided by the deadzone-adjusted "
            "expected stick-command magnitude (default: 0.75)."
        ),
    )
    parser.add_argument(
        "--max-actual-projection-slope",
        type=_nonnegative_float,
        default=0.20,
        help=(
            "Maximum absolute least-squares slope of actual direction projection over the "
            "final stable window, in m/s^2 (default: 0.20)."
        ),
    )
    parser.add_argument("--cmd-direction-cos", type=float, default=0.50)
    parser.add_argument("--min-cmd-projection", type=_nonnegative_float, default=0.10)
    parser.add_argument("--min-direction-speed", type=_positive_float, default=0.20)
    parser.add_argument("--skip-cmd-direction-check", action="store_true")
    parser.add_argument(
        "--max-cmd-odom-projection-error",
        type=_nonnegative_float,
        default=0.20,
        help=(
            "Maximum absolute difference between median PositionCommand and odometry "
            "direction projections in the stable window, in m/s (default: 0.20)."
        ),
    )
    parser.add_argument(
        "--skip-cmd-odom-projection-check",
        action="store_true",
        help="Disable the PositionCommand-versus-odometry projection tracking gate.",
    )
    parser.add_argument(
        "--min-empty-command-fraction",
        type=float,
        default=0.99,
        help="Minimum fraction of pos_cmd samples using TRAJECTORY_STATUS_EMPTY.",
    )
    parser.add_argument(
        "--center-command-tolerance",
        type=_nonnegative_float,
        default=0.05,
        help="Maximum center-phase p99 commanded velocity and acceleration norm.",
    )
    parser.add_argument("--full-half-ratio", type=_positive_float, default=1.50)
    parser.add_argument("--full-half-margin", type=_nonnegative_float, default=0.75)
    parser.add_argument(
        "--max-full-half-ratio-relative-error",
        type=_nonnegative_float,
        default=0.20,
        help=(
            "Maximum relative error between the observed full/half projection ratio and "
            "the deadzone-adjusted expected input ratio (default: 0.20, i.e. +/-20%%)."
        ),
    )
    parser.add_argument("--stop-speed", type=_nonnegative_float, default=0.35)
    parser.add_argument("--stop-speed-p95", type=_nonnegative_float, default=0.50)
    parser.add_argument("--stop-drift", type=_nonnegative_float, default=0.35)
    parser.add_argument("--min-vdes-hz", type=_nonnegative_float, default=8.0)
    parser.add_argument("--min-cmd-hz", type=_nonnegative_float, default=20.0)
    parser.add_argument("--min-odom-hz", type=_nonnegative_float, default=50.0)

    args = parser.parse_args(argv)
    args.invert_x = bool(args.invert_x)
    args.invert_y = bool(args.invert_y)
    args.swap_xy = bool(args.swap_xy)

    if args.axis_x == args.axis_y or min(args.axis_x, args.axis_y) < 0 or max(args.axis_x, args.axis_y) > 255:
        parser.error("--axis-x and --axis-y must be distinct values in [0,255]")
    if not 0.0 < args.axis_max <= 32767.0:
        parser.error("--axis-max must be in (0,32767]")
    if not 0.0 < args.half_fraction < 1.0:
        parser.error("--half-fraction must be in (0,1)")
    if not 0.0 <= args.deadzone < 1.0:
        parser.error("--deadzone must be in [0,1)")
    for name in ("vdes_direction_cos", "actual_direction_cos", "cmd_direction_cos"):
        if not -1.0 <= getattr(args, name) <= 1.0:
            parser.error("--%s must be in [-1,1]" % name.replace("_", "-"))
    if not 0.0 <= args.actual_positive_fraction <= 1.0:
        parser.error("--actual-positive-fraction must be in [0,1]")
    if not 0.0 <= args.min_empty_command_fraction <= 1.0:
        parser.error("--min-empty-command-fraction must be in [0,1]")
    if args.max_full_half_ratio_relative_error > 1.0:
        parser.error("--max-full-half-ratio-relative-error must be in [0,1]")
    if args.stop_speed_p95 < args.stop_speed:
        parser.error("--stop-speed-p95 must be >= --stop-speed")
    if args.safety_max_altitude <= args.safety_min_altitude:
        parser.error("--safety-max-altitude must be greater than --safety-min-altitude")
    if args.depth_width <= 0 or args.depth_height <= 0:
        parser.error("--depth-width and --depth-height must be positive")
    if not args.collision_topic.strip():
        parser.error("--collision-topic must not be empty")
    if args.publish_clear_depth and args.require_collision_topic:
        parser.error(
            "--publish-clear-depth and --require-collision-topic are mutually exclusive; "
            "the strict collision gate is only valid with the real sensor_simulator"
        )
    return args


def _default_output_path():
    stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("/tmp/joystick_e2e_%s.json" % stamp)


def _write_report(report, output_path):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(str(temporary), str(output_path))
    return output_path


def main(argv=None):
    args = parse_args(argv)
    output_path = Path(args.output).expanduser() if args.output else _default_output_path()
    fifo_path = None
    fifo_fd = None
    fifo_created = False
    acceptance = None
    report = None
    exit_code = 2

    try:
        fifo_path, fifo_fd, fifo_created = _open_fifo(args.fifo)
        rospy.init_node("yopo_joystick_e2e_test", anonymous=True)
        acceptance = JoystickAcceptance(args, fifo_path, fifo_fd, fifo_created)
        rospy.loginfo(
            "FIFO ready at %s. Waiting for an existing flight stack configured with this exact path...",
            fifo_path,
        )
        try:
            acceptance.execute()
        except AcceptanceSafetyAbort as exc:
            acceptance.aborted_reason = str(exc)
            if not exc.emergency_center_attempted:
                acceptance.send_emergency_center("AcceptanceSafetyAbort: %s" % exc)
            rospy.logerr("Safety abort: %s", exc)
        except (AcceptanceSetupError, rospy.ROSException) as exc:
            acceptance.runtime_error = "%s: %s" % (type(exc).__name__, exc)
            acceptance.send_emergency_center(acceptance.runtime_error)
            rospy.logerr("Acceptance setup/runtime error: %s", exc)
        report = acceptance.build_report()
        exit_code = 0 if report["passed"] else (2 if report["runtime_error"] else 1)
    except KeyboardInterrupt:
        if acceptance is not None:
            acceptance.runtime_error = "Interrupted by user"
            acceptance.send_emergency_center(acceptance.runtime_error)
            report = acceptance.build_report()
        else:
            report = {
                "schema_version": 1,
                "test": "YOPO joystick FIFO ROS closed-loop acceptance",
                "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
                "status": "error",
                "passed": False,
                "runtime_error": "Interrupted by user",
            }
        exit_code = 130
    except Exception as exc:
        if acceptance is not None:
            acceptance.runtime_error = "%s: %s" % (type(exc).__name__, exc)
            acceptance.send_emergency_center(acceptance.runtime_error)
            report = acceptance.build_report()
        else:
            report = {
                "schema_version": 1,
                "test": "YOPO joystick FIFO ROS closed-loop acceptance",
                "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
                "status": "error",
                "passed": False,
                "runtime_error": "%s: %s" % (type(exc).__name__, exc),
            }
        print("joystick_e2e_test: %s" % exc, file=sys.stderr)
        exit_code = 2
    finally:
        if acceptance is not None and fifo_fd is not None:
            acceptance.send_safe_center()
        if acceptance is not None:
            acceptance.stop_auxiliary_publishers()
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
            "test": "YOPO joystick FIFO ROS closed-loop acceptance",
            "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "status": "error",
            "passed": False,
            "runtime_error": "No report was produced",
        }
        exit_code = 2
    try:
        written_path = _write_report(report, output_path)
    except Exception as exc:
        print("Failed to write JSON report %s: %s" % (output_path, exc), file=sys.stderr)
        return 2

    print("joystick_e2e_report=%s" % written_path)
    print("joystick_e2e_status=%s" % report.get("status"))
    if report.get("aborted_reason"):
        print("joystick_e2e_abort=%s" % report["aborted_reason"])
    if report.get("runtime_error"):
        print("joystick_e2e_error=%s" % report["runtime_error"])
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
