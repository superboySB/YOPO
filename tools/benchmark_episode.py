#!/usr/bin/env python3
"""Record one paired YOPO-Simple/YOPO-MINCO closed-loop benchmark episode.

The node publishes the same goal(s) to both planners and records only common,
representation-independent measurements from odometry, commands, and the
simulator collision detector.  RViz candidate/best-trajectory topics are kept
as qualitative evidence, not mixed into the quantitative score.
"""

import argparse
import hashlib
import json
import math
import os
import threading
import time
from datetime import datetime, timezone

import numpy as np
import rosgraph
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32, Int32
from visualization_msgs.msg import Marker, MarkerArray


def vector3(value):
    return [float(value.x), float(value.y), float(value.z)]


def parse_goal(text):
    values = [float(value) for value in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("goal must be x,y,z")
    return values


def parse_json_dict(text):
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("configuration must be a JSON object")
    return value


def capture_label(label):
    """Stable JSON key for a logical vehicle, independent of its ROS lane."""
    return (label[5:] if label.startswith("yopo-") else label).replace(" ", "_")


class EpisodeRecorder:
    def __init__(self, args):
        self.args = args
        if args.left_label == args.right_label:
            raise ValueError("left-label and right-label must differ")
        self.vehicles = (args.left_label, args.right_label)
        self.lock = threading.Lock()
        self.active = False
        self.t0 = None
        self.final_goal_t = None
        self.current_goal = list(args.goal)
        self.samples = {name: [] for name in self.vehicles}
        self.commands = {name: [] for name in self.vehicles}
        self.arrival_s = {name: None for name in self.vehicles}
        self.collision_total = {name: 0 for name in self.vehicles}
        self.collision_samples_total = {name: 0 for name in self.vehicles}
        self.collision_state = {name: False for name in self.vehicles}
        self.collision_baseline = {name: 0 for name in self.vehicles}
        self.collision_samples_baseline = {name: 0 for name in self.vehicles}
        self.start_in_collision = {name: False for name in self.vehicles}
        self.clearance = {name: [] for name in self.vehicles}
        self.latest_position = {name: None for name in self.vehicles}
        self.map_points = None
        self.map_fingerprints = {}
        self.capture_times = [1.0]
        if args.scenario == "dynamic":
            self.capture_times.append(args.switch_time + 0.8)
        self.captures = {}

        rospy.init_node("yopo_benchmark_episode", anonymous=True)
        self.goal_pub = rospy.Publisher(args.goal_topic, PoseStamped, queue_size=1)

        for ros_name, label in (("yopo_simple", args.left_label), ("yopo_minco", args.right_label)):
            prefix = "/" + ros_name
            rospy.Subscriber(prefix + "/odom", Odometry, self._odom_cb,
                             callback_args=label, queue_size=20, tcp_nodelay=True)
            rospy.Subscriber(prefix + "/pos_cmd", PositionCommand, self._command_cb,
                             callback_args=label, queue_size=20, tcp_nodelay=True)
            rospy.Subscriber(prefix + "/collision_count", Int32, self._collision_cb,
                             callback_args=label, queue_size=1)
            rospy.Subscriber(prefix + "/collision_samples", Int32, self._collision_samples_cb,
                             callback_args=label, queue_size=1)
            rospy.Subscriber(prefix + "/collision_state", Bool, self._collision_state_cb,
                             callback_args=label, queue_size=1)
            rospy.Subscriber(prefix + "/clearance", Float32, self._clearance_cb,
                             callback_args=label, queue_size=10)

        self.map_subscribers = {
            args.left_label: rospy.Subscriber(
                "/yopo_simple/mock_map", PointCloud2, self._map_cb,
                callback_args=args.left_label, queue_size=1),
            args.right_label: rospy.Subscriber(
                "/yopo_minco/mock_map", PointCloud2, self._map_cb,
                callback_args=args.right_label, queue_size=1),
        }
        left_key = capture_label(args.left_label)
        right_key = capture_label(args.right_label)
        if args.left_best_type == "path":
            rospy.Subscriber("/yopo_simple/best_traj_visual", Path, self._path_cb,
                             callback_args=left_key + "_best", queue_size=1)
        else:
            rospy.Subscriber("/yopo_simple/best_traj_visual", MarkerArray, self._markers_cb,
                             callback_args=left_key + "_best", queue_size=1)
        rospy.Subscriber("/yopo_simple/trajs_visual", MarkerArray, self._markers_cb,
                         callback_args=left_key + "_candidates", queue_size=1)
        rospy.Subscriber("/yopo_simple/corridor_visual", MarkerArray, self._markers_cb,
                         callback_args=left_key + "_corridor", queue_size=1)
        rospy.Subscriber("/yopo_minco/best_traj_visual", MarkerArray, self._markers_cb,
                         callback_args=right_key + "_best", queue_size=1)
        rospy.Subscriber("/yopo_minco/trajs_visual", MarkerArray, self._markers_cb,
                         callback_args=right_key + "_candidates", queue_size=1)
        rospy.Subscriber("/yopo_minco/corridor_visual", MarkerArray, self._markers_cb,
                         callback_args=right_key + "_corridor", queue_size=1)

    def _elapsed(self):
        return None if self.t0 is None else time.monotonic() - self.t0

    def _odom_cb(self, msg, name):
        pos = vector3(msg.pose.pose.position)
        vel = vector3(msg.twist.twist.linear)
        quat = [float(msg.pose.pose.orientation.x), float(msg.pose.pose.orientation.y),
                float(msg.pose.pose.orientation.z), float(msg.pose.pose.orientation.w)]
        with self.lock:
            self.latest_position[name] = pos
            if not self.active:
                return
            elapsed = self._elapsed()
            self.samples[name].append([round(elapsed, 5), *pos, *vel, *quat])
            if self.arrival_s[name] is None and self.final_goal_t is not None:
                if np.linalg.norm(np.asarray(pos) - np.asarray(self.current_goal)) < self.args.arrival_radius:
                    self.arrival_s[name] = elapsed - self.final_goal_t

    def _command_cb(self, msg, name):
        with self.lock:
            if not self.active:
                return
            elapsed = self._elapsed()
            self.commands[name].append([
                round(elapsed, 5), *vector3(msg.position), *vector3(msg.velocity),
                *vector3(msg.acceleration), float(msg.yaw), float(msg.yaw_dot),
                int(msg.trajectory_flag),
            ])

    def _collision_cb(self, msg, name):
        with self.lock:
            self.collision_total[name] = int(msg.data)

    def _collision_samples_cb(self, msg, name):
        with self.lock:
            self.collision_samples_total[name] = int(msg.data)

    def _collision_state_cb(self, msg, name):
        with self.lock:
            self.collision_state[name] = bool(msg.data)

    def _clearance_cb(self, msg, name):
        with self.lock:
            if self.active:
                self.clearance[name].append([round(self._elapsed(), 5), float(msg.data)])

    def _map_cb(self, msg, name):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        hash_stride = max(1, len(raw) // 1000000)
        fingerprint = hashlib.sha256(raw[::hash_stride].tobytes()).hexdigest()
        with self.lock:
            self.map_fingerprints[name] = {"bytes": len(msg.data), "sample_sha256": fingerprint}
            if name != self.args.left_label or self.map_points is not None:
                return
        # Uniformly stride the complete cloud before spatial filtering.  This
        # avoids the strong x-axis bias produced by stopping an ordered cloud
        # iterator after its first N matches (especially for Perlin grids).
        offsets = {field.name: field.offset for field in msg.fields}
        endian = ">" if msg.is_bigendian else "<"
        dtype = np.dtype({"names": ["x", "y", "z"],
                          "formats": [endian + "f4"] * 3,
                          "offsets": [offsets["x"], offsets["y"], offsets["z"]],
                          "itemsize": msg.point_step})
        cloud = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
        stride = max(1, len(cloud) // 240000)
        xyz = np.column_stack((cloud["x"][::stride], cloud["y"][::stride], cloud["z"][::stride]))
        finite = np.isfinite(xyz).all(axis=1)
        bounded = (finite & (xyz[:, 0] >= -8.0) & (xyz[:, 0] <= 28.0) &
                   (xyz[:, 1] >= -20.0) & (xyz[:, 1] <= 22.0) &
                   (xyz[:, 2] >= 0.3) & (xyz[:, 2] <= 4.5))
        xyz = xyz[bounded]
        if len(xyz) > 30000:
            xyz = xyz[np.linspace(0, len(xyz) - 1, 30000, dtype=int)]
        points = np.round(xyz, 3).tolist()
        with self.lock:
            if self.map_points is None:
                self.map_points = points

    def _capture_slot(self, topic):
        if not self.active:
            return None
        elapsed = self._elapsed()
        for index, capture_t in enumerate(self.capture_times):
            key = "%s@%.1f" % (topic, capture_t)
            if elapsed >= capture_t and key not in self.captures:
                return key, elapsed
        return None

    def _path_cb(self, msg, topic):
        with self.lock:
            slot = self._capture_slot(topic)
            if slot is None:
                return
            key, elapsed = slot
            self.captures[key] = {
                "elapsed_s": round(elapsed, 4),
                "lines": [[vector3(pose.pose.position) for pose in msg.poses]],
            }

    def _markers_cb(self, msg, topic):
        with self.lock:
            slot = self._capture_slot(topic)
            if slot is None:
                return
            key, elapsed = slot
            lines, objects = [], []
            for marker in msg.markers:
                color = [float(marker.color.r), float(marker.color.g),
                         float(marker.color.b), float(marker.color.a)]
                if marker.type in (Marker.LINE_STRIP, Marker.LINE_LIST) and marker.points:
                    lines.append({"id": int(marker.id), "points": [vector3(p) for p in marker.points],
                                  "color": color})
                elif marker.type in (Marker.SPHERE, Marker.SPHERE_LIST):
                    centers = [vector3(p) for p in marker.points] if marker.points else [vector3(marker.pose.position)]
                    objects.append({"id": int(marker.id), "centers": centers,
                                    "scale": vector3(marker.scale), "color": color})
            self.captures[key] = {"elapsed_s": round(elapsed, 4), "lines": lines, "objects": objects}

    def _publish_goal(self, goal):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = goal
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def wait_ready(self):
        deadline = time.monotonic() + self.args.ready_timeout
        rate = rospy.Rate(10)
        master = rosgraph.Master(rospy.get_name())
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                odom_ready = all(value is not None for value in self.latest_position.values())
                map_ready = (len(self.map_fingerprints) == 2 and self.map_points is not None)
            try:
                _publishers, subscribers, _services = master.getSystemState()
            except Exception as error:
                rospy.logwarn_throttle(2.0, "waiting for ROS master: %s", error)
                rate.sleep()
                continue
            goal_subscribers = set()
            for topic, nodes in subscribers:
                if topic == self.args.goal_topic:
                    goal_subscribers.update(nodes)
            planners_ready = {self.args.left_node, self.args.right_node}.issubset(goal_subscribers)
            if odom_ready and map_ready and planners_ready and self.goal_pub.get_num_connections() >= 2:
                return
            rate.sleep()
        raise RuntimeError("timed out waiting for both odometry streams and planner goal subscribers")

    def run(self):
        self.wait_ready()
        # The map publisher is latched and the simulator republishes whenever
        # it has subscribers.  Forest clouds can exceed 800 MB, so unsubscribe
        # after the one-time identity/context capture to avoid perturbing odom.
        for subscriber in self.map_subscribers.values():
            subscriber.unregister()
        self.map_subscribers.clear()
        with self.lock:
            self.collision_baseline = dict(self.collision_total)
            self.collision_samples_baseline = dict(self.collision_samples_total)
            self.start_in_collision = dict(self.collision_state)
            self.active = True
            self.t0 = time.monotonic()
            self.final_goal_t = 0.0 if self.args.scenario == "straight" else None
        self._publish_goal(self.args.goal)

        switched = self.args.scenario == "straight"
        both_arrived_at = None
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            elapsed = self._elapsed()
            if not switched and elapsed >= self.args.switch_time:
                with self.lock:
                    self.current_goal = list(self.args.dynamic_goal)
                    self.final_goal_t = elapsed
                    self.arrival_s = {name: None for name in self.vehicles}
                self._publish_goal(self.args.dynamic_goal)
                switched = True

            with self.lock:
                complete = all(value is not None for value in self.arrival_s.values())
            if complete and both_arrived_at is None:
                both_arrived_at = elapsed
            if both_arrived_at is not None and elapsed - both_arrived_at >= 0.5:
                break
            if elapsed >= self.args.timeout:
                break
            rate.sleep()

        with self.lock:
            self.active = False
        return self._result()

    def _result(self):
        with self.lock:
            vehicles = {}
            for name in self.vehicles:
                collision_events = self.collision_total[name] - self.collision_baseline[name]
                collision_samples = self.collision_samples_total[name] - self.collision_samples_baseline[name]
                summary = compute_metrics(
                    self.samples[name], self.commands[name], self.current_goal,
                    self.final_goal_t or 0.0, self.arrival_s[name], collision_events,
                    collision_samples, self.start_in_collision[name], self.args.arrival_radius,
                )
                vehicles[name] = {"metrics": summary, "odom": self.samples[name],
                                  "commands": self.commands[name], "clearance": self.clearance[name]}
            return {
                "schema_version": 2,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "maze_type": self.args.maze_type,
                "map_seed": self.args.map_seed,
                "scenario": self.args.scenario,
                "initial_goal": self.args.goal,
                "final_goal": self.current_goal,
                "switch_time_s": self.final_goal_t,
                "timeout_s": self.args.timeout,
                "arrival_radius_m": self.args.arrival_radius,
                "velocity_mps": self.args.velocity,
                "safe_radius_m": self.args.safe_radius,
                "yaw_goal_weight": self.args.yaw_goal_weight,
                "lanes": {"yopo_simple": self.args.left_label,
                          "yopo_minco": self.args.right_label},
                "vehicle_configs": {self.args.left_label: self.args.left_config,
                                    self.args.right_label: self.args.right_config},
                "vehicles": vehicles,
                "captures": self.captures,
                "map_points": self.map_points or [],
                "map_fingerprints": self.map_fingerprints,
                "maps_identical": (len(self.map_fingerprints) == 2 and
                                   len({item["bytes"] for item in self.map_fingerprints.values()}) == 1 and
                                   len({item["sample_sha256"] for item in self.map_fingerprints.values()}) == 1),
            }


def compute_metrics(odom_rows, command_rows, goal, final_goal_t, arrival_s,
                    collision_events, collision_samples, start_in_collision, arrival_radius):
    odom = np.asarray(odom_rows, dtype=float)
    commands = np.asarray(command_rows, dtype=float)
    result = {
        "arrived": arrival_s is not None,
        "arrival_s": None if arrival_s is None else round(float(arrival_s), 4),
        "collision_events": int(collision_events),
        "collision_samples": int(collision_samples),
        "start_in_collision": bool(start_in_collision),
    }
    if odom.size == 0:
        return result

    t, pos, vel = odom[:, 0], odom[:, 1:4], odom[:, 4:7]
    use = t >= final_goal_t
    t, pos, vel = t[use], pos[use], vel[use]
    if len(t) < 2:
        return result
    dt = np.diff(t)
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    valid = (dt > 1e-4) & (dt < 0.2) & (steps < 1.0)
    path_length = float(steps[valid].sum())
    duration = float(t[-1] - t[0])
    direct = float(np.linalg.norm(pos[0] - np.asarray(goal)))
    speed = np.linalg.norm(vel, axis=1)
    result.update({
        "duration_s": round(duration, 4),
        "path_length_m": round(path_length, 4),
        "path_efficiency": round(direct / max(path_length, direct, 1e-9), 5),
        "mean_speed_mps": round(float(np.mean(speed)), 4),
        "max_speed_mps": round(float(np.max(speed)), 4),
        "final_error_m": round(float(np.linalg.norm(pos[-1] - np.asarray(goal))), 4),
        "collision_free_success": bool(arrival_s is not None and collision_events == 0 and not start_in_collision),
        "within_arrival_radius_at_end": bool(np.linalg.norm(pos[-1] - np.asarray(goal)) < arrival_radius),
    })

    if commands.size:
        ct, cvel, cacc = commands[:, 0], commands[:, 4:7], commands[:, 7:10]
        cuse = ct >= final_goal_t
        ct, cvel, cacc = ct[cuse], cvel[cuse], cacc[cuse]
        if len(ct) >= 2:
            cdt = np.diff(ct)
            ok = (cdt > 1e-4) & (cdt < 0.2)
            acc_norm = np.linalg.norm(cacc, axis=1)
            jerk = np.diff(cacc, axis=0) / np.maximum(cdt[:, None], 1e-6)
            jerk_norm = np.linalg.norm(jerk, axis=1)
            result.update({
                "acc_rms_mps2": round(float(np.sqrt(np.mean(acc_norm ** 2))), 4),
                "acc_peak_mps2": round(float(np.max(acc_norm)), 4),
                "jerk_rms_mps3": round(float(np.sqrt(np.mean(jerk_norm[ok] ** 2))) if ok.any() else 0.0, 4),
                "jerk_p95_mps3": round(float(np.percentile(jerk_norm[ok], 95)) if ok.any() else 0.0, 4),
            })
            goal_vec = np.asarray(goal)[None, :2] - commands[cuse, 1:3]
            vel_xy = cvel[:, :2]
            denom = np.linalg.norm(goal_vec, axis=1) * np.linalg.norm(vel_xy, axis=1)
            angle = np.arccos(np.clip(np.sum(goal_vec * vel_xy, axis=1) / np.maximum(denom, 1e-6), -1.0, 1.0))
            aligned = np.flatnonzero((angle < np.deg2rad(15.0)) & (np.linalg.norm(vel_xy, axis=1) > 0.5))
            result["turn_response_s"] = None if aligned.size == 0 else round(float(ct[aligned[0]] - final_goal_t), 4)
    return result


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--output", required=True)
    result.add_argument("--maze-type", type=int, required=True)
    result.add_argument("--map-seed", type=int, required=True)
    result.add_argument("--scenario", choices=("straight", "dynamic"), required=True)
    result.add_argument("--goal", type=parse_goal, default=[20.0, 0.0, 2.0])
    result.add_argument("--dynamic-goal", type=parse_goal, default=[10.0, 15.0, 2.0])
    result.add_argument("--switch-time", type=float, default=2.5)
    result.add_argument("--timeout", type=float, default=20.0)
    result.add_argument("--ready-timeout", type=float, default=50.0)
    result.add_argument("--arrival-radius", type=float, default=1.0)
    result.add_argument("--velocity", type=float, default=6.0)
    result.add_argument("--safe-radius", type=float, default=0.05)
    result.add_argument("--yaw-goal-weight", type=float, default=6.0)
    result.add_argument("--goal-topic", default="/move_base_simple/goal")
    result.add_argument("--left-label", default="yopo-simple")
    result.add_argument("--right-label", default="yopo-minco")
    result.add_argument("--left-node", default="/yopo_simple_planner")
    result.add_argument("--right-node", default="/yopo_minco_planner")
    result.add_argument("--left-config", type=parse_json_dict, default={})
    result.add_argument("--right-config", type=parse_json_dict, default={})
    result.add_argument("--left-best-type", choices=("path", "marker"), default="path")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    recorder = EpisodeRecorder(arguments)
    payload = recorder.run()
    output = os.path.abspath(arguments.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps({name: value["metrics"] for name, value in payload["vehicles"].items()}, sort_keys=True))
