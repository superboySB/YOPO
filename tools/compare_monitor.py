#!/usr/bin/env python3
"""Interactive YOPO-Simple vs YOPO-MINCO metrics and RViz annotations."""

import argparse
import json
import math
import os
import threading
from datetime import datetime, timezone

import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker, MarkerArray


COLORS = {
    "yopo-simple": (1.0, 0.35, 0.03, 1.0),
    "yopo-minco": (0.0, 0.72, 1.0, 1.0),
}


class VehicleState:
    def __init__(self, name):
        self.name = name
        self.position = None
        self.last_position = None
        self.path_length = 0.0
        self.arrival_seconds = None
        self.collision_total = 0
        self.collision_baseline = 0
        self.trail = []

    def reset(self):
        self.last_position = self.position
        self.path_length = 0.0
        self.arrival_seconds = None
        self.collision_baseline = self.collision_total
        self.trail = [] if self.position is None else [self.position]


class ComparisonMonitor:
    def __init__(self, args):
        rospy.init_node("yopo_comparison_monitor", anonymous=False)
        self.args = args
        self.lock = threading.Lock()
        self.goal = None
        self.goal_started = None
        self.run_id = 0
        self.completed_run = 0
        self.states = {
            "yopo-simple": VehicleState("yopo-simple"),
            "yopo-minco": VehicleState("yopo-minco"),
        }

        rospy.Subscriber(args.simple_odom, Odometry, self._odom_cb,
                         callback_args="yopo-simple", queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(args.minco_odom, Odometry, self._odom_cb,
                         callback_args="yopo-minco", queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(args.simple_collision, Int32, self._collision_cb,
                         callback_args="yopo-simple", queue_size=1)
        rospy.Subscriber(args.minco_collision, Int32, self._collision_cb,
                         callback_args="yopo-minco", queue_size=1)
        rospy.Subscriber(args.goal_topic, PoseStamped, self._goal_cb, queue_size=1)

        self.marker_pub = rospy.Publisher("/yopo_compare/markers", MarkerArray, queue_size=1)
        self.metrics_pub = rospy.Publisher("/yopo_compare/metrics", String, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(0.2), self._publish)
        rospy.loginfo("YOPO comparison monitor ready; publish one RViz 2D Nav Goal for both vehicles")
        rospy.spin()

    def _goal_cb(self, msg):
        with self.lock:
            self.goal = (msg.pose.position.x, msg.pose.position.y, self.args.goal_z)
            self.goal_started = rospy.Time.now()
            self.run_id += 1
            self.completed_run = 0
            for state in self.states.values():
                state.reset()
        rospy.loginfo("Comparison run %d: shared goal=(%.2f, %.2f, %.2f)",
                      self.run_id, *self.goal)

    def _collision_cb(self, msg, name):
        with self.lock:
            self.states[name].collision_total = int(msg.data)

    def _odom_cb(self, msg, name):
        p = msg.pose.pose.position
        position = (float(p.x), float(p.y), float(p.z))
        with self.lock:
            state = self.states[name]
            state.position = position
            if self.goal_started is None:
                state.last_position = position
                return
            if state.last_position is not None:
                step = math.dist(position, state.last_position)
                if step < 1.0:  # reject reset/teleport discontinuities
                    state.path_length += step
            state.last_position = position
            if not state.trail or math.dist(position, state.trail[-1]) >= self.args.trail_step:
                state.trail.append(position)
                state.trail = state.trail[-self.args.max_trail_points:]
            if (state.arrival_seconds is None and self.goal is not None and
                    math.dist(position, self.goal) < self.args.arrival_radius):
                state.arrival_seconds = (rospy.Time.now() - self.goal_started).to_sec()

    def _snapshot(self):
        elapsed = None if self.goal_started is None else (rospy.Time.now() - self.goal_started).to_sec()
        vehicles = {}
        for name, state in self.states.items():
            vehicles[name] = {
                "position": state.position,
                "path_length_m": round(state.path_length, 3),
                "arrival_seconds": None if state.arrival_seconds is None else round(state.arrival_seconds, 3),
                "collision_events": state.collision_total - state.collision_baseline,
                "arrived": state.arrival_seconds is not None,
            }
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "elapsed_seconds": None if elapsed is None else round(elapsed, 3),
            "arrival_radius_m": self.args.arrival_radius,
            "vehicles": vehicles,
        }

    def _publish(self, _event):
        with self.lock:
            snapshot = self._snapshot()
            markers = self._markers(snapshot)
            complete = (self.run_id > 0 and all(v["arrived"] for v in snapshot["vehicles"].values()))
            should_log = complete and self.completed_run != self.run_id
            if should_log:
                self.completed_run = self.run_id
        self.marker_pub.publish(markers)
        self.metrics_pub.publish(String(data=json.dumps(snapshot, sort_keys=True)))
        if should_log:
            self._append_result(snapshot)
            rospy.loginfo("Comparison run %d complete: %s", self.run_id,
                          json.dumps(snapshot["vehicles"], sort_keys=True))

    def _markers(self, snapshot):
        now = rospy.Time.now()
        result = MarkerArray()
        for marker_id, (name, state) in enumerate(self.states.items()):
            color = COLORS[name]
            if state.position is not None:
                label = Marker()
                label.header.frame_id = "world"
                label.header.stamp = now
                label.ns = "vehicle_labels"
                label.id = marker_id
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.position = Point(state.position[0], state.position[1], state.position[2] + 0.85)
                label.pose.orientation.w = 1.0
                label.scale.z = 0.45
                label.color.r, label.color.g, label.color.b, label.color.a = color
                metrics = snapshot["vehicles"][name]
                label.text = "%s\n%.1fm | collisions=%d" % (
                    name, metrics["path_length_m"], metrics["collision_events"])
                result.markers.append(label)

            trail = Marker()
            trail.header.frame_id = "world"
            trail.header.stamp = now
            trail.ns = "executed_trails"
            trail.id = marker_id
            trail.type = Marker.LINE_STRIP
            trail.action = Marker.ADD
            trail.pose.orientation.w = 1.0
            trail.scale.x = 0.08
            trail.color.r, trail.color.g, trail.color.b, trail.color.a = color
            trail.points = [Point(*p) for p in state.trail]
            result.markers.append(trail)

        if self.goal is not None:
            summary = Marker()
            summary.header.frame_id = "world"
            summary.header.stamp = now
            summary.ns = "comparison_summary"
            summary.id = 0
            summary.type = Marker.TEXT_VIEW_FACING
            summary.action = Marker.ADD
            summary.pose.position = Point(self.goal[0], self.goal[1], self.goal[2] + 2.0)
            summary.pose.orientation.w = 1.0
            summary.scale.z = 0.42
            summary.color.r = summary.color.g = summary.color.b = summary.color.a = 1.0
            lines = ["shared goal | run %d" % self.run_id]
            for name in ("yopo-simple", "yopo-minco"):
                m = snapshot["vehicles"][name]
                arrival = "%.2fs" % m["arrival_seconds"] if m["arrived"] else "flying"
                lines.append("%s: %s, %.1fm, C=%d" %
                             (name, arrival, m["path_length_m"], m["collision_events"]))
            summary.text = "\n".join(lines)
            result.markers.append(summary)
        return result

    def _append_result(self, snapshot):
        if not self.args.result_file:
            return
        result_path = os.path.abspath(self.args.result_file)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        record = dict(snapshot)
        record["completed_at"] = datetime.now(timezone.utc).isoformat()
        with open(result_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple-odom", default="/yopo_simple/odom")
    parser.add_argument("--minco-odom", default="/yopo_minco/odom")
    parser.add_argument("--simple-collision", default="/yopo_simple/collision_count")
    parser.add_argument("--minco-collision", default="/yopo_minco/collision_count")
    parser.add_argument("--goal-topic", default="/move_base_simple/goal")
    parser.add_argument("--goal-z", type=float, default=2.0)
    parser.add_argument("--arrival-radius", type=float, default=5.0)
    parser.add_argument("--trail-step", type=float, default=0.08)
    parser.add_argument("--max-trail-points", type=int, default=4000)
    parser.add_argument("--result-file", default="results/comparison_latest.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    ComparisonMonitor(parse_args())
