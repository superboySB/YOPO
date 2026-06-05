#!/usr/bin/env python3
import argparse
import json
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Int32


class SwarmTrackerMonitor:
    def __init__(self, uav_num: int):
        self.uav_num = uav_num
        self.arrived = [False] * uav_num
        self.arrived_seen = [False] * uav_num
        self.goal_distance = [None] * uav_num
        self.goal_position = [None] * uav_num
        self.odom_position = [None] * uav_num
        self.arrival_time = [None] * uav_num
        self.mask_frames = [0] * uav_num
        self.nonempty_mask_frames = [0] * uav_num
        self.uav_collision_total = 0
        self.occupied_collision_total = 0
        self.start_time = time.time()

        for idx in range(uav_num):
            rospy.Subscriber(f"/uav{idx}/yopo/arrived", Bool, self._arrived_cb(idx), queue_size=1)
            rospy.Subscriber(f"/uav{idx}/yopo/goal_distance", Float32, self._goal_cb(idx), queue_size=1)
            rospy.Subscriber(f"/uav{idx}/yopo/goal", PoseStamped, self._goal_pose_cb(idx), queue_size=1)
            rospy.Subscriber(f"/uav{idx}/sim/odom", Odometry, self._odom_cb(idx), queue_size=1)
            rospy.Subscriber(f"/uav{idx}/target_mask_image", Image, self._mask_cb(idx), queue_size=1)

        rospy.Subscriber("/yopo/uav_collision_counter_total", Int32, self._uav_collision_cb, queue_size=1)
        rospy.Subscriber("/yopo/collision_counter_total", Int32, self._occupied_collision_cb, queue_size=1)

    def _arrived_cb(self, idx):
        def callback(msg: Bool):
            self.arrived_seen[idx] = True
            self.arrived[idx] = bool(msg.data)
            if self.arrived[idx] and self.arrival_time[idx] is None:
                self.arrival_time[idx] = time.time() - self.start_time

        return callback

    def _goal_cb(self, idx):
        def callback(msg: Float32):
            self.goal_distance[idx] = float(msg.data)

        return callback

    def _goal_pose_cb(self, idx):
        def callback(msg: PoseStamped):
            pos = msg.pose.position
            self.goal_position[idx] = [float(pos.x), float(pos.y), float(pos.z)]

        return callback

    def _odom_cb(self, idx):
        def callback(msg: Odometry):
            pos = msg.pose.pose.position
            self.odom_position[idx] = [float(pos.x), float(pos.y), float(pos.z)]

        return callback

    def _mask_cb(self, idx):
        def callback(msg: Image):
            self.mask_frames[idx] += 1
            if msg.encoding in ("mono8", "8UC1"):
                mask = np.frombuffer(msg.data, dtype=np.uint8)
                nonempty = np.count_nonzero(mask) > 0
            elif msg.encoding == "32FC1":
                mask = np.frombuffer(msg.data, dtype=np.float32)
                nonempty = np.count_nonzero(mask > 0.5) > 0
            else:
                nonempty = len(msg.data) > 0
            if nonempty:
                self.nonempty_mask_frames[idx] += 1

        return callback

    def _uav_collision_cb(self, msg: Int32):
        self.uav_collision_total = int(msg.data)

    def _occupied_collision_cb(self, msg: Int32):
        self.occupied_collision_total = int(msg.data)

    def summary(self, reason: str):
        elapsed = time.time() - self.start_time
        return {
            "reason": reason,
            "elapsed_sec": round(elapsed, 3),
            "all_arrived": all(self.arrived),
            "uav_collision_total": self.uav_collision_total,
            "occupied_collision_total": self.occupied_collision_total,
            "uavs": [
                {
                    "name": f"uav{i}",
                    "arrived": self.arrived[i],
                    "arrived_seen": self.arrived_seen[i],
                    "arrival_time_sec": None if self.arrival_time[i] is None else round(self.arrival_time[i], 3),
                    "goal_distance": self.goal_distance[i],
                    "goal_position": self.goal_position[i],
                    "odom_position": self.odom_position[i],
                    "mask_frames": self.mask_frames[i],
                    "nonempty_mask_frames": self.nonempty_mask_frames[i],
                    "nonempty_mask_ratio": (
                        None if self.mask_frames[i] == 0 else round(self.nonempty_mask_frames[i] / self.mask_frames[i], 3)
                    ),
                }
                for i in range(self.uav_num)
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav-num", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=80.0)
    parser.add_argument("--require-arrival", type=int, default=1)
    parser.add_argument("--max-uav-collisions", type=int, default=0)
    parser.add_argument("--max-static-collisions", type=int, default=0)
    parser.add_argument("--min-nonempty-mask-frames", type=int, default=0)
    args = parser.parse_args()

    rospy.init_node("swarm_tracker_monitor", anonymous=True, disable_signals=True)
    monitor = SwarmTrackerMonitor(args.uav_num)
    rate = rospy.Rate(10)
    reason = "timeout"
    while not rospy.is_shutdown():
        if all(monitor.arrived):
            reason = "all_arrived"
            break
        if time.time() - monitor.start_time > args.timeout:
            break
        rate.sleep()

    summary = monitor.summary(reason)
    print(json.dumps(summary, indent=2, sort_keys=False))

    enough_masks = all(frames >= args.min_nonempty_mask_frames for frames in monitor.nonempty_mask_frames)
    success = (
        (not args.require_arrival or summary["all_arrived"])
        and summary["uav_collision_total"] <= args.max_uav_collisions
        and summary["occupied_collision_total"] <= args.max_static_collisions
        and enough_masks
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
