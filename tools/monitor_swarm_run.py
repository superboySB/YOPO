#!/usr/bin/env python3
import argparse
import json
import time

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Int32


class SwarmRunMonitor:
    def __init__(self, uav_num: int):
        self.uav_num = uav_num
        self.arrived = [False] * uav_num
        self.arrived_seen = [False] * uav_num
        self.goal_distance = [None] * uav_num
        self.odom_position = [None] * uav_num
        self.arrival_time = [None] * uav_num
        self.uav_collision_total = 0
        self.occupied_collision_total = 0

        self.start_time = time.time()

        for i in range(uav_num):
            rospy.Subscriber(f"/uav{i}/yopo/arrived", Bool, self.make_arrived_callback(i), queue_size=1)
            rospy.Subscriber(f"/uav{i}/yopo/goal_distance", Float32, self.make_goal_distance_callback(i), queue_size=1)
            rospy.Subscriber(f"/uav{i}/sim/odom", Odometry, self.make_odom_callback(i), queue_size=1)

        rospy.Subscriber("/yopo/uav_collision_counter_total", Int32, self.uav_collision_callback, queue_size=1)
        rospy.Subscriber("/yopo/collision_counter_total", Int32, self.occupied_collision_callback, queue_size=1)

    def make_arrived_callback(self, idx):
        def callback(msg: Bool):
            self.arrived_seen[idx] = True
            self.arrived[idx] = bool(msg.data)
            if self.arrived[idx] and self.arrival_time[idx] is None:
                self.arrival_time[idx] = time.time() - self.start_time

        return callback

    def make_goal_distance_callback(self, idx):
        def callback(msg: Float32):
            self.goal_distance[idx] = float(msg.data)

        return callback

    def make_odom_callback(self, idx):
        def callback(msg: Odometry):
            pos = msg.pose.pose.position
            self.odom_position[idx] = [float(pos.x), float(pos.y), float(pos.z)]

        return callback

    def uav_collision_callback(self, msg: Int32):
        self.uav_collision_total = int(msg.data)

    def occupied_collision_callback(self, msg: Int32):
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
                    "odom_position": self.odom_position[i],
                }
                for i in range(self.uav_num)
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav-num", type=int, default=4, help="number of UAVs to monitor")
    parser.add_argument("--timeout", type=float, default=35.0, help="monitor timeout in seconds")
    args = parser.parse_args()

    rospy.init_node("swarm_run_monitor", anonymous=True, disable_signals=True)
    monitor = SwarmRunMonitor(args.uav_num)

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

    success = summary["all_arrived"] and summary["uav_collision_total"] == 0
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
