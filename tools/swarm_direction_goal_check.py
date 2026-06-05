#!/usr/bin/env python3
import argparse
import json
import math
import time

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class GoalCollector:
    def __init__(self, uav_num):
        self.uav_num = uav_num
        self.goals = {}
        self.subs = [
            rospy.Subscriber(f"/uav{idx}/yopo/goal", PoseStamped, self._goal_cb(idx), queue_size=1)
            for idx in range(uav_num)
        ]

    def _goal_cb(self, idx):
        def callback(msg):
            self.goals[f"uav{idx}"] = [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ]
        return callback

    def wait_all(self, timeout):
        deadline = time.time() + timeout
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            if len(self.goals) == self.uav_num:
                return dict(self.goals)
            rate.sleep()
        raise TimeoutError("timed out waiting for all /uav*/yopo/goal topics")

    def wait_expected(self, before, expected_dx, expected_dy, tolerance, timeout):
        deadline = time.time() + timeout
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            if len(self.goals) == self.uav_num:
                ok = True
                for idx in range(self.uav_num):
                    name = f"uav{idx}"
                    b = before[name]
                    g = self.goals[name]
                    dx = g[0] - b[0]
                    dy = g[1] - b[1]
                    if math.hypot(dx - expected_dx, dy - expected_dy) > tolerance:
                        ok = False
                        break
                if ok:
                    return dict(self.goals)
            rate.sleep()
        return dict(self.goals)


def wait_odom_positions(uav_num, timeout):
    positions = {}
    for idx in range(uav_num):
        msg = rospy.wait_for_message(f"/uav{idx}/sim/odom", Odometry, timeout=timeout)
        positions[f"uav{idx}"] = [
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            float(msg.pose.pose.position.z),
        ]
    return positions


def wait_goal_positions(uav_num, timeout):
    goals = {}
    for idx in range(uav_num):
        msg = rospy.wait_for_message(f"/uav{idx}/yopo/goal", PoseStamped, timeout=timeout)
        goals[f"uav{idx}"] = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ]
    return goals


def publish_formation_goal(topic, target_x, target_y, publish_count):
    pub = rospy.Publisher(topic, PoseStamped, queue_size=1)
    rospy.sleep(1.0)

    msg = PoseStamped()
    msg.header.frame_id = "world"
    msg.pose.position.x = float(target_x)
    msg.pose.position.y = float(target_y)
    msg.pose.position.z = 0.0
    msg.pose.orientation.w = 1.0

    for _ in range(publish_count):
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rospy.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav-num", type=int, default=10)
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=20.0)
    parser.add_argument("--topic", type=str, default="/move_base_simple/goal")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--tolerance", type=float, default=1.5)
    parser.add_argument("--publish-count", type=int, default=5)
    args = parser.parse_args()

    rospy.init_node("swarm_formation_goal_check", anonymous=True)
    collector = GoalCollector(args.uav_num)

    before = wait_odom_positions(args.uav_num, args.timeout)
    collector.wait_all(args.timeout)
    expected_dx = args.target_x - before["uav0"][0]
    expected_dy = args.target_y - before["uav0"][1]
    publish_formation_goal(args.topic, args.target_x, args.target_y, args.publish_count)
    goals = collector.wait_expected(before, expected_dx, expected_dy, args.tolerance, args.timeout)

    rows = []
    ok = True
    for idx in range(args.uav_num):
        name = f"uav{idx}"
        b = before[name]
        g = goals[name]
        dx = g[0] - b[0]
        dy = g[1] - b[1]
        err = math.hypot(dx - expected_dx, dy - expected_dy)
        same_direction = err <= args.tolerance
        ok = ok and same_direction
        rows.append({
            "name": name,
            "before_xy": [round(b[0], 3), round(b[1], 3)],
            "goal_xy": [round(g[0], 3), round(g[1], 3)],
            "delta_xy": [round(dx, 3), round(dy, 3)],
            "expected_delta_xy": [round(expected_dx, 3), round(expected_dy, 3)],
            "direction_error": round(err, 3),
            "same_direction_distance": same_direction,
        })

    unique_goal_xy = len({(round(goal[0], 2), round(goal[1], 2)) for goal in goals.values()})
    summary = {
        "ok": ok and unique_goal_xy == args.uav_num,
        "uav0_target_xy": [round(args.target_x, 3), round(args.target_y, 3)],
        "expected_delta_xy": [round(expected_dx, 3), round(expected_dy, 3)],
        "unique_goal_xy_count": unique_goal_xy,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
