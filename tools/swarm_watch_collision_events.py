#!/usr/bin/env python3
import argparse
import time

import rospy
from std_msgs.msg import Int32


class CollisionWatcher:
    def __init__(self, uav_num: int):
        self.start_time = time.time()
        self.last_values = {}

        self.watch_topic("/yopo/collision_counter_total", "static_total")
        self.watch_topic("/yopo/uav_collision_counter_total", "uav_total")
        for i in range(uav_num):
            self.watch_topic(f"/uav{i}/yopo/collision_counter", f"uav{i}_static")
            self.watch_topic(f"/uav{i}/yopo/uav_collision_counter", f"uav{i}_uav")

    def watch_topic(self, topic: str, label: str):
        self.last_values[label] = None
        rospy.Subscriber(topic, Int32, self.make_callback(label, topic), queue_size=1)

    def make_callback(self, label: str, topic: str):
        def callback(msg: Int32):
            value = int(msg.data)
            last = self.last_values[label]
            if last is None:
                self.last_values[label] = value
                print(f"[{time.time() - self.start_time:7.3f}s] {topic} = {value}")
                return

            if value != last:
                delta = value - last
                print(f"[{time.time() - self.start_time:7.3f}s] {topic} = {value} (delta {delta:+d})")
                self.last_values[label] = value

        return callback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav-num", type=int, default=4, help="number of UAVs to monitor")
    parser.add_argument("--timeout", type=float, default=0.0, help="exit after timeout seconds; 0 means run forever")
    args = parser.parse_args()

    rospy.init_node("collision_event_watcher", anonymous=True, disable_signals=True)
    CollisionWatcher(args.uav_num)

    deadline = None if args.timeout <= 0 else time.time() + args.timeout
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if deadline is not None and time.time() >= deadline:
            break
        rate.sleep()


if __name__ == "__main__":
    main()
