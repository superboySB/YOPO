#!/usr/bin/env python3
"""Rate-limited adapter from YOPO camera targets to a two-axis ROS gimbal.

The output topics use ``std_msgs/Float64`` radians and can be remapped to
ros_control position controllers or to a board-specific servo driver.  If
joint-state feedback is available it is used for visualization; otherwise the
node publishes its rate-limited command estimate.
"""

import argparse
import math
from threading import Lock

import numpy as np
import rospy
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64


class Insight9GimbalBridge:
    def __init__(self, args):
        rospy.init_node("insight9_gimbal_bridge", anonymous=False)
        self.pitch_limit = math.radians(args.pitch_limit_deg)
        self.yaw_limit = math.radians(args.yaw_limit_deg)
        self.max_rate = math.radians(args.max_rate_deg_s)
        self.dt = 1.0 / args.rate
        self.servo_tau = args.servo_tau
        self.pitch_joint = args.pitch_joint
        self.yaw_joint = args.yaw_joint
        self.target = np.zeros(2, dtype=np.float64)
        self.actual = np.zeros(2, dtype=np.float64)
        self.feedback_time = rospy.Time(0)
        self.lock = Lock()

        self.pitch_pub = rospy.Publisher(args.pitch_topic, Float64, queue_size=1)
        self.yaw_pub = rospy.Publisher(args.yaw_topic, Float64, queue_size=1)
        self.state_pub = rospy.Publisher("/yopo/camera/orientation", Vector3, queue_size=1)
        self.command_sub = rospy.Subscriber(
            "/yopo/camera/command", Vector3, self.command_callback, queue_size=1,
            tcp_nodelay=True,
        )
        self.joint_sub = rospy.Subscriber(
            args.joint_states_topic, JointState, self.joint_state_callback, queue_size=1,
            tcp_nodelay=True,
        )
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.update)
        rospy.loginfo(
            "Insight 9 gimbal bridge ready: pitch=%s yaw=%s limits=(%.1f, %.1f)deg",
            args.pitch_topic, args.yaw_topic, args.pitch_limit_deg, args.yaw_limit_deg,
        )

    def command_callback(self, message):
        with self.lock:
            self.target[0] = np.clip(message.x, -self.pitch_limit, self.pitch_limit)
            self.target[1] = np.clip(message.y, -self.yaw_limit, self.yaw_limit)

    def joint_state_callback(self, message):
        positions = dict(zip(message.name, message.position))
        if self.pitch_joint not in positions or self.yaw_joint not in positions:
            return
        with self.lock:
            self.actual[:] = [positions[self.pitch_joint], positions[self.yaw_joint]]
            self.feedback_time = rospy.Time.now()

    def update(self, _event):
        with self.lock:
            max_step = self.max_rate * self.dt
            if (rospy.Time.now() - self.feedback_time).to_sec() > 0.25:
                first_order_step = (self.target - self.actual) * self.dt / self.servo_tau
                self.actual += np.clip(first_order_step, -max_step, max_step)
            pitch, yaw = self.actual.copy()
            pitch_command, yaw_command = self.target.copy()
        self.pitch_pub.publish(Float64(data=float(pitch_command)))
        self.yaw_pub.publish(Float64(data=float(yaw_command)))
        self.state_pub.publish(Vector3(x=float(pitch), y=float(yaw), z=0.0))


def parse_args():
    parser = argparse.ArgumentParser(description="YOPO to Insight 9 two-axis gimbal bridge")
    parser.add_argument("--pitch-topic", default="/gimbal/pitch_position_controller/command")
    parser.add_argument("--yaw-topic", default="/gimbal/yaw_position_controller/command")
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--pitch-joint", default="insight9_pitch_joint")
    parser.add_argument("--yaw-joint", default="insight9_yaw_joint")
    parser.add_argument("--pitch-limit-deg", type=float, default=60.0)
    parser.add_argument("--yaw-limit-deg", type=float, default=45.0)
    parser.add_argument("--max-rate-deg-s", type=float, default=120.0)
    parser.add_argument("--servo-tau", type=float, default=0.25)
    parser.add_argument("--rate", type=float, default=50.0)
    args = parser.parse_args()
    if args.rate <= 0.0 or args.servo_tau <= 0.0 or args.max_rate_deg_s <= 0.0:
        parser.error("rate, servo-tau and max-rate-deg-s must be positive")
    return args


if __name__ == "__main__":
    Insight9GimbalBridge(parse_args())
    rospy.spin()
