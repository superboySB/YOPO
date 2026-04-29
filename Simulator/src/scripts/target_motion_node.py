#!/usr/bin/env python3
import math

import rospy
from geometry_msgs.msg import Point, PoseStamped, Vector3
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker


class TargetMotionNode:
    def __init__(self):
        rospy.init_node("target_motion_node", anonymous=False)
        self.odom_topic = rospy.get_param("~odom_topic", "/target/odom")
        self.marker_topic = rospy.get_param("~marker_topic", "/target/marker")
        self.goal_topic = rospy.get_param("~goal_topic", "/move_base_simple/goal")
        self.frame_id = rospy.get_param("~frame_id", "world")
        self.rate = float(rospy.get_param("~rate", 50.0))
        self.mode = rospy.get_param("~mode", "circle")
        self.radius = float(rospy.get_param("~radius", 7.0))
        self.speed = float(rospy.get_param("~speed", 1.2))
        self.height = float(rospy.get_param("~height", 2.5))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.25))
        self.center_x = float(rospy.get_param("~center_x", 8.0))
        self.center_y = float(rospy.get_param("~center_y", 0.0))
        self.phase = float(rospy.get_param("~phase", 0.0))

        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=1)
        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=1)
        self.goal_sub = rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback, queue_size=1)
        self.start_time = rospy.Time.now()
        initial_pos, initial_vel, initial_yaw = self.sample_scripted(0.0)
        self.current_pos = list(initial_pos)
        self.current_vel = list(initial_vel)
        self.current_yaw = initial_yaw
        self.goal_pos = None

    def sample_scripted(self, t):
        omega = self.speed / max(self.radius, 1e-3)
        a = omega * t + self.phase
        if self.mode == "line":
            x = self.center_x + self.speed * t
            y = self.center_y + 2.0 * math.sin(0.5 * a)
            vx = self.speed
            vy = self.speed * math.cos(0.5 * a)
        elif self.mode == "figure8":
            x = self.center_x + self.radius * math.sin(a)
            y = self.center_y + 0.5 * self.radius * math.sin(2.0 * a)
            vx = self.radius * omega * math.cos(a)
            vy = self.radius * omega * math.cos(2.0 * a)
        else:
            x = self.center_x + self.radius * math.cos(a)
            y = self.center_y + self.radius * math.sin(a)
            vx = -self.radius * omega * math.sin(a)
            vy = self.radius * omega * math.cos(a)
        z = self.height + 0.4 * math.sin(0.7 * a)
        vz = 0.4 * 0.7 * omega * math.cos(0.7 * a)
        yaw = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-4 else 0.0
        return (x, y, z), (vx, vy, vz), yaw

    def goal_callback(self, msg):
        goal_z = msg.pose.position.z if msg.pose.position.z > 0.1 else self.height
        self.goal_pos = [msg.pose.position.x, msg.pose.position.y, goal_z]
        rospy.loginfo(
            "Target waypoint set from RViz: x=%.2f y=%.2f z=%.2f",
            self.goal_pos[0],
            self.goal_pos[1],
            self.goal_pos[2],
        )

    def sample_waypoint(self, dt):
        if self.goal_pos is None:
            return tuple(self.current_pos), tuple(self.current_vel), self.current_yaw

        delta = [self.goal_pos[i] - self.current_pos[i] for i in range(3)]
        dist = math.sqrt(sum(d * d for d in delta))
        if dist <= self.goal_tolerance:
            self.current_pos = list(self.goal_pos)
            self.current_vel = [0.0, 0.0, 0.0]
            return tuple(self.current_pos), tuple(self.current_vel), self.current_yaw

        direction = [d / max(dist, 1e-6) for d in delta]
        step = min(self.speed * dt, dist)
        self.current_vel = [direction[i] * step / max(dt, 1e-6) for i in range(3)]
        self.current_pos = [self.current_pos[i] + direction[i] * step for i in range(3)]
        self.current_yaw = math.atan2(self.current_vel[1], self.current_vel[0])
        return tuple(self.current_pos), tuple(self.current_vel), self.current_yaw

    def publish_marker(self, stamp, pos, yaw):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = "target_drone"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(*pos)
        marker.pose.orientation.z = math.sin(0.5 * yaw)
        marker.pose.orientation.w = math.cos(0.5 * yaw)
        marker.scale = Vector3(0.7, 0.7, 0.25)
        marker.color.r = 1.0
        marker.color.g = 0.25
        marker.color.b = 0.08
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def spin(self):
        rate = rospy.Rate(self.rate)
        last_stamp = rospy.Time.now()
        while not rospy.is_shutdown():
            stamp = rospy.Time.now()
            t = (stamp - self.start_time).to_sec()
            dt = max((stamp - last_stamp).to_sec(), 1.0 / max(self.rate, 1.0))
            last_stamp = stamp
            if self.goal_pos is None:
                pos, vel, yaw = self.sample_scripted(t)
                self.current_pos = list(pos)
                self.current_vel = list(vel)
                self.current_yaw = yaw
            else:
                pos, vel, yaw = self.sample_waypoint(dt)
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = self.frame_id
            odom.child_frame_id = "target"
            odom.pose.pose.position = Point(*pos)
            odom.pose.pose.orientation.z = math.sin(0.5 * yaw)
            odom.pose.pose.orientation.w = math.cos(0.5 * yaw)
            odom.twist.twist.linear.x = vel[0]
            odom.twist.twist.linear.y = vel[1]
            odom.twist.twist.linear.z = vel[2]
            self.odom_pub.publish(odom)
            self.publish_marker(stamp, pos, yaw)
            rate.sleep()


if __name__ == "__main__":
    TargetMotionNode().spin()
