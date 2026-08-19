import argparse
import os
import time
from threading import Lock

import cv2
import numpy as np
import rospy
import std_msgs.msg
import torch
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

from config.config import cfg
from control_msg import PositionCommand
from policy.poly_solver import Poly5Solver, Polys5Solver, calculate_yaw
from policy.yopo_network import YOPOOmniNetwork


def apply_runtime_overrides(args):
    if args.radius_min is not None:
        cfg["omni_radius_min"] = args.radius_min
    if args.radius_max is not None:
        cfg["omni_radius_max"] = args.radius_max
    if args.radio_range is not None:
        cfg["radio_range"] = args.radio_range
        cfg["goal_length"] = 2.0 * args.radio_range
        cfg["sgm_time"] = 2.0 * args.radio_range / float(cfg["vel_max_train"])
    if args.sgm_time is not None:
        cfg["sgm_time"] = args.sgm_time


class YopoActiveNet:
    def __init__(self, settings, weight):
        rospy.init_node("yopo_net", anonymous=False)
        cfg["train"] = False

        self.settings = settings
        self.weight = weight
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.max_depth = float(settings["max_depth"])
        self.velocity = float(settings["velocity"])
        self.traj_time = float(cfg["sgm_time"])
        self.ctrl_dt = float(settings["ctrl_dt"])
        self.arrive_dist = float(settings["arrive_dist"])
        self.depth_unit_logged = False
        self.verbose = bool(settings["verbose"])
        self.visualize = bool(settings["visualize"])
        self.plan_from_reference = bool(settings["plan_from_reference"])
        self.fixed_yaw = bool(settings["fixed_yaw"])
        self.fixed_yaw_value = settings["fixed_yaw_value"]
        self.goal = np.asarray(settings["goal"], dtype=np.float32)
        self.camera_orientation = np.zeros(2, dtype=np.float32)
        self.camera_target = np.zeros(2, dtype=np.float32)
        self.camera_mount = np.asarray([0.10, 0.0, 0.03], dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = YOPOOmniNetwork().to(self.device)
        state_dict = torch.load(weight, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(state_dict, strict=True)
        self.policy.eval()
        self.warm_up()

        self.odom = Odometry()
        self.odom_init = False
        self.goal_init = False
        self.desire_init = False
        self.arrive = False
        self.last_yaw = 0.0
        self.ctrl_time = None
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.last_control_msg = None
        self.lock = Lock()

        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_depth = 0.0
        self.time_visualize = 0.0
        self.count = 0

        self.lattice_traj_pub = rospy.Publisher("/yopo/topology_endstates_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo/trajs_visual", PointCloud2, queue_size=1)
        self.speed_text_pub = rospy.Publisher("/yopo/speed_text_visual", Marker, queue_size=1)
        self.active_camera_pub = rospy.Publisher("/yopo/active_camera_visual", MarkerArray, queue_size=1)
        self.camera_command_pub = rospy.Publisher("/yopo/camera/command", Vector3, queue_size=1)
        self.ctrl_pub = rospy.Publisher(settings["ctrl_topic"], PositionCommand, queue_size=1)

        self.odom_sub = rospy.Subscriber(settings["odom_topic"], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(
            settings["depth_topic"], Image, self.callback_depth, queue_size=1, tcp_nodelay=True
        )
        self.camera_state_sub = rospy.Subscriber(
            "/yopo/camera/orientation", Vector3, self.callback_camera_state, queue_size=1,
            tcp_nodelay=True,
        )
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)

        print("YOPO active-perception node ready!")
        print("Waiting for /move_base_simple/goal; the UAV will hold after takeoff until a target is received.")
        print("Insight 9 depth topic:", settings["depth_topic"])
        print("load weight from:", weight)
        print(
            "Runtime decode:",
            f"radius=[{float(cfg['omni_radius_min']):.2f}, {float(cfg['omni_radius_max']):.2f}],",
            f"sgm_time={float(cfg['sgm_time']):.3f}s,",
            f"velocity={self.velocity:.2f}m/s",
        )
        rospy.spin()

    def warm_up(self):
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        state = torch.zeros((1, 11), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            self.policy(depth, state)

    def callback_set_goal(self, data):
        z = self.goal[2] if len(self.goal) >= 3 else 2.0
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, z], dtype=np.float32)
        self.goal_init = True
        self.arrive = False
        print(f"New Goal: ({self.goal[0]:.1f}, {self.goal[1]:.1f}, {self.goal[2]:.1f})")

    def callback_camera_state(self, data):
        self.camera_orientation[:] = [data.x, data.y]

    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array(
                [data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z],
                dtype=np.float32,
            )
            self.desire_vel = np.array(
                [data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z],
                dtype=np.float32,
            )
            self.desire_acc = np.zeros(3, dtype=np.float32)
            ypr = R.from_quat(
                [
                    data.pose.pose.orientation.x,
                    data.pose.pose.orientation.y,
                    data.pose.pose.orientation.z,
                    data.pose.pose.orientation.w,
                ]
            ).as_euler("ZYX", degrees=False)
            self.last_yaw = ypr[0]
            if self.fixed_yaw and self.fixed_yaw_value is None:
                self.fixed_yaw_value = ypr[0]
                print(f"Fixed yaw locked to initial odometry yaw: {self.fixed_yaw_value:.3f} rad")
        self.odom_init = True

        pos = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z], dtype=np.float32)
        dist_to_goal = np.linalg.norm(pos - self.goal)
        if self.goal_init and dist_to_goal < self.arrive_dist and not self.arrive:
            print(f"Arrive! dist={dist_to_goal:.2f} m, threshold={self.arrive_dist:.2f} m")
            self.arrive = True
        self.publish_speed_text(data)
        self.publish_active_camera_visual(data)

    def publish_active_camera_visual(self, odom):
        if self.active_camera_pub.get_num_connections() <= 0:
            return
        stamp = rospy.Time.now()
        position = np.asarray(
            [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z],
            dtype=np.float32,
        )
        q = odom.pose.pose.orientation
        body_rotation = R.from_quat([q.x, q.y, q.z, q.w])
        camera_rotation = body_rotation * R.from_euler(
            "ZY", [float(self.camera_orientation[1]), float(self.camera_orientation[0])]
        )
        camera_position = position + body_rotation.apply(self.camera_mount)

        body = Marker()
        body.header.stamp, body.header.frame_id = stamp, "world"
        body.ns, body.id, body.type, body.action = "active_uav", 0, Marker.CUBE, Marker.ADD
        body.pose = odom.pose.pose
        body.scale.x, body.scale.y, body.scale.z = 0.45, 0.45, 0.12
        body.color.r, body.color.g, body.color.b, body.color.a = 0.15, 0.35, 0.95, 0.8

        camera = Marker()
        camera.header.stamp, camera.header.frame_id = stamp, "world"
        camera.ns, camera.id, camera.type, camera.action = "insight9", 1, Marker.CUBE, Marker.ADD
        camera.pose.position.x, camera.pose.position.y, camera.pose.position.z = map(float, camera_position)
        camera_q = camera_rotation.as_quat()
        camera.pose.orientation.x, camera.pose.orientation.y = float(camera_q[0]), float(camera_q[1])
        camera.pose.orientation.z, camera.pose.orientation.w = float(camera_q[2]), float(camera_q[3])
        # Insight 9 is 129 mm wide across the stereo baseline, 33.9 mm deep
        # along the optical axis, and 35 mm high.  The simulator camera looks
        # along local +X, so RViz dimensions are depth, width, height.
        camera.scale.x, camera.scale.y, camera.scale.z = 0.0339, 0.129, 0.035
        camera.color.r, camera.color.g, camera.color.b, camera.color.a = 1.0, 0.45, 0.05, 1.0

        frustum = Marker()
        frustum.header.stamp, frustum.header.frame_id = stamp, "world"
        frustum.ns, frustum.id, frustum.type, frustum.action = "insight9_fov", 2, Marker.LINE_LIST, Marker.ADD
        frustum.pose.orientation.w = 1.0
        frustum.scale.x = 0.025
        frustum.color.r, frustum.color.g, frustum.color.b, frustum.color.a = 1.0, 0.65, 0.05, 0.8
        distance = 2.0
        half_y = np.tan(np.deg2rad(float(cfg["insight9_horizontal_fov_deg"])) / 2.0) * distance
        half_z = np.tan(np.deg2rad(float(cfg["insight9_vertical_fov_deg"])) / 2.0) * distance
        origin = camera_position
        corners = [camera_position + camera_rotation.apply([distance, sy * half_y, sz * half_z])
                   for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        from geometry_msgs.msg import Point
        for corner in corners:
            for point in (origin, corner):
                frustum.points.append(Point(x=float(point[0]), y=float(point[1]), z=float(point[2])))
        for first, second in ((0, 1), (0, 2), (1, 3), (2, 3)):
            for point in (corners[first], corners[second]):
                frustum.points.append(Point(x=float(point[0]), y=float(point[1]), z=float(point[2])))
        self.active_camera_pub.publish(MarkerArray(markers=[body, camera, frustum]))

    def publish_speed_text(self, odom):
        if self.speed_text_pub.get_num_connections() <= 0:
            return

        vel = odom.twist.twist.linear
        speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
        pos = odom.pose.pose.position

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "world"
        marker.ns = "yopo_speed"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = pos.x
        marker.pose.position.y = pos.y
        marker.pose.position.z = pos.z + 1.6
        marker.pose.orientation.w = 1.0
        marker.scale.z = 3.0
        marker.color.r = 0.02
        marker.color.g = 0.12
        marker.color.b = 0.95
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0.3)
        marker.text = f"v = {speed:.2f} m/s"
        self.speed_text_pub.publish(marker)

    def preprocess_depth(self, data):
        if data.encoding == "32FC1":
            # Simulator convention: floating-point metric depth in metres.
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
            source_unit = "metres"
        elif data.encoding == "16UC1":
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32)
            # The Insight 9 Z16 stream and the ROS depth-image convention use
            # unsigned millimetres.  Do not infer units from the pixel range:
            # an all-near frame can legitimately contain only values <255 mm.
            depth = depth / 1000.0
            source_unit = "millimetres converted to metres"
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}")

        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        finite_depth = depth[np.isfinite(depth)]
        max_value = float(finite_depth.max()) if finite_depth.size else 0.0
        depth_norm = depth.astype(np.float32) / self.max_depth

        depth_norm = np.nan_to_num(depth_norm, nan=1.0, posinf=1.0, neginf=0.0)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        if not self.depth_unit_logged:
            print(
                f"Depth input: encoding={data.encoding}, raw_max={max_value:.3f}, "
                f"mode={source_unit} / {self.max_depth:g}, "
                f"normalized_range=[{depth_norm.min():.4f}, {depth_norm.max():.4f}]"
            )
            self.depth_unit_logged = True
        return depth_norm[None, ...]

    def process_state(self):
        q = self.odom.pose.pose.orientation
        rot_wb = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
        rot_bw = rot_wb.T

        if self.plan_from_reference:
            pos_w = self.desire_pos
            vel_w = self.desire_vel
        else:
            pos_w = np.array(
                [self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z],
                dtype=np.float32,
            )
            vel_w = np.array(
                [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z],
                dtype=np.float32,
            )
        acc_w = self.desire_acc
        goal_dir_w = self.goal - pos_w
        goal_norm = np.linalg.norm(goal_dir_w)
        if goal_norm < 1e-4:
            vdes_w = np.zeros(3, dtype=np.float32)
        else:
            vdes_w = self.velocity * goal_dir_w / goal_norm

        vel_b = rot_bw @ vel_w
        acc_b = rot_bw @ acc_w
        vdes_b = rot_bw @ vdes_w
        state_b = np.concatenate([vel_b, acc_b, vdes_b, self.camera_orientation]).astype(np.float32)
        return state_b, pos_w.astype(np.float32), vel_w.astype(np.float32), acc_w.astype(np.float32), rot_wb

    @torch.inference_mode()
    def callback_depth(self, depth_msg):
        if not self.odom_init or not self.goal_init:
            return

        time0 = time.time()
        depth = self.preprocess_depth(depth_msg).reshape(1, 1, self.height, self.width)
        time1 = time.time()

        state_b, start_pos, start_vel, start_acc, rot_wb = self.process_state()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        state_input = torch.from_numpy(state_b[None, :]).to(self.device, non_blocking=True)
        time2 = time.time()

        endstate_b, score, camera_target = self.policy(depth_input, state_input)
        endstate_b = endstate_b[0].detach().cpu().numpy()
        score = score[0].detach().cpu().numpy()
        camera_target = camera_target[0].detach().cpu().numpy()
        time3 = time.time()

        action_id = int(np.argmin(score))
        self.camera_target = camera_target.reshape(-1, 2)[action_id]
        command = Vector3(
            x=float(self.camera_target[0]), y=float(self.camera_target[1]), z=0.0
        )
        self.camera_command_pub.publish(command)
        endstate_cand_b = endstate_b.reshape(-1, 9)
        score_flat = score.reshape(-1)
        best_b = endstate_cand_b[action_id]
        best_state = best_b.reshape(3, 3).T
        best_state_w = rot_wb @ best_state
        # /move_base_simple/goal is the historical YOPO-Simple fixed-altitude
        # interface.  The learned policy plans horizontal avoidance while this
        # terminal constraint prevents accumulated vertical drift at 15 Hz.
        best_state_w[2, 0] = float(self.goal[2] - start_pos[2])
        best_state_w[2, 1] = 0.0
        best_state_w[2, 2] = 0.0

        with self.lock:
            self.optimal_poly_x = Poly5Solver(
                start_pos[0], start_vel[0], start_acc[0],
                best_state_w[0, 0] + start_pos[0], best_state_w[0, 1], best_state_w[0, 2], self.traj_time
            )
            self.optimal_poly_y = Poly5Solver(
                start_pos[1], start_vel[1], start_acc[1],
                best_state_w[1, 0] + start_pos[1], best_state_w[1, 1], best_state_w[1, 2], self.traj_time
            )
            self.optimal_poly_z = Poly5Solver(
                start_pos[2], start_vel[2], start_acc[2],
                best_state_w[2, 0] + start_pos[2], best_state_w[2, 1], best_state_w[2, 2], self.traj_time
            )
            self.ctrl_time = 0.0
        time4 = time.time()

        self.visualize_trajectory(start_pos, start_vel, start_acc, rot_wb, endstate_cand_b, score_flat)
        time5 = time.time()
        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return

        with self.lock:
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z], dtype=np.float32)
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z], dtype=np.float32)
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z], dtype=np.float32)
            if self.fixed_yaw:
                yaw = self.last_yaw if self.fixed_yaw_value is None else float(self.fixed_yaw_value)
                yaw_dot = 0.0
            else:
                goal_dir = self.goal - self.desire_pos
                yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
                self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def visualize_trajectory(self, start_pos, start_vel, start_acc, rot_wb, endstate_b, score):
        dt = self.traj_time / 20.0
        t_values = np.arange(0, self.traj_time, dt)

        if self.best_traj_pub.get_num_connections() > 0 and self.optimal_poly_x is not None:
            points_array = np.stack(
                [
                    self.optimal_poly_x.get_position(t_values),
                    self.optimal_poly_y.get_position(t_values),
                    self.optimal_poly_z.get_position(t_values),
                ],
                axis=-1,
            )
            header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id="world")
            self.best_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, points_array))

        if not self.visualize:
            return

        states_w = np.matmul(rot_wb, endstate_b.reshape(-1, 3, 3).transpose(0, 2, 1))
        if self.lattice_traj_pub.get_num_connections() > 0:
            header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id="world")
            points = states_w[:, :, 0] + start_pos[None, :]
            self.lattice_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, points))

        if self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], start_acc[0],
                                      states_w[:, 0, 0] + start_pos[0], states_w[:, 0, 1], states_w[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], start_acc[1],
                                      states_w[:, 1, 0] + start_pos[1], states_w[:, 1, 1], states_w[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], start_acc[2],
                                      states_w[:, 2, 0] + start_pos[2], states_w[:, 2, 1], states_w[:, 2, 2], self.traj_time)
            points_array = np.stack(
                [all_poly_x.get_position(t_values), all_poly_y.get_position(t_values), all_poly_z.get_position(t_values)],
                axis=-1,
            )
            scores = np.repeat(score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id="world")
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("intensity", 12, PointField.FLOAT32, 1),
            ]
            self.all_trajs_pub.publish(point_cloud2.create_cloud(header, fields, points_array))

    def print_time(self, time0, time1, time2, time3, time4, time5):
        self.time_depth += time1 - time0
        self.time_prepare += time2 - time1
        self.time_forward += time3 - time2
        self.time_process += time4 - time3
        self.time_visualize += time5 - time4
        self.count += 1
        if self.verbose or self.count % 30 == 0:
            print(
                f"YOPO-Active avg ms | depth {1000*self.time_depth/self.count:.2f}, "
                f"prepare {1000*self.time_prepare/self.count:.2f}, "
                f"forward {1000*self.time_forward/self.count:.2f}, "
                f"process {1000*self.time_process/self.count:.2f}, "
                f"visualize {1000*self.time_visualize/self.count:.2f}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="", help="Path to YOPO active-perception checkpoint.")
    parser.add_argument("--trial", type=int, default=0, help="Trial number under YOPO/saved/YOPO_{trial}.")
    parser.add_argument("--epoch", type=int, default=30, help="Checkpoint epoch.")
    parser.add_argument("--velocity", type=float, default=float(cfg["velocity"]), help="Desired speed magnitude.")
    parser.add_argument("--goal-height", type=float, default=2.0,
                        help="Fixed YOPO-Simple target altitude in world coordinates.")
    parser.add_argument("--odom-topic", default="/sim/odom", help="Vehicle odometry topic.")
    parser.add_argument("--depth-topic", default="/depth_image", help="Insight 9 Z16/32FC1 depth topic.")
    parser.add_argument("--ctrl-topic", default="/so3_control/pos_cmd",
                        help="quadrotor_msgs/PositionCommand output topic.")
    parser.add_argument("--max-depth", type=float, default=float(cfg["insight9_train_max_depth_m"]),
                        help="Insight 9 depth max range used for normalization.")
    parser.add_argument("--arrive-dist", type=float, default=2.0,
                        help="Braking trigger distance in meters; tuned for the simulator controller at 3 m/s.")
    parser.add_argument("--radius-min", type=float, default=None, help="Override omni_radius_min for checkpoint-consistent decoding.")
    parser.add_argument("--radius-max", type=float, default=None, help="Override omni_radius_max for checkpoint-consistent decoding.")
    parser.add_argument("--radio-range", type=float, default=None,
                        help="Override radio_range and recompute sgm_time unless --sgm-time is also set.")
    parser.add_argument("--sgm-time", type=float, default=None,
                        help="Override trajectory segment time. Must match training for fair tests.")
    parser.add_argument("--visualize", type=int, default=1, help="Publish all candidate trajectories.")
    parser.add_argument("--verbose", type=int, default=0, help="Print timing every frame.")
    parser.add_argument("--fixed-yaw", action="store_true", help="Keep yaw fixed instead of turning toward the goal.")
    parser.add_argument(
        "--fixed-yaw-value",
        type=float,
        default=None,
        help="Fixed yaw in radians. If omitted with --fixed-yaw, lock to the initial odometry yaw.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    apply_runtime_overrides(args)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = args.weight or os.path.join(base_dir, "saved", f"YOPO_{args.trial}", f"epoch{args.epoch}.pth")

    settings = {
        "goal": [50.0, 0.0, args.goal_height],
        "velocity": args.velocity,
        "max_depth": args.max_depth,
        "arrive_dist": args.arrive_dist,
        "ctrl_dt": 0.02,
        "odom_topic": args.odom_topic,
        "depth_topic": args.depth_topic,
        "ctrl_topic": args.ctrl_topic,
        "plan_from_reference": False,
        "verbose": bool(args.verbose),
        "visualize": bool(args.visualize),
        "fixed_yaw": args.fixed_yaw,
        "fixed_yaw_value": args.fixed_yaw_value,
    }
    YopoActiveNet(settings, weight)
