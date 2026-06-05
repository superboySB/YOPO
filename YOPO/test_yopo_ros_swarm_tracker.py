import argparse
import os
import time
from threading import Lock

import cv2
import numpy as np
import rospy
import std_msgs.msg
import torch
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool, Float32, Int32
from visualization_msgs.msg import Marker

from config.config import cfg
from control_msg import PositionCommand
from policy.poly_solver import Poly5Solver, Polys5Solver, calculate_yaw
from policy.primitive import LatticePrimitive
from policy.state_transform import StateTransform
from policy.yopo_network import YopoNetwork

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoSwarmTracker:
    def __init__(self, config, weight):
        self.config = config
        self.agent_name = self.config["agent_name"]
        rospy.init_node(self.config["node_name"], anonymous=False)
        cfg["train"] = False
        self.height = cfg["image_height"]
        self.width = cfg["image_width"]
        self.min_dis = 0.04
        self.max_dis = float(self.config["max_depth_dist"])
        self.goal = np.asarray(self.config["goal"], dtype=np.float64)
        self.arrive_radius = float(self.config["arrive_radius"])
        self.arrival_settle_radius = max(self.arrive_radius, float(self.config["arrival_settle_radius"]))
        self.arrival_settle_speed = float(self.config["arrival_settle_speed"])
        self.arrival_settle_time = float(self.config["arrival_settle_time"])
        self.arrival_stuck_radius = max(self.arrive_radius, float(self.config["arrival_stuck_radius"]))
        self.arrival_stuck_timeout = float(self.config["arrival_stuck_timeout"])
        self.arrival_min_progress = float(self.config["arrival_min_progress"])
        self.plan_from_reference = self.config["plan_from_reference"]
        self.use_trt = self.config["use_tensorrt"]
        self.verbose = self.config["verbose"]
        self.visualize = self.config["visualize"]
        self.target_mask_timeout = cfg["target_mask_timeout"]
        self.enable_rviz_goal = bool(self.config["enable_rviz_goal"])
        self.rviz_goal_reference_odom_topic = self.config["rviz_goal_reference_odom_topic"]
        self.min_altitude = float(self.config["min_altitude"])
        self.Rotation_bc = R.from_euler("ZYX", [0, self.config["pitch_angle_deg"], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.odom = Odometry()
        self.odom_init = False
        self.reference_odom = Odometry()
        self.reference_odom_init = False
        self.latest_target_mask = None
        self.latest_target_mask_stamp = None
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.static_collision_count = 0
        self.dynamic_collision_count = 0
        self.arrive_hold_pos = None
        self.arrival_close_since = None
        self.arrival_best_distance = np.inf
        self.arrival_best_stamp = None
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.hold_mode = False
        self.pending_rviz_goal = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time

        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = float(self.config["depth_fps"])

        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = YopoNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        self.lattice_traj_pub = rospy.Publisher(f"{self.config['visual_prefix']}/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher(f"{self.config['visual_prefix']}/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher(f"{self.config['visual_prefix']}/trajs_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        self.status_text_pub = rospy.Publisher(f"{self.config['visual_prefix']}/status_text", Marker, queue_size=1)
        self.arrive_pub = rospy.Publisher(f"{self.config['status_prefix']}/arrived", Bool, queue_size=1, latch=True)
        self.goal_distance_pub = rospy.Publisher(f"{self.config['status_prefix']}/goal_distance", Float32, queue_size=1, latch=True)
        self.goal_pub = rospy.Publisher(f"{self.config['status_prefix']}/goal", PoseStamped, queue_size=1, latch=True)

        self.odom_sub = rospy.Subscriber(self.config["odom_topic"], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.config["depth_topic"], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.target_mask_sub = rospy.Subscriber(self.config["target_mask_topic"], Image, self.callback_target_mask, queue_size=1, tcp_nodelay=True)
        self.static_collision_sub = rospy.Subscriber(
            f"{self.config['status_prefix']}/collision_counter",
            Int32,
            self.callback_static_collision_counter,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.dynamic_collision_sub = rospy.Subscriber(
            f"{self.config['status_prefix']}/uav_collision_counter",
            Int32,
            self.callback_dynamic_collision_counter,
            queue_size=1,
            tcp_nodelay=True,
        )
        if self.enable_rviz_goal:
            self.goal_sub = rospy.Subscriber(
                self.config["rviz_goal_topic"],
                PoseStamped,
                self.callback_rviz_goal,
                queue_size=1,
                tcp_nodelay=True,
            )
            self.reference_odom_sub = rospy.Subscriber(
                self.rviz_goal_reference_odom_topic,
                Odometry,
                self.callback_reference_odometry,
                queue_size=1,
                tcp_nodelay=True,
            )
        rospy.sleep(1.0)
        self.publish_status()
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print(f"[{self.agent_name}] YOPOv2 swarm tracker ready. goal={self.goal.tolist()}")
        rospy.spin()

    def callback_target_mask(self, data):
        if data.encoding in ("mono8", "8UC1"):
            mask = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width)
        elif data.encoding == "32FC1":
            mask = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
            mask = np.uint8(np.clip(mask, 0.0, 1.0) * 255)
        elif data.encoding in ("bgr8", "rgb8"):
            image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, 3)
            mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return
        self.latest_target_mask = mask.copy()
        self.latest_target_mask_stamp = data.header.stamp

    def callback_static_collision_counter(self, msg):
        self.static_collision_count = int(msg.data)

    def callback_dynamic_collision_counter(self, msg):
        self.dynamic_collision_count = int(msg.data)

    def callback_rviz_goal(self, data):
        if not self.odom_init or not self.reference_odom_init:
            self.pending_rviz_goal = data
            rospy.loginfo(f"[{self.agent_name}] queued RViz goal until odometry is ready.")
            return

        self.apply_rviz_goal(data)

    def callback_reference_odometry(self, data):
        self.reference_odom = data
        self.reference_odom_init = True
        self.apply_pending_rviz_goal()

    def apply_pending_rviz_goal(self):
        if self.pending_rviz_goal is None or not self.odom_init or not self.reference_odom_init:
            return
        pending_goal = self.pending_rviz_goal
        self.pending_rviz_goal = None
        self.apply_rviz_goal(pending_goal)

    def get_reference_position(self):
        return np.array((
            self.reference_odom.pose.pose.position.x,
            self.reference_odom.pose.pose.position.y,
            self.reference_odom.pose.pose.position.z,
        ), dtype=np.float64)

    def apply_rviz_goal(self, data):
        reference_target_xy = np.array((data.pose.position.x, data.pose.position.y), dtype=np.float64)
        if not np.all(np.isfinite(reference_target_xy)):
            rospy.logwarn(f"[{self.agent_name}] ignore RViz goal with invalid target position.")
            return

        reference_pos = self.get_reference_position()
        start_pos = self.get_current_position(from_odom=True)
        displacement_xy = reference_target_xy - reference_pos[:2]
        new_goal = start_pos.copy()
        new_goal[:2] += displacement_xy
        new_goal[2] = self.goal[2]
        current_vel = np.array((
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ), dtype=np.float64)

        with self.lock:
            self.goal = new_goal
            self.arrive = False
            self.arrive_hold_pos = None
            self.reset_arrival_state()
            self.hold_mode = False
            self.desire_pos = start_pos.copy()
            self.desire_vel = current_vel
            self.desire_acc = np.zeros(3)
            self.desire_init = False
            self.ctrl_time = None
            self.optimal_poly_x = None
            self.optimal_poly_y = None
            self.optimal_poly_z = None

        self.publish_status(np.linalg.norm(self.goal - start_pos))
        print(
            f"[{self.agent_name}] RViz formation goal: "
            f"uav0_target=({reference_target_xy[0]:.2f}, {reference_target_xy[1]:.2f}), "
            f"displacement=({displacement_xy[0]:.2f}, {displacement_xy[1]:.2f}), "
            f"goal={self.goal.tolist()}"
        )

    def callback_odometry(self, data):
        self.odom = data
        if self.config["odom_topic"] == self.rviz_goal_reference_odom_topic:
            self.reference_odom = data
            self.reference_odom_init = True
        if not self.desire_init:
            self.desire_pos = np.array((data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z))
            self.desire_vel = np.array((data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z))
            self.desire_acc = np.zeros(3)
            ypr = R.from_quat([data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                               data.pose.pose.orientation.z, data.pose.pose.orientation.w]).as_euler("ZYX", degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True
        self.apply_pending_rviz_goal()

        pos = self.get_current_position(from_odom=True)
        goal_distance = np.linalg.norm(self.goal - pos)
        if self.update_arrival_state(pos, goal_distance):
            goal_distance = np.linalg.norm(self.goal - pos)
        self.publish_status(goal_distance)

    def reset_arrival_state(self):
        self.arrival_close_since = None
        self.arrival_best_distance = np.inf
        self.arrival_best_stamp = None

    def update_arrival_state(self, pos, goal_distance):
        if self.arrive:
            return False

        now = rospy.Time.now()
        speed = np.linalg.norm((
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ))

        if goal_distance <= self.arrive_radius:
            self.mark_arrived(pos, adjusted=False, reason="within radius")
            return True

        if goal_distance < self.arrival_best_distance - self.arrival_min_progress:
            self.arrival_best_distance = goal_distance
            self.arrival_best_stamp = now
        elif self.arrival_best_stamp is None:
            self.arrival_best_distance = goal_distance
            self.arrival_best_stamp = now

        if goal_distance <= self.arrival_settle_radius and speed <= self.arrival_settle_speed:
            if self.arrival_close_since is None:
                self.arrival_close_since = now
            elif (now - self.arrival_close_since).to_sec() >= self.arrival_settle_time:
                self.mark_arrived(pos, adjusted=True, reason="settled near goal")
                return True
        else:
            self.arrival_close_since = None

        stuck_time = (now - self.arrival_best_stamp).to_sec() if self.arrival_best_stamp is not None else 0.0
        if goal_distance <= self.arrival_stuck_radius and stuck_time >= self.arrival_stuck_timeout:
            self.mark_arrived(pos, adjusted=True, reason="goal likely unreachable")
            return True

        return False

    def mark_arrived(self, pos, adjusted, reason):
        self.arrive = True
        self.arrive_hold_pos = pos.copy()
        if adjusted:
            self.goal = pos.copy()
        self.reset_arrival_state()
        mode = "adjusted" if adjusted else "arrived"
        print(f"[{self.agent_name}] {mode}: {reason}, hold at {self.arrive_hold_pos.tolist()}")

    def process_odom(self):
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        vel_w = self.desire_vel if self.plan_from_reference else np.array([
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_c = np.dot(Rotation_cw, self.desire_acc)
        start_pos = self.get_current_position()
        goal_c = np.dot(Rotation_cw, self.goal - start_pos)
        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        return self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))

    def get_current_position(self, from_odom=False):
        if not from_odom and self.plan_from_reference and self.desire_pos is not None:
            return self.desire_pos
        return np.array((
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ), dtype=np.float64)

    def get_current_velocity(self):
        if self.plan_from_reference and self.desire_vel is not None:
            return self.desire_vel
        return np.array((
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ), dtype=np.float64)

    def make_image_input(self, depth_msg):
        if depth_msg.encoding == "32FC1":
            depth_m = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
        elif depth_msg.encoding == "16UC1":
            depth_m = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(depth_msg.height, depth_msg.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {depth_msg.encoding}. Expected '32FC1' or '16UC1'.")

        depth = depth_m
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            depth_m = cv2.resize(depth_m, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth, self.max_dis) / self.max_dis
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        depth_u8 = np.uint8(np.nan_to_num(depth, nan=0.0) * 255)
        depth = cv2.inpaint(depth_u8, np.uint8(nan_mask), 1, cv2.INPAINT_NS).astype(np.float32) / 255.0

        target_mask = self.latest_target_mask
        mask_is_fresh = False
        if target_mask is not None and self.latest_target_mask_stamp is not None:
            stamp_delta = abs((depth_msg.header.stamp - self.latest_target_mask_stamp).to_sec())
            mask_is_fresh = stamp_delta <= self.target_mask_timeout
        if target_mask is None or not mask_is_fresh:
            target_mask = np.zeros((self.height, self.width), dtype=np.float32)
        else:
            target_mask = cv2.resize(target_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            target_mask = target_mask.astype(np.float32) / 255.0

        image = np.concatenate((depth[None, :, :], target_mask[None, :, :]), axis=0)
        return image.reshape(1, cfg["input_channels"], self.height, self.width).astype(np.float32)

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init:
            return
        if self.arrive:
            self.publish_status()
            return

        time0 = time.time()
        obs_norm = self.process_odom()
        image = self.make_image_input(data)

        time1 = time.time()
        image_input = torch.from_numpy(image).to(self.device, non_blocking=True)
        obs_norm = obs_norm.to(self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input(obs_norm)

        time2 = time.time()
        endstate_pred, score_pred = self.policy(image_input, obs_input)
        endstate_pred = endstate_pred.cpu().numpy()
        score_pred = score_pred.cpu().numpy()
        time3 = time.time()

        endstate, score = self.process_output_all(endstate_pred, score_pred)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)
        start_pos = self.get_current_position()
        action_id = self.select_action(score)

        with self.lock:
            start_vel = self.get_current_velocity()
            self.hold_mode = False
            end_pos = endstate_w[action_id, :, 0] + start_pos
            end_vel = endstate_w[action_id, :, 1]
            end_acc = endstate_w[action_id, :, 2]
            end_pos[2] = max(float(end_pos[2]), self.min_altitude)
            self.set_optimal_poly(start_pos, start_vel, self.desire_acc, end_pos, end_vel, end_acc)
            self.ctrl_time = 0.0

        time4 = time.time()
        self.visualize_trajectory(score, endstate_w)
        time5 = time.time()
        self.print_time(time0, time1, time2, time3, time4, time5)

    def select_action(self, score):
        return int(np.argmin(score))

    def set_optimal_poly(self, start_pos, start_vel, start_acc, end_pos, end_vel, end_acc):
        self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], start_acc[0], end_pos[0], end_vel[0], end_acc[0], self.traj_time)
        self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], start_acc[1], end_pos[1], end_vel[1], end_acc[1], self.traj_time)
        self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], start_acc[2], end_pos[2], end_vel[2], end_acc[2], self.traj_time)

    def build_hover_command(self):
        control_msg = PositionCommand()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
        hover_pos = self.arrive_hold_pos if self.arrive_hold_pos is not None else self.get_current_position()
        control_msg.position.x = float(hover_pos[0])
        control_msg.position.y = float(hover_pos[1])
        control_msg.position.z = max(float(hover_pos[2]), self.min_altitude)
        control_msg.velocity.x = 0.0
        control_msg.velocity.y = 0.0
        control_msg.velocity.z = 0.0
        control_msg.acceleration.x = 0.0
        control_msg.acceleration.y = 0.0
        control_msg.acceleration.z = 0.0
        control_msg.yaw = float(self.last_yaw)
        control_msg.yaw_dot = 0.0
        return control_msg

    def control_pub(self, _timer):
        if self.arrive:
            self.hold_mode = True
            self.ctrl_pub.publish(self.build_hover_command())
            self.publish_status()
            return
        if self.ctrl_time is None:
            return
        if self.ctrl_time > self.traj_time and not self.hold_mode:
            return

        with self.lock:
            self.ctrl_time += self.ctrl_dt
            eval_time = min(self.ctrl_time, self.traj_time) if self.hold_mode else self.ctrl_time
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY if self.hold_mode else control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(eval_time)
            control_msg.position.y = self.optimal_poly_y.get_position(eval_time)
            control_msg.position.z = max(self.optimal_poly_z.get_position(eval_time), self.min_altitude)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(eval_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(eval_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(eval_time)
            if control_msg.position.z <= self.min_altitude + 1e-3 and control_msg.velocity.z < 0.0:
                control_msg.velocity.z = 0.0
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(eval_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(eval_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(eval_time)
            if control_msg.position.z <= self.min_altitude + 1e-3 and control_msg.acceleration.z < 0.0:
                control_msg.acceleration.z = 0.0
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])

            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.ctrl_pub.publish(control_msg)
            self.publish_status(np.linalg.norm(self.goal - self.desire_pos))

    def process_output_all(self, endstate_pred, score_pred):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score = score_pred.reshape(self.lattice_primitive.traj_num)
        lattice_ids = torch.arange(self.lattice_primitive.traj_num - 1, -1, -1)
        endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, lattice_ids)
        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else self.get_current_position()
        start_vel = self.desire_vel if self.plan_from_reference else self.get_current_velocity()
        if self.best_traj_pub.get_num_connections() > 0 and self.optimal_poly_x is not None:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "world"
            self.best_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, points_array))

        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                          lattice_endstate[:, 0] + start_pos[0], zero_state[:, 0], zero_state[:, 0], self.traj_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                          lattice_endstate[:, 1] + start_pos[1], zero_state[:, 1], zero_state[:, 1], self.traj_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                          lattice_endstate[:, 2] + start_pos[2], zero_state[:, 2], zero_state[:, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "world"
            self.lattice_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, points_array))

        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                      pred_endstate[:, 0, 0] + start_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                      pred_endstate[:, 1, 0] + start_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                      pred_endstate[:, 2, 0] + start_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            # Match the YOPO-Simple RViz behavior: color candidates by network score.
            intensity = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, intensity))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "world"
            fields = [
                PointField("x", 0, PointField.FLOAT32, 1),
                PointField("y", 4, PointField.FLOAT32, 1),
                PointField("z", 8, PointField.FLOAT32, 1),
                PointField("intensity", 12, PointField.FLOAT32, 1),
            ]
            self.all_trajs_pub.publish(point_cloud2.create_cloud(header, fields, points_array))

    def publish_status(self, goal_distance=None):
        pos = self.desire_pos if self.desire_pos is not None else self.get_current_position() if self.odom_init else np.zeros(3)
        if goal_distance is None:
            goal_distance = np.linalg.norm(self.goal - pos)
        self.arrive_pub.publish(Bool(data=bool(self.arrive)))
        self.goal_distance_pub.publish(Float32(data=float(goal_distance)))
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "world"
        goal_msg.pose.position.x = float(self.goal[0])
        goal_msg.pose.position.y = float(self.goal[1])
        goal_msg.pose.position.z = float(self.goal[2])
        goal_msg.pose.orientation.w = 1.0
        self.goal_pub.publish(goal_msg)
        self.publish_status_text(pos, goal_distance)

    def publish_status_text(self, pos, goal_distance):
        if self.status_text_pub.get_num_connections() == 0:
            return

        speed = float(np.linalg.norm(self.get_current_velocity()))
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "world"
        marker.ns = "swarm_status"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(pos[0] + 0.45)
        marker.pose.position.y = float(pos[1] + 0.25)
        marker.pose.position.z = float(pos[2] + 0.85)
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.55
        marker.color.r = 0.08
        marker.color.g = 0.08
        marker.color.b = 0.08
        marker.color.a = 0.88
        marker.text = (
            f"{self.agent_name}\n"
            f"v {speed:.1f} m/s | d {float(goal_distance):.1f} m\n"
            f"col S {self.static_collision_count} | D {self.dynamic_collision_count}"
        )
        marker.lifetime = rospy.Duration(0.5)
        self.status_text_pub.publish(marker)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        self.time_interpolation += time1 - time0
        self.time_prepare += time2 - time1
        self.time_forward += time3 - time2
        self.time_process += time4 - time3
        self.time_visualize += time5 - time4
        self.count += 1
        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.depth_fps
        if self.verbose or total_time > tolerance:
            print(f"[{self.agent_name}] Average Time: "
                  f"image={1000 * self.time_interpolation / self.count:.2f} ms; "
                  f"prepare={1000 * self.time_prepare / self.count:.2f} ms; "
                  f"forward={1000 * self.time_forward / self.count:.2f} ms; "
                  f"post={1000 * self.time_process / self.count:.2f} ms; "
                  f"visual={1000 * self.time_visualize / self.count:.2f} ms")

    def warm_up(self):
        image = torch.zeros((1, cfg["input_channels"], self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, cfg["observation_dim"]), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        outputs = self.policy(image, obs)
        _ = self.state_transform.pred_to_endstate(outputs[0])


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--weights_root", type=str, default="saved")
    parser.add_argument("--agent_name", type=str, default="uav0")
    parser.add_argument("--node_name", type=str, default="yopo_tracker_uav0")
    parser.add_argument("--odom_topic", type=str, default="/uav0/sim/odom")
    parser.add_argument("--depth_topic", type=str, default="/uav0/depth_image")
    parser.add_argument("--target_mask_topic", type=str, default="/uav0/target_mask_image")
    parser.add_argument("--ctrl_topic", type=str, default="/uav0/so3_control/pos_cmd")
    parser.add_argument("--visual_prefix", type=str, default="/uav0/yopo_tracker")
    parser.add_argument("--status_prefix", type=str, default="/uav0/yopo")
    parser.add_argument("--rviz_goal_topic", type=str, default="/move_base_simple/goal")
    parser.add_argument("--goal_x", type=float, default=50.0)
    parser.add_argument("--goal_y", type=float, default=0.0)
    parser.add_argument("--goal_z", type=float, default=1.5)
    parser.add_argument("--arrive_radius", type=float, default=None)
    parser.add_argument("--arrival_settle_radius", type=float, default=None)
    parser.add_argument("--arrival_settle_speed", type=float, default=0.25)
    parser.add_argument("--arrival_settle_time", type=float, default=1.5)
    parser.add_argument("--arrival_stuck_radius", type=float, default=None)
    parser.add_argument("--arrival_stuck_timeout", type=float, default=5.0)
    parser.add_argument("--arrival_min_progress", type=float, default=0.15)
    parser.add_argument("--max_depth_dist", type=float, default=20.0)
    parser.add_argument("--depth_fps", type=float, default=10.0)
    parser.add_argument("--pitch_angle_deg", type=float, default=0.0)
    parser.add_argument("--plan_from_reference", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--visualize", type=int, default=0)
    parser.add_argument("--enable_rviz_goal", type=int, default=1)
    parser.add_argument("--rviz_goal_reference_odom_topic", type=str, default="/uav0/sim/odom")
    parser.add_argument("--min_altitude", type=float, default=1.5)
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_root = args.weights_root
    if not os.path.isabs(weights_root):
        weights_root = os.path.join(base_dir, weights_root)
    weight = "yopo_tracker_trt.pth" if args.use_tensorrt else os.path.join(
        weights_root, f"YOPO_{args.trial}", f"epoch{args.epoch}.pth"
    )
    print("load weight from:", weight)

    arrive_radius = args.arrive_radius
    if arrive_radius is None:
        arrive_radius = float(cfg["swarm_arrive_radius"])
    arrival_settle_radius = args.arrival_settle_radius
    if arrival_settle_radius is None:
        arrival_settle_radius = arrive_radius
    arrival_stuck_radius = args.arrival_stuck_radius
    if arrival_stuck_radius is None:
        arrival_stuck_radius = arrive_radius

    settings = {
        "use_tensorrt": args.use_tensorrt,
        "agent_name": args.agent_name,
        "node_name": args.node_name,
        "goal": [args.goal_x, args.goal_y, args.goal_z],
        "arrive_radius": arrive_radius,
        "arrival_settle_radius": arrival_settle_radius,
        "arrival_settle_speed": args.arrival_settle_speed,
        "arrival_settle_time": args.arrival_settle_time,
        "arrival_stuck_radius": arrival_stuck_radius,
        "arrival_stuck_timeout": args.arrival_stuck_timeout,
        "arrival_min_progress": args.arrival_min_progress,
        "pitch_angle_deg": -args.pitch_angle_deg,
        "odom_topic": args.odom_topic,
        "depth_topic": args.depth_topic,
        "target_mask_topic": args.target_mask_topic,
        "ctrl_topic": args.ctrl_topic,
        "visual_prefix": args.visual_prefix.rstrip("/"),
        "status_prefix": args.status_prefix.rstrip("/"),
        "rviz_goal_topic": args.rviz_goal_topic,
        "max_depth_dist": args.max_depth_dist,
        "depth_fps": args.depth_fps,
        "plan_from_reference": bool(args.plan_from_reference),
        "verbose": bool(args.verbose),
        "visualize": bool(args.visualize),
        "enable_rviz_goal": bool(args.enable_rviz_goal),
        "rviz_goal_reference_odom_topic": args.rviz_goal_reference_odom_topic,
        "min_altitude": args.min_altitude,
    }
    YopoSwarmTracker(settings, weight)
