import rospy
import std_msgs.msg
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker

import cv2
import os
import time
import torch
import numpy as np
import argparse
import re
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from control_msg import PositionCommand
from policy.yopo_network import YopoNetwork
from policy.poly_solver import *
from policy.state_transform import *

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")

try:
    import yaml
except ImportError:
    yaml = None


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis = float(cfg.get("depth_min_dis", 0.04))
        self.max_dis = float(cfg.get("depth_max_dis", 20.0))
        if self.max_dis <= self.min_dis:
            raise ValueError(f"Invalid depth range: min={self.min_dis}, max={self.max_dis}")
        self.goal = np.array(self.config['goal'])
        self.goal_altitude = float(self.goal[2])
        self.goal_length = float(cfg.get("goal_length", 0.0))
        self.require_user_goal = bool(cfg.get("require_user_goal", True))
        self.has_user_goal = not self.require_user_goal
        self.plan_from_reference = self.config['plan_from_reference']
        self.max_yaw_rate = float(cfg.get("max_yaw_rate", 0.5))
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.01
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.goal_dist = np.inf
        self.goal_best_dist = np.inf
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.current_speed = 0.0
        self.collision_counter_total = 0
        self.body_pitch_deg = 0.0
        self.body_roll_deg = 0.0
        self.camera_pitch_world_deg = 0.0
        self.traj_vis_cleared = False
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.depth_fps = int(self.config.get('depth_fps', 50))  # used only as processing time tolerance for printing logs
        # Open-space cruise bias (optional): boost forward speed only when depth is mostly far and goal is far.
        self.enable_cruise_boost = bool(cfg.get("enable_cruise_boost", False))
        self.cruise_far_depth_ratio = float(cfg.get("cruise_far_depth_ratio", 0.9))
        self.cruise_open_ratio_min = float(cfg.get("cruise_open_ratio_min", 0.75))
        self.cruise_near_depth_ratio_min = float(cfg.get("cruise_near_depth_ratio_min", 0.5))
        self.cruise_goal_dist_min = float(cfg.get("cruise_goal_dist_min", 80.0))
        self.cruise_target_speed_ratio = float(cfg.get("cruise_target_speed_ratio", 0.5))
        self.cruise_target_acc_ratio = float(cfg.get("cruise_target_acc_ratio", 0.35))
        self.cruise_max_extra_safety_penalty = float(cfg.get("cruise_max_extra_safety_penalty", 0.0))
        # Depth-based safety re-ranking for candidates (inference-time safety guard).
        self.enable_depth_safety_rerank = bool(cfg.get("enable_depth_safety_rerank", True))
        self.depth_safety_weight = float(cfg.get("depth_safety_weight", 3.0))
        self.depth_safety_margin = float(cfg.get("depth_safety_margin", 3.0))
        self.depth_safety_speed_margin_gain = float(cfg.get("depth_safety_speed_margin_gain", 0.0))
        self.depth_safety_samples = int(cfg.get("depth_safety_samples", 18))
        depth_safety_sample_spacing = float(cfg.get("depth_safety_sample_spacing", 0.0))
        if depth_safety_sample_spacing > 1e-6:
            auto_samples = int(np.ceil(self.traj_time * self.lattice_primitive.vel_max / depth_safety_sample_spacing))
            self.depth_safety_samples = max(self.depth_safety_samples, auto_samples)
        self.depth_safety_forward_min = float(cfg.get("depth_safety_forward_min", 0.8))
        self.depth_safety_behind_penalty = float(cfg.get("depth_safety_behind_penalty", 6.0))
        self.depth_safety_oob_penalty = float(cfg.get("depth_safety_oob_penalty", 1.5))
        # Use current depth to cap excessive speed when frontal clearance is short.
        self.enable_depth_speed_cap = bool(cfg.get("enable_depth_speed_cap", True))
        self.depth_speed_cap_brake_acc_ratio = float(cfg.get("depth_speed_cap_brake_acc_ratio", 0.55))
        self.depth_speed_cap_margin = float(cfg.get("depth_speed_cap_margin", 6.0))
        self.depth_speed_cap_center_percentile = float(cfg.get("depth_speed_cap_center_percentile", 20.0))
        self.depth_speed_cap_min_speed = float(cfg.get("depth_speed_cap_min_speed", 6.0))
        self.depth_speed_cap_max_speed_ratio = float(cfg.get("depth_speed_cap_max_speed_ratio", 1.0))
        self.depth_speed_cap_roi_h_min = float(cfg.get("depth_speed_cap_roi_h_min", 0.30))
        self.depth_speed_cap_roi_h_max = float(cfg.get("depth_speed_cap_roi_h_max", 0.70))
        self.depth_speed_cap_roi_w_min = float(cfg.get("depth_speed_cap_roi_w_min", 0.30))
        self.depth_speed_cap_roi_w_max = float(cfg.get("depth_speed_cap_roi_w_max", 0.70))
        hfov = np.deg2rad(float(cfg.get("horizon_camera_fov", 90.0)))
        vfov = np.deg2rad(float(cfg.get("vertical_camera_fov", 60.0)))
        self.depth_fx = float(self.width / (2.0 * np.tan(hfov / 2.0)))
        self.depth_fy = float(self.height / (2.0 * np.tan(vfov / 2.0)))
        self.depth_cx = float((self.width - 1) * 0.5)
        self.depth_cy = float((self.height - 1) * 0.5)
        # Goal handling for high-speed flight:
        # - arrive only when both near and slow
        # - keep publishing hold commands after arrival
        # - apply soft braking near goal based on stopping distance
        self.goal_reach_radius = float(cfg.get("goal_reach_radius", 5.0))
        self.goal_reach_speed = float(cfg.get("goal_reach_speed", 2.0))
        self.goal_resume_radius = float(cfg.get("goal_resume_radius", max(7.0, self.goal_reach_radius * 1.5)))
        self.goal_terminal_hold_radius = float(cfg.get("goal_terminal_hold_radius", 25.0))
        self.goal_hold_extra_margin = float(cfg.get("goal_hold_extra_margin", 0.0))
        self.goal_force_hold_radius = float(cfg.get("goal_force_hold_radius", 60.0))
        self.goal_rebound_enable_dist = float(cfg.get("goal_rebound_enable_dist", 90.0))
        self.goal_rebound_trigger = float(cfg.get("goal_rebound_trigger", 10.0))
        self.goal_hold_sticky = bool(cfg.get("goal_hold_sticky", True))
        self.enable_goal_soft_brake = bool(cfg.get("enable_goal_soft_brake", True))
        self.goal_brake_acc_ratio = float(cfg.get("goal_brake_acc_ratio", 0.55))
        self.goal_brake_buffer = float(cfg.get("goal_brake_buffer", 8.0))
        self.goal_min_approach_speed = float(cfg.get("goal_min_approach_speed", 1.5))
        self.goal_brake_max_extra_safety_penalty = float(cfg.get("goal_brake_max_extra_safety_penalty", 0.0))
        self.goal_dir_penalty_weight = float(cfg.get("goal_dir_penalty_weight", 0.0))
        self.goal_dir_forward_margin = float(cfg.get("goal_dir_forward_margin", 0.0))
        self.goal_dir_activation_goal_ratio = float(cfg.get("goal_dir_activation_goal_ratio", 0.4))
        self.enable_turn_speed_cap = bool(cfg.get("enable_turn_speed_cap", True))
        self.turn_speed_cap_start_deg = float(cfg.get("turn_speed_cap_start_deg", 20.0))
        self.turn_speed_cap_full_deg = float(cfg.get("turn_speed_cap_full_deg", 80.0))
        self.turn_speed_cap_min_ratio = float(cfg.get("turn_speed_cap_min_ratio", 0.45))
        # Runtime altitude guard keeps high-speed flight in a realistic altitude corridor.
        self.enable_altitude_guard = bool(cfg.get("enable_altitude_guard", True))
        self.altitude_guard_half_range = float(cfg.get("altitude_guard_half_range", 8.0))
        self.altitude_guard_vz_max = float(cfg.get("altitude_guard_vz_max", 12.0))
        self.altitude_guard_az_max = float(cfg.get("altitude_guard_az_max", 30.0))
        self.altitude_guard_weight = float(cfg.get("altitude_guard_weight", 2.0))
        self.enable_cmd_hard_limit = bool(cfg.get("enable_cmd_hard_limit", True))
        self.cmd_vel_limit_ratio = float(cfg.get("cmd_vel_limit_ratio", 1.0))
        self.cmd_acc_limit_ratio = float(cfg.get("cmd_acc_limit_ratio", 0.8))

        # Load Network
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

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", PointCloud2, queue_size=1)
        self.metrics_marker_pub = rospy.Publisher("/yopo/metrics_marker", Marker, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.collision_sub = rospy.Subscriber("/yopo/collision_counter_total", Int32, self.callback_collision_counter, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        # ros timer
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        self.timer_metrics = rospy.Timer(rospy.Duration(0.1), self.publish_metrics_marker)
        print("YOPO Net Node Ready!")
        rospy.spin()

    def callback_set_goal(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, self.goal_altitude])
        self.has_user_goal = True
        self.arrive = False
        self.traj_vis_cleared = False
        self.goal_dist = np.inf
        self.goal_best_dist = np.inf
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f}, {self.goal_altitude:.1f})")

    # the first frame
    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
            self.body_pitch_deg = float(np.degrees(ypr[1]))
            self.body_roll_deg = float(np.degrees(ypr[2]))
            if self.require_user_goal and not self.has_user_goal:
                self.arrive = True
                self.goal = np.array([self.desire_pos[0], self.desire_pos[1], self.goal_altitude], dtype=np.float32)
        self.odom_init = True
        ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                           self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
        self.body_pitch_deg = float(np.degrees(ypr[1]))
        self.body_roll_deg = float(np.degrees(ypr[2]))
        vel = data.twist.twist.linear
        self.current_speed = float(np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if self.require_user_goal and not self.has_user_goal:
            self.goal = np.array([pos[0], pos[1], self.goal_altitude], dtype=np.float32)
        self.goal_dist = float(np.linalg.norm(pos - self.goal))
        self.goal_best_dist = min(self.goal_best_dist, self.goal_dist)
        if self.require_user_goal and not self.has_user_goal:
            # Keep hover mode sticky before the first user goal.
            self.arrive = True
            return

        # Enter terminal hold as soon as we are inside the dynamic stopping corridor.
        # This prevents high-speed circling around the goal.
        if (not self.arrive) and self.should_enter_goal_hold(self.goal_dist, self.current_speed):
            self.arrive = True
            print(f"Enter goal-hold mode: dist={self.goal_dist:.1f}m speed={self.current_speed:.1f}m/s")
            return

        if not self.arrive:
            if self.goal_dist < self.goal_reach_radius and self.current_speed < self.goal_reach_speed:
                print("Arrive!")
                self.arrive = True
        elif (not self.goal_hold_sticky) and self.goal_dist > self.goal_resume_radius:
            # Leave terminal mode when drifting away or when a far new goal is set.
            self.arrive = False

    def callback_collision_counter(self, msg):
        self.collision_counter_total = int(msg.data)

    def publish_metrics_marker(self, _timer):
        if not self.odom_init:
            return

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "world"
        marker.ns = "metrics"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = self.odom.pose.pose.position.x
        marker.pose.position.y = self.odom.pose.pose.position.y
        marker.pose.position.z = self.odom.pose.pose.position.z + 6.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 4.0
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration(0.2)
        marker.text = (f"speed: {self.current_speed:.1f} m/s\n"
                       f"goal_mode: {'nav' if self.has_user_goal else 'hold'}\n"
                       f"body pitch/roll: {self.body_pitch_deg:.1f}/{self.body_roll_deg:.1f} deg\n"
                       f"cam pitch(world): {self.camera_pitch_world_deg:.1f} deg\n"
                       f"goal_dist: {self.goal_dist:.1f} m\n"
                       f"collision_total: {self.collision_counter_total}")
        self.metrics_marker_pub.publish(marker)

    def process_odom(self):
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        ypr_wc = R.from_matrix(self.Rotation_wc).as_euler('ZYX', degrees=True)
        self.camera_pitch_world_deg = float(ypr_wc[1])
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # goal_dir
        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init: return
        if self.require_user_goal and not self.has_user_goal:
            return
        if self.arrive:
            return

        # 1. Depth Image Process (Be careful with the depth units in your application)
        time0 = time.time()
        if data.encoding == "32FC1":    # Simulator, meter
            depth = np.frombuffer(data.data, dtype=np.float32).reshape(data.height, data.width)
        elif data.encoding == "16UC1":  # RealSense, millimeter
            depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width).astype(np.float32) / 1000.0
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}. Expected '32FC1' or '16UC1'.")

        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_metric = np.nan_to_num(depth, nan=self.max_dis, posinf=self.max_dis, neginf=self.min_dis)
        depth_metric = np.clip(depth_metric, self.min_dis, self.max_dis)
        depth = np.minimum(depth_metric, self.max_dis) / self.max_dis
        depth_raw_norm = depth.copy()

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        depth = interpolated_image.reshape([1, 1, self.height, self.width])
        # cv2.imshow("1", depth[0][0])
        # cv2.waitKey(1)

        # 2. YOPO Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
        obs_norm = self.process_odom().to(self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input(obs_norm)
        # torch.cuda.synchronize()

        time2 = time.time()
        # Forward (TensorRT: inference speed increased by 5x)
        endstate_pred, score_pred = self.policy(depth_input, obs_input)
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        time3 = time.time()

        # 3. Post-Processing
        # Replacing PyTorch operation on CUDA with NumPy operation on CPU (speed increased by 10x)
        endstate, score = self.process_output(endstate_pred, score_pred, return_all_preds=True)
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        safety_penalty = self.depth_safety_rerank(endstate_w, depth_metric, start_pos, start_vel, self.desire_acc)
        altitude_penalty = self.altitude_guard_penalty(endstate_w, start_pos)
        goal_dir_penalty = self.goal_direction_penalty(endstate_w, start_pos)
        score_for_select = score + safety_penalty + altitude_penalty + goal_dir_penalty
        action_id = int(np.argmin(score_for_select))
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            selected_endstate_w = endstate_w[action_id].copy()
            boosted_endstate_w = self.apply_cruise_boost(selected_endstate_w.copy(), depth_raw_norm, start_pos)
            if self.enable_depth_safety_rerank and self.cruise_max_extra_safety_penalty >= 0.0:
                base_pen = self.depth_safety_rerank(selected_endstate_w[np.newaxis, ...], depth_metric, start_pos, start_vel, self.desire_acc)[0]
                boost_pen = self.depth_safety_rerank(boosted_endstate_w[np.newaxis, ...], depth_metric, start_pos, start_vel, self.desire_acc)[0]
                if boost_pen <= base_pen + self.cruise_max_extra_safety_penalty:
                    selected_endstate_w = boosted_endstate_w
            else:
                selected_endstate_w = boosted_endstate_w
            braked_endstate_w = self.apply_goal_soft_brake(selected_endstate_w.copy(), start_pos, start_vel)
            if self.enable_depth_safety_rerank and self.goal_brake_max_extra_safety_penalty >= 0.0:
                base_pen = self.depth_safety_rerank(selected_endstate_w[np.newaxis, ...], depth_metric, start_pos, start_vel, self.desire_acc)[0]
                brake_pen = self.depth_safety_rerank(braked_endstate_w[np.newaxis, ...], depth_metric, start_pos, start_vel, self.desire_acc)[0]
                if brake_pen <= base_pen + self.goal_brake_max_extra_safety_penalty:
                    selected_endstate_w = braked_endstate_w
            else:
                selected_endstate_w = braked_endstate_w
            selected_endstate_w = self.apply_turn_speed_cap(selected_endstate_w, start_pos, start_vel)
            selected_endstate_w = self.apply_depth_speed_cap(selected_endstate_w, depth_metric)
            selected_endstate_w = self.apply_altitude_guard(selected_endstate_w, start_pos)

            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], selected_endstate_w[0, 0] + start_pos[0],
                                              selected_endstate_w[0, 1], selected_endstate_w[0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], selected_endstate_w[1, 0] + start_pos[1],
                                              selected_endstate_w[1, 1], selected_endstate_w[1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], selected_endstate_w[2, 0] + start_pos[2],
                                              selected_endstate_w[2, 1], selected_endstate_w[2, 2], self.traj_time)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score_for_select, endstate_w)
        time5 = time.time()

        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub(self, _timer):
        if not self.odom_init:
            return
        pos_now = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        vel_now = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        speed_now = float(np.linalg.norm(vel_now))
        self.current_speed = speed_now
        dist_now = float(np.linalg.norm(pos_now - self.goal))
        self.goal_dist = dist_now
        self.goal_best_dist = min(self.goal_best_dist, dist_now)
        if (not self.arrive) and self.should_enter_goal_hold(dist_now, speed_now):
            self.arrive = True
            print(f"Enter goal-hold mode(ctrl): dist={dist_now:.1f}m speed={speed_now:.1f}m/s")
        if self.arrive:
            if not self.traj_vis_cleared:
                self.clear_trajectory_visualization()
                self.traj_vis_cleared = True
            self.publish_goal_hold()
            return
        if self.ctrl_time is None:
            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            self.ctrl_time += self.ctrl_dt
            t_eval = min(self.ctrl_time, self.traj_time)
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(t_eval)
            control_msg.position.y = self.optimal_poly_y.get_position(t_eval)
            control_msg.position.z = self.optimal_poly_z.get_position(t_eval)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(t_eval)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(t_eval)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(t_eval)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(t_eval)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(t_eval)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(t_eval)
            control_msg = self.apply_command_hard_limits(control_msg)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt, self.max_yaw_rate)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def clear_trajectory_visualization(self):
        """
            Clear stale trajectory visuals in RViz when entering terminal hold mode.
            Otherwise the last predicted trajectories remain on screen after the drone stops.
        """
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'

        if self.best_traj_pub.get_num_connections() > 0:
            self.best_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, []))
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            self.lattice_traj_pub.publish(point_cloud2.create_cloud_xyz32(header, []))
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1)]
            self.all_trajs_pub.publish(point_cloud2.create_cloud(header, fields, []))

    def terminal_hold_radius(self, speed):
        brake_acc = max(0.5, self.goal_brake_acc_ratio * self.lattice_primitive.acc_max)
        stop_dist = (speed * speed) / (2.0 * brake_acc + 1e-6)
        return max(self.goal_terminal_hold_radius, stop_dist + self.goal_brake_buffer + self.goal_hold_extra_margin)

    def should_enter_goal_hold(self, goal_dist, speed):
        if goal_dist < self.goal_force_hold_radius:
            return True
        if self.goal_best_dist < self.goal_rebound_enable_dist and goal_dist > self.goal_best_dist + self.goal_rebound_trigger:
            return True
        return goal_dist < self.terminal_hold_radius(speed)

    def publish_goal_hold(self):
        control_msg = PositionCommand()
        control_msg.header.stamp = rospy.Time.now()
        # Use hover tracking mode in controller (position/velocity feedback)
        # to dissipate residual high-speed momentum near terminal states.
        control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
        control_msg.position.x = float(self.goal[0])
        control_msg.position.y = float(self.goal[1])
        control_msg.position.z = float(self.goal[2])
        control_msg.velocity.x = 0.0
        control_msg.velocity.y = 0.0
        control_msg.velocity.z = 0.0
        control_msg.acceleration.x = 0.0
        control_msg.acceleration.y = 0.0
        control_msg.acceleration.z = 0.0
        hold_dir = self.goal - np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        yaw, yaw_dot = calculate_yaw(np.zeros(3), hold_dir, self.last_yaw, self.ctrl_dt, self.max_yaw_rate)
        self.last_yaw = yaw
        control_msg.yaw = yaw
        control_msg.yaw_dot = yaw_dot
        self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
        self.desire_vel = np.array([0.0, 0.0, 0.0])
        self.desire_acc = np.array([0.0, 0.0, 0.0])
        self.desire_init = True
        self.last_control_msg = control_msg
        self.ctrl_pub.publish(control_msg)

    def apply_command_hard_limits(self, control_msg):
        if not self.enable_cmd_hard_limit:
            return control_msg

        z_min = self.goal_altitude - self.altitude_guard_half_range
        z_max = self.goal_altitude + self.altitude_guard_half_range
        control_msg.position.z = float(np.clip(control_msg.position.z, z_min, z_max))
        control_msg.velocity.z = float(np.clip(control_msg.velocity.z, -self.altitude_guard_vz_max, self.altitude_guard_vz_max))
        control_msg.acceleration.z = float(np.clip(control_msg.acceleration.z, -self.altitude_guard_az_max, self.altitude_guard_az_max))

        vel_vec = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z], dtype=np.float32)
        vel_limit = max(0.1, self.cmd_vel_limit_ratio * self.lattice_primitive.vel_max)
        vel_norm = float(np.linalg.norm(vel_vec))
        if vel_norm > vel_limit:
            vel_vec *= vel_limit / (vel_norm + 1e-8)
            control_msg.velocity.x = float(vel_vec[0])
            control_msg.velocity.y = float(vel_vec[1])
            control_msg.velocity.z = float(vel_vec[2])

        acc_vec = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z], dtype=np.float32)
        acc_limit = max(0.1, self.cmd_acc_limit_ratio * self.lattice_primitive.acc_max)
        acc_norm = float(np.linalg.norm(acc_vec))
        if acc_norm > acc_limit:
            acc_vec *= acc_limit / (acc_norm + 1e-8)
            control_msg.acceleration.x = float(acc_vec[0])
            control_msg.acceleration.y = float(acc_vec[1])
            control_msg.acceleration.z = float(acc_vec[2])

        return control_msg

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
        else:
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num-1, -1, -1))

        return endstate, score

    def apply_cruise_boost(self, endstate_w, depth_raw_norm, start_pos):
        """
            Boost forward speed in clearly open space.
            endstate_w shape: [3, 3], rows are xyz, columns are [pos, vel, acc].
        """
        if not self.enable_cruise_boost:
            return endstate_w

        goal_vec = self.goal - start_pos
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist < self.cruise_goal_dist_min:
            return endstate_w

        far_ratio = float(np.mean(depth_raw_norm >= self.cruise_far_depth_ratio))
        if far_ratio < self.cruise_open_ratio_min:
            return endstate_w

        h, w = depth_raw_norm.shape
        roi = depth_raw_norm[h // 3: (2 * h) // 3, w // 3: (2 * w) // 3]
        near_depth_ratio = float(np.nanpercentile(roi, 10))
        if near_depth_ratio < self.cruise_near_depth_ratio_min:
            return endstate_w

        goal_xy = np.array([goal_vec[0], goal_vec[1]])
        goal_xy_norm = np.linalg.norm(goal_xy)
        if goal_xy_norm < 1e-6:
            return endstate_w
        goal_dir_xy = goal_xy / (goal_xy_norm + 1e-8)

        vel_vec = endstate_w[:, 1].copy()
        vel_xy = vel_vec[:2]
        forward_speed = float(np.dot(vel_xy, goal_dir_xy))
        target_speed = self.cruise_target_speed_ratio * self.lattice_primitive.vel_max
        if forward_speed < target_speed:
            vel_xy = vel_xy + (target_speed - forward_speed) * goal_dir_xy
            vel_vec[:2] = vel_xy
            vel_norm = np.linalg.norm(vel_vec)
            if vel_norm > self.lattice_primitive.vel_max:
                vel_vec = vel_vec * (self.lattice_primitive.vel_max / (vel_norm + 1e-8))
            endstate_w[:, 1] = vel_vec

        acc_vec = endstate_w[:, 2].copy()
        acc_xy = acc_vec[:2]
        forward_acc = float(np.dot(acc_xy, goal_dir_xy))
        target_acc = self.cruise_target_acc_ratio * self.lattice_primitive.acc_max
        if forward_acc < target_acc:
            acc_xy = acc_xy + (target_acc - forward_acc) * goal_dir_xy
            acc_vec[:2] = acc_xy
            acc_norm = np.linalg.norm(acc_vec)
            if acc_norm > self.lattice_primitive.acc_max:
                acc_vec = acc_vec * (self.lattice_primitive.acc_max / (acc_norm + 1e-8))
            endstate_w[:, 2] = acc_vec

        return endstate_w

    def apply_goal_soft_brake(self, endstate_w, start_pos, start_vel):
        if not self.enable_goal_soft_brake:
            return endstate_w

        goal_vec = self.goal - start_pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist < 1e-3:
            endstate_w[:, 1] = 0.0
            endstate_w[:, 2] = 0.0
            return endstate_w

        goal_dir = goal_vec / (goal_dist + 1e-8)
        brake_acc = max(0.5, self.goal_brake_acc_ratio * self.lattice_primitive.acc_max)
        current_forward_speed = max(float(np.dot(start_vel, goal_dir)), 0.0)
        stop_dist = (current_forward_speed * current_forward_speed) / (2.0 * brake_acc + 1e-6) + self.goal_brake_buffer
        if goal_dist > stop_dist:
            return endstate_w

        vel_vec = endstate_w[:, 1].copy()
        forward_speed = float(np.dot(vel_vec, goal_dir))
        target_speed = np.sqrt(max(0.0, 2.0 * brake_acc * max(goal_dist - self.goal_reach_radius, 0.0)))
        if goal_dist > self.goal_reach_radius:
            target_speed = max(target_speed, self.goal_min_approach_speed)
        if forward_speed > target_speed:
            vel_vec = vel_vec + (target_speed - forward_speed) * goal_dir
            vel_norm = np.linalg.norm(vel_vec)
            if vel_norm > self.lattice_primitive.vel_max:
                vel_vec = vel_vec * (self.lattice_primitive.vel_max / (vel_norm + 1e-8))
            endstate_w[:, 1] = vel_vec

        acc_vec = endstate_w[:, 2].copy()
        acc_forward = float(np.dot(acc_vec, goal_dir))
        desired_forward_acc = -brake_acc if forward_speed > target_speed + 0.2 else acc_forward
        if desired_forward_acc < acc_forward:
            acc_vec = acc_vec + (desired_forward_acc - acc_forward) * goal_dir
            acc_norm = np.linalg.norm(acc_vec)
            if acc_norm > self.lattice_primitive.acc_max:
                acc_vec = acc_vec * (self.lattice_primitive.acc_max / (acc_norm + 1e-8))
            endstate_w[:, 2] = acc_vec

        return endstate_w

    def goal_direction_penalty(self, endstate_w, start_pos):
        traj_num = endstate_w.shape[0]
        if traj_num == 0 or self.goal_dir_penalty_weight <= 1e-8:
            return np.zeros((traj_num,), dtype=np.float32)

        goal_vec = self.goal - start_pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist < 1e-3:
            return np.zeros((traj_num,), dtype=np.float32)
        goal_dir = goal_vec / (goal_dist + 1e-8)

        forward_speed = np.einsum('ij,j->i', endstate_w[:, :, 1], goal_dir).astype(np.float32)
        reverse_gap = np.maximum(self.goal_dir_forward_margin - forward_speed, 0.0)
        pen = reverse_gap * reverse_gap

        activation_dist = self.goal_dir_activation_goal_ratio * max(self.goal_length, 1e-6)
        denom = max(self.goal_length - activation_dist, 1e-6)
        far_gate = np.clip((goal_dist - activation_dist) / denom, 0.0, 1.0)
        return self.goal_dir_penalty_weight * far_gate * pen

    def apply_turn_speed_cap(self, endstate_w, start_pos, start_vel):
        if not self.enable_turn_speed_cap:
            return endstate_w

        vel_xy = np.array([start_vel[0], start_vel[1]], dtype=np.float32)
        goal_vec = self.goal - start_pos
        goal_xy = np.array([goal_vec[0], goal_vec[1]], dtype=np.float32)
        vel_norm = float(np.linalg.norm(vel_xy))
        goal_norm = float(np.linalg.norm(goal_xy))
        if vel_norm < 1e-3 or goal_norm < 1e-3:
            return endstate_w

        cos_val = float(np.dot(vel_xy, goal_xy) / (vel_norm * goal_norm + 1e-8))
        turn_deg = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
        if turn_deg <= self.turn_speed_cap_start_deg:
            return endstate_w

        full_deg = max(self.turn_speed_cap_start_deg + 1.0, self.turn_speed_cap_full_deg)
        scale = np.clip((turn_deg - self.turn_speed_cap_start_deg) / (full_deg - self.turn_speed_cap_start_deg), 0.0, 1.0)
        cap_ratio = 1.0 - scale * (1.0 - self.turn_speed_cap_min_ratio)
        cap_speed = cap_ratio * self.lattice_primitive.vel_max

        des_vel = endstate_w[:, 1].copy()
        des_speed = float(np.linalg.norm(des_vel))
        if des_speed <= cap_speed:
            return endstate_w

        des_dir = des_vel / (des_speed + 1e-8)
        endstate_w[:, 1] = des_dir * cap_speed
        # remove positive tangential acceleration when turn-limited
        des_acc = endstate_w[:, 2].copy()
        tangential_acc = float(np.dot(des_acc, des_dir))
        if tangential_acc > 0.0:
            endstate_w[:, 2] = des_acc - tangential_acc * des_dir
        return endstate_w

    def estimate_forward_clearance(self, depth_metric):
        h, w = depth_metric.shape
        h0 = int(np.clip(self.depth_speed_cap_roi_h_min, 0.0, 1.0) * h)
        h1 = int(np.clip(self.depth_speed_cap_roi_h_max, 0.0, 1.0) * h)
        w0 = int(np.clip(self.depth_speed_cap_roi_w_min, 0.0, 1.0) * w)
        w1 = int(np.clip(self.depth_speed_cap_roi_w_max, 0.0, 1.0) * w)
        if h1 <= h0 or w1 <= w0:
            return None
        roi = depth_metric[h0:h1, w0:w1]
        finite = roi[np.isfinite(roi)]
        if finite.size < 16:
            return None
        p = float(np.clip(self.depth_speed_cap_center_percentile, 1.0, 99.0))
        return float(np.percentile(finite, p))

    def apply_depth_speed_cap(self, endstate_w, depth_metric):
        if not self.enable_depth_speed_cap:
            return endstate_w

        clearance = self.estimate_forward_clearance(depth_metric)
        if clearance is None:
            return endstate_w

        brake_acc = max(0.5, self.depth_speed_cap_brake_acc_ratio * self.lattice_primitive.acc_max)
        clearance_eff = max(0.0, clearance - self.depth_speed_cap_margin)
        safe_speed = np.sqrt(max(0.0, 2.0 * brake_acc * clearance_eff))
        max_speed = self.depth_speed_cap_max_speed_ratio * self.lattice_primitive.vel_max
        safe_speed = float(np.clip(safe_speed, self.depth_speed_cap_min_speed, max_speed))

        vel_vec = endstate_w[:, 1].copy()
        vel_norm = np.linalg.norm(vel_vec)
        if vel_norm <= safe_speed:
            return endstate_w

        vel_dir = vel_vec / (vel_norm + 1e-8)
        endstate_w[:, 1] = vel_vec * (safe_speed / (vel_norm + 1e-8))
        # Dampen tangential acceleration when speed is already clipped by clearance.
        acc_vec = endstate_w[:, 2].copy()
        tangential_acc = float(np.dot(acc_vec, vel_dir))
        if tangential_acc > 0.0:
            endstate_w[:, 2] = acc_vec - tangential_acc * vel_dir
        return endstate_w

    def altitude_guard_penalty(self, endstate_w, start_pos):
        if (not self.enable_altitude_guard) or endstate_w.shape[0] == 0:
            return np.zeros((endstate_w.shape[0],), dtype=np.float32)

        z_center = self.goal_altitude
        z_abs = start_pos[2] + endstate_w[:, 2, 0]
        z_err = np.maximum(np.abs(z_abs - z_center) - self.altitude_guard_half_range, 0.0)
        vz_err = np.maximum(np.abs(endstate_w[:, 2, 1]) - self.altitude_guard_vz_max, 0.0)
        az_err = np.maximum(np.abs(endstate_w[:, 2, 2]) - self.altitude_guard_az_max, 0.0)
        return self.altitude_guard_weight * (z_err * z_err + 0.05 * vz_err * vz_err + 0.01 * az_err * az_err)

    def apply_altitude_guard(self, endstate_w, start_pos):
        if not self.enable_altitude_guard:
            return endstate_w

        z_center = self.goal_altitude
        z_min = z_center - self.altitude_guard_half_range
        z_max = z_center + self.altitude_guard_half_range
        z_abs = start_pos[2] + endstate_w[2, 0]
        z_abs = float(np.clip(z_abs, z_min, z_max))
        endstate_w[2, 0] = z_abs - start_pos[2]
        endstate_w[2, 1] = float(np.clip(endstate_w[2, 1], -self.altitude_guard_vz_max, self.altitude_guard_vz_max))
        endstate_w[2, 2] = float(np.clip(endstate_w[2, 2], -self.altitude_guard_az_max, self.altitude_guard_az_max))
        return endstate_w

    def depth_safety_rerank(self, endstate_w, depth_metric, start_pos, start_vel, start_acc):
        """
            Add depth-consistency safety penalty for each candidate trajectory.
            This helps reject trajectories that are likely to collide in front of the camera.
        """
        traj_num = endstate_w.shape[0]
        if (not self.enable_depth_safety_rerank) or traj_num == 0:
            return np.zeros((traj_num,), dtype=np.float32)

        sample_n = max(6, self.depth_safety_samples)
        t_values = np.linspace(self.traj_time / sample_n, self.traj_time, sample_n, dtype=np.float32)

        poly_x = Polys5Solver(start_pos[0], start_vel[0], start_acc[0],
                              endstate_w[:, 0, 0] + start_pos[0], endstate_w[:, 0, 1], endstate_w[:, 0, 2], self.traj_time)
        poly_y = Polys5Solver(start_pos[1], start_vel[1], start_acc[1],
                              endstate_w[:, 1, 0] + start_pos[1], endstate_w[:, 1, 1], endstate_w[:, 1, 2], self.traj_time)
        poly_z = Polys5Solver(start_pos[2], start_vel[2], start_acc[2],
                              endstate_w[:, 2, 0] + start_pos[2], endstate_w[:, 2, 1], endstate_w[:, 2, 2], self.traj_time)

        px = poly_x.get_position(t_values).reshape(traj_num, sample_n)
        py = poly_y.get_position(t_values).reshape(traj_num, sample_n)
        pz = poly_z.get_position(t_values).reshape(traj_num, sample_n)
        points_w = np.stack((px, py, pz), axis=-1)  # [N, T, 3]

        rel_w = points_w - start_pos[None, None, :]
        rot_cw = self.Rotation_wc.T
        points_c = np.einsum('ij,ntj->nti', rot_cw, rel_w)
        x = points_c[:, :, 0]
        y = points_c[:, :, 1]
        z = points_c[:, :, 2]

        eps = 1e-4
        x_safe = np.maximum(x, eps)
        u = np.rint(self.depth_cx - self.depth_fx * (y / x_safe)).astype(np.int32)
        v = np.rint(self.depth_cy - self.depth_fy * (z / x_safe)).astype(np.int32)

        forward_mask = x > self.depth_safety_forward_min
        in_bound = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height) & forward_mask
        oob_mask = forward_mask & (~in_bound)

        penalties = np.zeros((traj_num, sample_n), dtype=np.float32)
        penalties[~forward_mask] += self.depth_safety_behind_penalty
        penalties[oob_mask] += self.depth_safety_oob_penalty

        valid_idx = np.where(in_bound)
        if valid_idx[0].size > 0:
            depth_obs = depth_metric[v[valid_idx], u[valid_idx]]
            depth_pred = x[valid_idx]
            clearance = depth_obs - depth_pred
            margin = np.full((traj_num, sample_n), self.depth_safety_margin, dtype=np.float32)
            if self.depth_safety_speed_margin_gain > 1e-8:
                speed_end = np.linalg.norm(endstate_w[:, :, 1], axis=1, keepdims=True).astype(np.float32)
                time_factor = (t_values / max(self.traj_time, 1e-3)).astype(np.float32)[None, :]
                margin += self.depth_safety_speed_margin_gain * speed_end * time_factor
            violation = np.maximum(margin[valid_idx] - clearance, 0.0)
            penalties[valid_idx] += violation * violation

        # Slightly increase penalty weight for farther future points.
        time_weight = np.linspace(0.5, 1.0, sample_n, dtype=np.float32)
        traj_penalty = (penalties * time_weight[None, :]).mean(axis=1)
        return self.depth_safety_weight * traj_penalty

    def visualize_trajectory(self, pred_score, pred_endstate):
        self.traj_vis_cleared = False
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        # lattice primitive
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
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        # all predicted trajectories
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
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1)]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        """
        Performance reference: PyTorch model should take < 5 ms; TensorRT model should take < 1 ms

        Notes:
        - Running program and enabling RViz under WSL greatly increase processing time, and Ubuntu does not have these issues
        - Even with queue_size=1, it may cause message accumulation and lag when processing time exceeds the image frequency
        """
        self.time_interpolation = self.time_interpolation + (time1 - time0)
        self.time_prepare = self.time_prepare + (time2 - time1)
        self.time_forward = self.time_forward + (time3 - time2)
        self.time_process = self.time_process + (time4 - time3)
        self.time_visualize = self.time_visualize + (time5 - time4)
        self.count = self.count + 1

        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.depth_fps
        if total_time > tolerance:
            rospy.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms, may cause message lag!")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m")
        if self.verbose or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"depth-interpolation: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.policy(depth, obs)
        _ = self.state_transform.pred_to_endstate(endstate_pred)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    return parser


def load_camera_pitch_from_sim_config(default_pitch_deg: float, base_dir: str) -> float:
    config_path = os.path.abspath(os.path.join(base_dir, "..", "Simulator", "src", "config", "config.yaml"))
    if not os.path.exists(config_path):
        return default_pitch_deg

    # Prefer YAML parser if available.
    if yaml is not None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_sim = yaml.safe_load(f)
            return float(cfg_sim.get("camera", {}).get("pitch", default_pitch_deg))
        except Exception as e:
            print(f"[warn] Failed to parse simulator config with yaml: {e}")

    # Fallback: regex parse "camera: ... pitch: X"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            text = f.read()
        m = re.search(r"camera:\s*(?:\n[ \t]+.*)*?\n[ \t]*pitch:\s*([-+]?\d*\.?\d+)", text)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return default_pitch_deg


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = "yopo_trt.pth" if args.use_tensorrt else base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    print("load weight from:", weight)
    default_pitch_deg = float(cfg.get("camera_pitch_deg", -25.0))
    pitch_angle_deg = load_camera_pitch_from_sim_config(default_pitch_deg, base_dir)
    print(f"camera pitch (deg): {pitch_angle_deg:.1f} (sim-config synced)")

    settings = {'use_tensorrt': args.use_tensorrt,
                'goal': [500, 0, 50],    # 默认设置远距离目标点，围绕50m高度飞行
                'pitch_angle_deg': pitch_angle_deg,   # 相机俯仰角(仰为负)，默认与Simulator配置同步
                'odom_topic': '/sim/odom',                   # 里程计话题
                'depth_topic': '/depth_image',               # 深度图话题
                'ctrl_topic': '/so3_control/pos_cmd',        # 控制器话题
                'plan_from_reference': False,   # 从参考状态规划？位置控制器: True, 神经网络直接控制: False
                'depth_fps': 50,                # 与仿真深度频率保持一致
                'verbose': False,               # 打印耗时？
                'visualize': True               # 可视化所有轨迹？(实飞改为False节省计算)
                }
    YopoNet(settings, weight)
