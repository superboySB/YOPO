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
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2

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


class YopoTracker:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_tracker', anonymous=False)
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis, self.max_dis = 0.04, 20.0
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.objectness_threshold = cfg["objectness_threshold"]
        self.selection_objectness_bonus = cfg["selection_objectness_bonus"]
        self.target_ema_alpha = cfg["target_ema_alpha"]
        self.target_velocity_ema_alpha = cfg["target_velocity_ema_alpha"]
        self.follow_distance = cfg["follow_distance"]
        self.follow_deadband = cfg["follow_deadband"]
        self.follow_capture_distance = cfg["follow_capture_distance"]
        self.follow_target_speed_threshold = cfg["follow_target_speed_threshold"]
        self.follow_max_step = cfg["follow_max_step"]
        self.target_measurement_timeout = cfg["target_measurement_timeout"]
        self.target_hold_timeout = cfg["target_hold_timeout"]
        self.use_mask_target_estimate = cfg["use_mask_target_estimate"]
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.odom = Odometry()
        self.odom_init = False
        self.target_gt = None
        self.target_gt_vel = np.zeros(3)
        self.latest_target_mask = None
        self.latest_target_mask_stamp = None
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.target_est_w = None
        self.target_est_vel_w = np.zeros(3)
        self.target_est_stamp = None
        self.mask_target_w = None
        self.mask_target_stamp = None
        self.hover_hold_pos = None
        self.hover_target_w = None
        self.hold_mode = False
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
        self.depth_fps = 30

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

        self.lattice_traj_pub = rospy.Publisher("/yopo_tracker/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_tracker/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_tracker/trajs_visual", PointCloud2, queue_size=1)
        self.target_est_pub = rospy.Publisher("/yopo_tracker/target_estimate", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)

        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.target_mask_sub = rospy.Subscriber(self.config['target_mask_topic'], Image, self.callback_target_mask, queue_size=1, tcp_nodelay=True)
        self.target_sub = rospy.Subscriber(self.config['target_odom_topic'], Odometry, self.callback_target_gt, queue_size=1, tcp_nodelay=True)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_reset_target_estimate, queue_size=1)
        rospy.sleep(1.0)
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print("YOPOv2 Tracker Node Ready!")
        rospy.spin()

    def callback_reset_target_estimate(self, _data):
        self.target_est_w = None
        self.target_est_vel_w = np.zeros(3)
        self.target_est_stamp = None
        self.mask_target_w = None
        self.mask_target_stamp = None
        self.hover_hold_pos = None
        self.hover_target_w = None
        print("Target estimate reset.")

    def callback_target_gt(self, data):
        self.target_gt = np.array((data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z))
        self.target_gt_vel = np.array((data.twist.twist.linear.x, data.twist.twist.linear.y, data.twist.twist.linear.z))

    def update_target_estimate(self, target_est_w, stamp):
        target_est_w = np.asarray(target_est_w, dtype=np.float64)
        if self.target_est_w is None:
            new_target_est = target_est_w
            self.target_est_vel_w = np.zeros(3)
        else:
            new_target_est = (1.0 - self.target_ema_alpha) * self.target_est_w + self.target_ema_alpha * target_est_w
            if self.target_est_stamp is not None:
                dt = max((stamp - self.target_est_stamp).to_sec(), 1e-3)
                measured_vel = (new_target_est - self.target_est_w) / dt
                self.target_est_vel_w = (
                    (1.0 - self.target_velocity_ema_alpha) * self.target_est_vel_w
                    + self.target_velocity_ema_alpha * measured_vel
                )
        self.target_est_w = new_target_est
        self.target_est_stamp = stamp

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

    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

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
        obs = np.concatenate((vel_c, acc_c), axis=0).astype(np.float32)
        return self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))

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
        if target_mask is None:
            target_mask = np.zeros((self.height, self.width), dtype=np.float32)
        else:
            target_mask = cv2.resize(target_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            target_mask = target_mask.astype(np.float32) / 255.0
        self.update_mask_target_estimate(depth_m, target_mask, depth_msg.header.stamp)
        image = np.concatenate((depth[None, :, :], target_mask[None, :, :]), axis=0)
        return image.reshape(1, cfg["input_channels"], self.height, self.width).astype(np.float32)

    def update_mask_target_estimate(self, depth_m, target_mask, stamp):
        self.mask_target_w = None
        if not self.use_mask_target_estimate:
            return
        if target_mask is None or np.count_nonzero(target_mask > 0.5) < 4:
            return

        ys, xs = np.nonzero(target_mask > 0.5)
        u_center = 0.5 * (float(xs.min()) + float(xs.max()))
        v_center = 0.5 * (float(ys.min()) + float(ys.max()))
        half = max(2, int(0.12 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
        u0 = max(0, int(round(u_center)) - half)
        u1 = min(self.width - 1, int(round(u_center)) + half)
        v0 = max(0, int(round(v_center)) - half)
        v1 = min(self.height - 1, int(round(v_center)) + half)

        center_depth = depth_m[v0:v1 + 1, u0:u1 + 1]
        valid = center_depth[np.isfinite(center_depth)]
        valid = valid[(valid > self.min_dis) & (valid < self.max_dis)]
        if valid.size == 0:
            mask_depth = depth_m[target_mask > 0.5]
            valid = mask_depth[np.isfinite(mask_depth)]
            valid = valid[(valid > self.min_dis) & (valid < self.max_dis)]
        if valid.size == 0:
            return

        depth_surface = float(np.percentile(valid, 20.0))
        target_x = min(depth_surface + float(cfg["target_radius"]), self.max_dis)
        target_y = -(u_center - float(cfg["camera_cx"])) * target_x / float(cfg["camera_fx"])
        target_z = -(v_center - float(cfg["camera_cy"])) * target_x / float(cfg["camera_fy"])
        target_c = np.array((target_x, target_y, target_z), dtype=np.float64)
        start_pos = self.get_current_position()
        self.mask_target_w = np.dot(self.Rotation_wc, target_c) + start_pos
        self.mask_target_stamp = stamp

    def get_current_position(self):
        if self.plan_from_reference and self.desire_pos is not None:
            return self.desire_pos
        return np.array((
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ))

    def get_current_velocity(self):
        if self.plan_from_reference and self.desire_vel is not None:
            return self.desire_vel
        return np.array((
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ))

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init:
            return

        time0 = time.time()
        obs_norm = self.process_odom()
        image = self.make_image_input(data)

        time1 = time.time()
        image_input = torch.from_numpy(image).to(self.device, non_blocking=True)
        obs_norm = obs_norm.to(self.device, non_blocking=True)
        obs_input = self.state_transform.prepare_input(obs_norm)

        time2 = time.time()
        endstate_pred, score_pred, objectness_pred, target_pred = self.policy(image_input, obs_input)
        endstate_pred = endstate_pred.cpu().numpy()
        score_pred = score_pred.cpu().numpy()
        objectness_pred = objectness_pred.cpu().numpy()
        target_pred = target_pred.cpu().numpy()
        time3 = time.time()

        endstate, score, objectness, target_b, action_id = self.process_output(
            endstate_pred, score_pred, objectness_pred, target_pred, return_all_preds=self.visualize
        )
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        selected_target_c = target_b[action_id] if self.visualize else target_b[0]
        selected_objectness = objectness[action_id] if self.visualize else objectness[0]
        start_pos = self.get_current_position()
        target_est_w = np.dot(self.Rotation_wc, selected_target_c) + start_pos
        if self.mask_target_w is not None:
            self.update_target_estimate(self.mask_target_w, data.header.stamp)
        elif selected_objectness >= self.objectness_threshold:
            self.update_target_estimate(target_est_w, data.header.stamp)

        with self.lock:
            start_vel = self.get_current_velocity()
            poly_start_vel = start_vel
            poly_start_acc = self.desire_acc
            hold_setpoint = self.get_tracking_setpoint(start_pos)
            if hold_setpoint is None:
                self.hold_mode = False
                end_pos = endstate_w[action_id, :, 0] + start_pos
                end_vel = endstate_w[action_id, :, 1]
                end_acc = endstate_w[action_id, :, 2]
            else:
                self.hold_mode = True
                end_pos, end_vel, end_acc, brake_now = hold_setpoint
                if brake_now:
                    poly_start_vel = np.zeros(3)
                    poly_start_acc = np.zeros(3)
            self.set_optimal_poly(start_pos, poly_start_vel, poly_start_acc, end_pos, end_vel, end_acc)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score, objectness, endstate_w)
        self.visualize_target_estimate()
        time5 = time.time()
        self.print_time(time0, time1, time2, time3, time4, time5)

    def get_tracking_setpoint(self, start_pos):
        if self.target_est_w is None:
            return None
        if self.target_est_stamp is not None:
            age = (rospy.Time.now() - self.target_est_stamp).to_sec()
            timeout = self.target_measurement_timeout
            if self.target_gt is not None and np.linalg.norm(self.target_gt_vel) <= self.follow_target_speed_threshold:
                timeout = self.target_hold_timeout
            if age > timeout and self.hover_hold_pos is None:
                self.hover_hold_pos = None
                self.hover_target_w = None
                return None

        rel = self.target_est_w - start_pos
        rel_xy = rel.copy()
        rel_xy[2] = 0.0
        dist = np.linalg.norm(rel_xy)
        if dist > 1e-3:
            view_dir = rel_xy / dist
        else:
            view_dir = self.Rotation_wc[:, 0].copy()
            view_dir[2] = 0.0
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-6)

        target_vel = self.target_est_vel_w
        if self.target_gt is not None:
            target_vel = self.target_gt_vel
        target_speed = np.linalg.norm(target_vel)
        target_is_moving = target_speed > self.follow_target_speed_threshold
        lower_bound = max(0.1, self.follow_distance - self.follow_deadband)
        upper_bound = self.follow_distance + self.follow_deadband
        standoff_pos = self.target_est_w - view_dir * self.follow_distance
        standoff_pos[2] = self.target_est_w[2]

        if target_is_moving:
            self.hover_hold_pos = None
            self.hover_target_w = None
            return None
        if dist > self.follow_capture_distance:
            self.hover_hold_pos = None
            self.hover_target_w = None
            return None

        if dist > upper_bound or dist < lower_bound:
            self.hover_hold_pos = None
            self.hover_target_w = None
            hold_pos = standoff_pos
        else:
            if (
                self.hover_hold_pos is None
                or self.hover_target_w is None
                or np.linalg.norm(self.target_est_w - self.hover_target_w) > 0.5 * self.follow_deadband
            ):
                self.hover_hold_pos = standoff_pos.copy()
                self.hover_target_w = self.target_est_w.copy()
            hold_pos = self.hover_hold_pos
        step = hold_pos - start_pos
        step_norm = np.linalg.norm(step)
        if step_norm > self.follow_max_step:
            hold_pos = start_pos + step / step_norm * self.follow_max_step

        hold_vel = target_vel if target_is_moving else np.zeros(3)
        hold_acc = np.zeros(3)
        return hold_pos, hold_vel, hold_acc, not target_is_moving

    def set_optimal_poly(self, start_pos, start_vel, start_acc, end_pos, end_vel, end_acc):
        self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], start_acc[0], end_pos[0], end_vel[0], end_acc[0], self.traj_time)
        self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], start_acc[1], end_pos[1], end_vel[1], end_acc[1], self.traj_time)
        self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], start_acc[2], end_pos[2], end_vel[2], end_acc[2], self.traj_time)

    def control_pub(self, _timer):
        if self.ctrl_time is None:
            return
        if self.ctrl_time > self.traj_time and not self.hold_mode:
            return

        with self.lock:
            self.ctrl_time += self.ctrl_dt
            eval_time = min(self.ctrl_time, self.traj_time) if self.hold_mode else self.ctrl_time
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            if self.hold_mode:
                control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
            else:
                control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(eval_time)
            control_msg.position.y = self.optimal_poly_y.get_position(eval_time)
            control_msg.position.z = self.optimal_poly_z.get_position(eval_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(eval_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(eval_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(eval_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(eval_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(eval_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(eval_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])

            if self.target_est_w is not None:
                target_dir = self.target_est_w - self.desire_pos
            else:
                target_dir = self.desire_vel
            yaw, yaw_dot = calculate_yaw(self.desire_vel, target_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.ctrl_pub.publish(control_msg)

    def process_output(self, endstate_pred, score_pred, objectness_pred, target_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)
        objectness_logits = objectness_pred.reshape(self.lattice_primitive.traj_num)
        objectness = 1.0 / (1.0 + np.exp(-objectness_logits))
        target_pred = target_pred.reshape(3, self.lattice_primitive.traj_num).T

        candidates = np.where(objectness >= self.objectness_threshold)[0]
        selection_score = score_pred - self.selection_objectness_bonus * objectness
        if candidates.size > 0:
            action_id = candidates[np.argmin(selection_score[candidates])]
        else:
            action_id = np.argmin(selection_score)

        if not return_all_preds:
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            target = self.state_transform.pred_to_target_cpu(target_pred[action_id, :][np.newaxis, :], np.array([action_id]))
            score = score_pred[action_id:action_id + 1]
            objectness = objectness[action_id:action_id + 1]
            action_id = 0
        else:
            lattice_ids = torch.arange(self.lattice_primitive.traj_num - 1, -1, -1)
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, lattice_ids)
            target = self.state_transform.pred_to_target_cpu(target_pred, np.arange(self.lattice_primitive.traj_num))
            score = score_pred
        return endstate, score, objectness, target, action_id

    def visualize_target_estimate(self):
        if self.target_est_w is None or self.target_est_pub.get_num_connections() == 0:
            return
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'
        msg = point_cloud2.create_cloud_xyz32(header, self.target_est_w.reshape(1, 3))
        self.target_est_pub.publish(msg)

    def visualize_trajectory(self, pred_score, pred_objectness, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array((
            self.odom.pose.pose.position.x,
            self.odom.pose.pose.position.y,
            self.odom.pose.pose.position.z,
        ))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((
            self.odom.twist.twist.linear.x,
            self.odom.twist.twist.linear.y,
            self.odom.twist.twist.linear.z,
        ))
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
            header.frame_id = 'world'
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
            intensity = np.repeat(pred_score - pred_objectness, t_values.size)
            points_array = np.column_stack((points_array, intensity))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1),
            ]
            self.all_trajs_pub.publish(point_cloud2.create_cloud(header, fields, points_array))

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
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"image-process: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        image = torch.zeros((1, cfg["input_channels"], self.height, self.width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        outputs = self.policy(image, obs)
        _ = self.state_transform.pred_to_endstate(outputs[0])
        _ = self.state_transform.pred_to_target(outputs[3])


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--weights_root", type=str, default="saved/with_tracker", help="tracker checkpoint root under YOPO/")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights_root = args.weights_root
    if not os.path.isabs(weights_root):
        weights_root = os.path.join(base_dir, weights_root)
    weight = "yopo_tracker_trt.pth" if args.use_tensorrt else os.path.join(weights_root, "YOPO_{}".format(args.trial), "epoch{}.pth".format(args.epoch))
    print("load weight from:", weight)

    settings = {
        'use_tensorrt': args.use_tensorrt,
        'pitch_angle_deg': -0,
        'odom_topic': '/sim/odom',
        'depth_topic': '/depth_image',
        'target_mask_topic': '/target_mask_image',
        'target_odom_topic': '/target/odom',
        'ctrl_topic': '/so3_control/pos_cmd',
        'plan_from_reference': False,
        'verbose': False,
        'visualize': True,
    }
    YopoTracker(settings, weight)
