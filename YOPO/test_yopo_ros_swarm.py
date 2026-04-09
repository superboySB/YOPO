import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if "YOPO_CONFIG_PATH" not in os.environ:
    os.environ["YOPO_CONFIG_PATH"] = os.path.join(BASE_DIR, "config", "swarm_traj_opt.yaml")

from ruamel.yaml import YAML

import rospy
import std_msgs.msg
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import time
import torch
import numpy as np
import argparse
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


def load_swarm_runtime_defaults():
    defaults = {
        "depth_fps": 10.0,
        "max_depth_dist": 20.0,
    }
    config_path = os.path.join(os.path.dirname(BASE_DIR), "Simulator", "src", "config", "swarm_config.yaml")
    if not os.path.exists(config_path):
        return defaults

    with open(config_path, "r", encoding="utf-8") as f:
        config = YAML(typ="safe").load(f)

    defaults["depth_fps"] = float(config["depth_fps"])
    defaults["max_depth_dist"] = float(config["camera"]["max_depth_dist"])
    return defaults


SWARM_RUNTIME_DEFAULTS = load_swarm_runtime_defaults()


def cleanup_cuda_memory():
    if not torch.cuda.is_available():
        return

    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        self.agent_name = self.config['agent_name']
        self.node_name = self.config['node_name']
        self.goal_topic = self.config['goal_topic']
        self.visual_prefix = self.config['visual_prefix'].rstrip('/')
        self.status_prefix = self.config['status_prefix'].rstrip('/')
        self.arrive_radius = self.config['arrive_radius']
        self.swarm_center = np.array(self.config['swarm_center'], dtype=np.float32)
        self.swarm_tangent_bias = self.config['swarm_tangent_bias']
        self.swarm_bias_radius = self.config['swarm_bias_radius']
        self.swarm_goal_center_mode = self.config['swarm_goal_center_mode']

        rospy.init_node(self.node_name, anonymous=False)
        # load params
        cfg["train"] = False
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.min_dis = 0.04
        self.max_dis = float(self.config['max_depth_dist'])
        self.goal = np.array(self.config['goal'])
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.arrive_hold_pos = None
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
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
        self.depth_fps = float(self.config['depth_fps'])  # used only as processing time tolerance for printing logs

        # Load Network
        cleanup_cuda_memory()
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
        cleanup_cuda_memory()

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher(f"{self.visual_prefix}/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher(f"{self.visual_prefix}/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher(f"{self.visual_prefix}/trajs_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        self.arrive_pub = rospy.Publisher(f"{self.status_prefix}/arrived", Bool, queue_size=1)
        self.goal_distance_pub = rospy.Publisher(f"{self.status_prefix}/goal_distance", Float32, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1, tcp_nodelay=True)
        self.goal_sub = None
        if self.goal_topic:
            self.goal_sub = rospy.Subscriber(self.goal_topic, PoseStamped, self.callback_set_goal, queue_size=1)
        # ros timer
        rospy.sleep(1.0)  # wait connection...
        self.publish_status()
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        print(f"[{self.agent_name}] YOPO Net Node Ready! goal={self.goal.tolist()}")
        rospy.spin()

    def build_hover_command(self):
        control_msg = PositionCommand()
        control_msg.header.stamp = rospy.Time.now()
        control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY

        if self.arrive_hold_pos is not None:
            hover_pos = np.array(self.arrive_hold_pos, dtype=np.float32)
        elif self.odom_init:
            hover_pos = np.array((
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ), dtype=np.float32)
        elif self.desire_pos is not None:
            hover_pos = np.array(self.desire_pos, dtype=np.float32)
        else:
            hover_pos = np.array(self.goal, dtype=np.float32)

        control_msg.position.x = float(hover_pos[0])
        control_msg.position.y = float(hover_pos[1])
        control_msg.position.z = float(hover_pos[2])
        control_msg.velocity.x = 0.0
        control_msg.velocity.y = 0.0
        control_msg.velocity.z = 0.0
        control_msg.acceleration.x = 0.0
        control_msg.acceleration.y = 0.0
        control_msg.acceleration.z = 0.0
        control_msg.yaw = float(self.last_yaw)
        control_msg.yaw_dot = 0.0
        return control_msg

    def get_current_position(self):
        if self.odom_init:
            return np.array((
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ), dtype=np.float32)
        if self.desire_pos is not None:
            return np.array(self.desire_pos, dtype=np.float32)
        return np.array(self.goal, dtype=np.float32)

    def callback_set_goal(self, data):
        nav_goal = np.asarray([
            data.pose.position.x,
            data.pose.position.y,
            self.swarm_center[2] if self.swarm_goal_center_mode else 2.0,
        ], dtype=np.float32)

        if self.swarm_goal_center_mode:
            current_pos = self.get_current_position()
            new_goal = nav_goal.copy()
            new_goal[:2] = current_pos[:2] + 2.0 * (nav_goal[:2] - current_pos[:2])
        else:
            new_goal = nav_goal

        self.goal = new_goal
        self.arrive = False
        self.arrive_hold_pos = None
        self.publish_status()
        if self.swarm_goal_center_mode:
            print(f"[{self.agent_name}] New Goal Center: ({nav_goal[0]:.1f}, {nav_goal[1]:.1f}) -> actual goal ({new_goal[0]:.1f}, {new_goal[1]:.1f}, {new_goal[2]:.1f})")
        else:
            print(f"[{self.agent_name}] New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

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
        self.odom_init = True

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        goal_distance = np.linalg.norm(pos - self.goal)
        if goal_distance < self.arrive_radius and not self.arrive:
            print(f"[{self.agent_name}] Arrive!")
            self.arrive = True
            self.arrive_hold_pos = pos.copy()
        self.publish_status(goal_distance)

    def process_odom(self):
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # goal_dir
        goal_w = self.goal - self.desire_pos
        goal_w = self.apply_swarm_goal_bias(goal_w)
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init:
            return
        if self.arrive:
            self.publish_status()
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
        depth = np.minimum(depth, self.max_dis) / self.max_dis

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
        endstate, score = self.process_output(endstate_pred, score_pred, return_all_preds=self.visualize)
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        action_id = np.argmin(score) if self.visualize else 0
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], endstate_w[action_id, 0, 0] + start_pos[0],
                                              endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], endstate_w[action_id, 1, 0] + start_pos[1],
                                              endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], endstate_w[action_id, 2, 0] + start_pos[2],
                                              endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], self.traj_time)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()

        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub(self, _timer):
        if self.arrive:
            self.desire_init = False
            self.ctrl_time = None
            if self.last_control_msg is not None:
                self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
                self.ctrl_pub.publish(self.last_control_msg)
            else:
                # Keep a safe fallback for the rare case where arrival is detected
                # before any READY command has been published.
                hover_msg = self.build_hover_command()
                self.last_control_msg = hover_msg
                self.ctrl_pub.publish(hover_msg)
            self.publish_status()
            return
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            self.publish_status()
            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
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
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)
            self.publish_status(np.linalg.norm(self.goal - self.desire_pos))

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
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num - 1, -1, -1))

        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
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

    def publish_status(self, goal_distance=None):
        if goal_distance is None:
            if self.desire_pos is not None:
                goal_distance = np.linalg.norm(self.goal - self.desire_pos)
            elif self.odom_init:
                pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
                goal_distance = np.linalg.norm(pos - self.goal)
            else:
                goal_distance = np.linalg.norm(self.goal)

        self.arrive_pub.publish(Bool(data=bool(self.arrive)))
        self.goal_distance_pub.publish(Float32(data=float(goal_distance)))

    def apply_swarm_goal_bias(self, goal_w):
        if self.swarm_tangent_bias <= 0.0 or self.swarm_bias_radius <= 0.0:
            return goal_w

        center_vec = self.desire_pos - self.swarm_center
        center_vec_xy = center_vec[:2]
        center_dist = np.linalg.norm(center_vec_xy)
        if center_dist < 1e-3 or center_dist >= self.swarm_bias_radius:
            return goal_w

        tangent = np.array([-center_vec_xy[1], center_vec_xy[0], 0.0], dtype=np.float32)
        tangent_norm = np.linalg.norm(tangent[:2])
        if tangent_norm < 1e-3:
            return goal_w

        tangent = tangent / tangent_norm
        bias_scale = 1.0 - center_dist / self.swarm_bias_radius
        return goal_w + tangent * (self.swarm_tangent_bias * bias_scale)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--weights_root", type=str, default="saved", help="checkpoint root under YOPO/")
    parser.add_argument("--agent_name", type=str, default="uav0", help="agent name for logging")
    parser.add_argument("--node_name", type=str, default="yopo_net_uav0", help="ROS node name")
    parser.add_argument("--odom_topic", type=str, default="/uav0/sim/odom", help="odometry topic")
    parser.add_argument("--depth_topic", type=str, default="/uav0/depth_image", help="depth image topic")
    parser.add_argument("--ctrl_topic", type=str, default="/uav0/so3_control/pos_cmd", help="controller command topic")
    parser.add_argument("--goal_topic", type=str, default="/move_base_simple/goal", help="goal topic; use empty string to disable")
    parser.add_argument("--visual_prefix", type=str, default="/uav0/yopo_net", help="visualization topic prefix")
    parser.add_argument("--status_prefix", type=str, default="/uav0/yopo", help="status topic prefix")
    parser.add_argument("--goal_x", type=float, default=50.0, help="goal x")
    parser.add_argument("--goal_y", type=float, default=0.0, help="goal y")
    parser.add_argument("--goal_z", type=float, default=2.0, help="goal z")
    parser.add_argument("--swarm_center_x", type=float, default=0.0, help="swarm center x")
    parser.add_argument("--swarm_center_y", type=float, default=0.0, help="swarm center y")
    parser.add_argument("--swarm_center_z", type=float, default=2.0, help="swarm center z")
    parser.add_argument("--swarm_tangent_bias", type=float, default=0.0, help="tangential bias magnitude near swarm center")
    parser.add_argument("--swarm_bias_radius", type=float, default=0.0, help="distance-to-center range where swarm bias is active")
    parser.add_argument("--swarm_goal_center_mode", type=int, default=0, help="treat 2D nav goal as a swarm center and mirror each UAV goal through it")
    parser.add_argument("--pitch_angle_deg", type=float, default=0.0, help="camera pitch angle")
    parser.add_argument("--plan_from_reference", type=int, default=0, help="plan from reference state or not")
    parser.add_argument("--verbose", type=int, default=0, help="print timing logs or not")
    parser.add_argument("--visualize", type=int, default=1, help="visualize all trajectories or not")
    parser.add_argument("--arrive_radius", type=float, default=5.0, help="goal arrival radius")
    parser.add_argument("--max_depth_dist", type=float, default=SWARM_RUNTIME_DEFAULTS["max_depth_dist"], help="depth image clipping distance used for normalization")
    parser.add_argument("--depth_fps", type=float, default=SWARM_RUNTIME_DEFAULTS["depth_fps"], help="depth callback frequency, used for timing tolerance logs")
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    weights_root = args.weights_root
    if not os.path.isabs(weights_root):
        weights_root = os.path.join(BASE_DIR, weights_root)
    weight = "yopo_trt.pth" if args.use_tensorrt else os.path.join(weights_root, "YOPO_{}".format(args.trial), "epoch{}.pth".format(args.epoch))
    print("load weight from:", weight)

    goal_topic = args.goal_topic.strip()
    if goal_topic.lower() in {"none", "null"}:
        goal_topic = ""

    settings = {'use_tensorrt': args.use_tensorrt,
                'agent_name': args.agent_name,
                'node_name': args.node_name,
                'goal': [args.goal_x, args.goal_y, args.goal_z],
                'goal_topic': goal_topic,
                'visual_prefix': args.visual_prefix,
                'status_prefix': args.status_prefix,
                'arrive_radius': args.arrive_radius,
                'swarm_center': [args.swarm_center_x, args.swarm_center_y, args.swarm_center_z],
                'swarm_tangent_bias': args.swarm_tangent_bias,
                'swarm_bias_radius': args.swarm_bias_radius,
                'swarm_goal_center_mode': bool(args.swarm_goal_center_mode),
                'pitch_angle_deg': -args.pitch_angle_deg,    # 相机俯仰角(仰为负)
                'odom_topic': args.odom_topic,
                'depth_topic': args.depth_topic,
                'ctrl_topic': args.ctrl_topic,
                'max_depth_dist': args.max_depth_dist,
                'depth_fps': args.depth_fps,
                'plan_from_reference': bool(args.plan_from_reference),
                'verbose': bool(args.verbose),
                'visualize': bool(args.visualize)
                }
    YopoNet(settings, weight)
