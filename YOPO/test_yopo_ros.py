import argparse
import os
import struct
import time
from threading import Lock, Thread

import cv2
import message_filters
import numpy as np
import rospy
import std_msgs.msg
import torch
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import Marker

from config.config import cfg
from control_msg import PositionCommand
from joystick_control import map_planar_stick
from joystick_safety import BodyPose, CameraIntrinsics, evaluate_altitude_lock_safety
from policy.poly_solver import Poly5Solver, Polys5Solver, calculate_yaw
from policy.yopo_network import YOPOOmniNetwork


_JS_EVENT = struct.Struct("IhBB")
_JS_AXIS = 0x02


def _joystick_plan_is_stale(last_commit_monotonic, now_monotonic, timeout):
    """Return whether an active joystick plan lacks a recent successful commit."""
    return last_commit_monotonic is None or now_monotonic - last_commit_monotonic > timeout


def _joystick_plan_is_current(input_active, sampled_epoch, current_epoch):
    """Gate inference results against center, direction, and watchdog invalidation."""
    return bool(input_active) and sampled_epoch == current_epoch


def _joystick_candidate_order(scores):
    """Return every finite candidate index in stable ascending score order."""
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite_ids = np.flatnonzero(np.isfinite(flat))
    return sorted((int(index) for index in finite_ids), key=lambda index: (float(flat[index]), index))


class JoystickIntent:
    def __init__(self, device, axis_x, axis_y, axis_max, deadzone, invert_x, invert_y, swap_xy, calibrate=False):
        self.device = device
        self.axis_x = int(axis_x)
        self.axis_y = int(axis_y)
        if self.axis_x < 0 or self.axis_y < 0 or self.axis_x == self.axis_y:
            raise ValueError("joystick axis indices must be distinct non-negative integers")
        self.axis_max = float(axis_max)
        self.deadzone = float(deadzone)
        self.invert_x = bool(invert_x)
        self.invert_y = bool(invert_y)
        self.swap_xy = bool(swap_xy)
        self.calibrate = bool(calibrate)
        self.lock = Lock()
        self.axes = {}
        self.enabled = True
        self.connected = False
        self.unlocked = False
        self.neutral_samples = 0
        self.neutral_samples_required = 5
        self.fd = None

        # Validate calibration values before starting the reader thread.
        map_planar_stick(
            0,
            0,
            axis_max=self.axis_max,
            deadzone=self.deadzone,
            invert_horizontal=self.invert_x,
            invert_vertical=self.invert_y,
            swap_xy=self.swap_xy,
        )

        # Opening a real js device returns immediately, but opening a read-only
        # FIFO blocks until a writer exists.  Leave all opens to the daemon
        # reader so ROS publishers/timers can still initialize fail-closed.
        print(f"Joystick reader starting for {self.device}; waiting for connection and centered axes.")
        Thread(target=self._reader, daemon=True).start()

    def _disconnect(self):
        with self.lock:
            self.axes.clear()
            self.connected = False
            self.unlocked = False
            self.neutral_samples = 0
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def _reconnect(self):
        while not rospy.is_shutdown():
            try:
                self.fd = os.open(self.device, os.O_RDONLY)
            except OSError as exc:
                rospy.logwarn_throttle(2.0, "Waiting for joystick %s: %s", self.device, exc)
                rospy.sleep(0.5)
                continue
            with self.lock:
                self.axes.clear()
                self.connected = True
                self.unlocked = False
                self.neutral_samples = 0
            rospy.loginfo("Joystick reconnected: %s", self.device)
            return True
        return False

    def _reader(self):
        buf = b""
        while not rospy.is_shutdown():
            if self.fd is None:
                if not self._reconnect():
                    return
                buf = b""
            try:
                data = os.read(self.fd, 64)
            except OSError as exc:
                rospy.logwarn_throttle(1.0, "Joystick read failed on %s: %s", self.device, exc)
                self._disconnect()
                buf = b""
                if not self._reconnect():
                    return
                continue
            if not data:
                rospy.logwarn_throttle(1.0, "Joystick disconnected: %s", self.device)
                self._disconnect()
                buf = b""
                if not self._reconnect():
                    return
                continue
            buf += data
            while len(buf) >= _JS_EVENT.size:
                _event_time, value, event_type, number = _JS_EVENT.unpack(buf[:_JS_EVENT.size])
                buf = buf[_JS_EVENT.size:]
                if event_type & _JS_AXIS:
                    with self.lock:
                        self.axes[number] = value
            if self.calibrate:
                with self.lock:
                    snap = dict(self.axes)
                body_fraction, magnitude, _active = self.read_body_velocity_fraction()
                connected, initialized, unlocked = self.get_status()
                rospy.loginfo_throttle(
                    0.2,
                    "joystick axes=%s body_fraction=[%.3f, %.3f, %.3f] magnitude=%.3f "
                    "connected=%s initialized=%s unlocked=%s",
                    " ".join("%d:%d" % (axis, snap[axis]) for axis in sorted(snap)),
                    body_fraction[0],
                    body_fraction[1],
                    body_fraction[2],
                    magnitude,
                    connected,
                    initialized,
                    unlocked,
                )

    def read_body_velocity_fraction(self):
        with self.lock:
            if not self.connected:
                return np.zeros(3, dtype=np.float32), 0.0, False
            initialized = self.axis_x in self.axes and self.axis_y in self.axes
            if not initialized:
                return np.zeros(3, dtype=np.float32), 0.0, False
            raw_horizontal = self.axes[self.axis_x]
            raw_vertical = self.axes[self.axis_y]
            # Keep mapping and arming under the same lock as the axis snapshot.
            # A disconnect/reconnect cannot therefore arm a new device using
            # values that belonged to the previous file descriptor.
            body_x, body_y, magnitude = map_planar_stick(
                raw_horizontal,
                raw_vertical,
                axis_max=self.axis_max,
                deadzone=self.deadzone,
                invert_horizontal=self.invert_x,
                invert_vertical=self.invert_y,
                swap_xy=self.swap_xy,
            )

            just_unlocked = False
            if not self.unlocked:
                if magnitude <= 0.0:
                    self.neutral_samples += 1
                    if self.neutral_samples >= self.neutral_samples_required:
                        self.unlocked = True
                        just_unlocked = True
                else:
                    self.neutral_samples = 0
                unlocked = self.unlocked
            else:
                unlocked = True

        if just_unlocked:
            rospy.loginfo("Joystick unlocked after both configured axes initialized and the stick was centered.")
        if not unlocked:
            return np.zeros(3, dtype=np.float32), 0.0, False
        if magnitude <= 0.0:
            return np.zeros(3, dtype=np.float32), 0.0, False
        command = np.asarray([body_x, body_y, 0.0], dtype=np.float32)
        return command, magnitude, True

    def get_status(self):
        with self.lock:
            initialized = self.axis_x in self.axes and self.axis_y in self.axes
            return self.connected, initialized, self.unlocked


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


class YopoOmniNet:
    def __init__(self, settings, weight):
        rospy.init_node("yopo_net", anonymous=False)
        cfg["train"] = False

        self.settings = settings
        self.weight = weight
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.max_depth = float(settings["max_depth"])
        self.depth_normalized = bool(settings.get("depth_normalized", False))
        if self.max_depth <= 0.0:
            raise ValueError("max_depth must be positive")
        self.velocity = float(settings["velocity"])
        self.traj_time = float(cfg["sgm_time"])
        self.ctrl_dt = float(settings["ctrl_dt"])
        self.arrive_dist = float(settings["arrive_dist"])
        self.depth_unit_logged = False
        self.verbose = bool(settings["verbose"])
        self.visualize = bool(settings["visualize"])
        self.plan_from_reference = bool(settings["plan_from_reference"])
        self.control_mode = settings["control_mode"]
        self.fixed_yaw_value = settings["fixed_yaw_value"]
        self.goal = np.asarray(settings["goal"], dtype=np.float32)
        self.view_names = ["front", "left", "right", "back"]
        self.joystick = None
        self.joystick_hold = False
        self.joystick_input_active = False
        self.joystick_plan_ready = False
        self.joystick_intent_epoch = 0
        self.joystick_depth_timeout = float(settings.get("joystick_depth_timeout", 0.20))
        self.joystick_last_plan_commit_monotonic = None
        self.joystick_fraction_h = np.zeros(3, dtype=np.float32)
        self.joystick_fraction_signature = np.zeros(3, dtype=np.float32)
        self.joystick_fraction_epoch_epsilon = 0.02
        self.joystick_hold_position = None
        # The right stick is strictly planar.  Before the first armed stick
        # command this target follows odometry so a planner started during the
        # controller's automatic takeoff cannot freeze a low startup height.
        # The first real active command locks it for every later active, center,
        # and disconnect/reconnect state.
        self.joystick_altitude_target = None
        self.joystick_altitude_locked = False
        self.joystick_alpha = 0.0
        # The checkpoint chooses a collision-aware endpoint but its intent loss
        # does not make the continuously replanned trajectory a velocity
        # controller.  This bounded outer loop adjusts only a scalar endpoint
        # contraction and, when contraction is insufficient, the time used to
        # traverse the original endpoint.  It never expands model geometry.
        self.joystick_speed_kp = float(settings["joystick_speed_kp"])
        self.joystick_speed_accel_max = float(settings["joystick_speed_accel_max"])
        self.joystick_min_traj_time = float(settings["joystick_min_traj_time"])
        self.joystick_altitude_safety = bool(settings.get("joystick_altitude_safety", True))
        self.joystick_strict_footprint = bool(settings.get("joystick_strict_footprint", False))
        self.joystick_rewrite_threshold = float(settings.get("joystick_rewrite_threshold", 0.20))
        self.joystick_vehicle_radius = float(settings.get("joystick_vehicle_radius", 0.30))
        self.joystick_safety_margin = float(settings.get("joystick_safety_margin", 0.15))
        self.joystick_speed_horizon_samples = 33
        self.joystick_horizon_filter_alpha = 0.35
        self.joystick_horizon_step_max = 0.05
        self.joystick_servo_horizon = self.traj_time
        self.joystick_brake_accel = 4.0
        self.joystick_brake_time_max = 1.5
        self.joystick_brake_distance_max = 3.0
        self.current_vdes_w = np.zeros(3, dtype=np.float32)
        self.current_vdes_b = np.zeros(3, dtype=np.float32)
        self.policy_vdes_b = np.zeros(3, dtype=np.float32)
        self.policy_intent_speed = float(cfg["omni_vdes_speed_max"])
        self.policy_velocity_scale = float(cfg["vel_max_train"])
        if self.policy_intent_speed <= 0.0 or self.policy_velocity_scale <= 0.0:
            raise ValueError("YOPO policy intent and velocity scales must be positive")
        if not np.isfinite(self.joystick_speed_kp) or self.joystick_speed_kp <= 0.0:
            raise ValueError("joystick speed feedback gain must be finite and positive")
        if not np.isfinite(self.joystick_speed_accel_max) or self.joystick_speed_accel_max <= 0.0:
            raise ValueError("joystick speed acceleration limit must be finite and positive")
        if (
            not np.isfinite(self.joystick_min_traj_time)
            or self.joystick_min_traj_time < 3.0 * self.ctrl_dt
            or self.joystick_min_traj_time > self.traj_time
        ):
            raise ValueError(
                "joystick minimum trajectory time must be finite, cover three control ticks, "
                "and not exceed the trained trajectory time"
            )
        if self.control_mode == "joystick" and (
            not np.isfinite(self.joystick_depth_timeout)
            or self.joystick_depth_timeout < 2.0 * self.ctrl_dt
        ):
            raise ValueError(
                "joystick depth timeout must be finite and cover at least two control ticks"
            )
        safety_scalars = {
            "joystick rewrite threshold": self.joystick_rewrite_threshold,
            "joystick vehicle radius": self.joystick_vehicle_radius,
            "joystick safety margin": self.joystick_safety_margin,
        }
        if self.control_mode == "joystick" and self.joystick_altitude_safety:
            for name, value in safety_scalars.items():
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative")
            if self.joystick_vehicle_radius <= 0.0:
                raise ValueError("joystick vehicle radius must be positive")
            horizontal_fov_deg = float(cfg["tof_horizontal_fov_deg"])
            vertical_fov_deg = float(cfg["tof_vertical_fov_deg"])
            if not 0.0 < horizontal_fov_deg < 180.0 or not 0.0 < vertical_fov_deg < 180.0:
                raise ValueError("ToF horizontal/vertical FOV calibration must lie in (0, 180) degrees")
            self.joystick_camera_intrinsics = CameraIntrinsics(
                fx=0.5 * self.width / np.tan(np.radians(horizontal_fov_deg) / 2.0),
                fy=0.5 * self.height / np.tan(np.radians(vertical_fov_deg) / 2.0),
                cx=0.5 * (self.width - 1),
                cy=0.5 * (self.height - 1),
            )
        else:
            self.joystick_camera_intrinsics = None
        self.control_source = "nav_goal"
        if self.control_mode == "joystick":
            self.joystick = JoystickIntent(**settings["joystick"])
            # Joystick mode is fail-closed: an unavailable device waits/reconnects
            # while the controller holds, instead of flying toward the RViz goal.
        # Body-frame lateral/backward commands must not also rotate the body,
        # otherwise the desired world velocity rotates continuously with yaw.
        self.fixed_yaw = bool(settings["fixed_yaw"]) or self.control_mode == "joystick"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = YOPOOmniNetwork().to(self.device)
        state_dict = torch.load(weight, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(state_dict, strict=True)
        self.policy.eval()
        self.warm_up()

        self.odom = Odometry()
        self.odom_init = False
        self.desire_init = False
        self.arrive = False
        self.last_yaw = 0.0
        self.ctrl_time = None
        self.plan_horizon = self.traj_time
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.last_control_msg = None
        self.lock = Lock()
        # Depth and control timers both poll the same device state.  Serialize
        # the complete sample/compute/commit operation so an older sample can
        # never commit after a newer neutral or direction sample.
        self.joystick_poll_lock = Lock()

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
        self.vdes_body_pub = rospy.Publisher("/yopo/vdes_body", Vector3Stamped, queue_size=1)
        self.ctrl_pub = rospy.Publisher(settings["ctrl_topic"], PositionCommand, queue_size=1)

        self.odom_sub = rospy.Subscriber(settings["odom_topic"], Odometry, self.callback_odometry, queue_size=1, tcp_nodelay=True)
        depth_subs = [
            message_filters.Subscriber(topic, Image, queue_size=1)
            for topic in settings["depth_topics"]
        ]
        self.depth_sync = message_filters.ApproximateTimeSynchronizer(depth_subs, queue_size=2, slop=0.03)
        self.depth_sync.registerCallback(self.callback_depths)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)

        print("YOPO-Omni Net Node Ready!")
        print("Depth topics:", settings["depth_topics"])
        print(
            "Depth input units:",
            "explicit normalized 32FC1"
            if self.depth_normalized
            else f"32FC1 meters / 16UC1 millimeters; normalization range={self.max_depth:g}m",
        )
        print("load weight from:", weight)
        if self.control_mode == "joystick":
            print(
                "Control mode: joystick desired-velocity vector. "
                "Right stick up/down -> body x forward/back, left/right -> body y left/right; "
                f"stick travel -> 0..{self.velocity:.2f}m/s; centered stick brakes/holds; yaw is locked."
            )
        else:
            print("Control mode: RViz 2D Nav Goal (/move_base_simple/goal), unchanged from the original flow.")
        print(
            "Runtime decode:",
            f"radius=[{float(cfg['omni_radius_min']):.2f}, {float(cfg['omni_radius_max']):.2f}],",
            f"sgm_time={float(cfg['sgm_time']):.3f}s,",
            f"velocity={self.velocity:.2f}m/s",
        )
        if self.control_mode == "joystick":
            print(
                "Joystick speed servo:",
                f"kp={self.joystick_speed_kp:.2f}/s,",
                f"horizontal_accel_max={self.joystick_speed_accel_max:.2f}m/s^2,",
                f"plan_horizon=[{self.joystick_min_traj_time:.2f}, {self.traj_time:.2f}]s,",
                "endpoint_scale=[0, 1],",
                f"depth_plan_timeout={self.joystick_depth_timeout:.3f}s",
            )
            if self.joystick_altitude_safety:
                footprint_mode = (
                    "strict full-sphere footprint certification"
                    if self.joystick_strict_footprint
                    else "center-in-FOV partial-footprint residual-risk mode"
                )
                print(
                    "Joystick altitude-rewrite safety: enabled, all network candidates checked,",
                    f"rewrite_threshold={self.joystick_rewrite_threshold:.2f}m,",
                    f"clearance={self.joystick_vehicle_radius + self.joystick_safety_margin:.2f}m,",
                    f"ToF={self.width}x{self.height} raw metric, range={self.max_depth:.2f}m,",
                    f"footprint={footprint_mode}",
                )
                if not self.joystick_strict_footprint:
                    print(
                        "Joystick footprint warning: clipped parts of a vehicle sphere are NOT certified; "
                        "only the visible valid footprint is checked.  Center-outside-FOV/blind, invalid, "
                        "and collision cases remain fail-closed."
                    )
            else:
                print("Joystick altitude-rewrite safety: DISABLED by command-line override.")
        rospy.spin()

    def warm_up(self):
        depth = torch.zeros((1, 4, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        state = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            self.policy(depth, state)

    def callback_set_goal(self, data):
        z = self.goal[2] if len(self.goal) >= 3 else 2.0
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, z], dtype=np.float32)
        self.arrive = False
        if self.control_mode == "joystick":
            print(
                f"RViz Goal stored: ({self.goal[0]:.1f}, {self.goal[1]:.1f}, {self.goal[2]:.1f}); "
                "joystick mode is currently driving vdes_b."
            )
        else:
            print(f"New Goal: ({self.goal[0]:.1f}, {self.goal[1]:.1f}, {self.goal[2]:.1f})")

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
        if self.control_mode == "joystick":
            with self.lock:
                if not self.joystick_altitude_locked:
                    self.joystick_altitude_target = float(pos[2])
                    if self.joystick_hold_position is not None:
                        self.joystick_hold_position[2] = self.joystick_altitude_target
            self.arrive = False
        else:
            dist_to_goal = np.linalg.norm(pos - self.goal)
            if dist_to_goal < self.arrive_dist and not self.arrive:
                print(f"Arrive! dist={dist_to_goal:.2f} m, threshold={self.arrive_dist:.2f} m")
                self.arrive = True
        self.publish_speed_text(data)

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
            itemsize = 4
            dtype_code = "f4"
            unit_msg = "explicit normalized 32FC1" if self.depth_normalized else "32FC1 meters"
        elif data.encoding == "16UC1":
            if self.depth_normalized:
                raise ValueError("Explicit normalized depth mode supports only 32FC1 images")
            itemsize = 2
            dtype_code = "u2"
            unit_msg = "16UC1 millimeters converted to meters"
        else:
            raise ValueError(f"Unsupported depth encoding: {data.encoding}")

        height = int(data.height)
        width = int(data.width)
        step = int(data.step)
        if height <= 0 or width <= 0 or step < width * itemsize or step % itemsize != 0:
            raise ValueError(
                f"Invalid {data.encoding} image layout: {width}x{height}, step={step}"
            )
        if len(data.data) != height * step:
            raise ValueError(
                f"Invalid {data.encoding} payload size: got {len(data.data)}, expected {height * step}"
            )
        byte_order = ">" if int(data.is_bigendian) else "<"
        row_items = step // itemsize
        depth = np.frombuffer(data.data, dtype=np.dtype(byte_order + dtype_code))
        depth = depth.reshape(height, row_items)[:, :width].astype(np.float32)
        if data.encoding == "16UC1":
            depth = depth / 1000.0

        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        finite_depth = depth[np.isfinite(depth)]
        max_value = float(finite_depth.max()) if finite_depth.size else 0.0
        if self.depth_normalized:
            depth_norm = depth.astype(np.float32)
        else:
            depth_norm = depth.astype(np.float32) / self.max_depth
            unit_msg = f"{unit_msg} / {self.max_depth:g}m"

        depth_norm = np.nan_to_num(depth_norm, nan=1.0, posinf=1.0, neginf=0.0)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        if not self.depth_unit_logged:
            print(
                f"Depth input: encoding={data.encoding}, raw_max={max_value:.3f}, mode={unit_msg}, "
                f"normalized_range=[{depth_norm.min():.4f}, {depth_norm.max():.4f}]"
            )
            self.depth_unit_logged = True
        return depth_norm[None, ...]

    def _decode_metric_depth_for_joystick_safety(self, data):
        """Strictly decode an untouched 8x8 image into meters plus validity.

        This path intentionally does not share the policy preprocessor: no
        resize, clipping, or nan_to_num operation may turn an unobserved zone
        into a clear max-range return for the altitude-rewrite veto.
        """
        if int(data.height) != self.height or int(data.width) != self.width:
            raise ValueError(
                f"expected {self.width}x{self.height}, got {int(data.width)}x{int(data.height)}"
            )
        if int(data.is_bigendian) not in (0, 1):
            raise ValueError(f"invalid Image.is_bigendian={data.is_bigendian!r}")

        if data.encoding == "32FC1":
            itemsize = 4
            dtype_code = "f4"
            scale_to_meters = self.max_depth if self.depth_normalized else 1.0
        elif data.encoding == "16UC1":
            if self.depth_normalized:
                raise ValueError("normalized depth mode supports only 32FC1")
            itemsize = 2
            dtype_code = "u2"
            scale_to_meters = 0.001
        else:
            raise ValueError(f"unsupported encoding {data.encoding!r}")

        step = int(data.step)
        minimum_step = self.width * itemsize
        if step < minimum_step or step % itemsize != 0:
            raise ValueError(
                f"invalid row step {step} for {data.encoding} width {self.width}; "
                f"expected a multiple of {itemsize} no smaller than {minimum_step}"
            )
        expected_bytes = self.height * step
        if len(data.data) != expected_bytes:
            raise ValueError(
                f"invalid payload size {len(data.data)} for height {self.height} and step {step}; "
                f"expected exactly {expected_bytes} bytes"
            )

        byte_order = ">" if int(data.is_bigendian) else "<"
        row_items = step // itemsize
        raw = np.frombuffer(data.data, dtype=np.dtype(byte_order + dtype_code))
        raw = raw.reshape(self.height, row_items)[:, : self.width]
        depth_m = raw.astype(np.float64) * scale_to_meters
        valid = (
            np.isfinite(depth_m)
            & (depth_m > 0.0)
            & (depth_m <= self.max_depth + 1e-6)
        )
        return depth_m, valid

    def _joystick_altitude_locked_value(self, freeze=False):
        """Track or freeze the joystick world-z target; caller holds self.lock."""
        current_altitude = float(self.odom.pose.pose.position.z)
        if not np.isfinite(current_altitude):
            raise ValueError("Cannot establish joystick altitude lock from non-finite odometry")
        if self.joystick_altitude_target is None or not self.joystick_altitude_locked:
            self.joystick_altitude_target = current_altitude
        if freeze:
            self.joystick_altitude_locked = True
        return float(self.joystick_altitude_target)

    def _capture_joystick_hold_locked(self):
        """Capture a reachable horizontal stopping point; caller holds self.lock."""
        pos = np.array(
            [self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z],
            dtype=np.float32,
        )
        vel = np.array(
            [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, 0.0],
            dtype=np.float32,
        )
        speed_xy = float(np.linalg.norm(vel[:2]))
        if speed_xy > 1e-4:
            stop_time = min(max(speed_xy / self.joystick_brake_accel, self.ctrl_dt), self.joystick_brake_time_max)
            displacement = 0.5 * vel * stop_time
            distance = float(np.linalg.norm(displacement[:2]))
            if distance > self.joystick_brake_distance_max:
                displacement *= self.joystick_brake_distance_max / distance
            pos += displacement
        # Releasing/reversing/disconnecting must recover the frozen world
        # altitude, rather than accepting any vertical error accumulated while
        # moving.  Prior to the first active input this still follows takeoff.
        pos[2] = self._joystick_altitude_locked_value()
        self.joystick_hold_position = pos
        self.joystick_plan_ready = False
        self.joystick_last_plan_commit_monotonic = None
        self.ctrl_time = None
        self.plan_horizon = self.traj_time
        self.joystick_servo_horizon = self.traj_time

    def _publish_vdes_body(self, vdes_b):
        vdes_msg = Vector3Stamped()
        # The body-frame vector was computed with the latest odometry attitude,
        # so publish that exact transform timestamp instead of callback wall
        # time.  Consumers can then pair vdes with the correct orientation even
        # while the vehicle is pitching during acceleration.
        odom_stamp = self.odom.header.stamp
        vdes_msg.header.stamp = odom_stamp if odom_stamp.to_sec() > 0.0 else rospy.Time.now()
        vdes_msg.header.frame_id = (self.odom.child_frame_id or "body").lstrip("/")
        vdes_msg.vector.x = float(vdes_b[0])
        vdes_msg.vector.y = float(vdes_b[1])
        vdes_msg.vector.z = float(vdes_b[2])
        self.vdes_body_pub.publish(vdes_msg)

    def _projected_plan_acceleration(
        self,
        best_state_w,
        start_vel,
        start_acc,
        direction_w,
        endpoint_scale,
        horizon,
    ):
        """Mean acceleration actually sampled before the next depth replan."""
        best_state_w = np.asarray(best_state_w, dtype=np.float64)
        direction_w = np.asarray(direction_w, dtype=np.float64)
        start_vel = np.asarray(start_vel, dtype=np.float64)
        start_acc = np.asarray(start_acc, dtype=np.float64)
        endpoint_scale = float(endpoint_scale)
        horizon = float(horizon)
        if (
            best_state_w.shape != (3, 3)
            or direction_w.shape != (3,)
            or start_vel.shape != (3,)
            or start_acc.shape != (3,)
            or not np.all(np.isfinite(best_state_w))
            or not np.all(np.isfinite(direction_w))
            or not np.all(np.isfinite(start_vel))
            or not np.all(np.isfinite(start_acc))
            or not np.isfinite(endpoint_scale)
            or not 0.0 <= endpoint_scale <= 1.0
            or not np.isfinite(horizon)
            or horizon < 3.0 * self.ctrl_dt
        ):
            return float("nan")

        projected_endpoint = direction_w @ best_state_w
        projected_poly = Poly5Solver(
            0.0,
            float(direction_w @ start_vel),
            float(direction_w @ start_acc),
            endpoint_scale * float(projected_endpoint[0]),
            endpoint_scale * float(projected_endpoint[1]),
            endpoint_scale * float(projected_endpoint[2]),
            horizon,
        )
        probe_times = self.ctrl_dt * np.arange(1.0, 4.0)
        accelerations = np.asarray(
            [projected_poly.get_acceleration(float(t)) for t in probe_times],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(accelerations)):
            return float("nan")
        return float(np.mean(accelerations))

    def _horizontal_plan_acceleration_peak(
        self,
        best_state_w,
        start_vel,
        start_acc,
        endpoint_scale,
        horizon,
    ):
        """Peak horizontal acceleration over control ticks before replanning."""
        best_state_w = np.asarray(best_state_w, dtype=np.float64)
        start_vel = np.asarray(start_vel, dtype=np.float64)
        start_acc = np.asarray(start_acc, dtype=np.float64)
        endpoint_scale = float(endpoint_scale)
        horizon = float(horizon)
        if (
            best_state_w.shape != (3, 3)
            or start_vel.shape != (3,)
            or start_acc.shape != (3,)
            or not np.all(np.isfinite(best_state_w))
            or not np.all(np.isfinite(start_vel))
            or not np.all(np.isfinite(start_acc))
            or not np.isfinite(endpoint_scale)
            or not 0.0 <= endpoint_scale <= 1.0
            or not np.isfinite(horizon)
            or horizon < 3.0 * self.ctrl_dt
        ):
            return float("nan")

        polynomials = [
            Poly5Solver(
                0.0,
                float(start_vel[axis]),
                float(start_acc[axis]),
                endpoint_scale * float(best_state_w[axis, 0]),
                endpoint_scale * float(best_state_w[axis, 1]),
                endpoint_scale * float(best_state_w[axis, 2]),
                horizon,
            )
            for axis in range(2)
        ]
        probe_times = self.ctrl_dt * np.arange(1.0, 4.0)
        accelerations = np.asarray(
            [
                [poly.get_acceleration(float(t)) for poly in polynomials]
                for t in probe_times
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(accelerations)):
            return float("nan")
        return float(np.max(np.linalg.norm(accelerations, axis=1)))

    def _select_joystick_speed_plan(
        self,
        best_state_w,
        start_vel,
        start_acc,
        operator_vdes_w,
        fallback_scale,
    ):
        """Return a bounded spatial scale and per-plan horizon.

        The model endpoint is first contracted at the checkpoint's trained
        horizon.  If a contraction cannot produce the requested velocity-loop
        acceleration, the endpoint remains unscaled and only its time
        parameterization is shortened.  Thus the selected endpoint is never
        extended beyond geometry scored by the checkpoint.
        """
        operator_vdes_w = np.asarray(operator_vdes_w, dtype=np.float64)
        start_vel = np.asarray(start_vel, dtype=np.float64)
        fallback_scale = float(np.clip(fallback_scale, 0.0, 1.0))
        fallback = (fallback_scale, self.traj_time, 0.0, float("nan"), 0.0, 0.0)
        if (
            operator_vdes_w.shape != (3,)
            or start_vel.shape != (3,)
            or not np.all(np.isfinite(operator_vdes_w))
            or not np.all(np.isfinite(start_vel))
        ):
            return fallback

        command_speed = float(np.linalg.norm(operator_vdes_w[:2]))
        if not np.isfinite(command_speed) or command_speed <= 1e-6:
            return (0.0, self.traj_time, 0.0, 0.0, 0.0, 0.0)
        direction_w = np.array(
            [operator_vdes_w[0] / command_speed, operator_vdes_w[1] / command_speed, 0.0],
            dtype=np.float64,
        )
        current_projection = float(direction_w @ start_vel)
        acceleration_target = float(
            np.clip(
                self.joystick_speed_kp * (command_speed - current_projection),
                -self.joystick_speed_accel_max,
                self.joystick_speed_accel_max,
            )
        )

        acceleration_at_zero = self._projected_plan_acceleration(
            best_state_w, start_vel, start_acc, direction_w, 0.0, self.traj_time
        )
        acceleration_at_one = self._projected_plan_acceleration(
            best_state_w, start_vel, start_acc, direction_w, 1.0, self.traj_time
        )
        endpoint_accelerations = np.asarray([acceleration_at_zero, acceleration_at_one], dtype=np.float64)
        if not np.all(np.isfinite(endpoint_accelerations)):
            return (fallback_scale, self.traj_time, acceleration_target, float("nan"), command_speed, current_projection)

        low_acceleration = float(np.min(endpoint_accelerations))
        high_acceleration = float(np.max(endpoint_accelerations))
        authority = acceleration_at_one - acceleration_at_zero
        if (
            abs(authority) > 1e-7
            and low_acceleration <= acceleration_target <= high_acceleration
        ):
            endpoint_scale = float(
                np.clip((acceleration_target - acceleration_at_zero) / authority, 0.0, 1.0)
            )
            predicted_acceleration = self._projected_plan_acceleration(
                best_state_w,
                start_vel,
                start_acc,
                direction_w,
                endpoint_scale,
                self.traj_time,
            )
            acceleration_peak = self._horizontal_plan_acceleration_peak(
                best_state_w,
                start_vel,
                start_acc,
                endpoint_scale,
                self.traj_time,
            )
            if (
                np.isfinite(acceleration_peak)
                and acceleration_peak <= self.joystick_speed_accel_max + 1e-6
            ):
                return (
                    endpoint_scale,
                    self.traj_time,
                    acceleration_target,
                    predicted_acceleration,
                    command_speed,
                    current_projection,
                )

        if acceleration_target > high_acceleration:
            boundary_scale = float(np.argmax(endpoint_accelerations))
        elif acceleration_target < low_acceleration:
            boundary_scale = float(np.argmin(endpoint_accelerations))
        else:
            # The affine solution exists but violates the acceleration bound,
            # or both endpoints have effectively identical authority.  Search
            # contractions at the trained horizon before changing time.
            boundary_scale = float(
                np.clip(
                    (acceleration_target - acceleration_at_zero) / authority
                    if abs(authority) > 1e-7
                    else fallback_scale,
                    0.0,
                    1.0,
                )
            )

        candidates = []
        unsafe_candidates = []
        for endpoint_scale in np.linspace(0.0, 1.0, self.joystick_speed_horizon_samples):
            predicted_acceleration = self._projected_plan_acceleration(
                best_state_w,
                start_vel,
                start_acc,
                direction_w,
                float(endpoint_scale),
                self.traj_time,
            )
            acceleration_peak = self._horizontal_plan_acceleration_peak(
                best_state_w,
                start_vel,
                start_acc,
                float(endpoint_scale),
                self.traj_time,
            )
            if np.isfinite(predicted_acceleration) and np.isfinite(acceleration_peak):
                candidate = (
                    abs(predicted_acceleration - acceleration_target),
                    -self.traj_time,
                    float(endpoint_scale),
                    self.traj_time,
                    predicted_acceleration,
                    acceleration_peak,
                )
                unsafe_candidates.append(candidate)
                if acceleration_peak <= self.joystick_speed_accel_max + 1e-6:
                    candidates.append(candidate)
        for horizon in np.linspace(
            self.joystick_min_traj_time,
            self.traj_time,
            self.joystick_speed_horizon_samples,
        ):
            predicted_acceleration = self._projected_plan_acceleration(
                best_state_w,
                start_vel,
                start_acc,
                direction_w,
                boundary_scale,
                float(horizon),
            )
            acceleration_peak = self._horizontal_plan_acceleration_peak(
                best_state_w,
                start_vel,
                start_acc,
                boundary_scale,
                float(horizon),
            )
            if np.isfinite(predicted_acceleration) and np.isfinite(acceleration_peak):
                # Prefer the longer (less aggressive) horizon when errors tie.
                candidate = (
                    abs(predicted_acceleration - acceleration_target),
                    -float(horizon),
                    boundary_scale,
                    float(horizon),
                    predicted_acceleration,
                    acceleration_peak,
                )
                unsafe_candidates.append(candidate)
                if acceleration_peak <= self.joystick_speed_accel_max + 1e-6:
                    candidates.append(candidate)
        if not candidates:
            if not unsafe_candidates:
                return (fallback_scale, self.traj_time, acceleration_target, float("nan"), command_speed, current_projection)
            # A non-finite/upstream state can make the current acceleration
            # impossible to recover within one replanning interval.  Choose
            # the least violating bounded geometry; the next 15 Hz cycle will
            # solve again from the newer state.
            candidates = sorted(unsafe_candidates, key=lambda item: (item[5], item[0], item[1]))[:1]
        _error, _negative_horizon, endpoint_scale, horizon, predicted_acceleration, _acceleration_peak = min(candidates)
        return (
            float(np.clip(endpoint_scale, 0.0, 1.0)),
            float(np.clip(horizon, self.joystick_min_traj_time, self.traj_time)),
            acceleration_target,
            predicted_acceleration,
            command_speed,
            current_projection,
        )

    def _evaluate_joystick_altitude_rewrite(
        self,
        flattened_polynomials,
        original_polynomials,
        start_pos,
        rot_wb,
        horizon,
        depth_images_m,
        valid_masks,
    ):
        """Apply the lock-height veto while retaining YOPO's far-range score.

        A collision is actionable even near the range boundary and is always
        rejected.  Invalid pixels and incomplete camera footprints inside the
        observable range are fail-closed.  If the *only* uncertainty is that a
        rewritten sphere reaches beyond the 4 m x-depth range, the currently
        observable trajectory pieces are checked independently and YOPO keeps
        responsibility for its already-scored far endpoint.
        """
        sample_count = max(17, int(np.ceil(float(horizon) / (2.0 * self.ctrl_dt))) + 1)
        sample_times = np.linspace(0.0, float(horizon), sample_count)
        evaluation_kwargs = {
            "body_pose": BodyPose(start_pos, rot_wb),
            "depth_images_m": depth_images_m,
            "valid_masks": valid_masks,
            "camera_intrinsics": self.joystick_camera_intrinsics,
            "max_depth_m": self.max_depth,
            "vehicle_radius_m": self.joystick_vehicle_radius,
            "safety_margin_m": self.joystick_safety_margin,
            "rewrite_threshold_m": self.joystick_rewrite_threshold,
            "require_full_footprint": self.joystick_strict_footprint,
        }
        result = evaluate_altitude_lock_safety(
            flattened_polynomials,
            original_trajectory=original_polynomials,
            sample_times_s=sample_times,
            **evaluation_kwargs,
        )
        if result.safe or result.first_collision is not None:
            return result.safe, result, result.reason

        first_unknown_reason = None if result.first_unknown is None else result.first_unknown.reason
        if first_unknown_reason != "sphere_extends_beyond_x_depth_range":
            return False, result, result.reason

        flat_points = np.stack(
            [[poly.get_position(float(timestamp)) for poly in flattened_polynomials] for timestamp in sample_times]
        ).astype(np.float64)
        original_points = np.stack(
            [[poly.get_position(float(timestamp)) for poly in original_polynomials] for timestamp in sample_times]
        ).astype(np.float64)
        rotation_bw = np.asarray(rot_wb, dtype=np.float64).T
        points_b = (rotation_bw @ (flat_points - np.asarray(start_pos, dtype=np.float64)).T).T
        clearance = self.joystick_vehicle_radius + self.joystick_safety_margin
        certainly_inside_range = np.linalg.norm(points_b[:, :2], axis=1) + clearance <= self.max_depth + 1e-9

        # Re-evaluate every contiguous in-range island.  This covers a curved
        # trajectory which leaves and later re-enters the sensor range without
        # joining the two islands by an artificial straight segment.
        run_start = None
        partial_footprint_used = False
        for index in range(sample_count + 1):
            in_range = index < sample_count and bool(certainly_inside_range[index])
            if in_range and run_start is None:
                run_start = index
            if not in_range and run_start is not None:
                run_stop = index
                local_result = evaluate_altitude_lock_safety(
                    flat_points[run_start:run_stop],
                    original_trajectory=original_points[run_start:run_stop],
                    **evaluation_kwargs,
                )
                if not local_result.safe:
                    return False, local_result, local_result.reason
                if "partial_footprint" in local_result.reason:
                    partial_footprint_used = True
                run_start = None

        if partial_footprint_used:
            reason = "material_rewrite_partial_footprint_and_beyond_depth_range_deferred"
        else:
            reason = "material_rewrite_beyond_depth_range_deferred_to_yopo"
        return True, result, reason

    def _prepare_joystick_candidate(
        self,
        action_id,
        score_value,
        candidate_state_b,
        start_pos,
        start_vel,
        start_acc,
        rot_wb,
        operator_vdes_w,
        command_alpha,
        command_altitude,
        previous_servo_horizon,
        depth_images_m,
        valid_masks,
    ):
        """Build and safety-check the trajectory that would actually execute."""
        candidate_state_b = np.asarray(candidate_state_b, dtype=np.float64)
        unscaled_state_w = np.asarray(rot_wb, dtype=np.float64) @ candidate_state_b
        speed_servo = self._select_joystick_speed_plan(
            unscaled_state_w,
            start_vel,
            start_acc,
            operator_vdes_w,
            command_alpha,
        )
        endpoint_scale, candidate_horizon = speed_servo[:2]
        endpoint_scale = float(np.clip(endpoint_scale, 0.0, 1.0))
        candidate_horizon = float(
            np.clip(candidate_horizon, self.joystick_min_traj_time, self.traj_time)
        )

        # Apply the same horizon smoothing to every candidate independently;
        # selection must compare the actual plans which this callback could
        # install, not raw network endpoints with different execution rules.
        previous_horizon = float(
            np.clip(previous_servo_horizon, self.joystick_min_traj_time, self.traj_time)
        )
        if candidate_horizon < previous_horizon:
            filtered_horizon = previous_horizon + self.joystick_horizon_filter_alpha * (
                candidate_horizon - previous_horizon
            )
            candidate_horizon = max(
                candidate_horizon,
                filtered_horizon,
                previous_horizon - self.joystick_horizon_step_max,
            )
            candidate_horizon = float(
                np.clip(candidate_horizon, self.joystick_min_traj_time, self.traj_time)
            )
            command_speed = float(np.linalg.norm(np.asarray(operator_vdes_w)[:2]))
            if command_speed > 1e-6:
                command_direction_w = np.array(
                    [operator_vdes_w[0] / command_speed, operator_vdes_w[1] / command_speed, 0.0],
                    dtype=np.float64,
                )
                predicted_acceleration = self._projected_plan_acceleration(
                    unscaled_state_w,
                    start_vel,
                    start_acc,
                    command_direction_w,
                    endpoint_scale,
                    candidate_horizon,
                )
                speed_servo = (
                    endpoint_scale,
                    candidate_horizon,
                    speed_servo[2],
                    predicted_acceleration,
                    speed_servo[4],
                    speed_servo[5],
                )

        acceleration_peak = self._horizontal_plan_acceleration_peak(
            unscaled_state_w,
            start_vel,
            start_acc,
            endpoint_scale,
            candidate_horizon,
        )
        if (
            not np.isfinite(acceleration_peak)
            or acceleration_peak > self.joystick_speed_accel_max + 1e-6
        ):
            return {
                "action_id": int(action_id),
                "score": float(score_value),
                "accepted": False,
                "rejection": "acceleration_limit",
                "acceleration_peak": float(acceleration_peak),
            }

        scaled_state_w = endpoint_scale * unscaled_state_w
        candidate_poly_x = Poly5Solver(
            start_pos[0], start_vel[0], start_acc[0],
            scaled_state_w[0, 0] + start_pos[0], scaled_state_w[0, 1], scaled_state_w[0, 2],
            candidate_horizon,
        )
        candidate_poly_y = Poly5Solver(
            start_pos[1], start_vel[1], start_acc[1],
            scaled_state_w[1, 0] + start_pos[1], scaled_state_w[1, 1], scaled_state_w[1, 2],
            candidate_horizon,
        )
        flattened_poly_z = Poly5Solver(
            start_pos[2], start_vel[2], start_acc[2],
            command_altitude, 0.0, 0.0,
            candidate_horizon,
        )

        safety_result = None
        safety_reason = "altitude_safety_disabled"
        if self.joystick_altitude_safety:
            original_poly_z = Poly5Solver(
                start_pos[2], start_vel[2], start_acc[2],
                scaled_state_w[2, 0] + start_pos[2], scaled_state_w[2, 1], scaled_state_w[2, 2],
                candidate_horizon,
            )
            try:
                accepted, safety_result, safety_reason = self._evaluate_joystick_altitude_rewrite(
                    (candidate_poly_x, candidate_poly_y, flattened_poly_z),
                    (candidate_poly_x, candidate_poly_y, original_poly_z),
                    start_pos,
                    rot_wb,
                    candidate_horizon,
                    depth_images_m,
                    valid_masks,
                )
            except (TypeError, ValueError, FloatingPointError) as exc:
                return {
                    "action_id": int(action_id),
                    "score": float(score_value),
                    "accepted": False,
                    "rejection": "safety_evaluation_error",
                    "safety_error": str(exc),
                }
            if not accepted:
                return {
                    "action_id": int(action_id),
                    "score": float(score_value),
                    "accepted": False,
                    "rejection": safety_reason,
                    "safety_result": safety_result,
                }

        return {
            "action_id": int(action_id),
            "score": float(score_value),
            "accepted": True,
            "polynomials": (candidate_poly_x, candidate_poly_y, flattened_poly_z),
            "candidate_horizon": candidate_horizon,
            "endpoint_scale": endpoint_scale,
            "speed_servo": speed_servo,
            "acceleration_peak": float(acceleration_peak),
            "safety_result": safety_result,
            "safety_reason": safety_reason,
        }

    def poll_joystick_intent(self, rot_wb=None):
        """Serially sample, transform, and commit the current stick intent."""
        with self.joystick_poll_lock:
            return self._poll_joystick_intent_serialized(rot_wb)

    def _poll_joystick_intent_serialized(self, rot_wb=None):
        """Poll the stick independently of depth and update operator/policy intents.

        The operator command is a yaw-frame horizontal velocity in [0, velocity].
        The policy receives only its direction at the checkpoint's trained speed;
        joystick magnitude is applied later to the selected endpoint.
        """
        zero = np.zeros(3, dtype=np.float32)
        if self.control_mode != "joystick" or self.joystick is None or not self.odom_init:
            return zero, zero, zero, 0.0, False, 0

        if rot_wb is None:
            q = self.odom.pose.pose.orientation
            rot_wb = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
        else:
            rot_wb = np.asarray(rot_wb, dtype=np.float32)
        rot_bw = rot_wb.T

        velocity_fraction_h, stick_magnitude, active = self.joystick.read_body_velocity_fraction()
        velocity_fraction_h = np.asarray(velocity_fraction_h, dtype=np.float32)
        if active:
            # Heading-frame x/y is intuitive planar teleoperation.  Using the
            # full roll/pitch attitude here would inject an unintended world-z
            # command whenever the vehicle tilts.
            yaw = float(np.arctan2(rot_wb[1, 0], rot_wb[0, 0]))
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rot_wh = np.array(
                [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            operator_vdes_h = (self.velocity * velocity_fraction_h).astype(np.float32)
            operator_vdes_w = (rot_wh @ operator_vdes_h).astype(np.float32)
            operator_vdes_b = (rot_bw @ operator_vdes_w).astype(np.float32)
            operator_speed = float(self.velocity * stick_magnitude)
            body_norm = float(np.linalg.norm(operator_vdes_b))
            if body_norm <= 1e-6:
                active = False
                operator_vdes_w = zero.copy()
                operator_vdes_b = zero.copy()
                policy_vdes_b = zero.copy()
                alpha = 0.0
            else:
                policy_vdes_b = (self.policy_intent_speed * operator_vdes_b / body_norm).astype(np.float32)
                alpha = float(np.clip(operator_speed / self.policy_velocity_scale, 0.0, 1.0))
        else:
            velocity_fraction_h = zero.copy()
            operator_vdes_w = zero.copy()
            operator_vdes_b = zero.copy()
            policy_vdes_b = zero.copy()
            alpha = 0.0

        entered_hold = False
        became_active = False
        altitude_was_locked = True
        altitude_target = None
        with self.lock:
            was_active = self.joystick_input_active
            committed_fraction_h = velocity_fraction_h if active else zero
            status_changed = bool(active) != was_active
            fraction_changed = bool(active) and was_active and (
                np.linalg.norm(committed_fraction_h - self.joystick_fraction_signature)
                >= self.joystick_fraction_epoch_epsilon
            )
            if status_changed or fraction_changed:
                # The signature is an epoch anchor rather than merely the last
                # poll.  Sub-threshold changes therefore accumulate, while
                # repeated reads of the same kernel joystick state do not churn
                # the epoch or starve inference.
                self.joystick_intent_epoch += 1
                self.joystick_fraction_signature = committed_fraction_h.copy()
                # A center/disconnect/re-arm/direction change invalidates both
                # the installed plan and the freshness proof for its old epoch.
                self.joystick_last_plan_commit_monotonic = None
            self.joystick_fraction_h = committed_fraction_h.copy()
            self.joystick_input_active = bool(active)
            self.joystick_hold = not active
            self.joystick_alpha = alpha
            self.current_vdes_w = operator_vdes_w
            self.current_vdes_b = operator_vdes_b
            self.policy_vdes_b = policy_vdes_b
            self.control_source = "joystick" if active else "joystick_hold"
            if active and not was_active:
                altitude_was_locked = self.joystick_altitude_locked
                altitude_target = self._joystick_altitude_locked_value(freeze=True)
                if self.joystick_hold_position is not None:
                    self.joystick_hold_position[2] = altitude_target
                # Keep publishing the previous hold until a depth callback has
                # generated a trajectory for this new command.
                self.joystick_plan_ready = False
                self.ctrl_time = None
                became_active = True
            elif fraction_changed:
                # A materially different active command must not keep executing
                # the already-installed trajectory.  Capture a reachable stop
                # from current odometry and hold there until a depth frame has
                # produced a trajectory for this exact intent epoch.
                self._capture_joystick_hold_locked()
            elif not active and (was_active or self.joystick_hold_position is None):
                self._capture_joystick_hold_locked()
                entered_hold = True
            intent_epoch = self.joystick_intent_epoch

        self._publish_vdes_body(operator_vdes_b)
        if entered_hold:
            connected, initialized, unlocked = self.joystick.get_status()
            rospy.loginfo(
                "Joystick hold captured: connected=%s initialized=%s unlocked=%s.",
                connected,
                initialized,
                unlocked,
            )
        elif became_active:
            if not altitude_was_locked:
                rospy.loginfo("Joystick world altitude locked at %.3f m.", altitude_target)
            rospy.loginfo("Joystick active; waiting for a fresh collision-avoidance trajectory.")
        return policy_vdes_b, operator_vdes_w, operator_vdes_b, alpha, bool(active), intent_epoch

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

        command_alpha = 1.0
        command_epoch = None
        operator_vdes_w = np.zeros(3, dtype=np.float32)
        if self.control_mode == "joystick" and self.joystick is not None:
            policy_vdes_b, operator_vdes_w, _operator_vdes_b, command_alpha, _active, command_epoch = (
                self.poll_joystick_intent(rot_wb)
            )
            vdes_b = policy_vdes_b
        else:
            goal_dir_w = self.goal - pos_w
            goal_norm = np.linalg.norm(goal_dir_w)
            if goal_norm < 1e-4:
                vdes_w = np.zeros(3, dtype=np.float32)
            else:
                vdes_w = self.velocity * goal_dir_w / goal_norm
            vdes_b = rot_bw @ vdes_w
            operator_vdes_w = vdes_w.astype(np.float32)
            self.joystick_hold = False
            self.control_source = "nav_goal"
            self.current_vdes_w = vdes_w.astype(np.float32)
            self.current_vdes_b = vdes_b.astype(np.float32)
            self._publish_vdes_body(vdes_b)

        vel_b = rot_bw @ vel_w
        acc_b = rot_bw @ acc_w
        state_b = np.concatenate([vel_b, acc_b, vdes_b]).astype(np.float32)
        return (
            state_b,
            pos_w.astype(np.float32),
            vel_w.astype(np.float32),
            acc_w.astype(np.float32),
            rot_wb,
            operator_vdes_w.astype(np.float32),
            command_alpha,
            command_epoch,
        )

    @torch.inference_mode()
    def callback_depths(self, front, left, right, back):
        if not self.odom_init:
            return

        time0 = time.time()
        if self.control_mode == "joystick":
            (
                state_b,
                start_pos,
                start_vel,
                start_acc,
                rot_wb,
                operator_vdes_w,
                command_alpha,
                command_epoch,
            ) = self.process_state()
            if self.joystick_hold:
                return
            with self.lock:
                command_altitude = self.joystick_altitude_target
                previous_servo_horizon = self.joystick_servo_horizon
            if command_altitude is None:
                return

        depth_messages = [front, left, right, back]
        safety_depth_m = None
        safety_valid_masks = None
        if self.control_mode == "joystick" and self.joystick_altitude_safety:
            try:
                safety_frames = [
                    self._decode_metric_depth_for_joystick_safety(message)
                    for message in depth_messages
                ]
                safety_depth_m = np.stack([frame[0] for frame in safety_frames], axis=0)
                safety_valid_masks = np.stack([frame[1] for frame in safety_frames], axis=0)
            except (TypeError, ValueError) as exc:
                with self.lock:
                    if _joystick_plan_is_current(
                        self.joystick_input_active,
                        command_epoch,
                        self.joystick_intent_epoch,
                    ) and (self.joystick_plan_ready or self.joystick_hold_position is None):
                        self._capture_joystick_hold_locked()
                rospy.logerr_throttle(
                    1.0,
                    "Joystick altitude safety rejected malformed or uncalibrated depth: %s; "
                    "publishing EMPTY hold.",
                    exc,
                )
                return

        depth = np.stack([self.preprocess_depth(msg) for msg in depth_messages], axis=0)
        depth = depth.reshape(1, 4, 1, self.height, self.width)
        time1 = time.time()

        if self.control_mode != "joystick":
            (
                state_b,
                start_pos,
                start_vel,
                start_acc,
                rot_wb,
                operator_vdes_w,
                command_alpha,
                command_epoch,
            ) = self.process_state()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        state_input = torch.from_numpy(state_b[None, :]).to(self.device, non_blocking=True)
        time2 = time.time()

        endstate_b, score = self.policy(depth_input, state_input)
        endstate_b = endstate_b[0].detach().cpu().numpy()
        score = score[0].detach().cpu().numpy()
        time3 = time.time()

        endstate_cand_b = endstate_b.reshape(-1, 9)
        score_flat = score.reshape(-1)
        candidate_horizon = self.traj_time
        speed_servo = None
        selected_candidate = None
        if self.control_mode == "joystick":
            expected_candidates = int(cfg["omni_topology_num"])
            if (
                endstate_cand_b.shape[0] != expected_candidates
                or score_flat.size != expected_candidates
            ):
                with self.lock:
                    if _joystick_plan_is_current(
                        self.joystick_input_active,
                        command_epoch,
                        self.joystick_intent_epoch,
                    ) and (self.joystick_plan_ready or self.joystick_hold_position is None):
                        self._capture_joystick_hold_locked()
                rospy.logerr_throttle(
                    1.0,
                    "Joystick candidate tensor malformed: endpoints=%d scores=%d expected=%d; "
                    "publishing EMPTY hold.",
                    endstate_cand_b.shape[0],
                    score_flat.size,
                    expected_candidates,
                )
                return

            candidate_results = []
            for action_id in _joystick_candidate_order(score_flat):
                candidate_state_b = endstate_cand_b[action_id].reshape(3, 3).T.copy()
                candidate_results.append(
                    self._prepare_joystick_candidate(
                        action_id,
                        score_flat[action_id],
                        candidate_state_b,
                        start_pos,
                        start_vel,
                        start_acc,
                        rot_wb,
                        operator_vdes_w,
                        command_alpha,
                        command_altitude,
                        previous_servo_horizon,
                        safety_depth_m,
                        safety_valid_masks,
                    )
                )
            finite_ids = {result["action_id"] for result in candidate_results}
            for action_id in range(expected_candidates):
                if action_id not in finite_ids:
                    candidate_results.append(
                        {
                            "action_id": action_id,
                            "score": float(score_flat[action_id]),
                            "accepted": False,
                            "rejection": "nonfinite_score",
                        }
                    )

            accepted_candidates = [result for result in candidate_results if result["accepted"]]
            rejection_counts = {}
            for result in candidate_results:
                if not result["accepted"]:
                    reason = result["rejection"]
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            rejection_summary = (
                "none"
                if not rejection_counts
                else ", ".join(f"{key}={value}" for key, value in sorted(rejection_counts.items()))
            )
            if not accepted_candidates:
                with self.lock:
                    # Capture the braking target only on the transition out of
                    # an executable plan.  Repeated rejected depth frames must
                    # not walk an already-held target forward with the vehicle.
                    if _joystick_plan_is_current(
                        self.joystick_input_active,
                        command_epoch,
                        self.joystick_intent_epoch,
                    ) and (self.joystick_plan_ready or self.joystick_hold_position is None):
                        self._capture_joystick_hold_locked()
                rospy.logwarn_throttle(
                    1.0,
                    "Joystick altitude/speed safety rejected all %d candidates (%s); "
                    "publishing EMPTY hold.",
                    expected_candidates,
                    rejection_summary,
                )
                first_safety_veto = next(
                    (result for result in candidate_results if result.get("safety_result") is not None),
                    None,
                )
                if first_safety_veto is not None:
                    veto_result = first_safety_veto["safety_result"]
                    if veto_result.first_collision is not None:
                        collision = veto_result.first_collision
                        rospy.logwarn_throttle(
                            1.0,
                            "First altitude veto: candidate=%d collision camera=%s pixel=(%d,%d) "
                            "depth=%.3fm clearance=%.3fm.",
                            first_safety_veto["action_id"],
                            self.view_names[collision.camera_index],
                            collision.pixel_row,
                            collision.pixel_col,
                            collision.measured_x_depth_m,
                            collision.required_clearance_m,
                        )
                    elif veto_result.first_unknown is not None:
                        unknown = veto_result.first_unknown
                        rospy.logwarn_throttle(
                            1.0,
                            "First altitude veto: candidate=%d unknown=%s camera=%s pixel=(%s,%s).",
                            first_safety_veto["action_id"],
                            unknown.reason,
                            "none" if unknown.camera_index is None else self.view_names[unknown.camera_index],
                            "none" if unknown.pixel_row is None else unknown.pixel_row,
                            "none" if unknown.pixel_col is None else unknown.pixel_col,
                        )
                return

            # Every finite candidate above has been evaluated.  Only now apply
            # the checkpoint's original score ordering among safe executable
            # trajectories; a lower-score vetoed path can never leak through.
            selected_candidate = min(
                accepted_candidates,
                key=lambda result: (result["score"], result["action_id"]),
            )
            action_id = selected_candidate["action_id"]
            candidate_poly_x, candidate_poly_y, candidate_poly_z = selected_candidate["polynomials"]
            candidate_horizon = selected_candidate["candidate_horizon"]
            speed_servo = selected_candidate["speed_servo"]
        else:
            action_id = int(np.argmin(score_flat))
            best_state = endstate_cand_b[action_id].reshape(3, 3).T.copy()
            best_state_w = rot_wb @ best_state
            candidate_poly_x = Poly5Solver(
                start_pos[0], start_vel[0], start_acc[0],
                best_state_w[0, 0] + start_pos[0], best_state_w[0, 1], best_state_w[0, 2], self.traj_time
            )
            candidate_poly_y = Poly5Solver(
                start_pos[1], start_vel[1], start_acc[1],
                best_state_w[1, 0] + start_pos[1], best_state_w[1, 1], best_state_w[1, 2], self.traj_time
            )
            candidate_poly_z = Poly5Solver(
                start_pos[2], start_vel[2], start_acc[2],
                best_state_w[2, 0] + start_pos[2], best_state_w[2, 1], best_state_w[2, 2], self.traj_time
            )
        with self.lock:
            plan_committed = False
            if self.control_mode == "joystick":
                # A center/disconnect/re-arm event may have arrived during
                # inference.  Never revive the result from the earlier intent.
                plan_is_current = _joystick_plan_is_current(
                    self.joystick_input_active,
                    command_epoch,
                    self.joystick_intent_epoch,
                )
                if plan_is_current:
                    self.optimal_poly_x = candidate_poly_x
                    self.optimal_poly_y = candidate_poly_y
                    self.optimal_poly_z = candidate_poly_z
                    self.ctrl_time = 0.0
                    self.plan_horizon = candidate_horizon
                    self.joystick_servo_horizon = candidate_horizon
                    self.joystick_plan_ready = True
                    # Timestamp only a successfully installed plan for the exact
                    # intent sampled before inference.  Raw depth arrival alone
                    # is not sufficient to keep active flight enabled.
                    self.joystick_last_plan_commit_monotonic = time.monotonic()
                    plan_committed = True
            else:
                self.optimal_poly_x = candidate_poly_x
                self.optimal_poly_y = candidate_poly_y
                self.optimal_poly_z = candidate_poly_z
                self.ctrl_time = 0.0
                self.plan_horizon = self.traj_time
        time4 = time.time()

        if plan_committed and speed_servo is not None:
            endpoint_scale, candidate_horizon, accel_target, accel_predicted, speed_target, speed_actual = speed_servo
            rospy.loginfo_throttle(
                1.0,
                "Joystick speed servo: target=%.2f actual_proj=%.2f accel_target=%.2f "
                "accel_plan=%.2f endpoint_scale=%.3f horizon=%.3fs",
                speed_target,
                speed_actual,
                accel_target,
                accel_predicted,
                endpoint_scale,
                candidate_horizon,
            )
            safety_result = selected_candidate["safety_result"]
            rewrite_deviation = (
                0.0 if safety_result is None else safety_result.max_rewrite_deviation_m
            )
            rejected_count = sum(not result["accepted"] for result in candidate_results)
            rospy.loginfo_throttle(
                1.0,
                "Joystick candidate selection: checked=%d selected=%d score=%.4f rejected=%d[%s] "
                "altitude_safety=%s max_rewrite=%.3fm.",
                len(candidate_results),
                selected_candidate["action_id"],
                selected_candidate["score"],
                rejected_count,
                rejection_summary,
                selected_candidate["safety_reason"],
                rewrite_deviation,
            )
            if "partial_footprint" in selected_candidate["safety_reason"]:
                rospy.logwarn_throttle(
                    2.0,
                    "Joystick partial-footprint mode selected candidate=%d: sphere center remained in "
                    "a camera FOV and every visible footprint pixel was valid/collision-free, but the "
                    "FOV-clipped sphere portion remains uncertified (residual risk deferred).",
                    selected_candidate["action_id"],
                )
            if "beyond_depth_range" in selected_candidate["safety_reason"]:
                rospy.logwarn_throttle(
                    2.0,
                    "Joystick altitude rewrite extends beyond %.2fm ToF range; all observable "
                    "segments passed the rewrite veto and the YOPO score governs the far endpoint.",
                    self.max_depth,
                )
        self.visualize_trajectory(
            start_pos,
            start_vel,
            start_acc,
            rot_wb,
            endstate_cand_b,
            score_flat,
            candidate_horizon,
        )
        time5 = time.time()
        self.print_time(time0, time1, time2, time3, time4, time5)

    def publish_joystick_hold(self):
        """Continuously publish a zero-velocity closed-loop hold command."""
        if not self.odom_init:
            return False
        with self.lock:
            if self.joystick_input_active and self.joystick_plan_ready:
                return False
            if self.joystick_hold_position is None:
                self._capture_joystick_hold_locked()
            hold_pos = self.joystick_hold_position.copy()
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            # NetworkControl treats non-READY messages as its position/velocity
            # closed-loop branch.  This must be republished because its own
            # timer stops updating after external position commands begin.
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
            control_msg.position.x = float(hold_pos[0])
            control_msg.position.y = float(hold_pos[1])
            control_msg.position.z = float(hold_pos[2])
            control_msg.velocity.x = 0.0
            control_msg.velocity.y = 0.0
            control_msg.velocity.z = 0.0
            control_msg.acceleration.x = 0.0
            control_msg.acceleration.y = 0.0
            control_msg.acceleration.z = 0.0
            control_msg.yaw = self.last_yaw if self.fixed_yaw_value is None else float(self.fixed_yaw_value)
            control_msg.yaw_dot = 0.0
            self.desire_pos = hold_pos
            self.desire_vel = np.zeros(3, dtype=np.float32)
            self.desire_acc = np.zeros(3, dtype=np.float32)
            self.desire_init = True
            self.last_control_msg = control_msg
        self.ctrl_pub.publish(control_msg)
        return True

    def control_pub(self, _timer):
        if self.control_mode == "joystick":
            # This 50 Hz poll makes center/disconnect fail-safe independent of
            # the four-depth ApproximateTimeSynchronizer.
            self.poll_joystick_intent()
            trajectory_expired = False
            depth_plan_stale = False
            depth_plan_age = None
            with self.lock:
                joystick_active = self.joystick_input_active
                plan_ready = self.joystick_plan_ready
                ctrl_time = self.ctrl_time
                plan_horizon = self.plan_horizon
                now_monotonic = time.monotonic()
                if joystick_active and plan_ready and _joystick_plan_is_stale(
                    self.joystick_last_plan_commit_monotonic,
                    now_monotonic,
                    self.joystick_depth_timeout,
                ):
                    if self.joystick_last_plan_commit_monotonic is not None:
                        depth_plan_age = max(
                            0.0,
                            now_monotonic - self.joystick_last_plan_commit_monotonic,
                        )
                    # Invalidate any inference that started before this timeout;
                    # it must not revive flight after the fail-closed hold.
                    self.joystick_intent_epoch += 1
                    self._capture_joystick_hold_locked()
                    plan_ready = False
                    depth_plan_stale = True
                elif (
                    joystick_active
                    and plan_ready
                    and (ctrl_time is None or ctrl_time >= plan_horizon)
                ):
                    self._capture_joystick_hold_locked()
                    trajectory_expired = True
            if depth_plan_stale:
                age_text = "missing" if depth_plan_age is None else f"{depth_plan_age:.3f}s"
                rospy.logwarn_throttle(
                    1.0,
                    "Joystick depth/plan watchdog expired: current-intent plan age=%s, "
                    "timeout=%.3fs; publishing EMPTY hold until a fresh depth plan commits.",
                    age_text,
                    self.joystick_depth_timeout,
                )
                self.publish_joystick_hold()
                return
            if not joystick_active or not plan_ready:
                self.publish_joystick_hold()
                return
            if trajectory_expired:
                rospy.logwarn_throttle(1.0, "Joystick trajectory expired; holding until a fresh depth plan arrives.")
                self.publish_joystick_hold()
                return
        elif self.ctrl_time is None or self.ctrl_time >= self.traj_time:
            return

        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return

        control_msg = None
        with self.lock:
            # Recheck after taking the lock: a depth callback or joystick event
            # may have invalidated the plan between the snapshot above and here.
            joystick_plan_invalid = self.control_mode == "joystick" and (
                not self.joystick_input_active or not self.joystick_plan_ready or self.ctrl_time is None
            )
            if not joystick_plan_invalid:
                # Never evaluate a fifth-order polynomial beyond its fitted horizon.
                plan_horizon = self.plan_horizon if self.control_mode == "joystick" else self.traj_time
                self.ctrl_time = min(self.ctrl_time + self.ctrl_dt, plan_horizon)
                control_msg = PositionCommand()
                control_msg.header.stamp = rospy.Time.now()
                control_msg.trajectory_flag = (
                    control_msg.TRAJECTORY_STATUS_EMPTY
                    if self.control_mode == "joystick"
                    else control_msg.TRAJECTORY_STATUS_READY
                )
                control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
                control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
                control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
                control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
                control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
                control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
                control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
                control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
                control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
                self.desire_pos = np.array(
                    [control_msg.position.x, control_msg.position.y, control_msg.position.z], dtype=np.float32
                )
                self.desire_vel = np.array(
                    [control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z], dtype=np.float32
                )
                self.desire_acc = np.array(
                    [control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z], dtype=np.float32
                )
                if self.fixed_yaw:
                    yaw = self.last_yaw if self.fixed_yaw_value is None else float(self.fixed_yaw_value)
                    yaw_dot = 0.0
                else:
                    if self.control_mode == "joystick":
                        goal_dir = self.current_vdes_w
                    else:
                        goal_dir = self.goal - self.desire_pos
                    yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
                    self.last_yaw = yaw
                control_msg.yaw = yaw
                control_msg.yaw_dot = yaw_dot
                self.desire_init = True
                self.last_control_msg = control_msg
        if control_msg is None:
            self.publish_joystick_hold()
            return
        self.ctrl_pub.publish(control_msg)

    def visualize_trajectory(self, start_pos, start_vel, start_acc, rot_wb, endstate_b, score, best_horizon):
        best_dt = best_horizon / 20.0
        best_t_values = np.arange(0, best_horizon, best_dt)

        if self.best_traj_pub.get_num_connections() > 0 and self.optimal_poly_x is not None:
            points_array = np.stack(
                [
                    self.optimal_poly_x.get_position(best_t_values),
                    self.optimal_poly_y.get_position(best_t_values),
                    self.optimal_poly_z.get_position(best_t_values),
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
            model_dt = self.traj_time / 20.0
            model_t_values = np.arange(0, self.traj_time, model_dt)
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], start_acc[0],
                                      states_w[:, 0, 0] + start_pos[0], states_w[:, 0, 1], states_w[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], start_acc[1],
                                      states_w[:, 1, 0] + start_pos[1], states_w[:, 1, 1], states_w[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], start_acc[2],
                                      states_w[:, 2, 0] + start_pos[2], states_w[:, 2, 1], states_w[:, 2, 2], self.traj_time)
            points_array = np.stack(
                [
                    all_poly_x.get_position(model_t_values),
                    all_poly_y.get_position(model_t_values),
                    all_poly_z.get_position(model_t_values),
                ],
                axis=-1,
            )
            scores = np.repeat(score, model_t_values.size)
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
                f"YOPO-Omni avg ms | depth {1000*self.time_depth/self.count:.2f}, "
                f"prepare {1000*self.time_prepare/self.count:.2f}, "
                f"forward {1000*self.time_forward/self.count:.2f}, "
                f"process {1000*self.time_process/self.count:.2f}, "
                f"visualize {1000*self.time_visualize/self.count:.2f}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", default="", help="Path to YOPO-Omni checkpoint.")
    parser.add_argument("--trial", type=int, default=0, help="Trial number under YOPO/saved/YOPO_{trial}.")
    parser.add_argument("--epoch", type=int, default=200, help="Checkpoint epoch.")
    parser.add_argument("--velocity", type=float, default=float(cfg["velocity"]), help="Desired speed magnitude.")
    parser.add_argument("--max-depth", type=float, default=4.0, help="ToF depth max range used for normalization.")
    parser.add_argument(
        "--depth-normalized",
        action="store_true",
        help="Treat 32FC1 depth input as already normalized [0,1]; default is 32FC1 meters / 16UC1 millimeters.",
    )
    parser.add_argument("--arrive-dist", type=float, default=1.0, help="Distance threshold in meters for stopping at the goal.")
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
    parser.add_argument(
        "--control-mode",
        choices=("nav_goal", "joystick"),
        default="nav_goal",
        help="nav_goal keeps the original RViz target-position flow; joystick drives desired body velocity directly.",
    )
    parser.add_argument("--joystick-device", default="/dev/input/js0", help="Linux joystick device path.")
    parser.add_argument("--joystick-axis-x", type=int, default=0, help="Right-stick horizontal axis index.")
    parser.add_argument("--joystick-axis-y", type=int, default=1, help="Right-stick vertical axis index.")
    parser.add_argument("--joystick-axis-max", type=float, default=32767.0, help="Absolute raw joystick axis maximum.")
    parser.add_argument("--joystick-deadzone", type=float, default=0.08,
                        help="Radial deadzone before the right stick is considered active.")
    parser.add_argument("--joystick-invert-x", type=int, choices=(0, 1), default=1,
                        help="Invert horizontal axis after normalization.")
    parser.add_argument("--joystick-invert-y", type=int, choices=(0, 1), default=0,
                        help="Invert vertical axis after normalization.")
    parser.add_argument("--joystick-swap-xy", type=int, choices=(0, 1), default=1,
                        help="Map vertical stick to body x and horizontal stick to body y.")
    parser.add_argument("--joystick-calibrate", type=int, choices=(0, 1), default=0,
                        help="Print raw joystick axes for calibration.")
    parser.add_argument(
        "--joystick-speed-kp",
        type=float,
        default=1.5,
        help="Desired horizontal acceleration per m/s of joystick speed error.",
    )
    parser.add_argument(
        "--joystick-speed-accel-max",
        type=float,
        default=4.0,
        help="Maximum joystick horizontal acceleration target/feed-forward in m/s^2.",
    )
    parser.add_argument(
        "--joystick-min-traj-time",
        type=float,
        default=1.0,
        help="Minimum per-plan joystick horizon; endpoint geometry is never expanded.",
    )
    parser.add_argument(
        "--joystick-depth-timeout",
        type=float,
        default=0.20,
        help="Fail closed if no current-intent depth plan commits within this many seconds.",
    )
    parser.add_argument(
        "--joystick-altitude-safety",
        type=int,
        choices=(0, 1),
        default=1,
        help="Check all eight lock-height trajectories against untouched metric depth (enabled by default).",
    )
    parser.add_argument(
        "--joystick-strict-footprint",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Require the whole vehicle-sphere projection inside one camera (1), or allow a clipped "
            "footprint only while its center stays in-FOV and all visible pixels pass (0, default)."
        ),
    )
    parser.add_argument(
        "--joystick-rewrite-threshold",
        type=float,
        default=0.20,
        help="Maximum lock-height trajectory deviation in meters still trusted to the original YOPO score.",
    )
    parser.add_argument(
        "--joystick-vehicle-radius",
        type=float,
        default=0.30,
        help="Vehicle sphere radius in meters for joystick altitude-rewrite collision checks.",
    )
    parser.add_argument(
        "--joystick-safety-margin",
        type=float,
        default=0.15,
        help="Extra obstacle clearance in meters added to --joystick-vehicle-radius.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    apply_runtime_overrides(args)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = args.weight or os.path.join(base_dir, "saved", f"YOPO_{args.trial}", f"epoch{args.epoch}.pth")

    depth_base = "/depth_image"
    settings = {
        "goal": [50.0, 0.0, 2.0],
        "velocity": args.velocity,
        "max_depth": args.max_depth,
        "depth_normalized": args.depth_normalized,
        "arrive_dist": args.arrive_dist,
        "ctrl_dt": 0.02,
        "odom_topic": "/sim/odom",
        "depth_topics": [
            f"{depth_base}_front",
            f"{depth_base}_left",
            f"{depth_base}_right",
            f"{depth_base}_back",
        ],
        "ctrl_topic": "/so3_control/pos_cmd",
        "plan_from_reference": False,
        "verbose": bool(args.verbose),
        "visualize": bool(args.visualize),
        "fixed_yaw": args.fixed_yaw,
        "fixed_yaw_value": args.fixed_yaw_value,
        "control_mode": args.control_mode,
        "joystick_speed_kp": args.joystick_speed_kp,
        "joystick_speed_accel_max": args.joystick_speed_accel_max,
        "joystick_min_traj_time": args.joystick_min_traj_time,
        "joystick_depth_timeout": args.joystick_depth_timeout,
        "joystick_altitude_safety": bool(args.joystick_altitude_safety),
        "joystick_strict_footprint": bool(args.joystick_strict_footprint),
        "joystick_rewrite_threshold": args.joystick_rewrite_threshold,
        "joystick_vehicle_radius": args.joystick_vehicle_radius,
        "joystick_safety_margin": args.joystick_safety_margin,
        "joystick": {
            "device": args.joystick_device,
            "axis_x": args.joystick_axis_x,
            "axis_y": args.joystick_axis_y,
            "axis_max": args.joystick_axis_max,
            "deadzone": args.joystick_deadzone,
            "invert_x": bool(args.joystick_invert_x),
            "invert_y": bool(args.joystick_invert_y),
            "swap_xy": bool(args.joystick_swap_xy),
            "calibrate": bool(args.joystick_calibrate),
        },
    }
    YopoOmniNet(settings, weight)
