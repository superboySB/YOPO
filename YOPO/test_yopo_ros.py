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
from joystick_control import map_dual_sticks
from policy.poly_solver import Poly5Solver, Polys5Solver, calculate_yaw
from policy.yopo_network import YOPOOmniNetwork


_JS_EVENT = struct.Struct("IhBB")
_JS_AXIS = 0x02


def _joystick_plan_is_current(input_active, sampled_epoch, current_epoch):
    """Gate inference results against center, direction, and watchdog invalidation."""
    return bool(input_active) and sampled_epoch == current_epoch


class JoystickIntent:
    def __init__(
        self,
        device,
        axis_x,
        axis_y,
        axis_z,
        axis_yaw,
        axis_max,
        deadzone,
        invert_x,
        invert_y,
        invert_z,
        invert_yaw,
        swap_xy,
        calibrate=False,
    ):
        self.device = device
        self.axis_x = int(axis_x)
        self.axis_y = int(axis_y)
        self.axis_z = int(axis_z)
        self.axis_yaw = int(axis_yaw)
        configured_axes = (self.axis_x, self.axis_y, self.axis_z, self.axis_yaw)
        if min(configured_axes) < 0 or len(set(configured_axes)) != len(configured_axes):
            raise ValueError("the four joystick axis indices must be distinct non-negative integers")
        self.axis_max = float(axis_max)
        self.deadzone = float(deadzone)
        self.invert_x = bool(invert_x)
        self.invert_y = bool(invert_y)
        self.invert_z = bool(invert_z)
        self.invert_yaw = bool(invert_yaw)
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
        map_dual_sticks(
            0,
            0,
            0,
            0,
            axis_max=self.axis_max,
            deadzone=self.deadzone,
            invert_right_horizontal=self.invert_x,
            invert_right_vertical=self.invert_y,
            swap_right_xy=self.swap_xy,
            invert_left_vertical=self.invert_z,
            invert_left_horizontal=self.invert_yaw,
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
                translation_fraction, yaw_fraction, magnitude, _active = self.read_control_fraction()
                connected, initialized, unlocked = self.get_status()
                rospy.loginfo_throttle(
                    0.2,
                    "joystick axes=%s heading_fraction=[%.3f, %.3f, %.3f] yaw_fraction=%.3f "
                    "translation_magnitude=%.3f "
                    "connected=%s initialized=%s unlocked=%s",
                    " ".join("%d:%d" % (axis, snap[axis]) for axis in sorted(snap)),
                    translation_fraction[0],
                    translation_fraction[1],
                    translation_fraction[2],
                    yaw_fraction,
                    magnitude,
                    connected,
                    initialized,
                    unlocked,
                )

    def read_control_fraction(self):
        with self.lock:
            if not self.connected:
                return np.zeros(3, dtype=np.float32), 0.0, 0.0, False
            initialized = all(
                axis in self.axes for axis in (self.axis_x, self.axis_y, self.axis_z, self.axis_yaw)
            )
            if not initialized:
                return np.zeros(3, dtype=np.float32), 0.0, 0.0, False
            raw_horizontal = self.axes[self.axis_x]
            raw_vertical = self.axes[self.axis_y]
            raw_vertical_speed = self.axes[self.axis_z]
            raw_yaw = self.axes[self.axis_yaw]
            # Keep mapping and arming under the same lock as the axis snapshot.
            # A disconnect/reconnect cannot therefore arm a new device using
            # values that belonged to the previous file descriptor.
            translation, yaw_fraction, _planar_magnitude, magnitude = map_dual_sticks(
                raw_horizontal,
                raw_vertical,
                raw_vertical_speed,
                raw_yaw,
                axis_max=self.axis_max,
                deadzone=self.deadzone,
                invert_right_horizontal=self.invert_x,
                invert_right_vertical=self.invert_y,
                swap_right_xy=self.swap_xy,
                invert_left_vertical=self.invert_z,
                invert_left_horizontal=self.invert_yaw,
            )

            just_unlocked = False
            if not self.unlocked:
                if magnitude <= 0.0 and abs(yaw_fraction) <= 0.0:
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
            rospy.loginfo("Joystick unlocked after all four configured axes initialized and both sticks were centered.")
        if not unlocked:
            return np.zeros(3, dtype=np.float32), 0.0, 0.0, False
        command = np.asarray(translation, dtype=np.float32)
        return command, float(yaw_fraction), float(magnitude), magnitude > 0.0

    def get_status(self):
        with self.lock:
            initialized = all(
                axis in self.axes for axis in (self.axis_x, self.axis_y, self.axis_z, self.axis_yaw)
            )
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
        self.joystick_vertical_velocity = float(settings.get("joystick_vertical_velocity", 2.0))
        self.joystick_yaw_rate_max = float(settings.get("joystick_yaw_rate", 1.0))
        if not np.isfinite(self.joystick_vertical_velocity) or self.joystick_vertical_velocity <= 0.0:
            raise ValueError("joystick vertical velocity must be finite and positive")
        if not np.isfinite(self.joystick_yaw_rate_max) or self.joystick_yaw_rate_max <= 0.0:
            raise ValueError("joystick yaw rate must be finite and positive")
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
        self.joystick = None
        self.joystick_hold = False
        self.joystick_input_active = False
        self.joystick_plan_ready = False
        self.joystick_intent_epoch = 0
        self.joystick_hover_command = None
        self.joystick_hover_polys = None
        self.joystick_hover_time = 0.0
        self.joystick_hover_duration = 0.0
        self.joystick_yaw_fraction = 0.0
        self.joystick_yaw_rate = 0.0
        self.joystick_yaw_target = None
        self.joystick_vertical_vdes = 0.0
        self.joystick_manual_position = None
        self.joystick_vertical_velocity_kp = 2.0
        self.joystick_vertical_accel_max = 4.0
        if self.control_mode == "joystick":
            self.joystick = JoystickIntent(**settings["joystick"])
            # Joystick mode is fail-closed: an unavailable device waits/reconnects
            # while the controller holds, instead of flying toward the RViz goal.
        self.fixed_yaw = bool(settings["fixed_yaw"])

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
                "Control mode: direct four-axis joystick. "
                f"right stick -> heading forward/left up to {self.velocity:.2f}m/s; "
                f"left vertical -> world climb up to {self.joystick_vertical_velocity:.2f}m/s; "
                f"left horizontal -> yaw rate up to {self.joystick_yaw_rate_max:.2f}rad/s."
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
                "Direct horizontal model path: body [velocity, acceleration, desired_velocity] -> "
                "checkpoint argmin -> unchanged trained-horizon trajectory. Left-stick vertical "
                "velocity and yaw rate use the flight controller directly."
            )
            print(
                "Centered translation switches READY to EMPTY and follows a continuous braking "
                "reference into hover, matching YOPO-Simple's mode transition without a navigation target."
            )
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
            if self.control_mode == "joystick" and self.joystick_yaw_target is None:
                self.joystick_yaw_target = float(ypr[0])
            if self.fixed_yaw and self.fixed_yaw_value is None:
                self.fixed_yaw_value = ypr[0]
                print(f"Fixed yaw locked to initial odometry yaw: {self.fixed_yaw_value:.3f} rad")
        self.odom_init = True

        pos = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z], dtype=np.float32)
        if self.control_mode == "joystick":
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

    def _capture_joystick_hover_locked(self):
        """Switch the latest model reference into a smooth EMPTY-mode stop.

        This keeps YOPO-Simple's important transition from READY model
        acceleration control to EMPTY position/velocity feedback, but avoids
        freezing a non-zero velocity forever.  A short quintic braking
        reference starts at the exact last p/v/a command and ends at zero
        velocity/acceleration; its endpoint then becomes the hover reference.
        This is a control-mode transition, not a navigation target or a safety
        candidate override. Caller holds ``self.lock``.
        """
        command = PositionCommand()
        command.header.stamp = rospy.Time.now()
        command.trajectory_flag = command.TRAJECTORY_STATUS_EMPTY
        source = self.last_control_msg
        if source is None:
            command.position.x = float(self.odom.pose.pose.position.x)
            command.position.y = float(self.odom.pose.pose.position.y)
            command.position.z = float(self.odom.pose.pose.position.z)
            command.velocity.x = 0.0
            command.velocity.y = 0.0
            command.velocity.z = 0.0
            command.acceleration.x = 0.0
            command.acceleration.y = 0.0
            command.acceleration.z = 0.0
        else:
            command.position.x = float(source.position.x)
            command.position.y = float(source.position.y)
            command.position.z = float(source.position.z)
            command.velocity.x = float(source.velocity.x)
            command.velocity.y = float(source.velocity.y)
            command.velocity.z = float(source.velocity.z)
            command.acceleration.x = float(source.acceleration.x)
            command.acceleration.y = float(source.acceleration.y)
            command.acceleration.z = float(source.acceleration.z)
        start_pos = np.asarray([command.position.x, command.position.y, command.position.z], dtype=np.float64)
        start_vel = np.asarray([command.velocity.x, command.velocity.y, command.velocity.z], dtype=np.float64)
        start_acc = np.asarray(
            [command.acceleration.x, command.acceleration.y, command.acceleration.z], dtype=np.float64
        )
        speed = float(np.linalg.norm(start_vel))
        duration = 0.0 if speed < 1e-3 else float(np.clip(speed / 3.0, 0.35, 1.5))
        if duration > 0.0:
            end_pos = start_pos + 0.5 * start_vel * duration
            self.joystick_hover_polys = tuple(
                Poly5Solver(start_pos[i], start_vel[i], start_acc[i], end_pos[i], 0.0, 0.0, duration)
                for i in range(3)
            )
        else:
            self.joystick_hover_polys = None
        self.joystick_hover_time = 0.0
        self.joystick_hover_duration = duration
        if self.joystick_yaw_target is None:
            self.joystick_yaw_target = float(self.last_yaw)
        command.yaw = float(self.joystick_yaw_target)
        command.yaw_dot = float(self.joystick_yaw_rate)
        self.joystick_hover_command = command
        self.joystick_plan_ready = False
        self.ctrl_time = None
        self.plan_horizon = self.traj_time

    def _capture_joystick_vertical_locked(self):
        """Start a direct vertical-velocity reference from current odometry."""
        self.joystick_manual_position = np.asarray(
            [
                self.odom.pose.pose.position.x,
                self.odom.pose.pose.position.y,
                self.odom.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        self.joystick_plan_ready = False
        self.ctrl_time = None

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

    def poll_joystick_intent(self, rot_wb=None):
        """Serially sample, transform, and commit the current stick intent."""
        with self.joystick_poll_lock:
            return self._poll_joystick_intent_serialized(rot_wb)

    def _poll_joystick_intent_serialized(self, rot_wb=None):
        """Poll both sticks and feed their desired velocity directly to YOPO."""
        zero = np.zeros(3, dtype=np.float32)
        if self.control_mode != "joystick" or self.joystick is None or not self.odom_init:
            return zero, zero, zero, False, 0

        if rot_wb is None:
            q = self.odom.pose.pose.orientation
            rot_wb = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
        else:
            rot_wb = np.asarray(rot_wb, dtype=np.float32)
        rot_bw = rot_wb.T

        velocity_fraction_h, yaw_fraction, _stick_magnitude, translation_requested = self.joystick.read_control_fraction()
        velocity_fraction_h = np.asarray(velocity_fraction_h, dtype=np.float32)
        if translation_requested:
            # Horizontal motion follows heading, while left-stick climb/descent
            # is world vertical. The full attitude transform then produces the
            # body-frame v_des consumed by the checkpoint.
            yaw = float(np.arctan2(rot_wb[1, 0], rot_wb[0, 0]))
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rot_wh = np.array(
                [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            operator_vdes_h = np.asarray(
                [
                    self.velocity * velocity_fraction_h[0],
                    self.velocity * velocity_fraction_h[1],
                    self.joystick_vertical_velocity * velocity_fraction_h[2],
                ],
                dtype=np.float32,
            )
            operator_vdes_w = (rot_wh @ operator_vdes_h).astype(np.float32)
            operator_vdes_b = (rot_bw @ operator_vdes_w).astype(np.float32)
            # The current checkpoint's eight topology queries are horizontal;
            # offline inspection shows a pure +/-z intent still produces a
            # horizontal endpoint. Keep the right-stick command in YOPO and
            # execute left-stick vertical velocity in the control layer.
            policy_vdes_w = operator_vdes_w.copy()
            policy_vdes_w[2] = 0.0
            planar_speed = float(np.linalg.norm(policy_vdes_w[:2]))
            planar_active = planar_speed > 1e-6
            policy_vdes_b = (rot_bw @ policy_vdes_w).astype(np.float32) if planar_active else zero.copy()
        else:
            velocity_fraction_h = zero.copy()
            operator_vdes_w = zero.copy()
            operator_vdes_b = zero.copy()
            policy_vdes_b = zero.copy()
            planar_active = False

        entered_hover = False
        became_active = False
        entered_vertical = False
        with self.lock:
            was_active = self.joystick_input_active
            was_vertical_active = abs(self.joystick_vertical_vdes) > 1e-6
            vertical_vdes = float(operator_vdes_w[2])
            vertical_active = abs(vertical_vdes) > 1e-6
            if bool(planar_active) != was_active:
                # Only crossing the centered translation state invalidates an
                # in-flight inference. Normal stick slews are applied on the
                # next depth frame and do not starve the planner.
                self.joystick_intent_epoch += 1
            self.joystick_input_active = bool(planar_active)
            self.joystick_hold = not planar_active
            self.joystick_yaw_rate = float(yaw_fraction * self.joystick_yaw_rate_max)
            self.joystick_vertical_vdes = vertical_vdes
            if planar_active and not was_active:
                self.joystick_plan_ready = False
                self.ctrl_time = None
                became_active = True
            elif not planar_active and vertical_active and (was_active or not was_vertical_active):
                self._capture_joystick_vertical_locked()
                entered_vertical = True
            elif not planar_active and not vertical_active and (
                was_active or was_vertical_active or self.joystick_hover_command is None
            ):
                self._capture_joystick_hover_locked()
                entered_hover = True
            intent_epoch = self.joystick_intent_epoch

        self._publish_vdes_body(operator_vdes_b)
        if entered_hover:
            connected, initialized, unlocked = self.joystick.get_status()
            rospy.loginfo(
                "Joystick hover reference captured: connected=%s initialized=%s unlocked=%s.",
                connected,
                initialized,
                unlocked,
            )
        elif became_active:
            rospy.loginfo("Right-stick translation active; waiting for the next direct YOPO trajectory.")
        elif entered_vertical:
            rospy.loginfo("Left-stick vertical velocity control active.")
        return policy_vdes_b, operator_vdes_w, operator_vdes_b, bool(planar_active), intent_epoch

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

        command_epoch = None
        operator_vdes_w = np.zeros(3, dtype=np.float32)
        if self.control_mode == "joystick" and self.joystick is not None:
            policy_vdes_b, operator_vdes_w, _operator_vdes_b, _active, command_epoch = (
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
            command_epoch,
        )

    @torch.inference_mode()
    def callback_depths(self, front, left, right, back):
        """Run the loaded YOPO checkpoint on each synchronized depth quartet."""
        if not self.odom_init:
            return

        time0 = time.time()
        (
            state_b,
            start_pos,
            start_vel,
            start_acc,
            rot_wb,
            _operator_vdes_w,
            command_epoch,
        ) = self.process_state()
        if self.control_mode == "joystick" and self.joystick_hold:
            return

        depth_messages = [front, left, right, back]
        depth = np.stack([self.preprocess_depth(message) for message in depth_messages], axis=0)
        depth = depth.reshape(1, 4, 1, self.height, self.width)
        time1 = time.time()

        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)
        state_input = torch.from_numpy(state_b[None, :]).to(self.device, non_blocking=True)
        time2 = time.time()

        # Runtime obstacle-avoidance inference using the checkpoint loaded
        # strictly in __init__; no hand-written candidate replaces its output.
        endstate_b, score = self.policy(depth_input, state_input)
        endstate_cand_b = endstate_b[0].detach().cpu().numpy().reshape(-1, 9)
        score_flat = score[0].detach().cpu().numpy().reshape(-1)
        time3 = time.time()

        action_id = int(np.argmin(score_flat))
        best_state_b = endstate_cand_b[action_id].reshape(3, 3).T
        best_state_w = rot_wb @ best_state_b
        candidate_poly_x = Poly5Solver(
            start_pos[0], start_vel[0], start_acc[0],
            best_state_w[0, 0] + start_pos[0], best_state_w[0, 1], best_state_w[0, 2], self.traj_time,
        )
        candidate_poly_y = Poly5Solver(
            start_pos[1], start_vel[1], start_acc[1],
            best_state_w[1, 0] + start_pos[1], best_state_w[1, 1], best_state_w[1, 2], self.traj_time,
        )
        candidate_poly_z = Poly5Solver(
            start_pos[2], start_vel[2], start_acc[2],
            best_state_w[2, 0] + start_pos[2], best_state_w[2, 1], best_state_w[2, 2], self.traj_time,
        )

        with self.lock:
            if self.control_mode == "joystick":
                if not _joystick_plan_is_current(
                    self.joystick_input_active,
                    command_epoch,
                    self.joystick_intent_epoch,
                ):
                    return
                self.joystick_plan_ready = True
            self.optimal_poly_x = candidate_poly_x
            self.optimal_poly_y = candidate_poly_y
            self.optimal_poly_z = candidate_poly_z
            self.ctrl_time = 0.0
            self.plan_horizon = self.traj_time
        time4 = time.time()

        self.visualize_trajectory(
            start_pos,
            start_vel,
            start_acc,
            rot_wb,
            endstate_cand_b,
            score_flat,
            self.traj_time,
        )
        time5 = time.time()
        self.print_time(time0, time1, time2, time3, time4, time5)

    def _advance_joystick_yaw_locked(self):
        if self.joystick_yaw_target is None:
            self.joystick_yaw_target = float(self.last_yaw)
        self.joystick_yaw_target += float(self.joystick_yaw_rate) * self.ctrl_dt
        self.joystick_yaw_target = float(
            np.arctan2(np.sin(self.joystick_yaw_target), np.cos(self.joystick_yaw_target))
        )
        self.last_yaw = self.joystick_yaw_target
        return self.joystick_yaw_target, float(self.joystick_yaw_rate)

    def publish_joystick_vertical(self):
        """Integrate the left-stick world-z velocity into an EMPTY reference."""
        if not self.odom_init:
            return False
        with self.lock:
            if self.joystick_input_active or abs(self.joystick_vertical_vdes) <= 1e-6:
                return False
            if self.joystick_manual_position is None:
                self._capture_joystick_vertical_locked()
            self.joystick_manual_position[2] += self.joystick_vertical_vdes * self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
            control_msg.position.x = float(self.joystick_manual_position[0])
            control_msg.position.y = float(self.joystick_manual_position[1])
            control_msg.position.z = float(self.joystick_manual_position[2])
            control_msg.velocity.x = 0.0
            control_msg.velocity.y = 0.0
            control_msg.velocity.z = float(self.joystick_vertical_vdes)
            control_msg.acceleration.x = 0.0
            control_msg.acceleration.y = 0.0
            control_msg.acceleration.z = 0.0
            control_msg.yaw, control_msg.yaw_dot = self._advance_joystick_yaw_locked()
            self.desire_pos = self.joystick_manual_position.copy()
            self.desire_vel = np.asarray([0.0, 0.0, self.joystick_vertical_vdes], dtype=np.float32)
            self.desire_acc = np.zeros(3, dtype=np.float32)
            self.desire_init = True
            self.last_control_msg = control_msg
        self.ctrl_pub.publish(control_msg)
        return True

    def publish_joystick_hold(self):
        """Publish a continuous EMPTY braking reference, then hover."""
        if not self.odom_init:
            return False
        with self.lock:
            if self.joystick_input_active and self.joystick_plan_ready:
                return False
            if self.joystick_hover_command is None:
                self._capture_joystick_hover_locked()
            source = self.joystick_hover_command
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
            if self.joystick_hover_polys is not None:
                sample_time = min(self.joystick_hover_time, self.joystick_hover_duration)
                position = np.asarray([poly.get_position(sample_time) for poly in self.joystick_hover_polys])
                velocity = np.asarray([poly.get_velocity(sample_time) for poly in self.joystick_hover_polys])
                acceleration = np.asarray([poly.get_acceleration(sample_time) for poly in self.joystick_hover_polys])
                self.joystick_hover_time = min(
                    self.joystick_hover_time + self.ctrl_dt, self.joystick_hover_duration
                )
            else:
                position = np.asarray([source.position.x, source.position.y, source.position.z])
                velocity = np.zeros(3)
                acceleration = np.zeros(3)
            control_msg.position.x, control_msg.position.y, control_msg.position.z = map(float, position)
            control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z = map(float, velocity)
            control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z = map(float, acceleration)
            control_msg.yaw, control_msg.yaw_dot = self._advance_joystick_yaw_locked()
            self.desire_pos = np.asarray(
                [control_msg.position.x, control_msg.position.y, control_msg.position.z], dtype=np.float32
            )
            self.desire_vel = np.asarray(
                [control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z], dtype=np.float32
            )
            self.desire_acc = np.asarray(
                [control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z], dtype=np.float32
            )
            self.desire_init = True
            self.last_control_msg = control_msg
        self.ctrl_pub.publish(control_msg)
        return True

    def control_pub(self, _timer):
        if self.control_mode == "joystick":
            # Input remains responsive even when depth callbacks are slower.
            self.poll_joystick_intent()
            with self.lock:
                joystick_active = self.joystick_input_active
                vertical_active = abs(self.joystick_vertical_vdes) > 1e-6
                plan_ready = self.joystick_plan_ready
                ctrl_time = self.ctrl_time
                plan_horizon = self.plan_horizon
            if not joystick_active:
                if vertical_active:
                    self.publish_joystick_vertical()
                else:
                    self.publish_joystick_hold()
                return
            if not plan_ready:
                self.publish_joystick_hold()
                return
            if ctrl_time is None or ctrl_time >= plan_horizon:
                # Match the original YOPO execution path: do not invent a
                # replacement plan when depth/model updates stop.
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
                if self.control_mode == "joystick":
                    current_z = float(self.odom.pose.pose.position.z)
                    current_vz = float(self.odom.twist.twist.linear.z)
                    vertical_acc = float(
                        np.clip(
                            self.joystick_vertical_velocity_kp * (self.joystick_vertical_vdes - current_vz),
                            -self.joystick_vertical_accel_max,
                            self.joystick_vertical_accel_max,
                        )
                    )
                    # READY ignores p/v in NetworkControl, but keeping these
                    # fields on the manual vertical reference makes the later
                    # READY->EMPTY hover transition continuous.
                    control_msg.position.z = current_z
                    control_msg.velocity.z = float(self.joystick_vertical_vdes)
                    control_msg.acceleration.z = vertical_acc
                self.desire_pos = np.array(
                    [control_msg.position.x, control_msg.position.y, control_msg.position.z], dtype=np.float32
                )
                self.desire_vel = np.array(
                    [control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z], dtype=np.float32
                )
                self.desire_acc = np.array(
                    [control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z], dtype=np.float32
                )
                if self.control_mode == "joystick":
                    yaw, yaw_dot = self._advance_joystick_yaw_locked()
                elif self.fixed_yaw:
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
    parser.add_argument("--joystick-axis-z", type=int, default=2, help="Left-stick vertical/climb axis index.")
    parser.add_argument("--joystick-axis-yaw", type=int, default=3, help="Left-stick horizontal/yaw axis index.")
    parser.add_argument("--joystick-axis-max", type=float, default=32767.0, help="Absolute raw joystick axis maximum.")
    parser.add_argument("--joystick-deadzone", type=float, default=0.08,
                        help="Radial deadzone before the right stick is considered active.")
    parser.add_argument("--joystick-invert-x", type=int, choices=(0, 1), default=1,
                        help="Invert horizontal axis after normalization.")
    parser.add_argument("--joystick-invert-y", type=int, choices=(0, 1), default=0,
                        help="Invert vertical axis after normalization.")
    parser.add_argument("--joystick-invert-z", type=int, choices=(0, 1), default=0,
                        help="Invert left-stick climb/descent axis after normalization.")
    parser.add_argument("--joystick-invert-yaw", type=int, choices=(0, 1), default=1,
                        help="Invert left-stick yaw axis; RadioMaster left then maps to positive yaw.")
    parser.add_argument("--joystick-swap-xy", type=int, choices=(0, 1), default=1,
                        help="Map vertical stick to body x and horizontal stick to body y.")
    parser.add_argument(
        "--joystick-vertical-velocity",
        type=float,
        default=2.0,
        help="Maximum left-stick climb/descent desired speed in m/s.",
    )
    parser.add_argument(
        "--joystick-yaw-rate",
        type=float,
        default=1.0,
        help="Maximum left-stick yaw rate in rad/s.",
    )
    parser.add_argument("--joystick-calibrate", type=int, choices=(0, 1), default=0,
                        help="Print raw joystick axes for calibration.")
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
        "joystick_vertical_velocity": args.joystick_vertical_velocity,
        "joystick_yaw_rate": args.joystick_yaw_rate,
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
        "joystick": {
            "device": args.joystick_device,
            "axis_x": args.joystick_axis_x,
            "axis_y": args.joystick_axis_y,
            "axis_z": args.joystick_axis_z,
            "axis_yaw": args.joystick_axis_yaw,
            "axis_max": args.joystick_axis_max,
            "deadzone": args.joystick_deadzone,
            "invert_x": bool(args.joystick_invert_x),
            "invert_y": bool(args.joystick_invert_y),
            "invert_z": bool(args.joystick_invert_z),
            "invert_yaw": bool(args.joystick_invert_yaw),
            "swap_xy": bool(args.joystick_swap_xy),
            "calibrate": bool(args.joystick_calibrate),
        },
    }
    YopoOmniNet(settings, weight)
