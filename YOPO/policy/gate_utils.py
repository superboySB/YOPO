import math
import numpy as np
import torch

from config.config import cfg


def cfg_get(key, default):
    try:
        return cfg[key]
    except KeyError:
        return default


def gate_enabled():
    return bool(cfg_get("gate_enabled", False))


def rpy_to_matrix_np(roll_deg, pitch_deg, yaw_deg):
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return rz @ ry @ rx


def _normalize_np(vec, eps=1.0e-6):
    return vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), eps)


class GateNumpy:
    def __init__(self, pose=None):
        if pose is None:
            pose = np.asarray(cfg_get("gate_pose", [85.0, 0.0, 0.0, 0.0, 0.0, 1.2]), dtype=np.float32)
            pose[0] = float(cfg_get("gate_slit_roll_deg", pose[0]))
        else:
            pose = np.asarray(pose, dtype=np.float32).copy()
        self.pose = pose.astype(np.float32)
        self.enabled = gate_enabled()
        self.rot = rpy_to_matrix_np(pose[0], pose[1], pose[2])
        self.center = pose[3:6].astype(np.float32)
        self.count = max(1, int(cfg_get("gate_count", 1)))
        self.spacing = max(0.0, float(cfg_get("gate_spacing", 3.0)))
        self.outer_width = float(cfg_get("gate_outer_width", 0.88))
        self.outer_length = float(cfg_get("gate_outer_length", 0.38))
        self.inner_width = float(cfg_get("gate_inner_width", 0.70))
        self.inner_length = float(cfg_get("gate_inner_length", 0.30))
        self.depth = float(cfg_get("gate_depth", 0.05))
        self.depth_margin = float(cfg_get("gate_depth_margin", 0.01))
        self.tunnel_half_depth = float(cfg_get("gate_tunnel_half_depth", 0.60))
        self.safe_margin = float(cfg_get("gate_safe_margin", 0.02))
        self.plane_sigma = float(cfg_get("gate_loss_plane_sigma", 0.35))
        self.plane_weight = float(cfg_get("gate_loss_plane_weight", 0.50))
        self.opening_weight = float(cfg_get("gate_loss_opening_weight", 1.00))
        self.crossing_weight = float(cfg_get("gate_loss_crossing_weight", 0.10))
        self.roll_align_weight = float(cfg_get("gate_loss_roll_align_weight", 0.0))
        self.entry_weight = float(cfg_get("gate_loss_entry_weight", 0.50))
        self.tunnel_weight = float(cfg_get("gate_loss_tunnel_weight", 1.00))
        self.exit_weight = float(cfg_get("gate_loss_exit_weight", 0.50))
        self.eval_points = int(cfg_get("gate_loss_eval_points", 30))
        self.test_weight = float(cfg_get("gate_test_weight", 1.00))
        self.goal_min_x = float(cfg_get("gate_goal_min_x", 1.20))
        self.goal_max_x = float(cfg_get("gate_goal_max_x", 2.60))
        self.objective_terminal_weight = float(cfg_get("gate_objective_terminal_weight", 1.00))
        self.objective_progress_weight = float(cfg_get("gate_objective_progress_weight", 1.20))
        self.objective_center_weight = float(cfg_get("gate_objective_center_weight", 0.30))
        self.objective_velocity_weight = float(cfg_get("gate_objective_velocity_weight", 0.15))
        self.objective_lateral_velocity_weight = float(cfg_get("gate_objective_lateral_velocity_weight", 0.0))
        self.objective_min_exit_speed_ratio = float(cfg_get("gate_objective_min_exit_speed_ratio", 0.20))
        self.vel_ref = float(cfg_get("vel_max_train", 6.0))
        self.goal_y_std = float(cfg_get("gate_goal_y_std", 0.10))
        self.goal_z_std = float(cfg_get("gate_goal_z_std", 0.03))
        self.sample_x_max = float(cfg_get("gate_sample_x_max", 3.00))
        self.sample_y_range = float(cfg_get("gate_sample_y_range", 0.90))
        self.sample_z_range = float(cfg_get("gate_sample_z_range", 0.45))
        self.floor_weight = float(cfg_get("gate_loss_floor_weight", 0.0))
        self.floor_sigma = float(cfg_get("gate_floor_sigma", 0.20))
        self.floor_min_z = float(cfg_get("gate_floor_min_z", float(self.center[2]) - self.sample_z_range))
        self.arrive_x_margin = float(cfg_get("gate_arrive_x_margin", 0.35))
        body_width = float(cfg_get("uav_collision_box_width", 0.34))
        body_depth = float(cfg_get("uav_collision_box_depth", 0.34))
        body_height = float(cfg_get("uav_collision_box_height", 0.13))
        body_half_xy = 0.5 * max(body_width, body_depth)
        self.body_radii = np.asarray([body_half_xy, body_half_xy, 0.5 * body_height], dtype=np.float32)

    @property
    def side_sep(self):
        return 0.5 * self.depth + self.depth_margin

    @property
    def normal(self):
        return self.rot[:, 0]

    @property
    def centers(self):
        offsets = np.arange(self.count, dtype=np.float32)[:, None] * self.spacing * self.normal[None, :]
        return self.center[None, :] + offsets

    def center_at(self, gate_idx):
        gate_idx = int(np.clip(gate_idx, 0, self.count - 1))
        return self.center + self.normal * (gate_idx * self.spacing)

    def to_local(self, points, center=None):
        points = np.asarray(points, dtype=np.float32)
        center = self.center if center is None else np.asarray(center, dtype=np.float32)
        return (np.atleast_2d(points) - center) @ self.rot

    def to_world(self, local_points, center=None):
        local_points = np.asarray(local_points, dtype=np.float32)
        center = self.center if center is None else np.asarray(center, dtype=np.float32)
        return np.atleast_2d(local_points) @ self.rot.T + center

    def nearest_gate_index(self, position):
        base_local_x = float(self.to_local(position, self.center)[0, 0])
        if self.spacing <= 1.0e-6:
            return 0
        return int(np.clip(np.round(base_local_x / self.spacing), 0, self.count - 1))

    def active_gate_index(self, start, goal):
        start_x = float(self.to_local(start, self.center)[0, 0])
        goal_x = float(self.to_local(goal, self.center)[0, 0])
        if self.spacing <= 1.0e-6:
            return 0

        direction = 1.0 if goal_x >= start_x else -1.0
        indices = range(self.count) if direction > 0.0 else range(self.count - 1, -1, -1)
        for idx in indices:
            gate_x = idx * self.spacing
            if (start_x - gate_x) * (goal_x - gate_x) <= 0.0:
                return idx
            if direction > 0.0 and start_x < gate_x + self.side_sep and goal_x > gate_x:
                return idx
            if direction < 0.0 and start_x > gate_x - self.side_sep and goal_x < gate_x:
                return idx
        return self.nearest_gate_index(start)

    def can_sample_gate_goal(self, position):
        center = self.center_at(self.nearest_gate_index(position))
        local = self.to_local(position, center)[0]
        return (abs(local[0]) <= self.sample_x_max + 0.5 and
                abs(local[1]) <= self.sample_y_range + 0.5 and
                abs(local[2]) <= self.sample_z_range + 0.5)

    def sample_goal(self, position, rng=np.random):
        center = self.center_at(self.nearest_gate_index(position))
        local = self.to_local(position, center)[0]
        side = -1.0 if local[0] < 0.0 else 1.0
        if abs(local[0]) < self.side_sep:
            side = -1.0 if rng.rand() < 0.5 else 1.0
        goal_side = -side
        goal_x = goal_side * rng.uniform(self.goal_min_x, self.goal_max_x)
        goal_y = rng.normal(0.0, self.goal_y_std)
        goal_z = rng.normal(0.0, self.goal_z_std)
        return self.to_world([goal_x, goal_y, goal_z], center)[0]

    def is_gate_task(self, start, goal):
        if not self.enabled:
            return False
        idx = self.active_gate_index(start, goal)
        center = self.center_at(idx)
        start_x = self.to_local(start, center)[0, 0]
        goal_x = self.to_local(goal, center)[0, 0]
        return start_x * goal_x < 0.0 and max(abs(start_x), abs(goal_x)) > self.side_sep

    def goal_reached(self, position, goal, radius):
        position = np.asarray(position, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        if np.linalg.norm(position - goal) >= radius:
            return False
        if not self.enabled:
            return True

        idx = self.active_gate_index(position, goal)
        center = self.center_at(idx)
        pos_local = self.to_local(position, center)[0]
        goal_local = self.to_local(goal, center)[0]
        goal_sign = 1.0 if goal_local[0] >= 0.0 else -1.0
        return goal_sign * (pos_local[0] - goal_local[0]) >= -self.arrive_x_margin


def body_axes_from_flatness_np(vel, acc, default_heading, grav=9.81):
    vel = np.atleast_2d(np.asarray(vel, dtype=np.float32))
    acc = np.atleast_2d(np.asarray(acc, dtype=np.float32))
    thrust = acc + np.asarray([0.0, 0.0, grav], dtype=np.float32)
    b3 = _normalize_np(thrust)

    heading = vel.copy()
    heading[:, 2] = 0.0
    default_heading = np.asarray(default_heading, dtype=np.float32).copy()
    default_heading[2] = 0.0
    if np.linalg.norm(default_heading) < 1.0e-6:
        default_heading = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    default_heading = default_heading / np.linalg.norm(default_heading)
    small = np.linalg.norm(heading, axis=1) < 1.0e-4
    heading[small] = default_heading
    b1d = _normalize_np(heading)

    b2 = np.cross(b3, b1d)
    small_b2 = np.linalg.norm(b2, axis=1) < 1.0e-5
    if np.any(small_b2):
        fallback = np.tile(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), (vel.shape[0], 1))
        b2[small_b2] = np.cross(b3[small_b2], fallback[small_b2])
    b2 = _normalize_np(b2)
    b1 = _normalize_np(np.cross(b2, b3))
    return b1, b2, b3


def gate_passage_cost_np(pos, vel, acc, gate=None, center=None):
    gate = gate or GateNumpy()
    if not gate.enabled:
        return 0.0
    pos = np.atleast_2d(np.asarray(pos, dtype=np.float32))
    vel = np.atleast_2d(np.asarray(vel, dtype=np.float32))
    acc = np.atleast_2d(np.asarray(acc, dtype=np.float32))
    center = gate.center if center is None else np.asarray(center, dtype=np.float32)
    local = gate.to_local(pos, center)
    b1, b2, b3 = body_axes_from_flatness_np(vel, acc, gate.normal)

    gate_y = gate.rot[:, 1]
    gate_z = gate.rot[:, 2]
    radii = gate.body_radii
    support_y = np.sqrt((radii[0] * (b1 @ gate_y)) ** 2 +
                        (radii[1] * (b2 @ gate_y)) ** 2 +
                        (radii[2] * (b3 @ gate_y)) ** 2) + gate.safe_margin
    support_z = np.sqrt((radii[0] * (b1 @ gate_z)) ** 2 +
                        (radii[1] * (b2 @ gate_z)) ** 2 +
                        (radii[2] * (b3 @ gate_z)) ** 2) + gate.safe_margin

    half_width = max(1.0e-3, 0.5 * gate.inner_width)
    half_height = max(1.0e-3, 0.5 * gate.inner_length)
    y_over = np.maximum(0.0, (np.abs(local[:, 1]) + support_y - half_width) / half_width)
    z_over = np.maximum(0.0, (np.abs(local[:, 2]) + support_z - half_height) / half_height)
    plane = (np.abs(local[:, 0]) / max(1.0e-3, gate.plane_sigma)) ** 2
    roll_align = (1.0 - np.abs(b3 @ gate_z)) ** 2
    point_cost = (gate.plane_weight * plane +
                  gate.opening_weight * (y_over ** 2 + z_over ** 2) +
                  gate.roll_align_weight * roll_align)
    num_points = local.shape[0]
    split_1 = max(1, num_points // 3)
    split_2 = min(num_points - 1, max(split_1 + 1, (2 * num_points) // 3))
    entry_local_x = local[:split_1, 0]
    tunnel_cost = float(np.mean(point_cost[split_1:split_2]))
    exit_local_x = local[split_2:, 0]
    start_x = float(local[0, 0])
    end_x = float(local[-1, 0])
    start_sign = 1.0 if start_x >= 0.0 else -1.0
    entry_cost = float(np.mean((np.maximum(0.0, gate.side_sep - start_sign * entry_local_x) /
                                max(0.05, gate.side_sep)) ** 2))
    exit_cost = float(np.mean((np.maximum(0.0, gate.side_sep + start_sign * exit_local_x) /
                               max(0.05, gate.side_sep)) ** 2))
    floor_cost = 0.0
    if gate.floor_weight > 0.0:
        floor_under = np.maximum(0.0, gate.floor_min_z - pos[:, 2]) / max(0.05, gate.floor_sigma)
        floor_cost = float(np.mean(floor_under ** 2))
    gate_cost = (gate.entry_weight * entry_cost +
                 gate.tunnel_weight * tunnel_cost +
                 gate.exit_weight * exit_cost +
                 gate.floor_weight * floor_cost)
    if abs(start_x) > gate.side_sep:
        crossing_over = max(0.0, math.copysign(1.0, start_x) * end_x + gate.side_sep)
        gate_cost += gate.crossing_weight * (crossing_over / max(0.05, gate.side_sep)) ** 2
    return gate_cost


def gate_objective_cost_np(start_pos, end_pos, end_vel, goal, gate=None, center=None):
    gate = gate or GateNumpy()
    if not gate.enabled:
        return 0.0

    center = gate.center if center is None else np.asarray(center, dtype=np.float32)
    start_local = gate.to_local(start_pos, center)[0]
    end_local = gate.to_local(end_pos, center)[0]
    goal_local = gate.to_local(goal, center)[0]
    vel_local = np.asarray(end_vel, dtype=np.float32) @ gate.rot

    terminal_scale = max(1.0e-3, gate.goal_max_x)
    terminal_delta = (np.asarray(end_pos, dtype=np.float32) - np.asarray(goal, dtype=np.float32)) / terminal_scale
    terminal = float(np.sum(np.where(np.abs(terminal_delta) < 1.0,
                                     0.5 * terminal_delta ** 2,
                                     np.abs(terminal_delta) - 0.5)))

    start_sign = 1.0 if start_local[0] >= 0.0 else -1.0
    desired_side_x = float(np.clip(abs(goal_local[0]), gate.goal_min_x, gate.goal_max_x))
    through_x = -start_sign * end_local[0]
    progress = (max(0.0, desired_side_x - through_x) / max(1.0e-3, gate.goal_min_x)) ** 2

    center_y = end_local[1] / max(1.0e-3, 0.5 * gate.inner_width)
    center_z = end_local[2] / max(1.0e-3, 0.5 * gate.inner_length)
    center_cost = center_y ** 2 + center_z ** 2

    through_speed = -start_sign * vel_local[0]
    min_exit_speed = gate.objective_min_exit_speed_ratio * gate.vel_ref
    velocity_cost = (max(0.0, min_exit_speed - through_speed) / max(1.0e-3, gate.vel_ref)) ** 2
    lateral_velocity = (vel_local[1] ** 2 + vel_local[2] ** 2) / max(1.0e-3, gate.vel_ref ** 2)

    return float(gate.objective_terminal_weight * terminal +
                 gate.objective_progress_weight * progress +
                 gate.objective_center_weight * center_cost +
                 gate.objective_velocity_weight * velocity_cost +
                 gate.objective_lateral_velocity_weight * lateral_velocity)


def gate_cost_for_polys_np(poly_x, poly_y, poly_z, traj_time, gate=None, center=None):
    gate = gate or GateNumpy()
    if not gate.enabled:
        return 0.0
    t_values = np.linspace(0.0, traj_time, max(2, gate.eval_points), dtype=np.float32)
    pos = np.stack((poly_x.get_position(t_values),
                    poly_y.get_position(t_values),
                    poly_z.get_position(t_values)), axis=1)
    vel = np.stack((poly_x.get_velocity(t_values),
                    poly_y.get_velocity(t_values),
                    poly_z.get_velocity(t_values)), axis=1)
    acc = np.stack((poly_x.get_acceleration(t_values),
                    poly_y.get_acceleration(t_values),
                    poly_z.get_acceleration(t_values)), axis=1)
    return gate_passage_cost_np(pos, vel, acc, gate, center=center)


def gate_total_cost_for_polys_np(poly_x, poly_y, poly_z, traj_time, start_pos, goal, gate=None, center=None):
    gate = gate or GateNumpy()
    if not gate.enabled:
        return 0.0
    passage_cost = gate_cost_for_polys_np(poly_x, poly_y, poly_z, traj_time, gate, center=center)
    end_pos = np.asarray([poly_x.get_position(traj_time),
                          poly_y.get_position(traj_time),
                          poly_z.get_position(traj_time)], dtype=np.float32)
    end_vel = np.asarray([poly_x.get_velocity(traj_time),
                          poly_y.get_velocity(traj_time),
                          poly_z.get_velocity(traj_time)], dtype=np.float32)
    objective_cost = gate_objective_cost_np(start_pos, end_pos, end_vel, goal, gate, center=center)
    return passage_cost + objective_cost


def gate_marker_edges_np(gate=None):
    gate = gate or GateNumpy()
    xi = 0.5 * gate.depth
    yi = 0.5 * gate.inner_width
    zi = 0.5 * gate.inner_length
    xo = xi + gate.depth_margin
    yo = 0.5 * gate.outer_width
    zo = 0.5 * gate.outer_length
    edges = []

    def add_box(x, y, z, center):
        pts = {
            "000": np.array([-x, -y, -z], dtype=np.float32),
            "001": np.array([-x, -y, z], dtype=np.float32),
            "010": np.array([-x, y, -z], dtype=np.float32),
            "011": np.array([-x, y, z], dtype=np.float32),
            "100": np.array([x, -y, -z], dtype=np.float32),
            "101": np.array([x, -y, z], dtype=np.float32),
            "110": np.array([x, y, -z], dtype=np.float32),
            "111": np.array([x, y, z], dtype=np.float32),
        }
        for a, b in [("000", "001"), ("000", "010"), ("000", "100"), ("001", "011"),
                     ("001", "101"), ("010", "011"), ("010", "110"), ("011", "111"),
                     ("100", "101"), ("100", "110"), ("101", "111"), ("110", "111")]:
            edges.append((gate.to_world(pts[a], center)[0], gate.to_world(pts[b], center)[0]))

    for center in gate.centers:
        add_box(xo, yo, zo, center)
        add_box(xi, yi, zi, center)
    return edges


def rpy_to_matrix_torch(roll_deg, pitch_deg, yaw_deg, device):
    roll = torch.tensor(math.radians(float(roll_deg)), device=device)
    pitch = torch.tensor(math.radians(float(pitch_deg)), device=device)
    yaw = torch.tensor(math.radians(float(yaw_deg)), device=device)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    rz = torch.stack((torch.stack((cy, -sy, torch.tensor(0.0, device=device))),
                      torch.stack((sy, cy, torch.tensor(0.0, device=device))),
                      torch.tensor([0.0, 0.0, 1.0], device=device)))
    ry = torch.stack((torch.stack((cp, torch.tensor(0.0, device=device), sp)),
                      torch.tensor([0.0, 1.0, 0.0], device=device),
                      torch.stack((-sp, torch.tensor(0.0, device=device), cp))))
    rx = torch.stack((torch.tensor([1.0, 0.0, 0.0], device=device),
                      torch.stack((torch.tensor(0.0, device=device), cr, -sr)),
                      torch.stack((torch.tensor(0.0, device=device), sr, cr))))
    return rz @ ry @ rx


def rpy_to_matrix_torch_batch(roll_deg, pitch_deg, yaw_deg):
    roll = torch.deg2rad(roll_deg)
    pitch = torch.deg2rad(pitch_deg)
    yaw = torch.deg2rad(yaw_deg)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    zeros = torch.zeros_like(roll)
    ones = torch.ones_like(roll)

    rz = torch.stack((
        torch.stack((cy, -sy, zeros), dim=-1),
        torch.stack((sy, cy, zeros), dim=-1),
        torch.stack((zeros, zeros, ones), dim=-1),
    ), dim=-2)
    ry = torch.stack((
        torch.stack((cp, zeros, sp), dim=-1),
        torch.stack((zeros, ones, zeros), dim=-1),
        torch.stack((-sp, zeros, cp), dim=-1),
    ), dim=-2)
    rx = torch.stack((
        torch.stack((ones, zeros, zeros), dim=-1),
        torch.stack((zeros, cr, -sr), dim=-1),
        torch.stack((zeros, sr, cr), dim=-1),
    ), dim=-2)
    return rz @ ry @ rx
