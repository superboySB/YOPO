import torch as th
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from policy.gate_utils import cfg_get, gate_enabled, rpy_to_matrix_torch, rpy_to_matrix_torch_batch


class GateSE3Loss(nn.Module):
    def __init__(self, L):
        super(GateSE3Loss, self).__init__()
        self.enabled = gate_enabled()
        self._L = L
        self.sgm_time = cfg["sgm_time"]
        self.eval_points = int(cfg_get("gate_loss_eval_points", 30))
        self.device = self._L.device
        pose = cfg_get("gate_pose", [85.0, 0.0, 0.0, 0.0, 0.0, 1.2])
        pose = list(pose)
        pose[0] = float(cfg_get("gate_slit_roll_deg", pose[0]))
        gate_rot = rpy_to_matrix_torch(pose[0], pose[1], pose[2], self.device).float()
        gate_center = th.tensor(pose[3:6], dtype=th.float32, device=self.device)
        body_width = float(cfg_get("uav_collision_box_width", 0.34))
        body_depth = float(cfg_get("uav_collision_box_depth", 0.34))
        body_height = float(cfg_get("uav_collision_box_height", 0.13))
        body_half_xy = 0.5 * max(body_width, body_depth)
        body_radii = th.tensor([body_half_xy, body_half_xy, 0.5 * body_height],
                               dtype=th.float32, device=self.device)

        self.register_buffer("gate_rot", gate_rot)
        self.register_buffer("gate_center", gate_center)
        self.register_buffer("gate_y", gate_rot[:, 1].clone())
        self.register_buffer("gate_z", gate_rot[:, 2].clone())
        self.register_buffer("gate_normal", gate_rot[:, 0].clone())
        self.register_buffer("body_radii", body_radii)

        self.inner_half_width = 0.5 * float(cfg_get("gate_inner_width", 0.64))
        self.inner_half_height = 0.5 * float(cfg_get("gate_inner_length", 0.22))
        self.safe_margin = float(cfg_get("gate_safe_margin", 0.02))
        self.plane_sigma = float(cfg_get("gate_loss_plane_sigma", 0.35))
        self.plane_weight = float(cfg_get("gate_loss_plane_weight", 0.50))
        self.opening_weight = float(cfg_get("gate_loss_opening_weight", 1.00))
        self.crossing_weight = float(cfg_get("gate_loss_crossing_weight", 0.10))
        self.roll_align_weight = float(cfg_get("gate_loss_roll_align_weight", 0.0))
        self.entry_weight = float(cfg_get("gate_loss_entry_weight", 0.50))
        self.tunnel_weight = float(cfg_get("gate_loss_tunnel_weight", 1.00))
        self.exit_weight = float(cfg_get("gate_loss_exit_weight", 0.50))
        self.floor_weight = float(cfg_get("gate_loss_floor_weight", 0.0))
        self.floor_sigma = float(cfg_get("gate_floor_sigma", 0.20))
        self.floor_min_z = float(cfg_get("gate_floor_min_z", pose[5] - float(cfg_get("gate_sample_z_range", 0.45))))
        self.side_sep = 0.5 * float(cfg_get("gate_depth", 0.05)) + float(cfg_get("gate_depth_margin", 0.01))
        self.gate_count = max(1, int(cfg_get("gate_count", 1)))
        self.gate_spacing = max(0.0, float(cfg_get("gate_spacing", 3.0)))
        self.gravity = 9.81

    def forward(self, Df, Dp, active_mask=None, gate_pose=None):
        batch_size = Dp.shape[0]
        if not self.enabled:
            return th.zeros(batch_size, dtype=Dp.dtype, device=Dp.device)
        gate_rot, gate_center, gate_normal, gate_y, gate_z = self.gate_geometry(batch_size, Dp.dtype, Dp.device, gate_pose)

        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        coe = self.get_coefficient_from_derivative(Dp, Df, L)
        dt = self.sgm_time / max(1, self.eval_points - 1)
        t_list = th.linspace(0.0, self.sgm_time, self.eval_points, device=Dp.device, dtype=Dp.dtype)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)
        pos = self.get_position_from_coeff(coe, t_list)
        vel = self.get_velocity_from_coeff(coe, t_list)
        acc = self.get_acceleration_from_coeff(coe, t_list)
        _ = dt  # keep the sampling definition explicit; the loss uses min-over-samples.

        center = self.active_gate_center(Df[:, :, 0], gate_rot, gate_center, gate_normal)
        local = th.bmm(pos - center.view(batch_size, 1, 3), gate_rot)
        b1, b2, b3 = self.body_axes_from_flatness(vel, acc, gate_normal)

        support_y = self.ellipsoid_support(gate_y, b1, b2, b3) + self.safe_margin
        support_z = self.ellipsoid_support(gate_z, b1, b2, b3) + self.safe_margin
        half_width = max(1.0e-3, self.inner_half_width)
        half_height = max(1.0e-3, self.inner_half_height)
        y_over = F.relu((local[:, :, 1].abs() + support_y - half_width) / half_width)
        z_over = F.relu((local[:, :, 2].abs() + support_z - half_height) / half_height)
        plane = (local[:, :, 0].abs() / max(1.0e-3, self.plane_sigma)) ** 2
        roll_align = (1.0 - (b3 * gate_z.view(batch_size, 1, 3)).sum(dim=-1).abs()) ** 2
        point_cost = (self.plane_weight * plane +
                      self.opening_weight * (y_over ** 2 + z_over ** 2) +
                      self.roll_align_weight * roll_align)
        split_1 = max(1, self.eval_points // 3)
        split_2 = min(self.eval_points - 1, max(split_1 + 1, (2 * self.eval_points) // 3))
        start_local_x = th.bmm((Df[:, :, 0] - center).unsqueeze(1), gate_rot).squeeze(1)[:, 0]
        end_local_x = th.bmm((Dp[:, :, 0] - center).unsqueeze(1), gate_rot).squeeze(1)[:, 0]
        start_sign = th.where(start_local_x >= 0.0, th.ones_like(start_local_x), -th.ones_like(start_local_x))
        entry_phase = F.relu(self.side_sep - start_sign.view(-1, 1) * local[:, :split_1, 0])
        exit_phase = F.relu(self.side_sep + start_sign.view(-1, 1) * local[:, split_2:, 0])
        entry_cost = (entry_phase / max(0.05, self.side_sep)).pow(2).mean(dim=1)
        tunnel_cost = point_cost[:, split_1:split_2].mean(dim=1)
        exit_cost = (exit_phase / max(0.05, self.side_sep)).pow(2).mean(dim=1)
        floor_cost = th.zeros_like(entry_cost)
        if self.floor_weight > 0.0:
            floor_under = F.relu(self.floor_min_z - pos[:, :, 2]) / max(0.05, self.floor_sigma)
            floor_cost = floor_under.pow(2).mean(dim=1)
        gate_cost = (self.entry_weight * entry_cost +
                     self.tunnel_weight * tunnel_cost +
                     self.exit_weight * exit_cost +
                     self.floor_weight * floor_cost)
        crossing_over = F.relu(start_sign * end_local_x + self.side_sep)
        crossing_cost = (crossing_over / max(0.05, self.side_sep)) ** 2
        crossing_cost = th.where(start_local_x.abs() > self.side_sep, crossing_cost, th.zeros_like(crossing_cost))
        gate_cost = gate_cost + self.crossing_weight * crossing_cost

        if active_mask is None:
            return gate_cost
        active_mask = active_mask.to(device=Dp.device, dtype=Dp.dtype).view(-1)
        return gate_cost * active_mask

    def gate_geometry(self, batch_size, dtype, device, gate_pose=None):
        if gate_pose is None:
            gate_rot = self.gate_rot.to(device=device, dtype=dtype).view(1, 3, 3).expand(batch_size, -1, -1)
            gate_center = self.gate_center.to(device=device, dtype=dtype).view(1, 3).expand(batch_size, -1)
        else:
            gate_pose = gate_pose.to(device=device, dtype=dtype).view(batch_size, 6)
            gate_rot = rpy_to_matrix_torch_batch(gate_pose[:, 0], gate_pose[:, 1], gate_pose[:, 2]).to(dtype=dtype)
            gate_center = gate_pose[:, 3:6]
        return gate_rot, gate_center, gate_rot[:, :, 0], gate_rot[:, :, 1], gate_rot[:, :, 2]

    def active_gate_center(self, start_pos, gate_rot, gate_center, gate_normal):
        if self.gate_count <= 1 or self.gate_spacing <= 1.0e-6:
            return gate_center

        base_local_x = th.bmm((start_pos - gate_center).unsqueeze(1), gate_rot).squeeze(1)[:, 0]
        gate_idx = th.round(base_local_x / self.gate_spacing).clamp(0, self.gate_count - 1)
        offsets = gate_idx.view(-1, 1) * self.gate_spacing * gate_normal
        return gate_center + offsets

    def body_axes_from_flatness(self, vel, acc, default_heading):
        thrust = acc + th.tensor([0.0, 0.0, self.gravity], dtype=acc.dtype, device=acc.device).view(1, 1, 3)
        b3 = thrust / thrust.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)

        heading = vel.clone()
        heading[:, :, 2] = 0.0
        default_heading = default_heading.to(dtype=vel.dtype, device=vel.device).clone()
        default_heading[:, 2] = 0.0
        fallback_heading = th.tensor([1.0, 0.0, 0.0], dtype=vel.dtype, device=vel.device).view(1, 3).expand_as(default_heading)
        default_heading = th.where(default_heading.norm(dim=-1, keepdim=True) < 1.0e-6,
                                   fallback_heading,
                                   default_heading)
        default_heading = default_heading / default_heading.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
        small = heading.norm(dim=-1, keepdim=True) < 1.0e-4
        heading = th.where(small, default_heading.view(vel.shape[0], 1, 3), heading)
        b1d = heading / heading.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)

        b2 = th.cross(b3, b1d, dim=-1)
        fallback = th.tensor([0.0, 1.0, 0.0], dtype=vel.dtype, device=vel.device).view(1, 1, 3)
        fallback_b2 = th.cross(b3, fallback.expand_as(b3), dim=-1)
        b2 = th.where(b2.norm(dim=-1, keepdim=True) < 1.0e-5, fallback_b2, b2)
        b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
        b1 = th.cross(b2, b3, dim=-1)
        b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp(min=1.0e-6)
        return b1, b2, b3

    def ellipsoid_support(self, axis, b1, b2, b3):
        r = self.body_radii.to(dtype=b1.dtype, device=b1.device)
        axis = axis.to(dtype=b1.dtype, device=b1.device).view(b1.shape[0], 1, 3)
        d1 = (b1 * axis).sum(dim=-1)
        d2 = (b2 * axis).sum(dim=-1)
        d3 = (b3 * axis).sum(dim=-1)
        return th.sqrt((r[0] * d1) ** 2 + (r[1] * d2) ** 2 + (r[2] * d3) ** 2 + 1.0e-9)

    def get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = th.zeros(Dp.shape[0], 18, dtype=Dp.dtype, device=Dp.device)
        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coe = (L @ d).squeeze(-1)
            coefficient[:, 6 * i: 6 * (i + 1)] = coe
        return coefficient

    def get_position_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1).squeeze(-2)
        x = th.sum(t_power * coe[:, 0: 6].unsqueeze(1), dim=-1)
        y = th.sum(t_power * coe[:, 6:12].unsqueeze(1), dim=-1)
        z = th.sum(t_power * coe[:, 12:18].unsqueeze(1), dim=-1)
        return th.stack([x, y, z], dim=-1)

    def get_velocity_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), 2 * t, 3 * t ** 2, 4 * t ** 3, 5 * t ** 4], dim=-1).squeeze(-2)
        vx = th.sum(t_power * coe[:, 1:6].unsqueeze(1), dim=-1)
        vy = th.sum(t_power * coe[:, 7:12].unsqueeze(1), dim=-1)
        vz = th.sum(t_power * coe[:, 13:18].unsqueeze(1), dim=-1)
        return th.stack([vx, vy, vz], dim=-1)

    def get_acceleration_from_coeff(self, coe, t):
        t_power = th.stack([2 * th.ones_like(t), 6 * t, 12 * t ** 2, 20 * t ** 3], dim=-1).squeeze(-2)
        ax = th.sum(t_power * coe[:, 2:6].unsqueeze(1), dim=-1)
        ay = th.sum(t_power * coe[:, 8:12].unsqueeze(1), dim=-1)
        az = th.sum(t_power * coe[:, 14:18].unsqueeze(1), dim=-1)
        return th.stack([ax, ay, az], dim=-1)


class GateObjectiveLoss(nn.Module):
    def __init__(self):
        super(GateObjectiveLoss, self).__init__()
        self.enabled = gate_enabled()
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        pose = list(cfg_get("gate_pose", [85.0, 0.0, 0.0, 0.0, 0.0, 1.2]))
        pose[0] = float(cfg_get("gate_slit_roll_deg", pose[0]))
        gate_rot = rpy_to_matrix_torch(pose[0], pose[1], pose[2], self.device).float()
        gate_center = th.tensor(pose[3:6], dtype=th.float32, device=self.device)

        self.register_buffer("gate_rot", gate_rot)
        self.register_buffer("gate_center", gate_center)
        self.register_buffer("gate_normal", gate_rot[:, 0].clone())

        self.gate_count = max(1, int(cfg_get("gate_count", 1)))
        self.gate_spacing = max(0.0, float(cfg_get("gate_spacing", 3.0)))
        self.goal_min_x = float(cfg_get("gate_goal_min_x", 1.20))
        self.goal_max_x = float(cfg_get("gate_goal_max_x", 2.60))
        self.inner_half_width = 0.5 * float(cfg_get("gate_inner_width", 0.64))
        self.inner_half_height = 0.5 * float(cfg_get("gate_inner_length", 0.22))
        self.terminal_weight = float(cfg_get("gate_objective_terminal_weight", 1.00))
        self.progress_weight = float(cfg_get("gate_objective_progress_weight", 1.20))
        self.center_weight = float(cfg_get("gate_objective_center_weight", 0.30))
        self.velocity_weight = float(cfg_get("gate_objective_velocity_weight", 0.15))
        self.lateral_velocity_weight = float(cfg_get("gate_objective_lateral_velocity_weight", 0.0))
        self.min_exit_speed_ratio = float(cfg_get("gate_objective_min_exit_speed_ratio", 0.20))
        self.vel_ref = float(cfg_get("vel_max_train", 6.0))

    def forward(self, Df, Dp, goal, active_mask=None, gate_pose=None):
        batch_size = Dp.shape[0]
        if not self.enabled:
            return th.zeros(batch_size, dtype=Dp.dtype, device=Dp.device)
        gate_rot, gate_center, gate_normal = self.gate_geometry(batch_size, Dp.dtype, Dp.device, gate_pose)

        start_pos = Df[:, :, 0]
        end_pos = Dp[:, :, 0]
        end_vel = Dp[:, :, 1]
        center = self.active_gate_center(start_pos, gate_rot, gate_center, gate_normal)

        local_start = th.bmm((start_pos - center).unsqueeze(1), gate_rot).squeeze(1)
        local_end = th.bmm((end_pos - center).unsqueeze(1), gate_rot).squeeze(1)
        local_goal = th.bmm((goal - center).unsqueeze(1), gate_rot).squeeze(1)
        local_vel = th.bmm(end_vel.unsqueeze(1), gate_rot).squeeze(1)

        terminal_scale = max(1.0e-3, self.goal_max_x)
        terminal = F.smooth_l1_loss((end_pos - goal) / terminal_scale,
                                    th.zeros_like(end_pos),
                                    reduction="none").sum(dim=1)

        start_sign = th.where(local_start[:, 0] >= 0.0,
                              th.ones_like(local_start[:, 0]),
                              -th.ones_like(local_start[:, 0]))
        desired_side_x = local_goal[:, 0].abs().clamp(min=self.goal_min_x, max=self.goal_max_x)
        through_x = -start_sign * local_end[:, 0]
        progress_over = F.relu(desired_side_x - through_x)
        progress = (progress_over / max(1.0e-3, self.goal_min_x)) ** 2

        center_y = local_end[:, 1] / max(1.0e-3, self.inner_half_width)
        center_z = local_end[:, 2] / max(1.0e-3, self.inner_half_height)
        center_cost = center_y ** 2 + center_z ** 2

        through_speed = -start_sign * local_vel[:, 0]
        min_exit_speed = self.min_exit_speed_ratio * self.vel_ref
        velocity_cost = (F.relu(min_exit_speed - through_speed) / max(1.0e-3, self.vel_ref)) ** 2
        lateral_velocity = (local_vel[:, 1] ** 2 + local_vel[:, 2] ** 2) / max(1.0e-3, self.vel_ref ** 2)

        objective = (self.terminal_weight * terminal +
                     self.progress_weight * progress +
                     self.center_weight * center_cost +
                     self.velocity_weight * velocity_cost +
                     self.lateral_velocity_weight * lateral_velocity)

        if active_mask is None:
            return objective
        active_mask = active_mask.to(device=Dp.device, dtype=Dp.dtype).view(-1)
        return objective * active_mask

    def gate_geometry(self, batch_size, dtype, device, gate_pose=None):
        if gate_pose is None:
            gate_rot = self.gate_rot.to(device=device, dtype=dtype).view(1, 3, 3).expand(batch_size, -1, -1)
            gate_center = self.gate_center.to(device=device, dtype=dtype).view(1, 3).expand(batch_size, -1)
        else:
            gate_pose = gate_pose.to(device=device, dtype=dtype).view(batch_size, 6)
            gate_rot = rpy_to_matrix_torch_batch(gate_pose[:, 0], gate_pose[:, 1], gate_pose[:, 2]).to(dtype=dtype)
            gate_center = gate_pose[:, 3:6]
        return gate_rot, gate_center, gate_rot[:, :, 0]

    def active_gate_center(self, start_pos, gate_rot, gate_center, gate_normal):
        if self.gate_count <= 1 or self.gate_spacing <= 1.0e-6:
            return gate_center

        base_local_x = th.bmm((start_pos - gate_center).unsqueeze(1), gate_rot).squeeze(1)[:, 0]
        gate_idx = th.round(base_local_x / self.gate_spacing).clamp(0, self.gate_count - 1)
        offsets = gate_idx.view(-1, 1) * self.gate_spacing * gate_normal
        return gate_center + offsets
