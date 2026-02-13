import torch as th
import torch.nn as nn
from config.config import cfg


class DynamicsLoss(nn.Module):
    """
    Trajectory-level dynamic feasibility penalty.
    Penalize velocity/acceleration norm violations along the whole polynomial segment
    (instead of only constraining terminal states).
    """

    def __init__(self, L):
        super(DynamicsLoss, self).__init__()
        self._L = L
        self.device = self._L.device
        self.sgm_time = cfg["sgm_time"]
        self.eval_points = int(cfg.get("dynamic_eval_points", 30))
        self.vel_limit = float(cfg.get("dynamic_vel_limit", cfg["vel_max_train"]))
        self.acc_limit = float(cfg.get("dynamic_acc_limit", cfg["acc_max_train"]))
        # Penalty starts at (limit - margin), offering a safety buffer for tracking errors.
        self.vel_margin = float(cfg.get("dynamic_vel_margin", 0.0))
        self.acc_margin = float(cfg.get("dynamic_acc_margin", 0.0))
        # Normalize violations by limit to keep loss scale stable across speeds.
        self.normalize_violation = bool(cfg.get("dynamic_normalize_violation", True))
        self.ratio_clip = float(cfg.get("dynamic_ratio_clip", 5.0))

    def forward(self, Df, Dp):
        """
        Args:
            Dp: decision parameters: (B, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters:    (B, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
        Returns:
            vel_cost: (B) mean velocity-limit violation along trajectory
            acc_cost: (B) mean acceleration-limit violation along trajectory
        """
        batch_size = Dp.shape[0]
        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        coe = self.get_coefficient_from_derivative(Dp, Df, L)

        dt = self.sgm_time / self.eval_points
        t_list = th.linspace(dt, self.sgm_time, self.eval_points, device=self.device)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)

        vel = self.get_velocity_from_coeff(coe, t_list)
        acc = self.get_acceleration_from_coeff(coe, t_list)
        vel_norm = vel.norm(dim=-1)
        acc_norm = acc.norm(dim=-1)

        vel_threshold = max(self.vel_limit - self.vel_margin, 1e-3)
        acc_threshold = max(self.acc_limit - self.acc_margin, 1e-3)
        vel_over = th.relu(vel_norm - vel_threshold)
        acc_over = th.relu(acc_norm - acc_threshold)

        if self.normalize_violation:
            vel_over = vel_over / vel_threshold
            acc_over = acc_over / acc_threshold

        if self.ratio_clip > 0.0:
            vel_over = th.clamp(vel_over, max=self.ratio_clip)
            acc_over = th.clamp(acc_over, max=self.ratio_clip)

        vel_cost = (vel_over * vel_over).mean(dim=1)
        acc_cost = (acc_over * acc_over).mean(dim=1)
        return vel_cost, acc_cost

    def get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = th.zeros(Dp.shape[0], 18, device=self.device)
        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coe = (L @ d).squeeze(-1)
            coefficient[:, 6 * i: 6 * (i + 1)] = coe
        return coefficient

    def get_velocity_from_coeff(self, coe, t):
        t_base = th.stack([th.ones_like(t), t, t ** 2, t ** 3, t ** 4], dim=-1).squeeze(-2)
        vel_scale = t_base.new_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        t_power = t_base * vel_scale

        coe_x = coe[:, 1:6]
        coe_y = coe[:, 7:12]
        coe_z = coe[:, 13:18]

        vx = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        vy = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        vz = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)
        return th.stack([vx, vy, vz], dim=-1)

    def get_acceleration_from_coeff(self, coe, t):
        t_base = th.stack([th.ones_like(t), t, t ** 2, t ** 3], dim=-1).squeeze(-2)
        acc_scale = t_base.new_tensor([2.0, 6.0, 12.0, 20.0])
        t_power = t_base * acc_scale

        coe_x = coe[:, 2:6]
        coe_y = coe[:, 8:12]
        coe_z = coe[:, 14:18]

        ax = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        ay = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        az = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)
        return th.stack([ax, ay, az], dim=-1)
