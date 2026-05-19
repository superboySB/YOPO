import torch as th
import torch.nn as nn
from config.config import cfg


class BodyRateLoss(nn.Module):
    def __init__(self, L):
        super(BodyRateLoss, self).__init__()
        self._L = L
        self.sgm_time = cfg["sgm_time"]
        self.eval_points = int(cfg["bdr_eval_points"])
        self.mass = float(cfg["vehicle_mass"])
        self.grav_acc = float(cfg["grav_acc"])
        self.horiz_drag = float(cfg["se3_horiz_drag"])
        self.paras_drag = float(cfg["se3_parasitic_drag"])
        self.speed_eps = float(cfg["se3_speed_eps"])
        self.max_bdr_mag = float(cfg["max_bdr_mag"])
        self.smooth_eps = float(cfg["bdr_smooth_eps"])

    def forward(self, Df, Dp):
        """
        Penalize trajectories whose SE(3) flatness body-rate exceeds MaxBdrMag.

        Args:
            Dp: decision parameters: (batch_size, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters:    (batch_size, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]

        Returns:
            body_rate_cost: (batch_size)
        """
        batch_size = Dp.shape[0]
        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        coe = self.get_coefficient_from_derivative(Dp, Df, L)

        dt = self.sgm_time / self.eval_points
        t_list = th.linspace(dt, self.sgm_time, self.eval_points,
                             device=Dp.device, dtype=Dp.dtype)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)

        vel = self.get_velocity_from_coeff(coe, t_list)
        acc = self.get_acceleration_from_coeff(coe, t_list)
        jerk = self.get_jerk_from_coeff(coe, t_list)
        body_rate = self.flatness_body_rate(vel, acc, jerk)

        violation = body_rate.square().sum(dim=-1) - self.max_bdr_mag ** 2
        return self.smoothed_l1_positive(violation, self.smooth_eps).mean(dim=-1)

    def flatness_body_rate(self, vel, acc, jerk):
        v0, v1, v2 = vel[..., 0], vel[..., 1], vel[..., 2]
        a0, a1, a2 = acc[..., 0], acc[..., 1], acc[..., 2]

        cp_term = th.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + self.speed_eps)
        w_term = 1.0 + self.paras_drag * cp_term
        w0, w1, w2 = w_term * v0, w_term * v1, w_term * v2
        dh_over_m = self.horiz_drag / self.mass

        zu0 = a0 + dh_over_m * w0
        zu1 = a1 + dh_over_m * w1
        zu2 = a2 + dh_over_m * w2 + self.grav_acc
        zu_sqr0 = zu0 * zu0
        zu_sqr1 = zu1 * zu1
        zu_sqr2 = zu2 * zu2
        zu01 = zu0 * zu1
        zu12 = zu1 * zu2
        zu02 = zu0 * zu2
        zu_sqr_norm = zu_sqr0 + zu_sqr1 + zu_sqr2
        zu_norm = th.sqrt(zu_sqr_norm.clamp_min(1.0e-12))

        ng_den = (zu_sqr_norm * zu_norm).clamp_min(1.0e-12)
        ng00 = (zu_sqr1 + zu_sqr2) / ng_den
        ng01 = -zu01 / ng_den
        ng02 = -zu02 / ng_den
        ng11 = (zu_sqr0 + zu_sqr2) / ng_den
        ng12 = -zu12 / ng_den
        ng22 = (zu_sqr0 + zu_sqr1) / ng_den

        v_dot_a = v0 * a0 + v1 * a1 + v2 * a2
        dw_term = self.paras_drag * v_dot_a / cp_term.clamp_min(1.0e-6)
        dw0 = w_term * a0 + dw_term * v0
        dw1 = w_term * a1 + dw_term * v1
        dw2 = w_term * a2 + dw_term * v2
        dz_term0 = jerk[..., 0] + dh_over_m * dw0
        dz_term1 = jerk[..., 1] + dh_over_m * dw1
        dz_term2 = jerk[..., 2] + dh_over_m * dw2
        dz0 = ng00 * dz_term0 + ng01 * dz_term1 + ng02 * dz_term2
        dz1 = ng01 * dz_term0 + ng11 * dz_term1 + ng12 * dz_term2
        dz2 = ng02 * dz_term0 + ng12 * dz_term1 + ng22 * dz_term2

        z0 = zu0 / zu_norm
        z1 = zu1 / zu_norm
        z2 = zu2 / zu_norm

        # Match gcopter's angular-rate penalty path: yaw=0, yaw_dot=0.
        omg_den = (z2 + 1.0).clamp_min(1.0e-6)
        omg_x = -dz1 + z1 * dz2 / omg_den
        omg_y = dz0 - z0 * dz2 / omg_den
        omg_z = (z1 * dz0 - z0 * dz1) / omg_den
        return th.stack([omg_x, omg_y, omg_z], dim=-1)

    def smoothed_l1_positive(self, x, mu):
        zero = th.zeros_like(x)
        linear = x - 0.5 * mu
        xdmu = x / mu
        smooth = (mu - 0.5 * x) * xdmu.square() * xdmu
        return th.where(x <= 0.0, zero, th.where(x > mu, linear, smooth))

    def get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = th.zeros(Dp.shape[0], 18, device=Dp.device, dtype=Dp.dtype)
        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coe = (L @ d).squeeze(-1)
            coefficient[:, 6 * i: 6 * (i + 1)] = coe
        return coefficient

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

    def get_jerk_from_coeff(self, coe, t):
        t_power = th.stack([6 * th.ones_like(t), 24 * t, 60 * t ** 2], dim=-1).squeeze(-2)
        jx = th.sum(t_power * coe[:, 3:6].unsqueeze(1), dim=-1)
        jy = th.sum(t_power * coe[:, 9:12].unsqueeze(1), dim=-1)
        jz = th.sum(t_power * coe[:, 15:18].unsqueeze(1), dim=-1)
        return th.stack([jx, jy, jz], dim=-1)
