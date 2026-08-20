import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss


class YOPOOmniLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.topology_num = int(cfg["omni_topology_num"])
        self.sgm_time = float(cfg["sgm_time"])
        self.eval_points = int(cfg["omni_loss_eval_points"])
        self.use_guidance_loss = bool(cfg["use_guidance_loss"])
        self.camera_weight = float(cfg["w_camera"])
        self.camera_smooth_weight = float(cfg["w_camera_smooth"])
        self.active_camera = bool(cfg["active_camera"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._L, self._RJ, self._RA = self.qp_generation()
        self._L = self._L.to(self.device)
        self._RJ = self._RJ.to(self.device)
        self._RA = self._RA.to(self.device)

        self.denormalize_weight()
        self.smoothness_loss = SmoothnessLoss(self._RJ, self._RA)
        self.safety_loss = SafetyLoss(self._L)
        self.safety_loss.traj_num = self.topology_num

        camera_mode = "active" if self.active_camera else "fixed-zero-label"
        print("------ YOPO Camera-A/B Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'intent':<12} = {self.intent_weight:6.4f} |")
        print(f"| {'altitude':<12} = {self.altitude_weight:6.4f} |")
        print(f"| {'explore':<12} = {self.explore_weight:6.4f} | beta={self.explore_beta:.3g}")
        print(f"| {'guide':<12} = {self.guide_weight:6.4f} | enabled={self.use_guidance_loss}")
        print(f"| {'camera':<12} = {self.camera_weight:6.4f} | smooth={self.camera_smooth_weight:.4f}")
        print(f"| {'camera mode':<12} = {camera_mode} | shared head/loss weights")
        print("----------------------------")

    def forward(self, start_state_w, end_state_w, endstate_b, state_b, guide_path_w,
                guide_mask, selected_topology, map_id, pred_score,
                pred_camera_target, camera_target, camera_orientation):
        B, K = endstate_b.shape[:2]
        flat_start = start_state_w[:, None, :, :].expand(B, K, 3, 3).reshape(B * K, 3, 3)
        flat_end = end_state_w.reshape(B * K, 3, 3)

        Df = flat_start.permute(0, 2, 1)
        Dp = flat_end.permute(0, 2, 1)
        smooth_cost, acc_cost = self.smoothness_loss(Df, Dp)
        safety_cost = self.safety_loss(Df, Dp, map_id)

        smooth_cost = smooth_cost.reshape(B, K)
        acc_cost = acc_cost.reshape(B, K)
        safety_cost = safety_cost.reshape(B, K)
        intent_cost = self.intent_loss(endstate_b[..., 0:3], state_b[:, 6:9])
        explore_cost = self.explore_loss(endstate_b[..., 0:3])
        guide_cost = self.guidance_loss(Df, Dp, guide_path_w, guide_mask, selected_topology, B, K)
        altitude_cost = self.altitude_loss(start_state_w, end_state_w)

        base_cost = (
            self.smoothness_weight * smooth_cost
            + self.accele_weight * acc_cost
            + self.safety_weight * safety_cost
            + self.intent_weight * intent_cost
            + self.altitude_weight * altitude_cost
        )
        explore_loss = explore_cost.mean()
        trajectory_cost = base_cost + self.explore_weight * explore_cost
        total_cost = trajectory_cost + self.guide_weight * guide_cost
        guide_denom = guide_mask.sum().clamp(min=1.0)
        guide_loss = guide_cost.sum() / guide_denom
        trajectory_loss = trajectory_cost.mean() + self.guide_weight * guide_loss
        score_loss = F.smooth_l1_loss(pred_score, total_cost.detach())
        rank_loss = self.ranking_loss(pred_score, total_cost.detach())
        selected_camera = pred_camera_target.gather(
            1, selected_topology[:, None, None].expand(-1, 1, 2)
        ).squeeze(1)
        camera_loss = F.smooth_l1_loss(selected_camera, camera_target)
        camera_smooth_loss = (selected_camera - camera_orientation).square().mean()
        loss = (
            trajectory_loss
            + self.score_weight * score_loss
            + self.rank_weight * rank_loss
            + self.camera_weight * camera_loss
            + self.camera_smooth_weight * camera_smooth_loss
        )

        return loss, {
            "trajectory": trajectory_loss,
            "score": score_loss,
            "rank": rank_loss,
            "smooth": smooth_cost.mean(),
            "acc": acc_cost.mean(),
            "safety": safety_cost.mean(),
            "intent": intent_cost.mean(),
            "altitude": altitude_cost.mean(),
            "explore": explore_loss,
            "guide": guide_loss,
            "camera": camera_loss,
            "camera_smooth": camera_smooth_loss,
        }

    def intent_loss(self, end_pos_b, vdes_b):
        vdes_norm = F.normalize(vdes_b, dim=-1, eps=1e-6)
        progress = (end_pos_b * vdes_norm[:, None, :]).sum(dim=-1)
        lateral = end_pos_b - progress[..., None] * vdes_norm[:, None, :]
        return F.softplus(float(cfg["omni_intent_min_progress"]) - progress) + 0.1 * lateral.norm(dim=-1)

    @staticmethod
    def altitude_loss(start_state_w, end_state_w):
        """Keep the fixed-altitude YOPO-Simple task level in world coordinates."""
        delta_z = end_state_w[:, :, 0, 2] - start_state_w[:, None, 0, 2]
        end_vz = end_state_w[:, :, 1, 2]
        end_az = end_state_w[:, :, 2, 2]
        return delta_z.square() + 0.25 * end_vz.square() + 0.05 * end_az.square()

    def explore_loss(self, end_pos_b):
        if self.explore_weight <= 0:
            return torch.zeros(end_pos_b.shape[:2], device=end_pos_b.device, dtype=end_pos_b.dtype)
        squared_dist = end_pos_b.square().sum(dim=-1)
        return torch.exp(-squared_dist / self.explore_beta)

    def guidance_loss(self, Df, Dp, guide_path_w, guide_mask, selected_topology, B, K):
        if not self.use_guidance_loss:
            return torch.zeros((B, K), device=Dp.device)

        coe = self.get_coefficient_from_derivative(Dp, Df)
        dt = self.sgm_time / self.eval_points
        t_list = torch.linspace(dt, self.sgm_time, self.eval_points, device=Dp.device)
        t_list = t_list.view(1, -1, 1).expand(B * K, -1, -1)
        traj = self.get_position_from_coeff(coe, t_list).reshape(B, K, self.eval_points, 3)

        dist = torch.cdist(traj.reshape(B * K, self.eval_points, 3),
                           guide_path_w[:, None, :, :].expand(B, K, -1, -1).reshape(B * K, guide_path_w.shape[1], 3))
        nearest = dist.min(dim=-1).values.mean(dim=-1).reshape(B, K)
        selected = nearest.gather(1, selected_topology[:, None]).squeeze(1) * guide_mask
        guide_cost = torch.zeros((B, K), device=Dp.device)
        guide_cost.scatter_(1, selected_topology[:, None], selected[:, None])
        return guide_cost

    def ranking_loss(self, pred_score, target_cost):
        if self.rank_weight <= 0:
            return pred_score.new_tensor(0.0)
        diff = target_cost[:, :, None] - target_cost[:, None, :]
        score_diff = pred_score[:, :, None] - pred_score[:, None, :]
        valid = diff < -float(cfg["omni_rank_min_gap"])
        if not valid.any():
            return pred_score.new_tensor(0.0)
        return F.relu(float(cfg["omni_rank_margin"]) + score_diff[valid]).mean()

    def get_coefficient_from_derivative(self, Dp, Df):
        L = self._L.unsqueeze(0).expand(Dp.shape[0], -1, -1)
        coefficient = torch.zeros(Dp.shape[0], 18, device=Dp.device)
        for i in range(3):
            d = torch.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coe = (L @ d).squeeze(-1)
            coefficient[:, 6 * i: 6 * (i + 1)] = coe
        return coefficient

    @staticmethod
    def get_position_from_coeff(coe, t):
        t_power = torch.stack([torch.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1).squeeze(-2)
        x = torch.sum(t_power * coe[:, 0:6].unsqueeze(1), dim=-1)
        y = torch.sum(t_power * coe[:, 6:12].unsqueeze(1), dim=-1)
        z = torch.sum(t_power * coe[:, 12:18].unsqueeze(1), dim=-1)
        return torch.stack([x, y, z], dim=-1)

    def denormalize_weight(self):
        vel_scale = cfg["vel_max_train"] / 1.0
        self.smoothness_weight = cfg["ws"] / vel_scale ** 5
        self.accele_weight = cfg["wa"] / vel_scale ** 3
        self.safety_weight = cfg["wc"]
        self.intent_weight = cfg["wi"]
        self.altitude_weight = float(cfg["wh"])
        self.explore_weight = float(cfg["w_explore"])
        self.explore_beta = float(cfg["omni_explore_beta"])
        if self.explore_beta <= 0:
            raise ValueError(f"omni_explore_beta must be positive, got {self.explore_beta}")
        self.guide_weight = cfg["w_guide_path"] if self.use_guidance_loss else 0.0
        self.score_weight = cfg["w_score"]
        self.rank_weight = cfg["w_rank"]

    def qp_generation(self):
        A = torch.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))

        H = torch.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        Q = torch.zeros((6, 6))
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3) * (self.sgm_time ** (i + j - 3))

        Ct = torch.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1
        C = torch.transpose(Ct, 0, 1)
        B = torch.inverse(A)
        L = B @ Ct
        R_jerk = C @ torch.transpose(B, 0, 1) @ H @ B @ Ct
        R_acc = C @ torch.transpose(B, 0, 1) @ Q @ B @ Ct
        return L, R_jerk, R_acc
