import torch.nn as nn
import torch as th
import torch.nn.functional as F
from config.config import cfg


class GuidanceLoss(nn.Module):
    def __init__(self):
        super(GuidanceLoss, self).__init__()
        self.goal_length = cfg['goal_length']
        self.vel_max = float(cfg["vel_max_train"])
        self.acc_max = float(cfg["acc_max_train"])
        self.vel_dir_weight = float(cfg.get("guidance_vel_dir_weight", 0.0))
        self.speed_weight = float(cfg.get("guidance_speed_weight", 0.0))
        self.speed_target_ratio = float(cfg.get("guidance_speed_target_ratio", 0.0))
        self.speed_activation_goal_ratio = float(cfg.get("guidance_speed_activation_goal_ratio", 0.5))
        self.speed_huber_delta = float(cfg.get("guidance_speed_huber_delta", 0.0))
        self.perp_weight_near = float(cfg.get("guidance_perp_weight_near", 0.5))
        self.perp_weight_far = float(cfg.get("guidance_perp_weight_far", 0.5))
        self.perp_activation_goal_ratio = float(cfg.get("guidance_perp_activation_goal_ratio", 0.5))
        self.speed_profile_weight = float(cfg.get("guidance_speed_profile_weight", 0.0))
        self.speed_profile_brake_acc_ratio = float(cfg.get("guidance_speed_profile_brake_acc_ratio", 0.55))
        self.speed_profile_buffer = float(cfg.get("guidance_speed_profile_buffer", 8.0))
        self.speed_profile_reach_radius = float(cfg.get("guidance_speed_profile_reach_radius", 5.0))
        self.speed_profile_activation_goal_ratio = float(cfg.get("guidance_speed_profile_activation_goal_ratio", 0.5))
        self.speed_profile_huber_delta = float(cfg.get("guidance_speed_profile_huber_delta", 0.0))
        self.reverse_speed_weight = float(cfg.get("guidance_reverse_speed_weight", 0.0))
        self.reverse_speed_margin = float(cfg.get("guidance_reverse_speed_margin", 0.0))
        self.reverse_activation_goal_ratio = float(cfg.get("guidance_reverse_activation_goal_ratio", 0.4))
        self.normalize_by_goal_length = bool(cfg.get("guidance_normalize_by_goal_length", False))
        self.guidance_ref_length = float(cfg.get("guidance_ref_length", self.goal_length))

    def forward(self, Df, Dp, goal):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            goal: (batch_size, 3)
        Returns:
            guidance_loss: (batch_size) → guidance loss

        GuidanceLoss: distance_loss (for straighter flight) or similarity_loss (for faster flight in large scenario)
        """
        cur_pos = Df[:, :, 0]
        end_pos = Dp[:, :, 0]
        end_vel = Dp[:, :, 1]

        traj_dir = end_pos - cur_pos  # [B, 3]
        goal_dir = goal - cur_pos  # [B, 3]

        # guidance_loss = self.distance_loss(traj_dir, goal_dir)
        guidance_loss = self.similarity_loss(traj_dir, goal_dir)
        if self.normalize_by_goal_length:
            # Keep guidance cost magnitude comparable when planning horizon/goal_length changes.
            guidance_loss = guidance_loss * (self.guidance_ref_length / max(self.goal_length, 1e-6))

        if self.vel_dir_weight > 0:
            vel_dir_loss = self.derivative_similarity_loss(end_vel, goal_dir)
            guidance_loss += self.vel_dir_weight * vel_dir_loss

        if self.speed_weight > 0 and self.speed_target_ratio > 0:
            speed_loss = self.forward_speed_loss(end_vel, goal_dir)
            guidance_loss += self.speed_weight * speed_loss
        if self.speed_profile_weight > 0:
            speed_profile_loss = self.forward_speed_profile_loss(end_vel, goal_dir)
            guidance_loss += self.speed_profile_weight * speed_profile_loss
        if self.reverse_speed_weight > 0:
            reverse_speed_loss = self.reverse_speed_loss(end_vel, goal_dir)
            guidance_loss += self.reverse_speed_weight * reverse_speed_loss

        return guidance_loss

    def distance_loss(self, traj_dir, goal_dir):
        """
        Returns:
            l1_distance: (batch_size) → guidance loss

        L1Loss: L1 distance (same scale as the similarity loss) to the normalized goal (for numerical stability).
                closer to the goal is preferred.
        Straighter flight and more precise near the goal, but slightly inferior in flight speed.
        """
        l1_distance = F.smooth_l1_loss(traj_dir, goal_dir, reduction='none')  # shape: (B, 3)
        l1_distance = l1_distance.sum(dim=1)  # (B)
        return l1_distance

    def similarity_loss(self, traj_dir, goal_dir):
        """
        Returns:
            similarity: (batch_size) → guidance loss

        SimilarityLoss: Projection length of the trajectory onto the goal direction:
                        higher cosine similarity and longer trajectory are preferred.

        Adjust perp_weight to penalize deviation perpendicular to the goal; equals the distance_loss() when perp_weight = 1.
        """
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)  # [B, 3]

        # projection length of trajectory on goal direction
        traj_along = (traj_dir * goal_dir_norm).sum(dim=1)  # [B]
        goal_length = goal_dir.norm(dim=1)  # [B]

        # length difference along goal direction (cosine similarity)
        parallel_diff = F.smooth_l1_loss(goal_length, traj_along, reduction='none')

        # length perpendicular to goal direction
        traj_perp = traj_dir - traj_along.unsqueeze(1) * goal_dir_norm  # [B, 3]
        perp_diff = traj_perp.norm(dim=1)  # [B]

        # Distance-adaptive perpendicular weighting:
        # far goal: smaller weight (allow larger high-speed lateral maneuvering)
        # near goal: larger weight (enforce tighter convergence to goal line)
        goal_dist = goal_dir.norm(dim=1)  # [B]
        activation_dist = self.perp_activation_goal_ratio * self.goal_length
        near_gate = (1.0 - goal_dist / max(activation_dist, 1e-6)).clamp(0.0, 1.0)
        perp_weight = self.perp_weight_far + (self.perp_weight_near - self.perp_weight_far) * near_gate
        similarity_loss = parallel_diff + perp_weight * perp_diff
        return similarity_loss

    def derivative_similarity_loss(self, derivative, goal_dir):
        """
            Constrain the velocity direction toward the goal
        """
        goal_dir_norm = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)  # [B, 3]
        derivative_norm = derivative / (derivative.norm(dim=1, keepdim=True) + 1e-8)  # [B, 3]

        similarity = (derivative_norm * goal_dir_norm).sum(dim=1)  # [B]
        return 1 - similarity

    def forward_speed_loss(self, end_vel, goal_dir):
        """
            Encourage high forward speed for far-goal samples.
            This term is active only when the goal is sufficiently far away.
        """
        goal_dist = goal_dir.norm(dim=1)  # [B]
        goal_dir_norm = goal_dir / (goal_dist.unsqueeze(1) + 1e-8)
        forward_speed = (end_vel * goal_dir_norm).sum(dim=1)  # [B]

        target_speed = self.speed_target_ratio * self.vel_max
        speed_gap = th.relu(target_speed - forward_speed)

        if self.speed_huber_delta > 0:
            delta = th.full_like(speed_gap, self.speed_huber_delta)
            quad = th.minimum(speed_gap, delta)
            linear = speed_gap - quad
            speed_loss = 0.5 * quad * quad + self.speed_huber_delta * linear
        else:
            speed_loss = speed_gap * speed_gap

        # Soft gate: no speed push near goal, fully active for far-goal supervision.
        activation_dist = self.speed_activation_goal_ratio * self.goal_length
        denom = max(self.goal_length - activation_dist, 1e-6)
        gate = ((goal_dist - activation_dist) / denom).clamp(0.0, 1.0)
        return gate * speed_loss

    def forward_speed_profile_loss(self, end_vel, goal_dir):
        """
            Near goal, constrain forward speed by braking-feasible profile:
                v_allowed(d) = sqrt(2 a_brake max(d - buffer, 0))
            This aligns training objective with high-speed terminal stop behavior in test stage.
        """
        goal_dist = goal_dir.norm(dim=1)  # [B]
        goal_dir_norm = goal_dir / (goal_dist.unsqueeze(1) + 1e-8)
        forward_speed = (end_vel * goal_dir_norm).sum(dim=1)  # [B]

        brake_acc = max(0.5, self.speed_profile_brake_acc_ratio * self.acc_max)
        brake_dist = th.relu(goal_dist - self.speed_profile_buffer)
        allowed_speed = th.sqrt(2.0 * brake_acc * brake_dist + 1e-8)
        # Keep a small near-goal crawl allowance.
        if self.speed_profile_reach_radius > 0:
            crawl_speed = th.sqrt(th.tensor(
                max(0.0, 2.0 * brake_acc * self.speed_profile_reach_radius),
                dtype=allowed_speed.dtype, device=allowed_speed.device
            )) * 0.1
            allowed_speed = th.maximum(allowed_speed, crawl_speed)

        overspeed = th.relu(forward_speed - allowed_speed)
        if self.speed_profile_huber_delta > 0:
            delta = th.full_like(overspeed, self.speed_profile_huber_delta)
            quad = th.minimum(overspeed, delta)
            linear = overspeed - quad
            overspeed_loss = 0.5 * quad * quad + self.speed_profile_huber_delta * linear
        else:
            overspeed_loss = overspeed * overspeed

        activation_dist = self.speed_profile_activation_goal_ratio * self.goal_length
        near_gate = (1.0 - goal_dist / max(activation_dist, 1e-6)).clamp(0.0, 1.0)
        return near_gate * overspeed_loss

    def reverse_speed_loss(self, end_vel, goal_dir):
        """
            Penalize anti-goal terminal velocity in far-goal supervision.
            This directly targets the 'mid-course random turning / reverse motion' failure mode.
        """
        goal_dist = goal_dir.norm(dim=1)  # [B]
        goal_dir_norm = goal_dir / (goal_dist.unsqueeze(1) + 1e-8)
        forward_speed = (end_vel * goal_dir_norm).sum(dim=1)  # [B]
        reverse_gap = th.relu(self.reverse_speed_margin - forward_speed)
        reverse_pen = reverse_gap * reverse_gap

        activation_dist = self.reverse_activation_goal_ratio * self.goal_length
        denom = max(self.goal_length - activation_dist, 1e-6)
        far_gate = ((goal_dist - activation_dist) / denom).clamp(0.0, 1.0)
        return far_gate * reverse_pen
