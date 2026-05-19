import torch as th
import torch.nn as nn
import torch.nn.functional as F
from config.config import cfg


class TrajectoryGuardLoss(nn.Module):
    def __init__(self):
        super(TrajectoryGuardLoss, self).__init__()
        self.min_progress = float(cfg["guard_min_progress"])
        self.min_vel_progress = float(cfg["guard_min_vel_progress"])
        self.min_height = float(cfg["guard_min_height"])
        self.max_height = float(cfg["guard_max_height"])
        self.max_vertical_speed = float(cfg["guard_max_vertical_speed"])
        self.max_terminal_speed = float(cfg["guard_max_terminal_speed"])
        self.lateral_weight = float(cfg["guard_lateral_weight"])

    def forward(self, Df, Dp, goal):
        """
        Penalize trajectories that are dynamically awkward for the SE(3)
        tracker even if their pure collision/smoothness costs are low.

        Args:
            Dp: (batch_size, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: (batch_size, 3, 3) -> [px, vx, ax; py, vy, ay; pz, vz, az]
            goal: (batch_size, 3)
        """
        cur_pos = Df[:, :, 0]
        end_pos = Dp[:, :, 0]
        end_vel = Dp[:, :, 1]

        goal_dir = goal - cur_pos
        goal_unit = goal_dir / goal_dir.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
        traj_dir = end_pos - cur_pos

        progress = (traj_dir * goal_unit).sum(dim=1)
        vel_progress = (end_vel * goal_unit).sum(dim=1)
        lateral = traj_dir - progress.unsqueeze(1) * goal_unit

        progress_cost = F.relu(self.min_progress - progress).square()
        reverse_vel_cost = F.relu(self.min_vel_progress - vel_progress).square()
        lateral_cost = lateral[:, 0:2].norm(dim=1).square()
        low_cost = F.relu(self.min_height - end_pos[:, 2]).square()
        high_cost = F.relu(end_pos[:, 2] - self.max_height).square()
        vz_cost = F.relu(end_vel[:, 2].abs() - self.max_vertical_speed).square()
        speed_cost = F.relu(end_vel.norm(dim=1) - self.max_terminal_speed).square()

        return (progress_cost + reverse_vel_cost +
                self.lateral_weight * lateral_cost +
                low_cost + high_cost + vz_cost + speed_cost)
