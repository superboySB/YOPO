import math
import torch as th
import torch.nn as nn
from config.config import cfg
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.guidance_loss import GuidanceLoss


class YOPOLoss(nn.Module):
    def __init__(self):
        """
        Compute the cost: including smoothness, safety, guidance, goal cost, etc.
        Currently, keeping multi-segment polynomial support (not yet verified), but only using a single-segment polynomial (m = 1) for now.
        dp: decision parameters
        df: fixed parameters
        """
        super(YOPOLoss, self).__init__()
        self.sgm_time = cfg["sgm_time"]
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self._C, self._B, self._L, self._RJ, self._RA = self.qp_generation()
        self._RJ = self._RJ.to(self.device)
        self._RA = self._RA.to(self.device)
        self._L = self._L.to(self.device)
        self.denormalize_weight()
        self.smoothness_loss = SmoothnessLoss(self._RJ, self._RA)
        self.safety_loss = SafetyLoss(self._L)
        self.goal_loss = GuidanceLoss()
        print("------ Loss Weights ------")
        print(f"| {'smooth raw':<16} = {self.raw_smoothness_weight:8.4f} | effective = {self.smoothness_weight:8.6f} |")
        print(f"| {'accel raw':<16} = {self.raw_acceleration_weight:8.4f} | effective = {self.acceleration_weight:8.6f} |")
        print(f"| {'static safety':<16} = {self.static_safety_weight:8.4f} |")
        print(f"| {'dynamic safety':<16} = {self.dynamic_safety_weight:8.4f} |")
        print(f"| {'goal':<16} = {self.goal_weight:8.4f} |")
        print("--------------------------")

    def qp_generation(self):
        # 论文中的映射矩阵
        A = th.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))

        # H海森矩阵，对应Jerk
        H = th.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        # Q海森矩阵，对应Accel
        Q = th.zeros((6, 6))
        for i in range(2, 6):
            for j in range(2, 6):
                Q[i, j] = (i * (i - 1)) * (j * (j - 1)) / (i + j - 3) * (self.sgm_time ** (i + j - 3))

        return self.stack_opt_dep(A, H, Q)

    def stack_opt_dep(self, A, H, Q):
        Ct = th.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1

        _C = th.transpose(Ct, 0, 1)

        B = th.inverse(A)

        B_T = th.transpose(B, 0, 1)

        _L = B @ Ct

        _R_Jerk = _C @ (B_T) @ H @ B @ Ct

        _R_Acc = _C @ (B_T) @ Q @ B @ Ct

        return _C, B, _L, _R_Jerk, _R_Acc

    def denormalize_weight(self):
        """
        Denormalize the cost weight to ensure consistency across different speeds to simplify parameter tuning.
        smoothness cost: time integral of jerk² is used as a smoothness cost.
                         If the speed is scaled by n, the cost is scaled by n⁵ (because jerk * n⁶ and time * 1/n).
        safety cost:     time integral of the distance from trajectory to obstacles.
                         If the speed is scaled by n, the cost is scaled by 1/n (because time * 1/n).
        goal cost:       projection of the trajectory onto goal direction.
                         Independent of speed.
        """
        vel_scale = cfg["vel_max_train"] / 1.0
        self.raw_smoothness_weight = float(cfg.get("smoothness_weight", cfg.get("ws", 10.0)))
        self.raw_acceleration_weight = float(cfg.get("acceleration_weight", cfg.get("wa", 1.0)))
        self.smoothness_weight = self.raw_smoothness_weight / vel_scale ** 5
        self.acceleration_weight = self.raw_acceleration_weight / vel_scale ** 3
        self.accele_weight = self.acceleration_weight
        self.static_safety_weight = float(cfg.get("static_safety_weight", cfg.get("wc", 1.0)))
        self.dynamic_safety_weight = float(cfg.get("dynamic_safety_weight", cfg.get("wc", 1.0)))
        self.goal_weight = float(cfg.get("goal_weight", cfg.get("wg", 0.15)))

    def forward(self, state, prediction, goal, map_id, dynamic_target_w=None, dynamic_target_visible=None):
        """
        Args:
            prediction: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            state: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame
            map_id: (batch_size) which ESDF map to query

        Returns:
            cost components: each cost tensor is (batch_size)
        """
        # Fixed part: initial pos, vel, acc → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Df = state.permute(0, 2, 1)

        # Decision parameters (local frame) → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Dp = prediction.permute(0, 2, 1)

        smoothness_cost, acceleration_cost = self.smoothness_loss(Df, Dp)
        safety_components = self.safety_loss.forward_components(
            Df, Dp, map_id, dynamic_target_w, dynamic_target_visible
        )
        goal_cost = self.goal_loss(Df, Dp, goal)

        static_safety_cost = safety_components["static_cost"]
        dynamic_safety_cost = safety_components["dynamic_cost"]
        return {
            "smoothness": self.smoothness_weight * smoothness_cost,
            "static_safety": self.static_safety_weight * static_safety_cost,
            "dynamic_safety": self.dynamic_safety_weight * dynamic_safety_cost,
            "goal": self.goal_weight * goal_cost,
            "acceleration": self.acceleration_weight * acceleration_cost,
            "raw_smoothness": smoothness_cost,
            "raw_static_safety": static_safety_cost,
            "raw_dynamic_safety": dynamic_safety_cost,
            "raw_goal": goal_cost,
            "raw_acceleration": acceleration_cost,
            "static_min_distance": safety_components["static_min_distance"],
            "dynamic_min_distance": safety_components["dynamic_min_distance"],
        }
