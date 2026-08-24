import numpy as np


def _poly_powers(t, deriv):
    """Quintic basis and its first three derivatives for an arbitrary time array."""
    if deriv == 0:
        return np.stack([np.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], axis=-1)
    if deriv == 1:
        return np.stack([np.zeros_like(t), np.ones_like(t), 2 * t, 3 * t ** 2,
                         4 * t ** 3, 5 * t ** 4], axis=-1)
    if deriv == 2:
        return np.stack([np.zeros_like(t), np.zeros_like(t), 2 * np.ones_like(t), 6 * t,
                         12 * t ** 2, 20 * t ** 3], axis=-1)
    if deriv == 3:
        return np.stack([np.zeros_like(t), np.zeros_like(t), np.zeros_like(t),
                         6 * np.ones_like(t), 24 * t, 60 * t ** 2], axis=-1)
    raise ValueError(f"unsupported deriv={deriv}")


class QuinticTraj:
    """Single quintic segment with position/velocity/acceleration boundary constraints.

    This class deliberately exposes the same evaluator API as :class:`MincoTraj`.  It is used by
    controlled decoder ablations where the MINCO network output, score and selected candidate stay
    fixed and only the polynomial representation changes.
    """
    def __init__(self):
        self.coeffs = None
        self.durations = None
        self._batched = False

    @staticmethod
    def _build_A(duration):
        duration = np.asarray(duration, dtype=np.float64)
        B = duration.shape[0]
        A = np.zeros((B, 6, 6), dtype=np.float64)
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 2.0
        t1, t2, t3, t4, t5 = (duration, duration ** 2, duration ** 3,
                               duration ** 4, duration ** 5)
        A[:, 3] = np.stack([np.ones(B), t1, t2, t3, t4, t5], axis=-1)
        A[:, 4, 1:] = np.stack([np.ones(B), 2 * t1, 3 * t2, 4 * t3, 5 * t4], axis=-1)
        A[:, 5, 2:] = np.stack([2 * np.ones(B), 6 * t1, 12 * t2, 20 * t3], axis=-1)
        return A

    def solve(self, head_pva, tail_pva, duration):
        head_pva = np.asarray(head_pva, dtype=np.float64)
        tail_pva = np.asarray(tail_pva, dtype=np.float64)
        if head_pva.ndim == 2:
            head_pva = head_pva[None]
            tail_pva = tail_pva[None]
            self._batched = False
        else:
            self._batched = True
        B = head_pva.shape[0]
        duration = np.asarray(duration, dtype=np.float64)
        if duration.ndim == 0:
            duration = np.full(B, float(duration))
        elif duration.shape == (1,):
            duration = np.full(B, float(duration[0]))
        assert duration.shape == (B,), f"duration shape {duration.shape} != ({B},)"
        self.durations = duration
        rhs = np.concatenate((head_pva, tail_pva), axis=1)
        self.coeffs = np.linalg.solve(self._build_A(duration), rhs)
        return self

    @property
    def total_time(self):
        return self.durations if self._batched else float(self.durations[0])

    def _eval_matrix(self, times, deriv):
        times = np.asarray(times, dtype=np.float64)
        clamped = np.minimum(np.maximum(times, 0.0), self.durations[:, None])
        return np.einsum('bki,bij->bkj', _poly_powers(clamped, deriv), self.coeffs)

    def _eval(self, t, deriv):
        scalar = np.isscalar(t)
        values = np.atleast_1d(np.asarray(t, dtype=np.float64))
        result = self._eval_matrix(np.broadcast_to(values, (self.coeffs.shape[0], values.size)), deriv)
        if not self._batched:
            result = result[0]
        if scalar:
            result = result[0] if not self._batched else result[:, 0]
        return result

    def sample(self, fractions, deriv=0):
        """Evaluate at normalized times independently for every trajectory in a batch."""
        fractions = np.atleast_1d(np.asarray(fractions, dtype=np.float64))
        result = self._eval_matrix(self.durations[:, None] * fractions[None], deriv)
        return result if self._batched else result[0]

    def position(self, t):     return self._eval(t, 0)
    def velocity(self, t):     return self._eval(t, 1)
    def acceleration(self, t): return self._eval(t, 2)
    def jerk(self, t):         return self._eval(t, 3)


class MincoTraj:
    """
    Numpy 2-piece MincoS3NU evaluator for inference (single & batch). Training uses the torch
    solver in minco/; at inference the batch is one image worth of trajectories, small enough
    that numpy wins — no kernel launches, and no device sync stalling on a busy GPU.

    All boundary conditions live in WORLD frame and follow MincoS3NU layout:
        head_pva, tail_pva: shape (..., 3, 3) where dim -2 = (pos, vel, acc) rows
                            and dim -1 = (x, y, z) cols.
        inner_pos: shape (..., 3) — one inner waypoint per trajectory.

    Use:
        traj = MincoTraj().solve(head_pva, tail_pva, inner_pos, durations=[T0, T1])
        pos = traj.position(t)         # scalar t → (3,) or (B, 3); array t → (K, 3) or (B, K, 3)
    """
    def __init__(self):
        self.coeffs = None       # (B, 12, 3) after solve()
        self.durations = None    # (B, 2) per-piece durations
        self.cum_durations = None  # (B, 2) cumulative end-times
        self._batched = False

    @staticmethod
    def _build_A(T0, T1):
        """Batched 12×12 constraint matrices for 2-piece MincoS3NU. T0/T1: (B,) piece durations."""
        B = T0.shape[0]
        A = np.zeros((B, 12, 12))
        one = np.ones(B)
        # head pos / vel / acc
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 2.0
        # --- piece 0 at t = T0: inner waypoint + C^4 continuity to piece 1 ---
        t1, t2, t3, t4, t5 = T0, T0 ** 2, T0 ** 3, T0 ** 4, T0 ** 5
        A[:, 3, 0:6] = np.stack([one, t1, t2, t3, t4, t5], axis=-1)              # inner waypoint position
        A[:, 4, 0:6] = A[:, 3, 0:6]                                              # pos continuity
        A[:, 4, 6] = -1.0
        A[:, 5, 1:6] = np.stack([one, 2 * t1, 3 * t2, 4 * t3, 5 * t4], axis=-1)  # vel continuity
        A[:, 5, 7] = -1.0
        A[:, 6, 2:6] = np.stack([2 * one, 6 * t1, 12 * t2, 20 * t3], axis=-1)    # acc continuity
        A[:, 6, 8] = -2.0
        A[:, 7, 3:6] = np.stack([6 * one, 24 * t1, 60 * t2], axis=-1)            # jerk continuity
        A[:, 7, 9] = -6.0
        A[:, 8, 4:6] = np.stack([24 * one, 120 * t1], axis=-1)                   # snap continuity
        A[:, 8, 10] = -24.0
        # --- piece 1 at t = T1: tail pos / vel / acc ---
        t1, t2, t3, t4, t5 = T1, T1 ** 2, T1 ** 3, T1 ** 4, T1 ** 5
        A[:, 9, 6:12] = np.stack([one, t1, t2, t3, t4, t5], axis=-1)
        A[:, 10, 7:12] = np.stack([one, 2 * t1, 3 * t2, 4 * t3, 5 * t4], axis=-1)
        A[:, 11, 8:12] = np.stack([2 * one, 6 * t1, 12 * t2, 20 * t3], axis=-1)
        return A

    def solve(self, head_pva, tail_pva, inner_pos, durations):
        """Solve the 2-piece quintic coefficients from world-frame boundary conditions (single or
        batched). durations: (2,) shared by every trajectory, or (B, 2) per trajectory."""
        head_pva = np.asarray(head_pva, dtype=np.float64)
        tail_pva = np.asarray(tail_pva, dtype=np.float64)
        inner_pos = np.asarray(inner_pos, dtype=np.float64)
        if head_pva.ndim == 2:
            head_pva = head_pva[None]
            tail_pva = tail_pva[None]
            inner_pos = inner_pos[None]
            self._batched = False
        else:
            self._batched = True
        B = head_pva.shape[0]

        durations = np.asarray(durations, dtype=np.float64)
        if durations.ndim == 1:
            durations = np.broadcast_to(durations, (B, 2)).copy()
        assert durations.shape == (B, 2), f"durations shape {durations.shape} != ({B}, 2)"
        self.durations = durations
        self.cum_durations = np.cumsum(durations, axis=1)                 # (B, 2)

        # Build b matrix
        b = np.zeros((B, 12, 3), dtype=np.float64)
        b[:, 0:3] = head_pva
        b[:, 3] = inner_pos
        b[:, 9:12] = tail_pva

        self.coeffs = np.linalg.solve(self._build_A(durations[:, 0], durations[:, 1]), b)
        return self

    @property
    def total_time(self):
        """Total trajectory time. Returns scalar for single, (B,) for batched."""
        total = self.cum_durations[:, -1]
        return total if self._batched else float(total[0])

    def _eval(self, t, deriv):
        """Evaluate the trajectory (deriv 0/1/2/3 = pos/vel/acc/jerk) at time(s) t, per-batch piece selection
        with time clamped to [0, total]. Returns (3,)/(K,3) for single, (B,3)/(B,K,3) for batched."""
        if np.isscalar(t):
            t_arr = np.array([t], dtype=np.float64)
            squeeze_k = True
        else:
            t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
            squeeze_k = False

        # Per-batch piece resolution: piece_idx[b, k] ∈ {0, 1}
        # rel_t[b, k] = t_arr[k] - (0 if piece==0 else durations[b, 0])
        B = self.coeffs.shape[0]
        total_per_b = self.cum_durations[:, -1]                                  # (B,)
        t_clamped = np.clip(t_arr[None, :], 0.0, total_per_b[:, None] - 1e-9)    # (B, K)
        piece_idx = (t_clamped >= self.durations[:, 0:1]).astype(int)            # (B, K)
        rel_t = t_clamped - piece_idx * self.durations[:, 0:1]                   # (B, K)

        powers = _poly_powers(rel_t, deriv)
        # powers: (B, K, 6)

        # Gather the per-piece coeff block, then contract on the 6 axis.
        # piece_coeffs[b, k, :, :] = coeffs[b, piece_idx[b, k]*6:(piece_idx[b, k]+1)*6, :]
        b_idx = np.arange(B)[:, None]                       # (B, 1)
        row_idx = piece_idx[:, :, None] * 6 + np.arange(6)[None, None, :]  # (B, K, 6)
        piece_coeffs = self.coeffs[b_idx[:, :, None], row_idx, :]          # (B, K, 6, 3)
        result = np.einsum('bki,bkij->bkj', powers, piece_coeffs)          # (B, K, 3)

        if not self._batched:
            result = result[0]                  # (K, 3)
        if squeeze_k:
            result = result[0] if not self._batched else result[:, 0, :]
        return result

    def position(self, t):     return self._eval(t, 0)
    def velocity(self, t):     return self._eval(t, 1)
    def acceleration(self, t): return self._eval(t, 2)
    def jerk(self, t):         return self._eval(t, 3)

    def sample(self, fractions, deriv=0):
        """Evaluate at normalized times independently for every trajectory in a batch."""
        fractions = np.atleast_1d(np.asarray(fractions, dtype=np.float64))
        times = self.cum_durations[:, -1:] * fractions[None]
        piece_idx = (times >= self.durations[:, 0:1]).astype(int)
        rel_t = times - piece_idx * self.durations[:, 0:1]
        powers = _poly_powers(rel_t, deriv)
        B = self.coeffs.shape[0]
        b_idx = np.arange(B)[:, None]
        row_idx = piece_idx[:, :, None] * 6 + np.arange(6)[None, None]
        piece_coeffs = self.coeffs[b_idx[:, :, None], row_idx]
        result = np.einsum('bki,bkij->bkj', powers, piece_coeffs)
        return result if self._batched else result[0]


def wrap_to_pi(angle):
    """将角度限制在 [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def calculate_yaw(vel_dir, goal_dir, last_yaw, dt, max_yaw_rate=0.5, goal_weight_scale=2.0):
    """Desired yaw + yaw-rate: blend the velocity heading with the goal heading (goal weight grows
    with heading error, ~equal at 60°), then rate-limit the change to max_yaw_rate·π per second."""
    # Normalize velocity and goal directions
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)
    goal_dist = np.linalg.norm(goal_dir)
    goal_dir = goal_dir / (goal_dist + 1e-5)

    # Goal yaw and weighting
    goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
    delta_yaw = wrap_to_pi(goal_yaw - last_yaw)
    weight = goal_weight_scale * abs(delta_yaw) / np.pi

    # Desired direction and yaw
    dir_des = vel_dir + weight * goal_dir
    yaw_desired = np.arctan2(dir_des[1], dir_des[0]) if goal_dist > 0.5 else last_yaw

    # Yaw difference and limit
    yaw_diff = wrap_to_pi(yaw_desired - last_yaw)
    max_yaw_change = max_yaw_rate * np.pi * dt
    yaw_change = np.clip(yaw_diff, -max_yaw_change, max_yaw_change)

    # Updated yaw and yaw rate
    yaw = wrap_to_pi(last_yaw + yaw_change)
    yawdot = yaw_change / dt

    return yaw, yawdot
