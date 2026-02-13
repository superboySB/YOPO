import torch as th
import torch.nn as nn
from config.config import cfg


class AltitudeLoss(nn.Module):
    """
    Explicit altitude regularization for high-speed flight.

    Modes:
    - none: disable altitude constraint
    - fixed: constrain end z to a global target altitude
    - goal: constrain end z to goal z
    - relative: constrain end z to current z (+ optional offset)
    """
    def __init__(self):
        super(AltitudeLoss, self).__init__()
        self.mode = str(cfg.get("altitude_mode", "none")).lower()
        self.target_altitude = float(cfg.get("target_altitude", 20.0))
        self.relative_offset = float(cfg.get("altitude_relative_offset", 0.0))
        self.tolerance = float(cfg.get("altitude_tolerance", 0.0))
        # Robust settings for high-speed training:
        # - huber_delta <= 0: use pure L2 penalty.
        # - altitude_cost_clip <= 0: disable clipping.
        self.huber_delta = float(cfg.get("altitude_huber_delta", 0.0))
        self.altitude_cost_clip = float(cfg.get("altitude_cost_clip", -1.0))

    def forward(self, Df, Dp, goal):
        """
        Args:
            Dp: (B, 3, 3), decision parameters [px,vx,ax; py,vy,ay; pz,vz,az]
            Df: (B, 3, 3), fixed parameters [px,vx,ax; py,vy,ay; pz,vz,az]
            goal: (B, 3), absolute goal position in world frame
        Returns:
            altitude_loss: (B)
        """
        if self.mode == "none":
            return th.zeros(Dp.shape[0], device=Dp.device, dtype=Dp.dtype)

        end_z = Dp[:, 2, 0]
        if self.mode == "fixed":
            target_z = th.full_like(end_z, self.target_altitude)
        elif self.mode == "goal":
            target_z = goal[:, 2]
        elif self.mode == "relative":
            target_z = Df[:, 2, 0] + self.relative_offset
        else:
            raise ValueError(f"Unsupported altitude_mode: {self.mode}")

        # Dead-zone penalty around target altitude to avoid over-constraining small oscillations.
        abs_err = th.abs(end_z - target_z)
        clipped_err = th.clamp(abs_err - self.tolerance, min=0.0)

        if self.huber_delta > 0.0:
            # Huber-style penalty: quadratic near the target, linear for large deviation.
            delta = th.full_like(clipped_err, self.huber_delta)
            quad = th.minimum(clipped_err, delta)
            linear = clipped_err - quad
            altitude_loss = 0.5 * quad * quad + self.huber_delta * linear
        else:
            altitude_loss = clipped_err * clipped_err

        if self.altitude_cost_clip > 0.0:
            altitude_loss = th.clamp(altitude_loss, max=self.altitude_cost_clip)
        return altitude_loss
