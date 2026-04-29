import torch
import numpy as np
from config.config import cfg
from policy.primitive import LatticePrimitive


class StateTransform:
    def __init__(self):
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.goal_length = cfg['goal_length']
        self.target_position_scale = cfg["target_position_scale"]
        self.image_width = cfg["image_width"]
        self.image_height = cfg["image_height"]
        self.camera_fx = cfg["camera_fx"]
        self.camera_fy = cfg["camera_fy"]
        self.camera_cx = cfg["camera_cx"]
        self.camera_cy = cfg["camera_cy"]

    def pred_to_endstate(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
            Transform the predicted state to the body frame (Original prediction → Primitive frame → Body frame).
            endstate_pred: [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h]
            :return [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h] in body frame
        """
        B, V, H = endstate_pred.shape[0], endstate_pred.shape[2], endstate_pred.shape[3]

        # [B, 9, 3, 5] -> [B, 3, 5, 9] -> [B, 15, 9]
        endstate_pred = endstate_pred.permute(0, 2, 3, 1).reshape(B, V * H, 9)

        # 获取 lattice angle 和 rotation (.flip: 由于lattice和grid的顺序相反)
        yaw, pitch = self.lattice_primitive.getAngleLattice()  # [15]
        yaw = yaw.to(device=endstate_pred.device, dtype=endstate_pred.dtype)
        pitch = pitch.to(device=endstate_pred.device, dtype=endstate_pred.dtype)
        yaw = yaw.flip(0)[None, :].expand(B, -1)  # [B, 15]
        pitch = pitch.flip(0)[None, :].expand(B, -1)  # [B, 15]
        Rbp = self.lattice_primitive.getRotation().flip(0)  # [15, 3, 3]
        Rbp = Rbp.to(device=endstate_pred.device, dtype=endstate_pred.dtype)
        Rbp = Rbp[None, :, :, :].expand(B, -1, -1, -1)  # [B, 15, 3, 3]

        delta_yaw = endstate_pred[:, :, 0] * self.lattice_primitive.yaw_diff  # [B, 15]
        delta_pitch = endstate_pred[:, :, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, :, 2] + 1.0) * self.lattice_primitive.radio_range

        cos_pitch = torch.cos(pitch + delta_pitch)
        endstate_x = cos_pitch * torch.cos(yaw + delta_yaw) * radio
        endstate_y = cos_pitch * torch.sin(yaw + delta_yaw) * radio
        endstate_z = torch.sin(pitch + delta_pitch) * radio
        endstate_p = torch.stack([endstate_x, endstate_y, endstate_z], dim=-1)  # [B, 15, 3]

        # vel / acc
        endstate_vp = endstate_pred[:, :, 3:6] * self.lattice_primitive.vel_max  # [B, 15, 3]
        endstate_ap = endstate_pred[:, :, 6:9] * self.lattice_primitive.acc_max  # [B, 15, 3]

        # v/a 变换到 body frame
        endstate_vb = torch.matmul(Rbp, endstate_vp.unsqueeze(-1)).squeeze(-1)  # [B, 15, 3]
        endstate_ab = torch.matmul(Rbp, endstate_ap.unsqueeze(-1)).squeeze(-1)

        endstate = torch.cat([endstate_p, endstate_vb, endstate_ab], dim=-1)  # [B, 15, 9]

        endstate = endstate.permute(0, 2, 1).reshape(B, 9, V, H)  # [B, 9, 3, 5]
        return endstate

    def pred_to_target(self, target_pred: torch.Tensor) -> torch.Tensor:
        """
            Decode YOPOv2-Tracker target predictions.
            target_pred: [batch; du dv depth; primitive_v; primitive_h].
            du/dv are cell-local logits and depth is a normalized range logit.
            return: [batch; x y z; primitive_v; primitive_h] in the camera/body frame.
        """
        B, V, H = target_pred.shape[0], target_pred.shape[2], target_pred.shape[3]
        device = target_pred.device
        dtype = target_pred.dtype
        stride_u = float(self.image_width) / float(H)
        stride_v = float(self.image_height) / float(V)

        h_idx = torch.arange(H, dtype=dtype, device=device).view(1, 1, H)
        v_idx = torch.arange(V, dtype=dtype, device=device).view(1, V, 1)

        u = (h_idx + torch.sigmoid(target_pred[:, 0])) * stride_u
        v = (v_idx + torch.sigmoid(target_pred[:, 1])) * stride_v
        depth = torch.sigmoid(target_pred[:, 2]) * self.target_position_scale

        x = depth
        y = -(u - self.camera_cx) / self.camera_fx * depth
        z = -(v - self.camera_cy) / self.camera_fy * depth
        return torch.stack([x, y, z], dim=1)

    def pred_to_target_cpu(self, target_pred: np.ndarray, grid_id=None) -> np.ndarray:
        """
            CPU decoder used by the ROS node.
            target_pred: [N, 3] raw network predictions in flattened image-grid order.
            grid_id: optional flattened grid indices matching target_pred rows.
        """
        target_pred = np.asarray(target_pred, dtype=np.float32)
        if target_pred.ndim == 1:
            target_pred = target_pred[None, :]

        if grid_id is None:
            grid_id = np.arange(target_pred.shape[0], dtype=np.int64)
        if isinstance(grid_id, torch.Tensor):
            grid_id = grid_id.cpu().numpy()
        grid_id = np.asarray(grid_id, dtype=np.int64).reshape(-1)

        H = self.lattice_primitive.horizon_num
        V = self.lattice_primitive.vertical_num
        stride_u = float(self.image_width) / float(H)
        stride_v = float(self.image_height) / float(V)

        h_idx = grid_id % H
        v_idx = grid_id // H
        h_idx = np.clip(h_idx, 0, H - 1)
        v_idx = np.clip(v_idx, 0, V - 1)

        sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
        u = (h_idx + sigmoid(target_pred[:, 0])) * stride_u
        v = (v_idx + sigmoid(target_pred[:, 1])) * stride_v
        depth = sigmoid(target_pred[:, 2]) * self.target_position_scale

        x = depth
        y = -(u - self.camera_cx) / self.camera_fx * depth
        z = -(v - self.camera_cy) / self.camera_fy * depth
        return np.stack((x, y, z), axis=1)

    def pred_to_endstate_cpu(self, endstate_pred: np.ndarray, lattice_id: torch.Tensor) -> np.ndarray:
        """
            Used during test:
            Numpy version of pred_to_endstate() on CPU (used in test, x10 times faster than torch on CUDA)
            :return [B; px py pz vx vy vz ax ay az] in body frame
        """
        delta_yaw = endstate_pred[:, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = endstate_pred[:, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, 2] + 1.0) * self.lattice_primitive.radio_range

        yaw, pitch = self.lattice_primitive.getAngleLattice(lattice_id)
        yaw, pitch = yaw.cpu().numpy(), pitch.cpu().numpy()
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.stack((endstate_x, endstate_y, endstate_z), axis=1)

        endstate_vp = endstate_pred[:, 3:6] * self.lattice_primitive.vel_max
        endstate_ap = endstate_pred[:, 6:9] * self.lattice_primitive.acc_max

        Rpb = self.lattice_primitive.getRotation(lattice_id).cpu().numpy()
        endstate_vb = np.matmul(Rpb, endstate_vp[:, :, np.newaxis]).squeeze(-1)
        endstate_ab = np.matmul(Rpb, endstate_ap[:, :, np.newaxis]).squeeze(-1)

        return np.concatenate((endstate_p, endstate_vb, endstate_ab), axis=1)


    def prepare_input(self, obs):
        """
            Transform the observation to the primitive frame (Body frame → Primitive frame → Body frame).
            obs: [batch; vx, vy, vz, ax, ay, az] in body frame
            :return [batch; vx, vy, vz, ax, ay, az; primitive_v; primitive_h] in primitive frame
        """
        B, N = obs.shape[0], self.lattice_primitive.traj_num
        obs_dim = obs.shape[1]
        if obs_dim % 3 != 0:
            raise ValueError(f"Observation dimension must be a multiple of 3, got {obs_dim}")
        obs_rows = obs_dim // 3

        # 获取所有 Rbp 并倒序排列 (由于lattice和grid的顺序相反)
        Rbp_all = self.lattice_primitive.getRotation().flip(0)  # shape: [N, 3, 3]
        Rbp_all = Rbp_all.to(device=obs.device, dtype=obs.dtype)

        obs = obs.view(B, obs_rows, 3)  # [B, obs_rows, 3]

        # 扩展 obs 和 Rbp 到 [B, N, 3, 3]
        obs_exp = obs[:, None, :, :].expand(B, N, obs_rows, 3)
        Rbp_exp = Rbp_all[None, :, :, :].expand(B, N, 3, 3)

        # 执行批量坐标变换
        transformed = torch.matmul(obs_exp, Rbp_exp)  # [B, N, 3, 3]

        transformed_flat = transformed.view(B, N, obs_dim)  # [B, N, obs_dim]
        out = transformed_flat.permute(0, 2, 1).contiguous()  # [B, obs_dim, N]
        out = out.view(B, obs_dim, self.lattice_primitive.vertical_num, self.lattice_primitive.horizon_num)
        return out

    def unnormalize_obs(self, vel_acc):
        vel_acc[:, 0:3] = vel_acc[:, 0:3] * self.lattice_primitive.vel_max
        vel_acc[:, 3:6] = vel_acc[:, 3:6] * self.lattice_primitive.acc_max
        return vel_acc

    def normalize_obs(self, vel_acc_goal):
        vel_acc_goal[:, 0:3] = vel_acc_goal[:, 0:3] / self.lattice_primitive.vel_max
        vel_acc_goal[:, 3:6] = vel_acc_goal[:, 3:6] / self.lattice_primitive.acc_max

        if vel_acc_goal.shape[1] < 9:
            return vel_acc_goal

        # Clamp the goal direction to unit length
        goal_norm = vel_acc_goal[:, 6:9].norm(dim=1, keepdim=True)
        vel_acc_goal[:, 6:9] = vel_acc_goal[:, 6:9] / goal_norm.clamp(min=self.goal_length)
        return vel_acc_goal


def rotate_body2world(rot_wb, pos_b):
    """
    Rotate pos_b from body frame to world frame using quaternion q_wb.
    rot_wb: (..., 3, 3)
    pos_b: (..., 3)
    """
    pos_w = torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)
    return pos_w


def transform_body2world(rot_wb, t_w, pos_b):
    """
    Transform pos_b from body frame to world frame using quaternion q_wb and t_w.
    rot_wb: (..., 3, 3)
    t_w: (..., 3)
    pos_b: (..., 3)
    """
    return rotate_body2world(rot_wb, pos_b) + t_w


def state_body2world(pos_w, rot_wb, pos_b, vel_b, acc_b):
    pos_b = transform_body2world(rot_wb, pos_w, pos_b)
    vel_b = rotate_body2world(rot_wb, vel_b)
    acc_b = rotate_body2world(rot_wb, acc_b)
    return pos_b, vel_b, acc_b


if __name__ == '__main__':
    CoordTransform = StateTransform()
