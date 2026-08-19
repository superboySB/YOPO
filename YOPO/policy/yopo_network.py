"""YOPO active-perception policy.

The high-resolution Insight 9 depth image is encoded by the ResNet-18
backbone that existed before the 8x8 ToF migration. Spatial CNN features
become Transformer memory tokens; topology queries jointly predict a
trajectory end state, its cost, and the target two-axis camera orientation.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from config.config import cfg
from policy.models.backbone import YopoBackbone


class YOPOActivePerceptionNetwork(nn.Module):
    """Joint trajectory and Insight 9 gimbal policy."""

    def __init__(self):
        super().__init__()
        self.d_model = int(cfg["omni_d_model"])
        self.topology_num = int(cfg["omni_topology_num"])
        self.vel_max = float(cfg["vel_max_train"])
        self.acc_max = float(cfg["acc_max_train"])
        self.radius_min = float(cfg["omni_radius_min"])
        self.radius_max = float(cfg["omni_radius_max"])
        self.pitch_max = math.radians(float(cfg["omni_pitch_max_deg"]))
        self.camera_pitch_max = math.radians(float(cfg["camera_pitch_limit_deg"]))
        self.camera_yaw_max = math.radians(float(cfg["camera_yaw_limit_deg"]))

        self.image_backbone = YopoBackbone(self.d_model)
        self.image_position_encoder = nn.Sequential(
            nn.Linear(2, self.d_model), nn.SiLU(), nn.Linear(self.d_model, self.d_model)
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(11, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
        )
        self.topology_geo_encoder = nn.Sequential(
            nn.Linear(2, self.d_model), nn.SiLU(), nn.Linear(self.d_model, self.d_model)
        )
        self.topology_embedding = nn.Embedding(self.topology_num, self.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(cfg["omni_num_heads"]),
            dim_feedforward=int(cfg["omni_ffn_dim"]),
            dropout=float(cfg["omni_dropout"]),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(cfg["omni_decoder_layers"])
        )
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 12),
        )

        theta = torch.arange(self.topology_num, dtype=torch.float32)
        theta = theta * 2.0 * math.pi / self.topology_num
        self.register_buffer("topology_theta", theta)

    def forward(self, depth, state):
        """Return trajectory, score and associated camera-target candidates.

        ``depth`` is ``[B,1,H,W]``. ``state`` is ``[B,11]`` or
        ``[B,D,11]`` and contains body-frame velocity, acceleration, desired
        velocity, then current camera pitch/yaw.
        """
        image_tokens = self.encode_depth(depth)
        state_feature = self.state_encoder(self.prepare_topology_state(state))
        topology_feature = self.build_topology_feature(depth.device)

        if state_feature.dim() == 3:
            queries = state_feature + topology_feature[None, :, :]
            raw = self.head(self.decoder(tgt=queries, memory=image_tokens))
        elif state_feature.dim() == 4:
            batch, directions = state_feature.shape[:2]
            queries = state_feature + topology_feature[None, None, :, :]
            queries = queries.reshape(batch * directions, self.topology_num, self.d_model)
            memory = image_tokens[:, None, :, :].expand(
                batch, directions, -1, -1
            ).reshape(batch * directions, image_tokens.shape[1], self.d_model)
            raw = self.head(self.decoder(tgt=queries, memory=memory))
            raw = raw.reshape(batch, directions, self.topology_num, 12)
        else:
            raise ValueError(
                "YOPO active-perception state must be [B,11] or [B,D,11], "
                f"got {tuple(state.shape)}"
            )

        return (
            self.decode_endstate(raw[..., :9]),
            F.softplus(raw[..., 9]),
            self.decode_camera_target(raw[..., 10:12]),
        )

    def encode_depth(self, depth):
        if depth.dim() != 4 or depth.shape[1] != 1:
            raise ValueError(
                "Insight 9 depth input must be [B,1,H,W], "
                f"got {tuple(depth.shape)}"
            )
        feature = self.image_backbone(depth)
        _, _, height, width = feature.shape
        ys = torch.linspace(-1.0, 1.0, height, device=feature.device, dtype=feature.dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=feature.device, dtype=feature.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        position = torch.stack((grid_x, grid_y), dim=-1).reshape(height * width, 2)
        position = self.image_position_encoder(position)[None, :, :]
        return feature.flatten(2).transpose(1, 2) + position

    def build_topology_feature(self, device):
        theta = self.topology_theta.to(device)
        geometry = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
        ids = torch.arange(self.topology_num, device=device)
        return self.topology_geo_encoder(geometry) + self.topology_embedding(ids)

    def build_topology_rotation(self, device):
        theta = self.topology_theta.to(device)
        cos_yaw, sin_yaw = torch.cos(theta), torch.sin(theta)
        zeros, ones = torch.zeros_like(theta), torch.ones_like(theta)
        return torch.stack(
            (
                torch.stack((cos_yaw, -sin_yaw, zeros), dim=-1),
                torch.stack((sin_yaw, cos_yaw, zeros), dim=-1),
                torch.stack((zeros, zeros, ones), dim=-1),
            ),
            dim=-2,
        )

    def prepare_topology_state(self, state):
        if state.shape[-1] != 11:
            raise ValueError(f"Expected 11 state values, got {state.shape[-1]}")
        rotation = self.build_topology_rotation(state.device)
        motion = state[..., :9].reshape(*state.shape[:-1], 3, 3)
        camera = state[..., 9:11]

        if state.dim() == 2:
            local_motion = torch.einsum("bnc,kcl->bknl", motion, rotation)
            camera = camera[:, None, :].expand(-1, self.topology_num, -1)
        elif state.dim() == 3:
            local_motion = torch.einsum("bdnc,kcl->bdknl", motion, rotation)
            camera = camera[:, :, None, :].expand(-1, -1, self.topology_num, -1)
        else:
            raise ValueError("State must have shape [B,11] or [B,D,11]")

        local_motion = local_motion.reshape(*local_motion.shape[:-2], 9)
        local_motion = self.normalize_motion_state(local_motion)
        camera = camera.clone()
        camera[..., 0] /= self.camera_pitch_max
        camera[..., 1] /= self.camera_yaw_max
        return torch.cat((local_motion, camera), dim=-1)

    def decode_endstate(self, raw):
        delta_yaw = torch.tanh(raw[..., 0]) * (math.pi / self.topology_num)
        delta_pitch = torch.tanh(raw[..., 1]) * self.pitch_max
        radius = self.radius_min + (self.radius_max - self.radius_min) * torch.sigmoid(raw[..., 2])
        view_shape = [1] * (raw.dim() - 2) + [self.topology_num]
        theta = self.topology_theta.to(raw.device).reshape(view_shape)
        yaw = theta + delta_yaw
        cos_pitch = torch.cos(delta_pitch)
        position = torch.stack(
            (
                radius * torch.cos(yaw) * cos_pitch,
                radius * torch.sin(yaw) * cos_pitch,
                radius * torch.sin(delta_pitch),
            ),
            dim=-1,
        )
        rotation = self.build_topology_rotation(raw.device)
        velocity = torch.einsum(
            "kcd,...kd->...kc", rotation, self.vel_max * torch.tanh(raw[..., 3:6])
        )
        acceleration = torch.einsum(
            "kcd,...kd->...kc", rotation, self.acc_max * torch.tanh(raw[..., 6:9])
        )
        return torch.cat((position, velocity, acceleration), dim=-1)

    def decode_camera_target(self, raw):
        return torch.stack(
            (
                torch.tanh(raw[..., 0]) * self.camera_pitch_max,
                torch.tanh(raw[..., 1]) * self.camera_yaw_max,
            ),
            dim=-1,
        )

    def normalize_motion_state(self, state):
        state = state.clone()
        state[..., 0:3] /= self.vel_max
        state[..., 3:6] /= self.acc_max
        state[..., 6:9] /= self.vel_max
        return state


# Existing entrypoints keep working while the explicit class name documents
# that this is no longer the four-ToF omni policy.
YOPOOmniNetwork = YOPOActivePerceptionNetwork
