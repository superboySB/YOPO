import math

import torch
import torch.nn.functional as F
from torch import nn

from config.config import cfg
from policy.models.backbone import YopoBackbone


class OmniDepthBackbone(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = YopoBackbone(d_model)
        # Keep the original YOPO ResNet18 capacity, but stop the last stage from
        # downsampling so each 96x160 view yields about 6x10 tokens instead of 3x5.
        self.net.cnn.layer4[0].conv1.stride = (1, 1)
        self.net.cnn.layer4[0].downsample[0].stride = (1, 1)

    def forward(self, depth):
        return self.net(depth)


class YOPOOmniNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = int(cfg["omni_d_model"])
        self.topology_num = int(cfg["omni_topology_num"])
        self.vel_max = float(cfg["vel_max_train"])
        self.acc_max = float(cfg["acc_max_train"])
        self.radius_min = float(cfg["omni_radius_min"])
        self.radius_max = float(cfg["omni_radius_max"])
        self.pitch_max = math.radians(float(cfg["omni_pitch_max_deg"]))

        self.depth_backbone = OmniDepthBackbone(self.d_model)
        self.view_embedding = nn.Embedding(4, self.d_model)
        self.ray_mlp = nn.Sequential(
            nn.Linear(3, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(9, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
        )
        self.geo_mlp = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
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
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(cfg["omni_decoder_layers"]))
        self.head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, 10),
        )

        theta = torch.arange(self.topology_num, dtype=torch.float32) * 2.0 * math.pi / self.topology_num
        self.register_buffer("topology_theta", theta)
        self.register_buffer("view_yaw", torch.tensor([0.0, math.pi / 2.0, -math.pi / 2.0, math.pi], dtype=torch.float32))

    def forward(self, depth, state):
        """
        depth: [B, 4, 1, H, W], normalized depth.
        state: [B, 9] or [B, D, 9], body-frame [v, a, v_des].
        """
        image_tokens = self.encode_depth(depth)
        state = self.prepare_topology_state(state)
        state_feature = self.state_encoder(state)
        topology_feature = self.build_topology_feature(depth.device)

        if state_feature.dim() == 3:
            queries = state_feature + topology_feature[None, :, :]
            latent = self.decoder(tgt=queries, memory=image_tokens)
            raw = self.head(latent)
        elif state_feature.dim() == 4:
            B, D = state_feature.shape[:2]
            queries = state_feature + topology_feature[None, None, :, :]
            queries = queries.reshape(B * D, self.topology_num, self.d_model)
            memory = image_tokens[:, None, :, :].expand(B, D, -1, -1).reshape(B * D, image_tokens.shape[1], self.d_model)
            latent = self.decoder(tgt=queries, memory=memory)
            raw = self.head(latent).reshape(B, D, self.topology_num, 10)
        else:
            raise ValueError(f"YOPOOmniNetwork expects state shape [B,9] or [B,D,9], got {tuple(state.shape)}")

        endstate = self.decode_endstate(raw[..., :9])
        score = F.softplus(raw[..., 9])
        return endstate, score

    def encode_depth(self, depth):
        B, V, C, H, W = depth.shape
        feature = self.depth_backbone(depth.reshape(B * V, C, H, W))
        _, D, h, w = feature.shape
        tokens = feature.flatten(2).transpose(1, 2).reshape(B, V, h * w, D)

        view_ids = torch.arange(V, device=depth.device)
        view_embed = self.view_embedding(view_ids)[None, :, None, :]
        ray_embed = self.ray_mlp(self.build_ray_grid(h, w, depth.device))[None, :, :, :]
        tokens = tokens + view_embed + ray_embed
        return tokens.reshape(B, V * h * w, D)

    def build_ray_grid(self, h, w, device):
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        xs = torch.linspace(-1.0, 1.0, w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        cam_ray = torch.stack(
            [
                torch.ones_like(grid_x),
                -grid_x,
                -grid_y,
            ],
            dim=-1,
        )
        cam_ray = F.normalize(cam_ray, dim=-1).reshape(1, h * w, 3).repeat(4, 1, 1)

        yaw = self.view_yaw.to(device)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        ray_x = cos_yaw[:, None] * cam_ray[..., 0] - sin_yaw[:, None] * cam_ray[..., 1]
        ray_y = sin_yaw[:, None] * cam_ray[..., 0] + cos_yaw[:, None] * cam_ray[..., 1]
        ray_z = cam_ray[..., 2]
        return torch.stack([ray_x, ray_y, ray_z], dim=-1)

    def build_topology_feature(self, device):
        theta = self.topology_theta.to(device)
        geo = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        topology_ids = torch.arange(self.topology_num, device=device)
        return self.geo_mlp(geo) + self.topology_embedding(topology_ids)

    def build_topology_rotation(self, device):
        theta = self.topology_theta.to(device)
        cos_yaw = torch.cos(theta)
        sin_yaw = torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        return torch.stack(
            [
                torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
                torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=-2,
        )

    def prepare_topology_state(self, state):
        rot_bp = self.build_topology_rotation(state.device)

        if state.dim() == 2:
            state_vec = state.reshape(state.shape[0], 3, 3)
            state_local = torch.einsum("bnc,kcl->bknl", state_vec, rot_bp)
        elif state.dim() == 3:
            state_vec = state.reshape(state.shape[0], state.shape[1], 3, 3)
            state_local = torch.einsum("bdnc,kcl->bdknl", state_vec, rot_bp)
        else:
            raise ValueError(f"YOPOOmniNetwork expects state shape [B,9] or [B,D,9], got {tuple(state.shape)}")

        state_local = state_local.reshape(*state_local.shape[:-2], 9)
        return self.normalize_state(state_local)

    def decode_endstate(self, raw):
        delta_yaw = torch.tanh(raw[..., 0]) * (math.pi / self.topology_num)
        delta_pitch = torch.tanh(raw[..., 1]) * self.pitch_max
        radius = self.radius_min + (self.radius_max - self.radius_min) * torch.sigmoid(raw[..., 2])

        theta = self.topology_theta.to(raw.device)[None, :]
        yaw = theta + delta_yaw
        cos_pitch = torch.cos(delta_pitch)
        pos = torch.stack(
            [
                radius * torch.cos(yaw) * cos_pitch,
                radius * torch.sin(yaw) * cos_pitch,
                radius * torch.sin(delta_pitch),
            ],
            dim=-1,
        )
        rot_bp = self.build_topology_rotation(raw.device)
        vel_local = self.vel_max * torch.tanh(raw[..., 3:6])
        acc_local = self.acc_max * torch.tanh(raw[..., 6:9])
        vel = torch.einsum("kcd,...kd->...kc", rot_bp, vel_local)
        acc = torch.einsum("kcd,...kd->...kc", rot_bp, acc_local)
        return torch.cat([pos, vel, acc], dim=-1)

    def normalize_state(self, state):
        state = state.clone()
        state[..., 0:3] = state[..., 0:3] / self.vel_max
        state[..., 3:6] = state[..., 3:6] / self.acc_max
        state[..., 6:9] = state[..., 6:9] / self.vel_max
        return state
