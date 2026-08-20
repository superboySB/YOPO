#!/usr/bin/env python3
"""Fast contract test for the joint Insight 9 trajectory/camera policy."""

import argparse
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "YOPO"))

from config.config import cfg  # noqa: E402
from policy.yopo_network import YOPOActivePerceptionNetwork  # noqa: E402


def str2bool(value):
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--active-camera", type=str2bool, default=bool(cfg["active_camera"]))
    args = parser.parse_args()

    cfg["active_camera"] = args.active_camera
    assert float(cfg["w_camera"]) > 0.0
    assert float(cfg["w_camera_smooth"]) > 0.0
    device = torch.device(args.device)
    model = YOPOActivePerceptionNetwork().to(device)
    assert model.state_encoder[0].in_features == 11
    assert model.head[-1].out_features == 12
    model.train(args.backward)
    batch, directions = 2, int(cfg["omni_topology_num"])
    depth = torch.rand(
        batch, 1, int(cfg["image_height"]), int(cfg["image_width"]),
        device=device,
    )
    state = torch.randn(batch, 11, device=device)
    if args.active_camera:
        state[:, 9] = torch.tanh(state[:, 9]) * math.radians(float(cfg["camera_pitch_limit_deg"]))
        state[:, 10] = torch.tanh(state[:, 10]) * math.radians(float(cfg["camera_yaw_limit_deg"]))
    else:
        state[:, 9:11] = 0.0

    endstate, score, camera = model(depth, state)
    assert endstate.shape == (batch, directions, 9)
    assert score.shape == (batch, directions)
    assert camera.shape == (batch, directions, 2)
    assert torch.isfinite(endstate).all() and torch.isfinite(score).all() and torch.isfinite(camera).all()
    assert camera[..., 0].abs().max() <= math.radians(float(cfg["camera_pitch_limit_deg"])) + 1e-5
    assert camera[..., 1].abs().max() <= math.radians(float(cfg["camera_yaw_limit_deg"])) + 1e-5

    pose_state = state[:, None, :].expand(-1, directions, -1).contiguous()
    pose_endstate, pose_score, pose_camera = model(depth, pose_state)
    assert pose_endstate.shape == (batch, directions, directions, 9)
    assert pose_score.shape == (batch, directions, directions)
    assert pose_camera.shape == (batch, directions, directions, 2)

    if args.backward:
        (endstate.square().mean() + score.mean() + camera.square().mean()).backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    mode = "active" if args.active_camera else "fixed"
    print(f"PASS device={device} mode={mode} parameters={parameters:,} state_dim=11 head_dim=12")
    print(f"single={tuple(endstate.shape)}, pose={tuple(pose_endstate.shape)}, backward={args.backward}")


if __name__ == "__main__":
    main()
