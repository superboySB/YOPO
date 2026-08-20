"""
    将yopo模型转换为Tensorrt
    prepare:
        1 pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
        2 git clone https://github.com/NVIDIA-AI-IOT/torch2trt
          cd torch2trt
          python setup.py install
"""

import os
import argparse
import time
from pathlib import Path

import torch
from ruamel.yaml import YAML
from config.config import cfg
from policy.yopo_network import YOPOOmniNetwork


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in ("1", "true", "yes", "y", "on"):
        return True
    if lowered in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    parser.add_argument("--weight", default="", help="Explicit PyTorch checkpoint path.")
    parser.add_argument("--active-camera", type=str2bool, default=bool(cfg["active_camera"]),
                        help="Checkpoint camera mode: true or false.")
    parser.add_argument("--dir", type=str, default='yopo_trt.pth', help="output file name")
    return parser


def validate_resolved_config(weight, active_camera):
    resolved_path = Path(weight).resolve().parent / "resolved_config.yaml"
    if not resolved_path.is_file():
        print(f"WARNING: {resolved_path} not found; converting a legacy unaudited checkpoint.")
        return
    with resolved_path.open("r", encoding="utf-8") as stream:
        resolved = YAML(typ="safe").load(stream)
    if not isinstance(resolved, dict):
        raise ValueError(f"Checkpoint resolved config must be a YAML mapping: {resolved_path}")
    expected = {
        "active_camera": bool(active_camera),
        "image_height": int(cfg["image_height"]),
        "image_width": int(cfg["image_width"]),
        "insight9_train_max_depth_m": float(cfg["insight9_train_max_depth_m"]),
        "depth_preprocess": str(cfg["depth_preprocess"]),
    }
    missing = [key for key in expected if key not in resolved]
    if missing:
        raise ValueError(f"Checkpoint resolved config is missing keys: {missing}")
    mismatch = [
        f"{key}: checkpoint={resolved[key]!r}, conversion={value!r}"
        for key, value in expected.items() if resolved[key] != value
    ]
    if mismatch:
        raise ValueError("Checkpoint/TensorRT contract mismatch: " + "; ".join(mismatch))


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        from torch2trt import torch2trt
    except ImportError as exc:
        raise SystemExit(
            "torch2trt is required for conversion; install NVIDIA-AI-IOT/torch2trt first."
        ) from exc
    cfg["active_camera"] = bool(args.active_camera)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = args.weight or base_dir + "/saved/YOPO_{}/epoch{}.pth".format(args.trial, args.epoch)
    validate_resolved_config(weight, args.active_camera)

    print("Loading Network...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(weight, map_location=device, weights_only=True)
    policy = YOPOOmniNetwork()
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()

    # The inputs should be consistent with training
    depth_in = torch.zeros((1, 1, cfg["image_height"], cfg["image_width"]), dtype=torch.float32, device=device)
    obs_in = torch.zeros((1, 11), dtype=torch.float32, device=device)

    print("TensorRT Transfer...")
    model_trt = torch2trt(policy, [depth_in, obs_in], fp16_mode=True)
    torch.save(model_trt.state_dict(), args.dir)


    print("Evaluation...")
    # Warm Up...
    traj_trt, score_trt, camera_trt = model_trt(depth_in, obs_in)
    traj, score, camera = policy(depth_in, obs_in)
    if device == "cuda":
        torch.cuda.synchronize()

    # PyTorch Latency
    torch_start = time.time()
    traj, score, camera = policy(depth_in, obs_in)
    if device == "cuda":
        torch.cuda.synchronize()
    torch_end = time.time()

    # TensorRT Latency
    trt_start = time.time()
    traj_trt, score_trt, camera_trt = model_trt(depth_in, obs_in)
    if device == "cuda":
        torch.cuda.synchronize()
    trt_end = time.time()

    # Transfer Error
    traj_error = torch.mean(torch.abs(traj - traj_trt))
    score_error = torch.mean(torch.abs(score - score_trt))
    camera_error = torch.mean(torch.abs(camera - camera_trt))

    print(f"Torch Latency: {1000 * (torch_end - torch_start):.3f} ms, "
          f"TensorRT Latency: {1000 * (trt_end - trt_start):.3f} ms, "
          f"Transfer Endstate Error: {traj_error.item():.6f},"
          f"Transfer Score Error: {score_error.item():.6f},"
          f"Transfer Camera Error: {camera_error.item():.6f}")
