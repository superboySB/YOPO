#!/usr/bin/env python3
import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from ruamel.yaml import YAML
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YOPO_ROOT = os.path.join(REPO_ROOT, "YOPO")
if YOPO_ROOT not in sys.path:
    sys.path.insert(0, YOPO_ROOT)

from config.config import cfg
from policy.yopo_dataset import YOPOOmniPoseDataset
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


def parse_args():
    parser = argparse.ArgumentParser(description="Offline YOPO active-perception checkpoint test.")
    parser.add_argument("--weight", required=True, help="Path to YOPO active-perception .pth checkpoint.")
    parser.add_argument("--dataset-path", default=None, help='Override cfg["dataset_path"], e.g. "../dataset_active".')
    parser.add_argument("--split", choices=["train", "valid"], default="valid", help="Dataset split to test.")
    parser.add_argument("--batch-size", type=int, default=8, help="Pose-level batch size.")
    parser.add_argument("--num-batches", type=int, default=8, help="Number of batches to run.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for the test.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--active-camera", type=str2bool, default=bool(cfg["active_camera"]),
                        help="Expected dataset/checkpoint camera mode: true or false.")
    parser.add_argument("--strict-depth-range", action="store_true", help="Fail if normalized depth is outside [0, 1].")
    return parser.parse_args()


def validate_checkpoint_config(weight, active_camera):
    resolved_path = Path(weight).resolve().parent / "resolved_config.yaml"
    if not resolved_path.is_file():
        print(f"WARNING: legacy checkpoint without resolved config: {resolved_path}")
        return
    with resolved_path.open("r", encoding="utf-8") as stream:
        resolved = YAML(typ="safe").load(stream)
    if not isinstance(resolved, dict):
        raise ValueError(f"Invalid resolved checkpoint config: {resolved_path}")
    expected = {
        "active_camera": bool(active_camera),
        "image_width": int(cfg["image_width"]),
        "image_height": int(cfg["image_height"]),
        "insight9_train_max_depth_m": float(cfg["insight9_train_max_depth_m"]),
        "depth_preprocess": str(cfg["depth_preprocess"]),
    }
    missing = [key for key in expected if key not in resolved]
    if missing:
        raise ValueError(f"Resolved checkpoint config is missing keys: {missing}")
    mismatch = []
    for key, value in expected.items():
        actual = resolved[key]
        if key == "insight9_train_max_depth_m":
            try:
                matches = math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-6)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == value
        if not matches:
            mismatch.append(f"{key}: checkpoint={actual!r}, expected={value!r}")
    if mismatch:
        raise ValueError("Checkpoint contract mismatch: " + "; ".join(mismatch))
    print(f"Resolved checkpoint config validated: {resolved_path}")


def inspect_raw_depth(dataset, batch):
    map_id = int(batch[-1][0].item())
    pose_key = int(dataset.indices[0])
    pose_id = int(dataset.arrays["pose_id"][dataset.start_by_key[pose_key]])
    image_path = os.path.join(dataset.data_dir, str(map_id), f"img_{pose_id}_depth.png")
    image = cv2.imread(image_path, -1)
    if image is None:
        raise FileNotFoundError(f"Missing depth image: {image_path}")
    return image_path, image.dtype, float(image.min()), float(image.max())


def main():
    args = parse_args()
    cfg["active_camera"] = bool(args.active_camera)
    if args.dataset_path is not None:
        cfg["dataset_path"] = args.dataset_path
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset = YOPOOmniPoseDataset(mode=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = YOPOOmniNetwork().to(device)
    validate_checkpoint_config(args.weight, args.active_camera)
    state_dict = torch.load(args.weight, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Loaded checkpoint: {args.weight}")
    print(f"Device: {device}")
    print(f"Dataset path: {dataset.data_dir}")
    print(
        f"Camera mode: {'active' if args.active_camera else 'fixed'}; "
        f"preprocess={cfg['depth_preprocess']}"
    )
    print(f"Depth normalization: cv2.imread(..., -1) uint16 -> float32 / 65535.0")
    print(f"Network expects normalized Insight 9 depth in [0, 1], shape [B,1,H,W].")

    total_poses = 0
    total_dirs = 0
    forward_times = []
    score_min = []
    score_max = []

    with torch.inference_mode():
        for step, batch in enumerate(loader):
            if step >= args.num_batches:
                break
            depth, _, _, state_b, _, _, guide_mask, selected_topology, camera_target, _ = batch
            if not args.active_camera:
                camera_state_max = float(state_b[..., 9:11].abs().max().item())
                camera_target_max = float(camera_target.abs().max().item())
                if camera_state_max > 1e-6 or camera_target_max > 1e-6:
                    raise ValueError(
                        "Fixed-camera batch contains non-zero camera labels: "
                        f"state={camera_state_max:.3g}, target={camera_target_max:.3g}"
                    )
            if step == 0:
                image_path, raw_dtype, raw_min, raw_max = inspect_raw_depth(dataset, batch)
                print(f"Raw depth sample: {image_path}")
                print(f"Raw depth dtype/range: {raw_dtype}, min={raw_min:.1f}, max={raw_max:.1f}")

            depth_min = float(depth.min().item())
            depth_max = float(depth.max().item())
            print(
                f"Batch {step}: depth range=[{depth_min:.6f}, {depth_max:.6f}], "
                f"state={tuple(state_b.shape)}, guide_mask_valid={float(guide_mask.sum().item()):.0f}, "
                f"selected_topology_range=[{int(selected_topology.min().item())}, {int(selected_topology.max().item())}]"
            )
            if args.strict_depth_range and (depth_min < -1e-6 or depth_max > 1.0 + 1e-6):
                raise ValueError(f"Normalized depth out of [0,1]: [{depth_min}, {depth_max}]")

            depth = depth.to(device, non_blocking=True)
            state_b = state_b.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            endstate, score, pred_camera_target = model(depth, state_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_times.append(time.time() - t0)

            print(
                f"         output endstate={tuple(endstate.shape)}, score={tuple(score.shape)}, "
                f"camera={tuple(pred_camera_target.shape)}, target={tuple(camera_target.shape)}"
            )
            score_min.append(float(score.min().item()))
            score_max.append(float(score.max().item()))
            total_poses += depth.shape[0]
            total_dirs += depth.shape[0] * state_b.shape[1]

    if not forward_times:
        raise RuntimeError("No batches were tested.")

    mean_forward = sum(forward_times) / len(forward_times)
    print("----- Summary -----")
    print(f"Tested poses: {total_poses}")
    print(f"Tested direction samples: {total_dirs}")
    print(f"Mean forward time: {mean_forward * 1000.0:.2f} ms/batch")
    print(f"Score range over tested batches: [{min(score_min):.6f}, {max(score_max):.6f}]")
    print("Depth normalization check passed: this test uses the same Dataset path as training and does not re-normalize ROS topics.")


if __name__ == "__main__":
    main()
