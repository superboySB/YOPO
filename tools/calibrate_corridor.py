#!/usr/bin/env python3
"""Calibrate the MINCO corridor lower bound on the deterministic validation split.

The script does not tune against closed-loop benchmark outcomes.  It replays the
same validation split and state RNG stream for every sigma, computes the SDF
labels used by training, and selects the smallest sigma whose admitted
candidate precision reaches the requested target.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
YOPO_DIR = ROOT / "YOPO"
sys.path.insert(0, str(YOPO_DIR))

from config.config import cfg  # noqa: E402
from loss.loss_function import YOPOLoss  # noqa: E402
from policy.state_transform import (rotate_body2world, state_body2world,
                                    transform_body2world)  # noqa: E402
from policy.yopo_dataset import YOPODataset  # noqa: E402
from policy.yopo_network import YopoNetwork  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def predict_world(policy, depth, pos, rot, obs_b):
    """The validation lift in YopoTrainer._predict_world, with dynamic batch size."""
    batch = depth.shape[0]
    traj_num = int(cfg["traj_num"])
    count = batch * traj_num
    goal_w, start_vel_w, start_acc_w = state_body2world(
        pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6])
    inner_b, tail_pva_b, durations, score, inner_dr, radius = policy.inference(depth, obs_b)
    inner_b = inner_b.reshape(count, 3)
    tail_b = tail_pva_b.reshape(count, 3, 3)
    pos_ex = pos.repeat_interleave(traj_num, dim=0)
    rot_ex = rot.repeat_interleave(traj_num, dim=0)
    inner_w = transform_body2world(rot_ex, pos_ex, inner_b)
    tail_w = torch.stack([
        transform_body2world(rot_ex, pos_ex, tail_b[:, 0]),
        rotate_body2world(rot_ex, tail_b[:, 1]),
        rotate_body2world(rot_ex, tail_b[:, 2]),
    ], dim=1)
    head_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1).repeat_interleave(traj_num, dim=0)
    return {
        "head_w": head_w,
        "tail_w": tail_w,
        "inner_w": inner_w,
        "goal_w": goal_w.repeat_interleave(traj_num, dim=0),
        "durations": durations.reshape(count, 2),
        "score": score,
        "inner_dr": inner_dr,
        "radius": radius.reshape(count, -1),
    }


def wilson_lower(successes, total, z=1.959963984540054):
    if total == 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    spread = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return float((center - spread) / denominator)


def evaluate_sweep(mu, scale, truth, sigmas, safe_radius, radius_lambda,
                   target_precision, min_admissions):
    safe_mu = 1.0 - np.exp(-safe_radius / radius_lambda)
    true_safe = truth.min(axis=1) >= safe_radius
    rows = []
    for sigma in sigmas:
        lower = mu - sigma * scale
        admitted = lower.min(axis=1) >= safe_mu
        admitted_n = int(admitted.sum())
        true_positive = int((admitted & true_safe).sum())
        precision = None if admitted_n == 0 else true_positive / admitted_n
        recall = (true_positive / int(true_safe.sum())) if true_safe.any() else None
        coverage = float(np.mean((1.0 - np.exp(-truth / radius_lambda)) >= lower))
        rows.append({
            "sigma": float(sigma),
            "pointwise_coverage": coverage,
            "admitted_candidates": admitted_n,
            "admission_rate": float(admitted.mean()),
            "admission_precision": precision,
            "admission_precision_wilson95_lower": wilson_lower(true_positive, admitted_n),
            "safe_candidate_recall": recall,
            "unsafe_admissions": admitted_n - true_positive,
        })
    eligible = [row for row in rows
                if row["admitted_candidates"] >= min_admissions
                and row["admission_precision"] is not None
                and row["admission_precision"] >= target_precision]
    if not eligible:
        raise RuntimeError("no sigma satisfies target precision and minimum admissions")
    selected = eligible[0]
    return rows, selected


def plot_diagnostics(mu, scale, truth, rows, selected, safe_radius, radius_lambda, output):
    sigmas = np.asarray([row["sigma"] for row in rows])
    coverage = np.asarray([row["pointwise_coverage"] for row in rows])
    precision = np.asarray([np.nan if row["admission_precision"] is None else row["admission_precision"]
                            for row in rows])
    recall = np.asarray([np.nan if row["safe_candidate_recall"] is None else row["safe_candidate_recall"]
                         for row in rows])
    admission = np.asarray([row["admission_rate"] for row in rows])
    sigma = selected["sigma"]
    safe_mu = 1.0 - np.exp(-safe_radius / radius_lambda)
    lower = (mu - sigma * scale).min(axis=1)
    true_min = truth.min(axis=1)
    admitted = lower >= safe_mu
    lower_m = -radius_lambda * np.log1p(-np.clip(lower, 0.0, 1.0 - 1e-6))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].plot(sigmas, coverage, color="#0072b2", lw=2)
    axes[0].axvline(sigma, color="#d55e00", ls="--", label=f"selected k={sigma:g}")
    axes[0].axhline(0.95, color="#777777", ls=":", lw=1)
    axes[0].axhline(0.99, color="#777777", ls=":", lw=1)
    axes[0].set(xlabel="uncertainty multiplier k", ylabel="pointwise coverage",
                title="A. Coverage of y >= mu-kb", ylim=(0.0, 1.01))
    axes[0].legend(frameon=False)

    axes[1].plot(sigmas, precision, label="admission precision", color="#009e73", lw=2)
    axes[1].plot(sigmas, recall, label="safe-candidate recall", color="#cc79a7", lw=2)
    axes[1].plot(sigmas, admission, label="admission rate", color="#56b4e9", lw=2)
    axes[1].axvline(sigma, color="#d55e00", ls="--")
    axes[1].set(xlabel="uncertainty multiplier k", ylabel="fraction",
                title="B. Candidate admission", ylim=(0.0, 1.01))
    axes[1].legend(frameon=False, fontsize=8)

    rng = np.random.default_rng(0)
    indices = rng.choice(len(lower_m), size=min(120000, len(lower_m)), replace=False)
    axes[2].hexbin(lower_m[indices], true_min[indices], gridsize=70, bins="log",
                   mincnt=1, cmap="viridis", extent=(0, 2.5, 0, 2.5))
    axes[2].axvline(safe_radius, color="#d55e00", ls="--", lw=1)
    axes[2].axhline(safe_radius, color="#d55e00", ls="--", lw=1)
    axes[2].plot([0, 2.5], [0, 2.5], color="white", ls=":", lw=1)
    axes[2].set(xlabel="predicted min lower bound (m)", ylabel="SDF min label (m)",
                title=(f"C. Candidate calibration (k={sigma:g})\n"
                       f"accepted={admitted.sum():,}, unsafe={selected['unsafe_admissions']:,}"),
                xlim=(0, 2.5), ylim=(0, 2.5))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="YOPO/saved/yopo-minco/epoch50.pth")
    parser.add_argument("--dataset-path", default="../dataset_minco")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="development-only prefix limit; omit for the report calibration")
    parser.add_argument("--safe-radius", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=6.0)
    parser.add_argument("--sigma-step", type=float, default=0.1)
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--min-admissions", type=int, default=1000)
    parser.add_argument("--output-json", default="docs/report_assets/corridor_calibration.json")
    parser.add_argument("--output-plot", default="docs/report_assets/corridor_calibration.png")
    args = parser.parse_args()

    checkpoint = (ROOT / args.checkpoint).resolve() if not os.path.isabs(args.checkpoint) else Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    seed_everything(args.seed)
    cfg["dataset_path"] = args.dataset_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YOPODataset(mode="valid")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    policy = YopoNetwork().to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    policy.eval()
    objective = YOPOLoss()

    radii, labels = [], []
    processed_images = 0
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if args.max_batches is not None and index >= args.max_batches:
                break
            depth, pos, rot, obs_b, map_id = [value.to(device) for value in batch]
            traj = predict_world(policy, depth, pos, rot, obs_b)
            _costs, aux = objective(
                traj["head_w"], traj["tail_w"], traj["inner_w"], traj["goal_w"], map_id,
                durations=traj["durations"])
            radii.append(traj["radius"].detach().cpu().numpy())
            labels.append(aux["radius_dist"].detach().cpu().numpy())
            processed_images += int(depth.shape[0])
            if (index + 1) % 25 == 0:
                print(f"[calibration] batches={index + 1}/{len(loader)}")
    radius = np.concatenate(radii)
    truth = np.concatenate(labels)
    radius_num = int(cfg["radius_num"])
    mu, scale = radius[:, :radius_num], radius[:, radius_num:]
    sigmas = np.arange(0.0, args.sigma_max + args.sigma_step / 2.0, args.sigma_step)
    rows, selected = evaluate_sweep(
        mu, scale, truth, sigmas, args.safe_radius, float(cfg["radius_warp_lambda"]),
        args.target_precision, args.min_admissions)

    payload = {
        "method": "smallest sigma with empirical candidate precision >= target on fixed validation split",
        "checkpoint": os.path.relpath(checkpoint, ROOT),
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_path": args.dataset_path,
        "dataset_split": "YOPODataset(mode=valid, val_ratio=0.1, split_seed=0)",
        "state_seed": args.seed,
        "batch_size": args.batch_size,
        "processed_batches": len(radii),
        "validation_images": processed_images,
        "candidate_trajectories": int(len(mu)),
        "corridor_points": int(mu.size),
        "radius_num": radius_num,
        "radius_warp_lambda_m": float(cfg["radius_warp_lambda"]),
        "safe_radius_m": args.safe_radius,
        "target_precision": args.target_precision,
        "min_admissions": args.min_admissions,
        "selected_sigma": selected["sigma"],
        "selected_metrics": selected,
        "sweep": rows,
    }
    output_json = ROOT / args.output_json
    output_plot = ROOT / args.output_plot
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_diagnostics(mu, scale, truth, rows, selected, args.safe_radius,
                     float(cfg["radius_warp_lambda"]), output_plot)
    print(json.dumps({"selected_sigma": selected["sigma"],
                      "selected_metrics": selected,
                      "output_json": str(output_json),
                      "output_plot": str(output_plot)}, indent=2))


if __name__ == "__main__":
    main()
