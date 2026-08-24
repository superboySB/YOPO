#!/usr/bin/env python3
"""Fixed-checkpoint, fixed-input trajectory-representation ablation.

Every validation image is forwarded exactly once through the MINCO policy.  The same head/tail
boundary states, candidate scores and selected candidate IDs are then decoded as (1) one fixed-time
quintic, (2) two fixed-time MINCO pieces, and (3) two predicted-time MINCO pieces.  All modes use the
same time-normalized samples and the same ESDF/free-ball collision certificate.
"""

import argparse
import csv
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
sys.path.insert(0, str(ROOT / "YOPO"))
sys.path.insert(0, str(ROOT / "tools"))

from config.config import cfg  # noqa: E402
from loss.safety_loss import SafetyLoss  # noqa: E402
from policy.poly_solver import MincoTraj, QuinticTraj  # noqa: E402
from policy.yopo_dataset import YOPODataset  # noqa: E402
from policy.yopo_network import YopoNetwork  # noqa: E402
from calibrate_corridor import predict_world  # noqa: E402


MODES = ("single_fixed", "minco_fixed", "minco_variable")
LABELS = {
    "single_fixed": "single quintic / fixed T",
    "minco_fixed": "2-piece MINCO / fixed T",
    "minco_variable": "2-piece MINCO / predicted T",
}
COLORS = {"single_fixed": "#0072b2", "minco_fixed": "#d55e00", "minco_variable": "#009e73"}
METRICS = (
    "safe_candidate_fraction", "any_safe", "selected_safe", "selected_min_clearance_m",
    "selected_path_length_m", "selected_jerk_energy", "selected_acc_energy",
    "selected_max_speed_mps", "selected_max_acc_mps2", "selected_max_jerk_mps3",
    "selected_duration_s", "safe_oracle_min_jerk_energy",
)


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


def build_trajectory(mode, head, tail, inner, durations, nominal_total):
    if mode == "single_fixed":
        return QuinticTraj().solve(head, tail, np.full(head.shape[0], nominal_total))
    if mode == "minco_fixed":
        return MincoTraj().solve(head, tail, inner, np.full((head.shape[0], 2), nominal_total / 2.0))
    return MincoTraj().solve(head, tail, inner, durations)


def bootstrap_ci(values, rng, draws=10000, chunk=200):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None]
    samples = []
    for begin in range(0, draws, chunk):
        count = min(chunk, draws - begin)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        samples.append(values[indices].mean(axis=1))
    samples = np.concatenate(samples)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def evaluate_mode(mode, traj, safety, map_id, batch_size, traj_num, fractions, scores):
    sample = {name: traj.sample(fractions, deriv)
              for deriv, name in enumerate(("pos", "vel", "acc", "jerk"))}
    pos_t = torch.from_numpy(sample["pos"]).to(safety.device, dtype=torch.float32)
    with torch.no_grad():
        _cost, dist = safety.get_distance_cost(
            pos_t.reshape(batch_size, traj_num * len(fractions), 3), map_id)
        dist = dist.reshape(batch_size * traj_num, len(fractions))
        collided, _first = safety._collision_certificate(pos_t, dist)
    dist = dist.cpu().numpy()
    safe = ~collided.cpu().numpy()

    total = np.asarray(traj.total_time, dtype=np.float64)
    if total.ndim == 0:
        total = np.full(batch_size * traj_num, float(total))
    pos, vel, acc, jerk = (sample[name] for name in ("pos", "vel", "acc", "jerk"))
    path = np.linalg.norm(np.diff(pos, axis=1), axis=-1).sum(axis=1)
    jerk_energy = np.trapz(np.square(jerk).sum(axis=-1), x=fractions, axis=1) * total
    acc_energy = np.trapz(np.square(acc).sum(axis=-1), x=fractions, axis=1) * total
    values = {
        "safe": safe.reshape(batch_size, traj_num),
        "min_clearance": dist.min(axis=1).reshape(batch_size, traj_num),
        "path_length": path.reshape(batch_size, traj_num),
        "jerk_energy": jerk_energy.reshape(batch_size, traj_num),
        "acc_energy": acc_energy.reshape(batch_size, traj_num),
        "max_speed": np.linalg.norm(vel, axis=-1).max(axis=1).reshape(batch_size, traj_num),
        "max_acc": np.linalg.norm(acc, axis=-1).max(axis=1).reshape(batch_size, traj_num),
        "max_jerk": np.linalg.norm(jerk, axis=-1).max(axis=1).reshape(batch_size, traj_num),
        "duration": total.reshape(batch_size, traj_num),
        "sample": sample,
        "distance": dist,
    }
    selected = scores.argmax(axis=1)
    rows = []
    for image in range(batch_size):
        chosen = selected[image]
        safe_jerk = values["jerk_energy"][image][values["safe"][image]]
        rows.append({
            "mode": mode,
            "selected_candidate": int(chosen),
            "safe_candidate_fraction": float(values["safe"][image].mean()),
            "any_safe": int(values["safe"][image].any()),
            "selected_safe": int(values["safe"][image, chosen]),
            "selected_min_clearance_m": float(values["min_clearance"][image, chosen]),
            "selected_path_length_m": float(values["path_length"][image, chosen]),
            "selected_jerk_energy": float(values["jerk_energy"][image, chosen]),
            "selected_acc_energy": float(values["acc_energy"][image, chosen]),
            "selected_max_speed_mps": float(values["max_speed"][image, chosen]),
            "selected_max_acc_mps2": float(values["max_acc"][image, chosen]),
            "selected_max_jerk_mps3": float(values["max_jerk"][image, chosen]),
            "selected_duration_s": float(values["duration"][image, chosen]),
            "safe_oracle_min_jerk_energy": (float(safe_jerk.min()) if len(safe_jerk) else float("nan")),
        })
    return rows, values, selected


def plot_summary(rows, summary, output):
    definitions = [
        ("safe_candidate_fraction", "Safe candidates", 100.0, "%"),
        ("any_safe", "Frames with ≥1 safe", 100.0, "%"),
        ("selected_safe", "Score-selected safe", 100.0, "%"),
        ("selected_min_clearance_m", "Selected min clearance", 1.0, "m"),
        ("selected_path_length_m", "Selected path length", 1.0, "m"),
        ("selected_jerk_energy", "Selected ∫||jerk||²dt", 1.0, "m²/s⁵"),
        ("selected_max_acc_mps2", "Selected max acceleration", 1.0, "m/s²"),
        ("selected_duration_s", "Selected duration", 1.0, "s"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.0), constrained_layout=True)
    for ax, (metric, title, scale, unit) in zip(axes.flat, definitions):
        means = [summary["mode_means"][mode][metric] * scale for mode in MODES]
        ax.bar(range(3), means, color=[COLORS[m] for m in MODES], alpha=0.9)
        ax.set_xticks(range(3), ["single\nfixed", "MINCO\nfixed", "MINCO\nvariable"], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25)
        for index, value in enumerate(means):
            ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Fixed checkpoint/input/score decoder ablation (validation ESDF)", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_distributions(rows, output):
    definitions = [
        ("safe_candidate_fraction", "Safe-candidate fraction", 1.0, None),
        ("selected_min_clearance_m", "Selected minimum clearance (m)", 1.0, (-0.5, 3.0)),
        ("selected_path_length_m", "Selected path length (m)", 1.0, (0.0, 15.0)),
        ("selected_jerk_energy", "Selected jerk energy (m²/s⁵)", 1.0, (1.0, 1e5)),
        ("selected_max_acc_mps2", "Selected maximum acceleration (m/s²)", 1.0, (0.0, 30.0)),
        ("selected_duration_s", "Selected duration (s)", 1.0, (0.0, 3.4)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    for ax, (metric, title, scale, limits) in zip(axes.flat, definitions):
        for mode in MODES:
            values = np.sort(np.asarray([r[metric] for r in rows if r["mode"] == mode], dtype=float) * scale)
            finite = np.isfinite(values); values = values[finite]
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=COLORS[mode],
                    lw=1.8, label=LABELS[mode])
        if metric == "selected_jerk_energy":
            ax.set_xscale("log")
        if limits is not None:
            ax.set_xlim(*limits)
        ax.set(title=title, ylabel="empirical CDF")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Decoder distributions over 10,000 matched validation observations", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_example(example, output):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    axes[0, 0].imshow(example["depth"], cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("A. Network input depth")
    axes[0, 0].axis("off")
    for mode in MODES:
        item = example["modes"][mode]
        pos = item["pos"]
        axes[0, 1].plot(pos[:, 0], pos[:, 1], color=COLORS[mode], lw=2, label=LABELS[mode])
        axes[0, 2].plot(pos[:, 0], pos[:, 2], color=COLORS[mode], lw=2)
        axes[1, 0].plot(example["fractions"], item["clearance"], color=COLORS[mode], lw=2)
        axes[1, 1].plot(example["fractions"], item["speed"], color=COLORS[mode], lw=2)
        axes[1, 2].plot(example["fractions"], item["jerk"], color=COLORS[mode], lw=2)
    axes[0, 1].set(title="B. World XY path", xlabel="x (m)", ylabel="y (m)")
    axes[0, 2].set(title="C. World XZ path", xlabel="x (m)", ylabel="z (m)")
    axes[1, 0].axhline(0, color="black", ls="--", lw=1)
    axes[1, 0].set(title="D. ESDF clearance", xlabel="normalized time", ylabel="m")
    axes[1, 1].set(title="E. Speed", xlabel="normalized time", ylabel="m/s")
    axes[1, 2].set(title="F. Jerk magnitude", xlabel="normalized time", ylabel="m/s³")
    axes[0, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat[1:]:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Matched selected candidate, validation image {example['image_index']}", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="YOPO/saved/yopo-minco/epoch50.pth")
    parser.add_argument("--dataset-path", default="../dataset_minco")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--raw-csv", default="results/decoder_validation_raw.csv")
    parser.add_argument("--output-json", default="docs/report_assets/decoder_validation.json")
    parser.add_argument("--output-plot", default="docs/report_assets/decoder_validation.png")
    parser.add_argument("--distribution-plot", default="docs/report_assets/decoder_distributions.png")
    parser.add_argument("--example-plot", default="docs/report_assets/decoder_example.png")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()
    seed_everything(args.seed)
    cfg["dataset_path"] = args.dataset_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = YOPODataset(mode="valid")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    policy = YopoNetwork().to(device)
    policy.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    policy.eval()
    safety = SafetyLoss()
    fractions = np.linspace(0.0, 1.0, args.samples)
    nominal_total = float(cfg["sgm_time"])
    traj_num = int(cfg["traj_num"])

    all_rows = []
    example = None
    example_separation = -np.inf
    processed = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            depth, pos, rot, obs_b, map_id = [value.to(device) for value in batch]
            predicted = predict_world(policy, depth, pos, rot, obs_b)
            count = depth.shape[0] * traj_num
            head = predicted["head_w"].cpu().numpy()
            tail = predicted["tail_w"].cpu().numpy()
            inner = predicted["inner_w"].cpu().numpy()
            durations = predicted["durations"].cpu().numpy()
            scores = predicted["score"].reshape(depth.shape[0], traj_num).cpu().numpy()
            batch_outputs = {}
            for mode in MODES:
                traj = build_trajectory(mode, head, tail, inner, durations, nominal_total)
                mode_rows, values, selected = evaluate_mode(
                    mode, traj, safety, map_id, depth.shape[0], traj_num, fractions, scores)
                for local_index, row in enumerate(mode_rows):
                    row["image_index"] = processed + local_index
                    row["map_id"] = int(map_id[local_index].item())
                all_rows.extend(mode_rows)
                batch_outputs[mode] = (values, selected)

            for local_index in range(depth.shape[0]):
                chosen = batch_outputs["single_fixed"][1][local_index]
                flat_id = local_index * traj_num + chosen
                p0 = batch_outputs["single_fixed"][0]["sample"]["pos"][flat_id]
                p1 = batch_outputs["minco_fixed"][0]["sample"]["pos"][flat_id]
                separation = float(np.linalg.norm(p1 - p0, axis=1).max())
                if separation > example_separation:
                    example_separation = separation
                    example = {"image_index": processed + local_index,
                               "depth": depth[local_index, 0].cpu().numpy(),
                               "fractions": fractions, "modes": {}}
                    for mode in MODES:
                        values = batch_outputs[mode][0]
                        mode_flat_id = local_index * traj_num + chosen
                        example["modes"][mode] = {
                            "pos": values["sample"]["pos"][mode_flat_id],
                            "speed": np.linalg.norm(values["sample"]["vel"][mode_flat_id], axis=1),
                            "jerk": np.linalg.norm(values["sample"]["jerk"][mode_flat_id], axis=1),
                            "clearance": values["distance"][mode_flat_id],
                        }
            processed += int(depth.shape[0])
            if (batch_index + 1) % 25 == 0:
                print(f"[decoder] batches={batch_index + 1}/{len(loader)}, images={processed}")

    raw_path = ROOT / args.raw_csv
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    mode_rows = {mode: sorted((row for row in all_rows if row["mode"] == mode),
                              key=lambda row: row["image_index"]) for mode in MODES}
    summary = {"mode_means": {}, "mode_medians": {}, "paired_effects": {}}
    for mode in MODES:
        summary["mode_means"][mode] = {
            metric: float(np.nanmean([row[metric] for row in mode_rows[mode]])) for metric in METRICS}
        summary["mode_medians"][mode] = {
            metric: float(np.nanmedian([row[metric] for row in mode_rows[mode]])) for metric in METRICS}
    rng = np.random.default_rng(20260824)
    comparisons = {
        "geometry_minco_fixed_minus_single_fixed": ("single_fixed", "minco_fixed"),
        "duration_variable_minus_minco_fixed": ("minco_fixed", "minco_variable"),
    }
    for name, (baseline, variant) in comparisons.items():
        summary["paired_effects"][name] = {}
        for metric in METRICS:
            diff = np.asarray([v[metric] - b[metric]
                               for b, v in zip(mode_rows[baseline], mode_rows[variant])], dtype=float)
            summary["paired_effects"][name][metric] = {
                "mean": float(np.nanmean(diff)),
                "bootstrap_95ci": bootstrap_ci(diff, rng),
                "n": int(np.isfinite(diff).sum()),
            }
    payload = {
        "design": "one MINCO forward per image; identical head/tail/score/candidate ID; decoder only",
        "checkpoint": os.path.relpath(checkpoint, ROOT),
        "checkpoint_sha256": sha256(checkpoint),
        "dataset_path": args.dataset_path,
        "split": "YOPODataset(valid), sklearn split seed 0",
        "state_seed": args.seed,
        "validation_images": processed,
        "candidate_trajectories_per_mode": processed * traj_num,
        "samples_per_trajectory": args.samples,
        "collision_rule": "shared ESDF and SafetyLoss free-ball chain certificate",
        "nominal_total_time_s": nominal_total,
        **summary,
        "raw_csv": os.path.relpath(raw_path, ROOT),
    }
    output_json = ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    plot_summary(all_rows, payload, ROOT / args.output_plot)
    plot_distributions(all_rows, ROOT / args.distribution_plot)
    plot_example(example, ROOT / args.example_plot)
    print(json.dumps({"validation_images": processed, "rows": len(all_rows),
                      "output_json": str(output_json), "raw_csv": str(raw_path)}, indent=2))


if __name__ == "__main__":
    main()
