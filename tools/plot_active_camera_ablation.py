#!/usr/bin/env python3
"""Plot rates, paired outcomes and map-overlaid trajectories for the A/B run."""

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path


COLORS = {"active": "#1976d2", "fixed": "#ef6c00"}


def load_summary(path):
    path = Path(path).resolve()
    return path, json.loads(path.read_text())


def resolve_artifact(summary_path, artifact):
    if not artifact:
        return None
    path = Path(artifact)
    if not path.is_absolute():
        path = summary_path.parent / path
    return path


def read_columns(path, names, compressed=False):
    if path is None or not path.exists():
        return {name: [] for name in names}
    opener = gzip.open if compressed or path.suffix == ".gz" else open
    output = {name: [] for name in names}
    with opener(str(path), "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            valid = True
            parsed = {}
            for name in names:
                try:
                    parsed[name] = float(row[name])
                except (KeyError, TypeError, ValueError):
                    valid = False
                    break
            if valid:
                for name in names:
                    output[name].append(parsed[name])
    return output


def rate_error(aggregate, arm, metric):
    value = aggregate.get(arm, {}).get(metric)
    interval = aggregate.get(arm, {}).get(metric + "_wilson95", [None, None])
    if value is None or interval[0] is None or interval[1] is None:
        return None, None, None
    return value, value - interval[0], interval[1] - value


def comparison_plot(summary, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = summary.get("aggregate", {})
    paired = summary.get("paired", {}).get("per_seed", [])
    arms = ["active", "fixed"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    x = [0, 1]
    width = 0.34
    for offset, (metric, title) in enumerate(
        [
            ("collision_rate", "Collision rate"),
            ("collision_free_success_rate", "Collision-free success"),
        ]
    ):
        values, lower, upper = [], [], []
        for arm in arms:
            value, low_error, high_error = rate_error(aggregate, arm, metric)
            values.append(float("nan") if value is None else value)
            lower.append(0.0 if low_error is None else low_error)
            upper.append(0.0 if high_error is None else high_error)
        locations = [value + (offset - 0.5) * width for value in x]
        bars = axes[0].bar(
            locations,
            values,
            width=width,
            label=title,
            yerr=[lower, upper],
            capsize=4,
            alpha=0.88,
        )
        for bar, value, arm in zip(bars, values, arms):
            evaluated = aggregate.get(arm, {}).get("evaluated_runs", 0)
            label = (
                "N/A (n=0)" if math.isnan(value)
                else "{:.1f}% (n={})".format(100 * value, evaluated)
            )
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                0.03 if math.isnan(value) else min(1.08, value + 0.035),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Active", "Fixed"])
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Run fraction (Wilson 95% CI)")
    axes[0].set_title("Outcome rates")
    axes[0].legend(loc="upper center", fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    seeds = [item["seed"] for item in paired]
    collision_delta = [item["collision_active_minus_fixed"] for item in paired]
    success_delta = [item["collision_free_success_active_minus_fixed"] for item in paired]
    locations = list(range(len(seeds)))
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].plot(locations, collision_delta, "o-", label="Collision A-F", color="#c62828")
    axes[1].plot(
        locations, success_delta, "s-",
        label="Collision-free success A-F", color="#2e7d32",
    )
    axes[1].set_xticks(locations)
    axes[1].set_xticklabels([str(seed) for seed in seeds], rotation=35, ha="right")
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_ylim(-1.25, 1.25)
    axes[1].set_xlabel("Holdout map seed")
    axes[1].set_ylabel("Paired indicator difference")
    axes[1].set_title("Per-seed paired outcomes")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    clearance_active = [
        item.get("min_clearance_active_m")
        for item in paired
        if item.get("min_clearance_active_m") is not None
        and item.get("min_clearance_fixed_m") is not None
    ]
    clearance_fixed = [
        item.get("min_clearance_fixed_m")
        for item in paired
        if item.get("min_clearance_active_m") is not None
        and item.get("min_clearance_fixed_m") is not None
    ]
    axes[2].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    for index, (active, fixed) in enumerate(zip(clearance_active, clearance_fixed)):
        axes[2].plot([0, 1], [active, fixed], color="#777777", alpha=0.45, linewidth=1)
        axes[2].scatter([0], [active], color=COLORS["active"], s=28, zorder=3)
        axes[2].scatter([1], [fixed], color=COLORS["fixed"], s=28, zorder=3)
    axes[2].set_xlim(-0.4, 1.4)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Active", "Fixed"])
    axes[2].set_ylabel("Minimum surface clearance (m)")
    axes[2].set_title("Paired minimum clearance")
    axes[2].grid(axis="y", alpha=0.25)

    complete_pairs = summary.get("paired", {}).get("complete_pairs", 0)
    fig.suptitle("Insight 9 active-camera ablation ({} complete seed pairs)".format(complete_pairs))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def trajectory_plot(summary_path, summary, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = {}
    for run in summary.get("runs", []):
        if run.get("status") not in ("ok", "collision", "incomplete"):
            continue
        grouped.setdefault(run.get("seed"), {})[run.get("arm")] = run
    seeds = [seed for seed in sorted(grouped) if set(grouped[seed]) == {"active", "fixed"}]
    if not seeds:
        raise ValueError("No complete active/fixed trajectory pairs to plot")

    columns = min(3, len(seeds))
    rows = int(math.ceil(len(seeds) / float(columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 4.5 * rows), squeeze=False)
    for axis, seed in zip([item for row in axes for item in row], seeds):
        runs = grouped[seed]
        if runs["active"].get("map_sha256") != runs["fixed"].get("map_sha256"):
            raise ValueError("Seed {} active/fixed map hashes differ".format(seed))
        map_run = runs["active"] if runs["active"].get("map_csv_gz") else runs["fixed"]
        map_path = resolve_artifact(summary_path, map_run.get("map_csv_gz"))
        points = read_columns(map_path, ["x", "y", "z"], compressed=True)
        if points["x"]:
            obstacle = [index for index, z in enumerate(points["z"]) if 0.45 <= z <= 4.55]
            axis.scatter(
                [points["x"][index] for index in obstacle],
                [points["y"][index] for index in obstacle],
                s=0.35, c="#666666", alpha=0.28, rasterized=True,
            )

        for arm in ("fixed", "active"):
            run = runs[arm]
            trajectory_path = resolve_artifact(summary_path, run.get("trajectory_csv"))
            trajectory = read_columns(
                trajectory_path, ["x", "y", "body_yaw_rad", "camera_yaw_rad"]
            )
            if trajectory["x"]:
                axis.plot(
                    trajectory["x"],
                    trajectory["y"],
                    color=COLORS[arm],
                    linewidth=1.8,
                    label="{} ({}; collisions={})".format(
                        arm.capitalize(),
                        "success" if run.get("success") else run.get("status"),
                        run.get("collision_episode_count", 0),
                    ),
                )
                axis.scatter(
                    [trajectory["x"][0]], [trajectory["y"][0]], marker="o", s=35,
                    color=COLORS[arm], edgecolor="black", linewidth=0.4, zorder=4,
                )
                axis.scatter(
                    [trajectory["x"][-1]], [trajectory["y"][-1]], marker="^", s=45,
                    color=COLORS[arm], edgecolor="black", linewidth=0.4, zorder=4,
                )
                stride = max(1, len(trajectory["x"]) // 12)
                indices = list(range(0, len(trajectory["x"]), stride))[:12]
                view_yaw = [
                    trajectory["body_yaw_rad"][index] + trajectory["camera_yaw_rad"][index]
                    for index in indices
                ]
                axis.quiver(
                    [trajectory["x"][index] for index in indices],
                    [trajectory["y"][index] for index in indices],
                    [math.cos(yaw) for yaw in view_yaw],
                    [math.sin(yaw) for yaw in view_yaw],
                    color=COLORS[arm], alpha=0.55, scale=18, width=0.003, zorder=3,
                )
            collision_path = resolve_artifact(summary_path, run.get("collision_csv"))
            collisions = read_columns(collision_path, ["start_x", "start_y"])
            if collisions["start_x"]:
                axis.scatter(
                    collisions["start_x"],
                    collisions["start_y"],
                    marker="x",
                    s=55,
                    linewidth=1.8,
                    color=COLORS[arm],
                    zorder=5,
                )
        axis.set_title("Holdout seed {}".format(seed))
        axis.set_xlabel("World x (m)")
        axis.set_ylabel("World y (m)")
        axis.axis("equal")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="best")

    flat_axes = [item for row in axes for item in row]
    for axis in flat_axes[len(seeds):]:
        axis.set_visible(False)
    fig.suptitle("Active/fixed camera trajectories on identical Insight 9 maps")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_plots(summary_json, output_dir):
    summary_path, summary = load_summary(summary_json)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = [comparison_plot(summary, output_dir / "metrics_comparison.png")]
    try:
        generated.append(
            trajectory_plot(summary_path, summary, output_dir / "trajectory_overlay.png")
        )
    except ValueError as exc:
        print("warning: {}".format(exc), file=sys.stderr)
    return generated


def make_parser():
    parser = argparse.ArgumentParser(description="Plot a YOPO active-camera A/B summary.")
    parser.add_argument("--summary", required=True, help="summary.json")
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    summary = Path(args.summary).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else summary.parent / "plots"
    try:
        paths = generate_plots(summary, output_dir)
        for path in paths:
            print(path)
        return 0
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
