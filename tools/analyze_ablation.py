#!/usr/bin/env python3
"""Aggregate paired MINCO selection/corridor ablations and render report assets."""

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_comparison import common_metrics  # noqa: E402


BASELINE = "minco-baseline"
EXPERIMENTS = ("inner_topk3", "jerk_topk3", "corridor_calibrated")
LABELS = {
    "inner_topk3": "top-3 / inner waypoint",
    "jerk_topk3": "top-3 / jerk continuity",
    "corridor_calibrated": "calibrated corridor",
}
COLORS = {
    "inner_topk3": "#0072b2",
    "jerk_topk3": "#d55e00",
    "corridor_calibrated": "#009e73",
}
MAZE_NAMES = {1: "Perlin", 2: "Columns", 5: "Forest", 7: "Walls"}
METRICS = (
    "collision_free_success", "collision_events", "timeout_penalized_s",
    "path_efficiency", "min_clearance_m", "command_acc_rms_mps2",
    "command_jerk_p95_mps3", "command_active_fraction", "turn_response_penalized_s",
)


def finite_mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not values else float(np.mean(values))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_runs(raw_root):
    runs = []
    for experiment in EXPERIMENTS:
        for path in sorted(glob.glob(os.path.join(raw_root, experiment, "*.json"))):
            with open(path, "r", encoding="utf-8") as handle:
                run = json.load(handle)
            if BASELINE not in run["vehicles"] or experiment not in run["vehicles"]:
                raise ValueError(f"unexpected vehicle labels in {path}")
            run["_path"] = path
            run["_experiment"] = experiment
            runs.append(run)
    return runs


def build_rows(runs):
    rows = []
    for run in runs:
        experiment = run["_experiment"]
        source = os.path.basename(run["_path"])
        condition = source.replace("_swapped.json", ".json")
        lane_order = ("normal" if run.get("lanes", {}).get("yopo_simple") == BASELINE
                      else "swapped")
        for model, role in ((BASELINE, "baseline"), (experiment, "variant")):
            row = common_metrics(run, model)
            row["experiment"] = experiment
            row["role"] = role
            row["model"] = model
            row["condition"] = condition
            row["lane_order"] = lane_order
            rows.append(row)
    return rows


def aggregate(rows):
    output = []
    for experiment in EXPERIMENTS:
        for role in ("baseline", "variant"):
            for scenario in ("straight", "dynamic", "all"):
                for maze_type in (1, 2, 5, 7, "all"):
                    group = [row for row in rows if row["experiment"] == experiment
                             and row["role"] == role and row["valid_start"]]
                    if scenario != "all":
                        group = [row for row in group if row["scenario"] == scenario]
                    if maze_type != "all":
                        group = [row for row in group if row["maze_type"] == maze_type]
                    if not group:
                        continue
                    output.append({
                        "experiment": experiment,
                        "role": role,
                        "scenario": scenario,
                        "maze_type": maze_type,
                        "n": len(group),
                        "collision_free_success_rate": finite_mean(
                            [row["collision_free_success"] for row in group]),
                        "collision_episode_rate": finite_mean(
                            [row["collision_events"] > 0 for row in group]),
                        "timeout_penalized_time_mean_s": finite_mean(
                            [row["timeout_penalized_s"] for row in group]),
                        "path_efficiency_mean": finite_mean(
                            [row.get("path_efficiency") for row in group]),
                        "min_clearance_mean_m": finite_mean(
                            [row.get("min_clearance_m") for row in group]),
                        "command_acc_rms_mean_mps2": finite_mean(
                            [row.get("command_acc_rms_mps2") for row in group]),
                        "command_jerk_p95_mean_mps3": finite_mean(
                            [row.get("command_jerk_p95_mps3") for row in group]),
                        "command_active_fraction_mean": finite_mean(
                            [row.get("command_active_fraction") for row in group]),
                        "turn_response_penalized_mean_s": finite_mean(
                            [row.get("turn_response_penalized_s") for row in group]),
                    })
    return output


def paired_statistics(rows):
    rng = np.random.default_rng(20260824)
    output = {}
    for experiment in EXPERIMENTS:
        output[experiment] = {}
        for scenario in ("straight", "dynamic", "all"):
            subset = [row for row in rows if row["experiment"] == experiment
                      and (scenario == "all" or row["scenario"] == scenario)]
            pairs = []
            for source in sorted({row["source"] for row in subset}):
                pair = {row["role"]: row for row in subset if row["source"] == source}
                if set(pair) == {"baseline", "variant"} and all(
                        row["valid_start"] for row in pair.values()):
                    pairs.append(pair)
            result = {"paired_episodes": len(pairs), "metrics": {}}
            for metric in METRICS:
                diffs_by_condition = {}
                for pair in pairs:
                    base, variant = pair["baseline"].get(metric), pair["variant"].get(metric)
                    if base is not None and variant is not None:
                        condition = pair["baseline"]["condition"]
                        diffs_by_condition.setdefault(condition, []).append(float(variant) - float(base))
                if not diffs_by_condition:
                    continue
                # Lane orders are repeated measurements of one map/seed/scenario condition.
                # Average within condition, then bootstrap condition clusters.
                values = np.asarray([np.mean(items) for items in diffs_by_condition.values()])
                bootstrap = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
                result["metrics"][metric] = {
                    "paired_episodes": sum(len(items) for items in diffs_by_condition.values()),
                    "condition_clusters": len(values),
                    "mean_variant_minus_baseline": float(values.mean()),
                    "bootstrap_95ci": [float(np.percentile(bootstrap, 2.5)),
                                       float(np.percentile(bootstrap, 97.5))],
                }
            output[experiment][scenario] = result
    return output


def plot_effects(paired, output):
    definitions = [
        ("collision_free_success", "Success (pp)", 100.0, "all"),
        ("timeout_penalized_s", "Penalized time (s)", 1.0, "all"),
        ("path_efficiency", "Path efficiency (pp)", 100.0, "all"),
        ("min_clearance_m", "Min clearance (cm)", 100.0, "all"),
        ("command_acc_rms_mps2", "Acceleration RMS (m/s²)", 1.0, "all"),
        ("command_jerk_p95_mps3", "Jerk p95 (m/s³)", 1.0, "all"),
        ("command_active_fraction", "Active commands (pp)", 100.0, "all"),
        ("turn_response_penalized_s", "Dynamic turn response (s)", 1.0, "dynamic"),
    ]
    fig, axes = plt.subplots(1, len(definitions), figsize=(18, 4.8), constrained_layout=True)
    y = np.arange(len(EXPERIMENTS))
    for ax, (metric, title, scale, scenario) in zip(axes, definitions):
        means, low, high = [], [], []
        for experiment in EXPERIMENTS:
            item = paired[experiment][scenario]["metrics"].get(metric)
            if item is None:
                means.append(np.nan); low.append(np.nan); high.append(np.nan)
            else:
                means.append(item["mean_variant_minus_baseline"] * scale)
                low.append(item["bootstrap_95ci"][0] * scale)
                high.append(item["bootstrap_95ci"][1] * scale)
        means, low, high = map(np.asarray, (means, low, high))
        for index, experiment in enumerate(EXPERIMENTS):
            ax.errorbar(means[index], y[index],
                        xerr=[[means[index] - low[index]], [high[index] - means[index]]],
                        fmt="o", color=COLORS[experiment], capsize=3, ms=6)
        ax.axvline(0, color="#333333", lw=1)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels([LABELS[item] for item in EXPERIMENTS], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
    fig.suptitle("Variant − baseline paired mean and condition-cluster bootstrap 95% CI", fontsize=13)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_by_map(rows, output):
    metrics = [
        ("collision_free_success", "Success difference (pp)", 100.0, "RdBu", -100, 100),
        ("timeout_penalized_s", "Penalized-time difference (s)", 1.0, "RdBu_r", -8, 8),
        ("command_jerk_p95_mps3", "Jerk-p95 difference (m/s³)", 1.0, "RdBu_r", -15, 15),
        ("command_active_fraction", "Active-command difference (pp)", 100.0, "RdBu", -100, 100),
    ]
    columns = [(maze, scenario) for maze in (1, 2, 5, 7) for scenario in ("straight", "dynamic")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 10.0), constrained_layout=True)
    for ax, (metric, title, scale, cmap, vmin, vmax) in zip(axes, metrics):
        matrix = np.full((len(EXPERIMENTS), len(columns)), np.nan)
        for i, experiment in enumerate(EXPERIMENTS):
            for j, (maze, scenario) in enumerate(columns):
                group = [row for row in rows if row["experiment"] == experiment
                         and row["maze_type"] == maze and row["scenario"] == scenario]
                per_source = []
                for source in sorted({row["source"] for row in group}):
                    pair = {row["role"]: row for row in group if row["source"] == source}
                    if set(pair) == {"baseline", "variant"}:
                        a, b = pair["variant"].get(metric), pair["baseline"].get(metric)
                        if a is not None and b is not None:
                            per_source.append((float(a) - float(b)) * scale)
                matrix[i, j] = finite_mean(per_source)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        ax.set_yticks(range(len(EXPERIMENTS)), [LABELS[item] for item in EXPERIMENTS])
        ax.set_xticks(range(len(columns)),
                      [f"{MAZE_NAMES[maze]}\n{scenario[0].upper()}" for maze, scenario in columns])
        ax.set_title(title, loc="left", fontsize=11)
        fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def command_series(payload):
    command = np.asarray(payload["commands"], dtype=float)
    active = command[command[:, -1] == 1]
    t = active[:, 0]
    acc = np.linalg.norm(active[:, 7:10], axis=1)
    dt = np.diff(t)
    jerk = np.linalg.norm(np.diff(active[:, 7:10], axis=0)
                          / np.maximum(dt[:, None], 1e-9), axis=1)
    return t, acc, t[1:], jerk


def select_representative(runs):
    candidates = [run for run in runs if run["_experiment"] == "jerk_topk3"
                  and run["scenario"] == "dynamic"]
    def gain(run):
        left = common_metrics(run, BASELINE).get("command_jerk_p95_mps3")
        right = common_metrics(run, "jerk_topk3").get("command_jerk_p95_mps3")
        return -np.inf if left is None or right is None else left - right
    return max(candidates, key=gain)


def plot_profiles(run, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    points = np.asarray(run.get("map_points", []), dtype=float)
    if points.size:
        axes[0, 0].scatter(points[::max(1, len(points) // 9000), 0],
                           points[::max(1, len(points) // 9000), 1],
                           s=1, c="#bbbbbb", alpha=0.25, rasterized=True)
    for label, color, name in ((BASELINE, "#555555", "baseline"),
                               ("jerk_topk3", COLORS["jerk_topk3"], "jerk top-3")):
        payload = run["vehicles"][label]
        odom = np.asarray(payload["odom"], dtype=float)
        axes[0, 0].plot(odom[:, 1], odom[:, 2], color=color, lw=2, label=name)
        t, acc, tj, jerk = command_series(payload)
        axes[0, 1].plot(t, acc, color=color, lw=1.3, label=name)
        axes[1, 0].plot(tj, jerk, color=color, lw=1.0, alpha=0.9, label=name)
        goal = np.asarray(run["final_goal"])[None, :2] - np.asarray(payload["commands"])[:, 1:3]
        velocity = np.asarray(payload["commands"])[:, 4:6]
        denominator = np.linalg.norm(goal, axis=1) * np.linalg.norm(velocity, axis=1)
        angle = np.degrees(np.arccos(np.clip(np.sum(goal * velocity, axis=1)
                                             / np.maximum(denominator, 1e-9), -1, 1)))
        axes[1, 1].plot(np.asarray(payload["commands"])[:, 0], angle,
                        color=color, lw=1.3, label=name)
    switch = float(run["switch_time_s"])
    for ax in axes.flat[1:]:
        ax.axvline(switch, color="#7b3294", ls="--", lw=1, label="goal switch")
        ax.grid(alpha=0.25)
    axes[0, 0].scatter([run["initial_goal"][0], run["final_goal"][0]],
                       [run["initial_goal"][1], run["final_goal"][1]], marker="*", s=120,
                       c=["#e69f00", "#7b3294"], edgecolors="black", linewidths=.4)
    axes[0, 0].set(xlabel="x (m)", ylabel="y (m)", title="A. Closed-loop XY path")
    axes[0, 0].axis("equal"); axes[0, 0].legend(frameon=False)
    axes[0, 1].set(xlabel="episode time (s)", ylabel="|a_cmd| (m/s²)", title="B. Acceleration command")
    axes[1, 0].set(xlabel="episode time (s)", ylabel="|Δa/Δt| (m/s³)", title="C. Command jerk")
    axes[1, 1].axhline(15, color="#777777", ls=":", lw=1)
    axes[1, 1].set(xlabel="episode time (s)", ylabel="goal/velocity angle (deg)",
                   title="D. Reorientation after goal switch", ylim=(0, 180))
    fig.suptitle(f"High jerk-effect paired episode: maze {run['maze_type']}, seed {run['map_seed']}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def select_corridor_stall(runs):
    candidates = []
    for run in runs:
        if run["_experiment"] != "corridor_calibrated":
            continue
        baseline = common_metrics(run, BASELINE)
        variant = common_metrics(run, "corridor_calibrated")
        if baseline.get("collision_free_success") and not variant.get("collision_free_success"):
            candidates.append((variant.get("command_active_fraction", 1.0), run))
    if not candidates:
        candidates = [(common_metrics(run, "corridor_calibrated").get(
            "command_active_fraction", 1.0), run) for run in runs
                      if run["_experiment"] == "corridor_calibrated"]
    return min(candidates, key=lambda item: item[0])[1]


def plot_corridor_stall(run, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    points = np.asarray(run.get("map_points", []), dtype=float)
    if points.size:
        stride = max(1, len(points) // 9000)
        axes[0].scatter(points[::stride, 0], points[::stride, 1], s=1,
                        c="#b0b0b0", alpha=.24, rasterized=True)
    styles = ((BASELINE, "#555555", "baseline"),
              ("corridor_calibrated", COLORS["corridor_calibrated"], "calibrated k=1.8"))
    goal = np.asarray(run["final_goal"], dtype=float)
    for label, color, name in styles:
        payload = run["vehicles"][label]
        odom = np.asarray(payload["odom"], dtype=float)
        axes[0].plot(odom[:, 1], odom[:, 2], color=color, lw=2, label=name)
        distance = np.linalg.norm(odom[:, 1:4] - goal[None, :], axis=1)
        axes[1].plot(odom[:, 0], distance, color=color, lw=1.8, label=name)
        commands = np.asarray(payload["commands"], dtype=float)
        active = (commands[:, -1] == 1).astype(float)
        window = min(25, len(active))
        rolling = np.convolve(active, np.ones(window) / window, mode="same")
        axes[2].plot(commands[:, 0], rolling, color=color, lw=1.8, label=name)
    axes[0].scatter([goal[0]], [goal[1]], marker="*", s=140, color="#202020")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="A. Closed-loop XY path")
    axes[0].axis("equal"); axes[0].legend(frameon=False)
    axes[1].axhline(run["arrival_radius_m"], color="#777777", ls=":", lw=1)
    axes[1].set(xlabel="episode time (s)", ylabel="distance to target (m)",
                title="B. Target-distance trace")
    axes[2].set(xlabel="episode time (s)", ylabel="active-command fraction",
                title="C. 0.5 s rolling planner availability", ylim=(-.03, 1.03))
    for ax in axes[1:]:
        ax.grid(alpha=.25)
    fig.suptitle(f"Corridor over-conservatism audit: maze {run['maze_type']}, "
                 f"seed {run['map_seed']}, {run['scenario']}")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def capture_lines(run, label, kind, target_t):
    prefix = label.replace("yopo-", "") + "_" + kind + "@"
    matches = [(abs(float(key.rsplit("@", 1)[1]) - target_t), value)
               for key, value in run.get("captures", {}).items() if key.startswith(prefix)]
    if not matches:
        return []
    item = min(matches, key=lambda pair: pair[0])[1]
    output = []
    for line in item.get("lines", []):
        output.append(line["points"] if isinstance(line, dict) else line)
    return output


def plot_candidates(run, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    target = float(run["switch_time_s"]) + 0.8
    for ax, label, title, color in ((axes[0], BASELINE, "baseline: score top-1", "#555555"),
                                    (axes[1], "jerk_topk3", "top-3: minimum jerk jump", COLORS["jerk_topk3"])):
        candidates = capture_lines(run, label, "candidates", target)
        best = capture_lines(run, label, "best", target)
        for line in candidates:
            values = np.asarray(line)
            if len(values): ax.plot(values[:, 0], values[:, 1], color="#87b5d8", alpha=.35, lw=.7)
        for line in best:
            values = np.asarray(line)
            if len(values): ax.plot(values[:, 0], values[:, 1], color=color, lw=3)
        odom = np.asarray(run["vehicles"][label]["odom"], dtype=float)
        idx = np.argmin(np.abs(odom[:, 0] - target))
        center = odom[idx, 1:3]
        ax.scatter(*center, marker="^", s=80, color=color)
        ax.set(xlabel="x (m)", ylabel="y (m)", title=title,
               xlim=(center[0] - 2, center[0] + 11), ylim=(center[1] - 7, center[1] + 7))
        ax.grid(alpha=.2); ax.set_aspect("equal", adjustable="box")
    fig.suptitle(f"Candidate trajectories at t≈{target:.1f} s (after target switch)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_gif(run, output):
    series = {}
    end = 0.0
    for label in (BASELINE, "jerk_topk3"):
        odom = np.asarray(run["vehicles"][label]["odom"], dtype=float)
        series[label] = odom
        end = max(end, float(odom[-1, 0]))
    points = np.asarray(run.get("map_points", []), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    if points.size:
        stride = max(1, len(points) // 6000)
        ax.scatter(points[::stride, 0], points[::stride, 1], s=1, c="#aaaaaa", alpha=.22)
    styles = ((BASELINE, "#555555", "baseline"),
              ("jerk_topk3", COLORS["jerk_topk3"], "jerk top-3"))
    lines, dots = {}, {}
    for label, color, name in styles:
        lines[label], = ax.plot([], [], color=color, lw=2, label=name)
        dots[label], = ax.plot([], [], "o", color=color, ms=6)
    ax.scatter([run["initial_goal"][0], run["final_goal"][0]],
               [run["initial_goal"][1], run["final_goal"][1]], marker="*", s=140,
               c=["#e69f00", "#7b3294"], edgecolors="black", linewidths=.4)
    all_pos = np.vstack([value[:, 1:3] for value in series.values()])
    ax.set_xlim(all_pos[:, 0].min() - 2, all_pos[:, 0].max() + 2)
    ax.set_ylim(all_pos[:, 1].min() - 2, all_pos[:, 1].max() + 2)
    ax.set(xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2); ax.legend(frameon=False)
    title = ax.set_title("")
    frame_times = np.linspace(0, end, 80)
    def update(frame):
        for label, _color, _name in styles:
            data = series[label]
            use = data[:, 0] <= frame
            lines[label].set_data(data[use, 1], data[use, 2])
            if use.any(): dots[label].set_data([data[use][-1, 1]], [data[use][-1, 2]])
        phase = "new target" if frame >= float(run["switch_time_s"]) else "initial target"
        title.set_text(f"t={frame:4.1f} s | {phase}")
        return [*lines.values(), *dots.values(), title]
    movie = animation.FuncAnimation(fig, update, frames=frame_times, interval=80, blit=False)
    movie.save(output, writer=animation.PillowWriter(fps=12), dpi=100)
    plt.close(fig)


def validate(runs):
    condition_lanes = {}
    config_variants = {}
    for run in runs:
        experiment = run["_experiment"]
        condition = (experiment, int(run["maze_type"]), int(run["map_seed"]), run["scenario"])
        lane = ("normal" if run.get("lanes", {}).get("yopo_simple") == BASELINE
                else "swapped")
        condition_lanes.setdefault(condition, set()).add(lane)
        for label, config in run.get("vehicle_configs", {}).items():
            key = (experiment, label)
            normalized = dict(config)
            normalized.pop("lane_order", None)
            config_variants.setdefault(key, set()).add(json.dumps(normalized, sort_keys=True))
    checks = {
        "run_count": len(runs),
        "runs_per_experiment": {experiment: sum(run["_experiment"] == experiment for run in runs)
                                for experiment in EXPERIMENTS},
        "all_maps_identical_between_lanes": all(run.get("maps_identical", False) for run in runs),
        "invalid_start_runs": 0,
        "max_initial_position_delta_m": 0.0,
        "condition_clusters": len(condition_lanes),
        "all_conditions_have_both_lane_orders": all(
            lanes == {"normal", "swapped"} for lanes in condition_lanes.values()),
        "maze_types": sorted({int(run["maze_type"]) for run in runs}),
        "map_seeds": sorted({int(run["map_seed"]) for run in runs}),
        "scenarios": sorted({run["scenario"] for run in runs}),
        "unique_configs_per_experiment_role": {
            f"{experiment}:{label}": len(values)
            for (experiment, label), values in sorted(config_variants.items())
        },
        "lane_order_counts": {
            "normal": sum(run.get("lanes", {}).get("yopo_simple") == BASELINE for run in runs),
            "swapped": sum(run.get("lanes", {}).get("yopo_minco") == BASELINE for run in runs),
        },
    }
    for run in runs:
        experiment = run["_experiment"]
        left = run["vehicles"][BASELINE]
        right = run["vehicles"][experiment]
        if left["metrics"].get("start_in_collision") or right["metrics"].get("start_in_collision"):
            checks["invalid_start_runs"] += 1
        p0 = np.asarray(left["odom"][0][1:4]); p1 = np.asarray(right["odom"][0][1:4])
        checks["max_initial_position_delta_m"] = max(
            checks["max_initial_position_delta_m"], float(np.linalg.norm(p0 - p1)))
    return checks


def require_complete_design(checks):
    expected_runs = len(EXPERIMENTS) * 4 * 5 * 2 * 2
    failures = []
    if checks["run_count"] != expected_runs:
        failures.append(f"run_count={checks['run_count']} (expected {expected_runs})")
    if set(checks["runs_per_experiment"].values()) != {80}:
        failures.append(f"runs_per_experiment={checks['runs_per_experiment']}")
    if checks["condition_clusters"] != 120 or not checks["all_conditions_have_both_lane_orders"]:
        failures.append("incomplete normal/swapped condition pairs")
    if checks["maze_types"] != [1, 2, 5, 7] or checks["map_seeds"] != [1, 2, 3, 4, 5]:
        failures.append("unexpected map/seed design")
    if checks["scenarios"] != ["dynamic", "straight"]:
        failures.append("unexpected scenarios")
    if not checks["all_maps_identical_between_lanes"] or checks["invalid_start_runs"]:
        failures.append("map fingerprint mismatch or invalid start")
    if checks["max_initial_position_delta_m"] > 0.01:
        failures.append("initial position mismatch exceeds 1 cm")
    if set(checks["unique_configs_per_experiment_role"].values()) != {1}:
        failures.append("planner configuration changed within an experiment role")
    if failures:
        raise RuntimeError("invalid ablation design: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="results/minco_ablation_raw")
    parser.add_argument("--output-dir", default="docs/report_assets")
    parser.add_argument("--skip-gif", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.raw_root)
    missing = [experiment for experiment in EXPERIMENTS
               if not any(run["_experiment"] == experiment for run in runs)]
    if missing:
        raise RuntimeError(f"no raw runs for: {', '.join(missing)}")
    rows = build_rows(runs)
    aggregates = aggregate(rows)
    paired = paired_statistics(rows)
    checks = validate(runs)
    require_complete_design(checks)
    write_csv(output / "ablation_episodes.csv", rows)
    write_csv(output / "ablation_aggregate.csv", aggregates)
    with open(output / "ablation_paired.json", "w", encoding="utf-8") as handle:
        json.dump(paired, handle, indent=2, sort_keys=True); handle.write("\n")
    representative = select_representative(runs)
    corridor_stall = select_corridor_stall(runs)
    metadata = {
        "checks": checks,
        "experiments": list(EXPERIMENTS),
        "representative": os.path.relpath(representative["_path"]),
        "corridor_stall_example": os.path.relpath(corridor_stall["_path"]),
        "representative_selection": "maximum baseline-minus-variant jerk-p95 in dynamic jerk_topk3 runs",
        "corridor_stall_selection": "minimum active-command fraction among baseline-success/variant-failure runs",
        "bootstrap_seed": 20260824,
        "bootstrap_resamples": 10000,
    }
    with open(output / "ablation_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True); handle.write("\n")
    plot_effects(paired, output / "ablation_effects.png")
    plot_by_map(rows, output / "ablation_by_map.png")
    plot_profiles(representative, output / "ablation_profiles.png")
    plot_candidates(representative, output / "ablation_candidates.png")
    plot_corridor_stall(corridor_stall, output / "ablation_corridor_stall.png")
    if not args.skip_gif:
        make_gif(representative, output / "ablation_dynamic_goal.gif")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
