#!/usr/bin/env python3
"""Aggregate paired benchmark JSON and render the report's reproducible assets."""

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


MODELS = ("yopo-simple", "yopo-minco")
COLORS = {"yopo-simple": "#f06b24", "yopo-minco": "#00a6d6"}
LABELS = {"yopo-simple": "YOPO-Simple", "yopo-minco": "YOPO-MINCO"}
MAZE_NAMES = {1: "Perlin 3D", 2: "Columns", 5: "Forest", 7: "Walls"}


def load_runs(raw_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as handle:
            run = json.load(handle)
        run["_path"] = path
        runs.append(run)
    return runs


def sustained_alignment(commands, goal, start_t, end_t, degrees=15.0, hold_s=0.24):
    rows = np.asarray(commands, dtype=float)
    if rows.size == 0:
        return None
    use = (rows[:, 0] >= start_t) & (rows[:, 0] <= end_t) & (rows[:, -1] == 1)
    rows = rows[use]
    if len(rows) < 3:
        return None
    direction = np.asarray(goal)[None, :2] - rows[:, 1:3]
    velocity = rows[:, 4:6]
    denom = np.linalg.norm(direction, axis=1) * np.linalg.norm(velocity, axis=1)
    angle = np.arccos(np.clip(np.sum(direction * velocity, axis=1) / np.maximum(denom, 1e-9), -1, 1))
    good = (angle < np.deg2rad(degrees)) & (np.linalg.norm(velocity, axis=1) > 0.5)
    for index in np.flatnonzero(good):
        stop = np.searchsorted(rows[:, 0], rows[index, 0] + hold_s)
        if stop < len(rows) and good[index:stop + 1].all():
            return float(rows[index, 0] - start_t)
    return None


def common_metrics(run, model):
    payload = run["vehicles"][model]
    stored = payload["metrics"]
    odom = np.asarray(payload["odom"], dtype=float)
    commands = np.asarray(payload["commands"], dtype=float)
    clearance = np.asarray(payload.get("clearance", []), dtype=float)
    start_t = float(run.get("switch_time_s") or 0.0)
    final_budget = float(run["timeout_s"]) - start_t
    arrival = stored.get("arrival_s")
    end_t = min(float(run["timeout_s"]), start_t + (float(arrival) if arrival is not None else final_budget))
    goal = np.asarray(run["final_goal"], dtype=float)
    result = {
        "source": os.path.basename(run["_path"]),
        "maze_type": int(run["maze_type"]),
        "map_seed": int(run["map_seed"]),
        "scenario": run["scenario"],
        "model": model,
        "valid_start": not bool(stored.get("start_in_collision", False)),
        "arrived": bool(stored.get("arrived", False)),
        "collision_events": int(stored.get("collision_events", 0)),
        "collision_samples": int(stored.get("collision_samples", 0)),
        "arrival_s": None if arrival is None else float(arrival),
        "timeout_penalized_s": float(arrival) if arrival is not None else final_budget,
    }
    result["collision_free_success"] = bool(result["valid_start"] and result["arrived"] and result["collision_events"] == 0)
    if odom.size == 0:
        return result
    use = (odom[:, 0] >= start_t) & (odom[:, 0] <= end_t)
    segment = odom[use]
    if len(segment) < 2:
        return result
    t, pos, vel = segment[:, 0], segment[:, 1:4], segment[:, 4:7]
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    valid = (dt > 1e-4) & (dt < 0.2) & (step < 1.0)
    length = float(step[valid].sum())
    direct = float(np.linalg.norm(goal - pos[0]))
    speed = np.linalg.norm(vel, axis=1)
    result.update({
        "duration_s": float(t[-1] - t[0]),
        "path_length_m": length,
        "path_efficiency": direct / max(direct, length, 1e-9),
        "mean_speed_mps": float(np.trapz(speed, t) / max(t[-1] - t[0], 1e-9)),
        "max_speed_mps": float(np.max(speed)),
        "final_error_m": float(np.linalg.norm(goal - pos[-1])),
    })
    if clearance.size:
        clearance = clearance[(clearance[:, 0] >= start_t) & (clearance[:, 0] <= end_t), 1]
        if clearance.size:
            result["min_clearance_m"] = float(np.min(clearance))
            result["clearance_p05_m"] = float(np.percentile(clearance, 5))

    if commands.size:
        in_window = (commands[:, 0] >= start_t) & (commands[:, 0] <= end_t)
        window = commands[in_window]
        if len(window):
            result["command_active_fraction"] = float(np.mean(window[:, -1] == 1))
        cmd = commands[in_window & (commands[:, -1] == 1)]
        if len(cmd) >= 3:
            ct, acc = cmd[:, 0], cmd[:, 7:10]
            cdt = np.diff(ct)
            ok = (cdt > 1e-4) & (cdt < 0.2)
            acc_norm = np.linalg.norm(acc, axis=1)
            jerk = np.diff(acc, axis=0) / np.maximum(cdt[:, None], 1e-9)
            jerk_norm = np.linalg.norm(jerk, axis=1)
            result.update({
                "command_acc_rms_mps2": float(np.sqrt(np.mean(acc_norm ** 2))),
                "command_acc_peak_mps2": float(np.max(acc_norm)),
                "command_jerk_rms_mps3": float(np.sqrt(np.mean(jerk_norm[ok] ** 2))) if ok.any() else None,
                "command_jerk_p95_mps3": float(np.percentile(jerk_norm[ok], 95)) if ok.any() else None,
            })
    result["turn_response_s"] = sustained_alignment(payload["commands"], goal, start_t, end_t)
    result["turn_response_penalized_s"] = (result["turn_response_s"] if result["turn_response_s"] is not None
                                             else float(end_t - start_t))
    return result


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not values else float(np.mean(values))


def median(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not values else float(np.median(values))


def aggregate(rows):
    output = []
    for scenario in ("straight", "dynamic", "all"):
        for maze_type in (1, 2, 5, 7, "all"):
            for model in MODELS:
                group = [row for row in rows if row["model"] == model and row["valid_start"]]
                if scenario != "all":
                    group = [row for row in group if row["scenario"] == scenario]
                if maze_type != "all":
                    group = [row for row in group if row["maze_type"] == maze_type]
                if not group:
                    continue
                output.append({
                    "scenario": scenario,
                    "maze_type": maze_type,
                    "model": model,
                    "n": len(group),
                    "collision_free_success_rate": mean([row["collision_free_success"] for row in group]),
                    "arrival_rate": mean([row["arrived"] for row in group]),
                    "collision_episode_rate": mean([row["collision_events"] > 0 for row in group]),
                    "collision_events_mean": mean([row["collision_events"] for row in group]),
                    "timeout_penalized_time_mean_s": mean([row["timeout_penalized_s"] for row in group]),
                    "arrival_median_s_success_only": median([row["arrival_s"] for row in group]),
                    "path_efficiency_mean": mean([row.get("path_efficiency") for row in group]),
                    "path_length_mean_m": mean([row.get("path_length_m") for row in group]),
                    "min_clearance_mean_m": mean([row.get("min_clearance_m") for row in group]),
                    "clearance_p05_mean_m": mean([row.get("clearance_p05_m") for row in group]),
                    "command_acc_rms_mean_mps2": mean([row.get("command_acc_rms_mps2") for row in group]),
                    "command_jerk_p95_mean_mps3": mean([row.get("command_jerk_p95_mps3") for row in group]),
                    "turn_response_median_s": median([row.get("turn_response_s") for row in group]),
                    "turn_response_penalized_mean_s": mean([row.get("turn_response_penalized_s") for row in group]),
                })
    return output


def paired_differences(rows):
    """MINCO minus Simple paired means with an episode bootstrap CI."""
    rng = np.random.default_rng(0)
    metrics = ("collision_free_success", "collision_events", "timeout_penalized_s",
               "path_efficiency", "min_clearance_m", "command_acc_rms_mps2", "command_jerk_p95_mps3",
               "turn_response_penalized_s")
    result = {}
    for scenario in ("straight", "dynamic", "all"):
        sources = sorted({row["source"] for row in rows if scenario == "all" or row["scenario"] == scenario})
        pairs = []
        for source in sources:
            pair = {row["model"]: row for row in rows if row["source"] == source}
            if set(pair) == set(MODELS) and all(pair[model]["valid_start"] for model in MODELS):
                pairs.append(pair)
        scenario_result = {"paired_episodes": len(pairs), "metrics": {}}
        for metric in metrics:
            diffs = []
            for pair in pairs:
                simple = pair["yopo-simple"].get(metric)
                minco = pair["yopo-minco"].get(metric)
                if simple is not None and minco is not None:
                    diffs.append(float(minco) - float(simple))
            if not diffs:
                continue
            values = np.asarray(diffs)
            samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            scenario_result["metrics"][metric] = {
                "n": len(values),
                "mean_minco_minus_simple": float(values.mean()),
                "bootstrap_95ci": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
            }
        result[scenario] = scenario_result
    return result


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def grouped_bars(ax, labels, simple, minco, ylabel, percent=False):
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, simple, width, color=COLORS["yopo-simple"], label=LABELS["yopo-simple"])
    ax.bar(x + width / 2, minco, width, color=COLORS["yopo-minco"], label=LABELS["yopo-minco"])
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if percent:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
    ax.grid(axis="y", alpha=0.25)


def plot_overview(rows, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = []
    success = {model: [] for model in MODELS}
    collision = {model: [] for model in MODELS}
    for maze in (1, 2, 5, 7):
        for scenario in ("straight", "dynamic"):
            labels.append(f"{MAZE_NAMES[maze]}\n{scenario}")
            for model in MODELS:
                group = [r for r in rows if r["valid_start"] and r["maze_type"] == maze and
                         r["scenario"] == scenario and r["model"] == model]
                success[model].append(mean([r["collision_free_success"] for r in group]) or 0)
                collision[model].append(mean([r["collision_events"] > 0 for r in group]) or 0)
    grouped_bars(axes[0, 0], labels, success[MODELS[0]], success[MODELS[1]],
                 "Collision-free success", percent=True)
    grouped_bars(axes[0, 1], labels, collision[MODELS[0]], collision[MODELS[1]],
                 "Episodes with collision", percent=True)
    if not any(value for model in MODELS for value in collision[model]):
        axes[0, 1].text(0.5, 0.5, "0 collision episodes\nin all 80 rollouts",
                        ha="center", va="center", transform=axes[0, 1].transAxes,
                        fontsize=14, color="#444444")

    for ax, scenario in zip(axes[1], ("straight", "dynamic")):
        data, positions, colors, tick_labels = [], [], [], []
        position = 1
        for maze in (1, 2, 5, 7):
            for model in MODELS:
                values = [r["timeout_penalized_s"] for r in rows if r["valid_start"] and
                          r["maze_type"] == maze and r["scenario"] == scenario and r["model"] == model]
                data.append(values); positions.append(position); colors.append(COLORS[model]); position += 1
            tick_labels.append(MAZE_NAMES[maze]); position += 0.5
        boxes = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showmeans=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.65)
        centers = [(positions[i] + positions[i + 1]) / 2 for i in range(0, len(positions), 2)]
        ax.set_xticks(centers, tick_labels, rotation=20, ha="right")
        ax.set_ylabel("Timeout-penalized time (s)")
        ax.set_title(f"{scenario.capitalize()} goal")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="upper center", ncol=2)
    fig.suptitle("Closed-loop paired benchmark (same map, seed, start, goal and controller)", fontsize=14)
    fig.savefig(os.path.join(out_dir, "benchmark_overview.png"), dpi=180)
    plt.close(fig)


def plot_quality(rows, out_dir):
    metrics = [
        ("path_efficiency", "Path efficiency (higher is better)"),
        ("min_clearance_m", "Minimum voxel clearance (m)"),
        ("command_acc_rms_mps2", "Command acceleration RMS (m/s²)"),
        ("command_jerk_p95_mps3", "Command jerk p95 (m/s³)"),
        ("turn_response_penalized_s", "Turn response, timeout-penalized (s)"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        data = []
        for model in MODELS:
            values = [r.get(metric) for r in rows if r["valid_start"] and r.get(metric) is not None and
                      (r["scenario"] == "dynamic" if metric == "turn_response_penalized_s" else True) and r["model"] == model]
            data.append(values)
        boxes = ax.boxplot(data, labels=[LABELS[m] for m in MODELS], patch_artist=True,
                           showmeans=True, widths=0.6)
        for patch, model in zip(boxes["boxes"], MODELS):
            patch.set_facecolor(COLORS[model]); patch.set_alpha(0.65)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Common metrics recomputed from the same odometry/command streams", fontsize=14)
    fig.savefig(os.path.join(out_dir, "benchmark_quality.png"), dpi=180)
    plt.close(fig)


def representative_run(runs, rows, maze, scenario, require_both_success=False):
    candidates = []
    for run in runs:
        if run["maze_type"] != maze or run["scenario"] != scenario:
            continue
        paired = {m: next(r for r in rows if r["source"] == os.path.basename(run["_path"]) and r["model"] == m)
                  for m in MODELS}
        if not all(r["valid_start"] for r in paired.values()):
            continue
        if require_both_success and not all(r["collision_free_success"] for r in paired.values()):
            continue
        delta = paired["yopo-minco"].get("path_efficiency", 0) - paired["yopo-simple"].get("path_efficiency", 0)
        candidates.append((run, delta))
    if not candidates and require_both_success:
        return representative_run(runs, rows, maze, scenario, False)
    if not candidates:
        return None
    target = float(np.median([delta for _run, delta in candidates]))
    return min(candidates, key=lambda item: abs(item[1] - target))[0]


def trajectory_segment(run, model):
    odom = np.asarray(run["vehicles"][model]["odom"], dtype=float)
    if odom.size == 0:
        return odom
    start = float(run.get("switch_time_s") or 0.0)
    arrival = run["vehicles"][model]["metrics"].get("arrival_s")
    end = float(run["timeout_s"]) if arrival is None else start + float(arrival)
    return odom[(odom[:, 0] >= start) & (odom[:, 0] <= end)]


def map_scatter(ax, run):
    points = np.asarray(run.get("map_points", []), dtype=float)
    if points.size:
        ax.scatter(points[:, 0], points[:, 1], s=0.15, c="#5a5a5a", alpha=0.22, rasterized=True)


def plot_trajectories(runs, rows, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    selected = {}
    for ax, maze in zip(axes.flat, (1, 2, 5, 7)):
        run = representative_run(runs, rows, maze, "straight")
        if run is None:
            ax.set_visible(False); continue
        selected[str(maze)] = os.path.basename(run["_path"])
        map_scatter(ax, run)
        for model in MODELS:
            traj = trajectory_segment(run, model)
            if traj.size:
                ax.plot(traj[:, 1], traj[:, 2], color=COLORS[model], lw=2.0, label=LABELS[model])
                ax.scatter(traj[0, 1], traj[0, 2], color=COLORS[model], marker="o", s=24)
        goal = run["final_goal"]
        ax.scatter(goal[0], goal[1], marker="*", s=150, color="#202020", label="Goal")
        ax.set_title(f"{MAZE_NAMES[maze]} · seed {run['map_seed']}")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal"); ax.grid(alpha=0.2)
    axes.flat[0].legend(loc="best")
    fig.suptitle("Median paired-gap straight-goal episodes (top view)", fontsize=14)
    fig.savefig(os.path.join(out_dir, "benchmark_trajectories.png"), dpi=180)
    plt.close(fig)
    return selected


def capture_lines(run, prefix):
    options = [(key, value) for key, value in run.get("captures", {}).items() if key.startswith(prefix + "@")]
    if not options:
        return [], []
    _key, payload = sorted(options)[0]
    lines = [entry["points"] if isinstance(entry, dict) else entry for entry in payload.get("lines", [])]
    return lines, payload.get("objects", [])


def plot_candidates(runs, rows, out_dir):
    run = representative_run(runs, rows, 5, "straight") or representative_run(runs, rows, 1, "straight")
    if run is None:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, model, prefix, best_prefix in (
        (axes[0], "yopo-simple", "simple_candidates", "simple_best"),
        (axes[1], "yopo-minco", "minco_candidates", "minco_best"),
    ):
        map_scatter(ax, run)
        lines, _objects = capture_lines(run, prefix)
        for line in lines:
            pts = np.asarray(line)
            if pts.size:
                ax.plot(pts[:, 0], pts[:, 1], color=COLORS[model], alpha=0.28, lw=1.0)
        best, _ = capture_lines(run, best_prefix)
        for line in best:
            pts = np.asarray(line)
            if pts.size:
                ax.plot(pts[:, 0], pts[:, 1], color="#1b9e3f", lw=3, label="selected")
        if model == "yopo-minco":
            _corridor_lines, corridor = capture_lines(run, "minco_corridor")
            for item in corridor:
                for center in item.get("centers", []):
                    radius = item.get("scale", [0, 0, 0])[0] / 2
                    ax.add_patch(plt.Circle((center[0], center[1]), radius, color=COLORS[model], alpha=0.08))
        ax.set_title(f"{LABELS[model]} candidates at ≈1 s")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.axis("equal"); ax.grid(alpha=0.2)
    fig.suptitle(f"Same Forest map/seed/start/goal · seed {run['map_seed']}", fontsize=14)
    fig.savefig(os.path.join(out_dir, "benchmark_candidates.png"), dpi=180)
    plt.close(fig)
    return os.path.basename(run["_path"])


def scalar_series(path, tag):
    accumulator = EventAccumulator(path, size_guidance={"scalars": 0})
    accumulator.Reload()
    values = accumulator.Scalars(tag)
    return np.asarray([value.step for value in values]), np.asarray([value.value for value in values])


def plot_training_curves(repo, out_dir):
    simple_event = os.path.join(repo, "YOPO/saved/yopo-simple/events.out.tfevents.simple.latest")
    minco_event = os.path.join(repo, "YOPO/saved/yopo-minco/events.out.tfevents.minco.latest")
    curves = [
        ("Trajectory objective", simple_event, "Eval/TrajLoss", minco_event, "Eval/MeanTrajLoss"),
        ("Selection objective", simple_event, "Eval/ScoreLoss", minco_event, "Eval/RankLoss"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    summary = {}
    for ax, (title, sp, stag, mp, mtag) in zip(axes, curves):
        for model, path, tag in (("yopo-simple", sp, stag), ("yopo-minco", mp, mtag)):
            steps, values = scalar_series(path, tag)
            relative = values / values[0] * 100.0
            ax.plot(steps + 1, relative, color=COLORS[model], lw=2, label=LABELS[model])
            summary[f"{model}:{tag}"] = {"first": float(values[0]), "last": float(values[-1]),
                                             "relative_change_pct": float((values[-1] / values[0] - 1) * 100)}
        ax.axhline(100, color="#777777", lw=0.8, ls="--")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Relative validation objective (%)")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Within-model convergence only: raw losses have different definitions and scales", fontsize=13)
    fig.savefig(os.path.join(out_dir, "training_curves_fair.png"), dpi=180)
    plt.close(fig)
    return summary


def make_dynamic_gif(runs, rows, out_dir):
    candidates = []
    for maze in (1, 2, 5, 7):
        run = representative_run(runs, rows, maze, "dynamic", require_both_success=True)
        if run is not None:
            paired = [r for r in rows if r["source"] == os.path.basename(run["_path"])]
            if all(r["collision_free_success"] for r in paired):
                candidates.append(run)
    if not candidates:
        candidates = [run for run in runs if run["scenario"] == "dynamic"]
    if not candidates:
        return None
    def efficiency_gap(candidate):
        source = os.path.basename(candidate["_path"])
        paired = {row["model"]: row for row in rows if row["source"] == source}
        return paired.get("yopo-minco", {}).get("path_efficiency", 0) - paired.get("yopo-simple", {}).get("path_efficiency", 0)
    gaps = [efficiency_gap(candidate) for candidate in candidates]
    target_gap = float(np.median(gaps))
    run = min(candidates, key=lambda candidate: abs(efficiency_gap(candidate) - target_gap))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    map_scatter(ax, run)
    all_positions = []
    trajectories = {}
    for model in MODELS:
        traj = np.asarray(run["vehicles"][model]["odom"], dtype=float)
        trajectories[model] = traj
        if traj.size:
            all_positions.append(traj[:, 1:3])
    all_positions.append(np.asarray([run["initial_goal"][:2], run["final_goal"][:2]]))
    extent = np.vstack(all_positions)
    margin = 2.0
    ax.set_xlim(extent[:, 0].min() - margin, extent[:, 0].max() + margin)
    ax.set_ylim(extent[:, 1].min() - margin, extent[:, 1].max() + margin)
    ax.set_aspect("equal"); ax.grid(alpha=0.2); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    lines, dots = {}, {}
    for model in MODELS:
        lines[model], = ax.plot([], [], color=COLORS[model], lw=2.2, label=LABELS[model])
        dots[model], = ax.plot([], [], "o", color=COLORS[model], ms=7)
    goal_dot = ax.scatter([], [], marker="*", s=180, color="#202020", label="Active goal")
    title = ax.set_title("")
    ax.legend(loc="best")
    end = min(float(run["timeout_s"]), max(traj[-1, 0] for traj in trajectories.values() if traj.size))
    frames = np.linspace(0, end, min(100, max(30, int(end * 5))))

    def update(frame_t):
        for model, traj in trajectories.items():
            segment = traj[traj[:, 0] <= frame_t]
            if len(segment):
                lines[model].set_data(segment[:, 1], segment[:, 2])
                dots[model].set_data([segment[-1, 1]], [segment[-1, 2]])
        switched = run.get("switch_time_s") is not None and frame_t >= float(run["switch_time_s"])
        goal = run["final_goal"] if switched else run["initial_goal"]
        goal_dot.set_offsets(np.asarray([[goal[0], goal[1]]]))
        title.set_text(f"Dynamic 2D goal · {MAZE_NAMES[run['maze_type']]} seed {run['map_seed']} · t={frame_t:.1f}s")
        return [*lines.values(), *dots.values(), goal_dot, title]

    movie = animation.FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
    output = os.path.join(out_dir, "benchmark_dynamic_goal.gif")
    movie.save(output, writer=animation.PillowWriter(fps=8))
    plt.close(fig)
    return os.path.basename(run["_path"])


def main(args):
    repo = os.path.abspath(args.repo)
    out_dir = os.path.join(repo, args.output_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runs = load_runs(os.path.join(repo, args.raw_dir))
    if not runs:
        raise SystemExit("no benchmark JSON files found")
    expected = {(maze, seed, scenario) for maze in (1, 2, 5, 7)
                for seed in (1, 2, 3, 4, 5) for scenario in ("straight", "dynamic")}
    observed = {(int(run["maze_type"]), int(run["map_seed"]), run["scenario"])
                for run in runs}
    invalid_starts = sum(any(run["vehicles"][model]["metrics"].get("start_in_collision", False)
                             for model in MODELS) for run in runs)
    if len(runs) != len(expected) or observed != expected:
        raise RuntimeError(f"incomplete benchmark design: files={len(runs)}, conditions={len(observed)}")
    if not all(run.get("maps_identical", False) for run in runs) or invalid_starts:
        raise RuntimeError("map fingerprint mismatch or invalid benchmark start")
    rows = [common_metrics(run, model) for run in runs for model in MODELS]
    aggregates = aggregate(rows)
    paired = paired_differences(rows)
    write_csv(os.path.join(out_dir, "benchmark_episodes.csv"), rows)
    write_csv(os.path.join(out_dir, "benchmark_aggregate.csv"), aggregates)
    with open(os.path.join(out_dir, "benchmark_paired.json"), "w", encoding="utf-8") as handle:
        json.dump(paired, handle, indent=2, sort_keys=True)
    plot_overview(rows, out_dir)
    plot_quality(rows, out_dir)
    selected = {
        "trajectory_panels": plot_trajectories(runs, rows, out_dir),
        "candidate_panel": plot_candidates(runs, rows, out_dir),
        "dynamic_gif": make_dynamic_gif(runs, rows, out_dir),
    }
    training = plot_training_curves(repo, out_dir)
    metadata = {
        "episode_files": len(runs),
        "paired_vehicle_rollouts": len(rows),
        "maze_types": sorted({run["maze_type"] for run in runs}),
        "map_seeds": sorted({run["map_seed"] for run in runs}),
        "scenarios": sorted({run["scenario"] for run in runs}),
        "identical_map_fingerprints": sum(bool(run.get("maps_identical")) for run in runs),
        "invalid_start_episodes": invalid_starts,
        "selected_examples": selected,
        "training_curve_endpoints": training,
        "protocol": {
            "same_start": [0.0, 0.0, 2.0],
            "straight_goal": [20.0, 0.0, 2.0],
            "dynamic_initial_goal": [20.0, 0.0, 2.0],
            "dynamic_final_goal": [10.0, 15.0, 2.0],
            "dynamic_switch_s": 2.5,
            "arrival_radius_m": 1.0,
            "requested_velocity_mps": 6.0,
            "common_yaw_goal_weight": 6.0,
            "episode_timeout_s": 20.0,
        },
    }
    with open(os.path.join(out_dir, "benchmark_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/workspace/YOPO")
    parser.add_argument("--raw-dir", default="results/benchmark_raw")
    parser.add_argument("--output-dir", default="docs/report_assets")
    main(parser.parse_args())
