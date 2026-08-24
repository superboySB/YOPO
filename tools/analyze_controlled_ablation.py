#!/usr/bin/env python3
"""Aggregate counterbalanced closed-loop experiments for controlled YOPO-MINCO factors."""

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

EXPERIMENTS = ("geometry", "duration", "corridor", "temporal")
DEFINITIONS = {
    "geometry": ("single_fixed", "minco_fixed", "MINCO geometry"),
    "duration": ("minco_fixed", "minco_variable", "predicted duration"),
    "corridor": ("corridor_off", "corridor_filter", "corridor filter"),
    "temporal": ("score_top1", "jerk_top3", "jerk top-3"),
}
COLORS = {"geometry": "#0072b2", "duration": "#d55e00", "corridor": "#009e73", "temporal": "#cc79a7"}
MAZE_NAMES = {1: "Perlin", 2: "Columns", 5: "Forest", 7: "Walls"}
METRICS = ("collision_free_success", "collision_events", "timeout_penalized_s", "path_efficiency",
           "min_clearance_m", "command_acc_rms_mps2", "command_jerk_p95_mps3",
           "command_active_fraction", "turn_response_penalized_s")


def finite_mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not values else float(np.mean(values))


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_runs(root):
    runs = []
    for experiment in EXPERIMENTS:
        baseline, variant, _label = DEFINITIONS[experiment]
        for path in sorted(glob.glob(os.path.join(root, experiment, "*.json"))):
            with open(path, encoding="utf-8") as handle:
                run = json.load(handle)
            if baseline not in run["vehicles"] or variant not in run["vehicles"]:
                raise ValueError(f"unexpected vehicle labels in {path}")
            run["_path"] = path; run["_experiment"] = experiment
            runs.append(run)
    return runs


def build_rows(runs):
    rows = []
    for run in runs:
        experiment = run["_experiment"]
        baseline, variant, _label = DEFINITIONS[experiment]
        source = os.path.basename(run["_path"])
        condition = source.replace("_swapped.json", ".json")
        lane_order = "normal" if run.get("lanes", {}).get("yopo_simple") == baseline else "swapped"
        for model, role in ((baseline, "baseline"), (variant, "variant")):
            row = common_metrics(run, model)
            row.update(experiment=experiment, role=role, condition=condition, lane_order=lane_order)
            rows.append(row)
    return rows


def aggregate(rows):
    output = []
    for experiment in EXPERIMENTS:
        for role in ("baseline", "variant"):
            for scenario in ("straight", "dynamic", "all"):
                for maze in (1, 2, 5, 7, "all"):
                    group = [r for r in rows if r["experiment"] == experiment and r["role"] == role and r["valid_start"]]
                    if scenario != "all": group = [r for r in group if r["scenario"] == scenario]
                    if maze != "all": group = [r for r in group if r["maze_type"] == maze]
                    if not group: continue
                    item = {"experiment": experiment, "role": role, "scenario": scenario,
                            "maze_type": maze, "n": len(group)}
                    for metric in METRICS:
                        item[metric + "_mean"] = finite_mean([r.get(metric) for r in group])
                    output.append(item)
    return output


def paired_statistics(rows):
    rng = np.random.default_rng(20260824)
    output = {}
    for experiment in EXPERIMENTS:
        output[experiment] = {}
        for scenario in ("straight", "dynamic", "all"):
            subset = [r for r in rows if r["experiment"] == experiment and
                      (scenario == "all" or r["scenario"] == scenario)]
            result = {"paired_records": 0, "condition_clusters": 0, "metrics": {}}
            for metric in METRICS:
                by_condition = {}
                for source in sorted({r["source"] for r in subset}):
                    pair = {r["role"]: r for r in subset if r["source"] == source}
                    if set(pair) != {"baseline", "variant"} or not all(r["valid_start"] for r in pair.values()):
                        continue
                    base, variant = pair["baseline"].get(metric), pair["variant"].get(metric)
                    if base is not None and variant is not None:
                        by_condition.setdefault(pair["baseline"]["condition"], []).append(float(variant) - float(base))
                if not by_condition: continue
                values = np.asarray([np.mean(v) for v in by_condition.values()])
                bootstrap = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
                result["metrics"][metric] = {
                    "paired_records": sum(map(len, by_condition.values())), "condition_clusters": len(values),
                    "mean_variant_minus_baseline": float(values.mean()),
                    "bootstrap_95ci": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
                }
                result["paired_records"] = max(result["paired_records"], sum(map(len, by_condition.values())))
                result["condition_clusters"] = max(result["condition_clusters"], len(values))
            output[experiment][scenario] = result
    return output


def plot_effects(paired, output):
    definitions = [("collision_free_success", "Success", 100, "pp", "all"),
                   ("collision_events", "Collision events", 1, "count", "all"),
                   ("timeout_penalized_s", "Penalized time", 1, "s", "all"),
                   ("path_efficiency", "Path efficiency", 100, "pp", "all"),
                   ("min_clearance_m", "Min clearance", 100, "cm", "all"),
                   ("command_acc_rms_mps2", "Acceleration RMS", 1, "m/s²", "all"),
                   ("command_jerk_p95_mps3", "Jerk p95", 1, "m/s³", "all"),
                   ("command_active_fraction", "Active commands", 100, "pp", "all"),
                   ("turn_response_penalized_s", "Dynamic turn response", 1, "s", "dynamic")]
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11), constrained_layout=True)
    y = np.arange(len(EXPERIMENTS))
    for ax, (metric, title, scale, unit, scenario) in zip(axes.flat, definitions):
        for index, experiment in enumerate(EXPERIMENTS):
            item = paired[experiment][scenario]["metrics"].get(metric)
            if not item: continue
            mean = item["mean_variant_minus_baseline"] * scale
            low, high = np.asarray(item["bootstrap_95ci"]) * scale
            ax.errorbar(mean, index, xerr=[[mean-low], [high-mean]], fmt="o",
                        color=COLORS[experiment], capsize=3)
        ax.axvline(0, color="#333333", lw=1); ax.grid(axis="x", alpha=.25)
        ax.set(title=title, xlabel=f"variant − baseline ({unit})", yticks=y)
        ax.set_yticklabels([DEFINITIONS[e][2] for e in EXPERIMENTS] if ax in axes[:, 0] else [])
        ax.invert_yaxis()
    fig.suptitle("Counterbalanced closed-loop effects; condition-cluster bootstrap 95% CI", fontsize=13)
    fig.savefig(output, dpi=180); plt.close(fig)


def plot_by_map(rows, output):
    definitions = [("collision_free_success", "Success (pp)", 100, "RdBu", -100, 100),
                   ("timeout_penalized_s", "Penalized time (s)", 1, "RdBu_r", -8, 8),
                   ("min_clearance_m", "Min clearance (cm)", 100, "RdBu", -20, 20),
                   ("command_jerk_p95_mps3", "Jerk p95 (m/s³)", 1, "RdBu_r", -15, 15)]
    columns = [(m, s) for m in (1, 2, 5, 7) for s in ("straight", "dynamic")]
    fig, axes = plt.subplots(4, 1, figsize=(14.5, 10.5), constrained_layout=True)
    for ax, (metric, title, scale, cmap, vmin, vmax) in zip(axes, definitions):
        matrix = np.full((4, 8), np.nan)
        for i, experiment in enumerate(EXPERIMENTS):
            for j, (maze, scenario) in enumerate(columns):
                group = [r for r in rows if r["experiment"] == experiment and r["maze_type"] == maze and r["scenario"] == scenario]
                diffs = []
                for source in {r["source"] for r in group}:
                    pair = {r["role"]: r for r in group if r["source"] == source}
                    if set(pair) == {"baseline", "variant"}:
                        a, b = pair["variant"].get(metric), pair["baseline"].get(metric)
                        if a is not None and b is not None: diffs.append((float(a)-float(b))*scale)
                matrix[i, j] = finite_mean(diffs)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(4):
            for j in range(8):
                if np.isfinite(matrix[i, j]): ax.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center", fontsize=8)
        ax.set_yticks(range(4), [DEFINITIONS[e][2] for e in EXPERIMENTS])
        ax.set_xticks(range(8), [f"{MAZE_NAMES[m]}\n{s[0].upper()}" for m, s in columns])
        ax.set_title(title, loc="left", fontsize=10); fig.colorbar(image, ax=ax, fraction=.018, pad=.01)
    fig.savefig(output, dpi=180); plt.close(fig)


def representative(runs, experiment):
    baseline, variant, _label = DEFINITIONS[experiment]
    candidates = [r for r in runs if r["_experiment"] == experiment and r["scenario"] == "dynamic" and
                  r.get("lanes", {}).get("yopo_simple") == baseline]
    values = []
    for run in candidates:
        b, v = common_metrics(run, baseline), common_metrics(run, variant)
        values.append((run, v["timeout_penalized_s"] - b["timeout_penalized_s"]))
    target = np.median([v for _r, v in values])
    return min(values, key=lambda item: abs(item[1] - target))[0]


def command_series(payload):
    cmd = np.asarray(payload["commands"], dtype=float)
    cmd = cmd[cmd[:, -1] == 1]
    if len(cmd) < 3: return np.array([]), np.array([])
    dt = np.diff(cmd[:, 0]); jerk = np.linalg.norm(np.diff(cmd[:, 7:10], axis=0) / np.maximum(dt[:, None], 1e-9), axis=1)
    return cmd[1:, 0], jerk


def plot_profiles(runs, output):
    fig, axes = plt.subplots(4, 3, figsize=(14.5, 14), constrained_layout=True)
    selected = {}
    for row, experiment in enumerate(EXPERIMENTS):
        run = representative(runs, experiment); selected[experiment] = os.path.relpath(run["_path"])
        baseline, variant, label = DEFINITIONS[experiment]
        points = np.asarray(run.get("map_points", []), dtype=float)
        if points.size:
            stride=max(1,len(points)//7000); axes[row,0].scatter(points[::stride,0],points[::stride,1],s=.5,c="#aaa",alpha=.2)
        for model, color, name in ((baseline,"#555555","baseline"),(variant,COLORS[experiment],"variant")):
            payload=run["vehicles"][model]; odom=np.asarray(payload["odom"],dtype=float)
            axes[row,0].plot(odom[:,1],odom[:,2],color=color,lw=1.8,label=name)
            t,j=command_series(payload); axes[row,1].plot(t,j,color=color,lw=.9)
            goal=np.asarray(run["final_goal"]); axes[row,2].plot(odom[:,0],np.linalg.norm(odom[:,1:4]-goal,axis=1),color=color,lw=1.5)
        axes[row,0].scatter([run["initial_goal"][0],run["final_goal"][0]], [run["initial_goal"][1],run["final_goal"][1]], marker="*",s=80,c=["#e69f00","#7b3294"])
        axes[row,0].set(title=f"{label}: XY path",xlabel="x (m)",ylabel="y (m)"); axes[row,0].legend(frameon=False,fontsize=8)
        axes[row,1].set(title="command jerk",xlabel="time (s)",ylabel="m/s³",ylim=(0,80))
        axes[row,2].set(title="distance to final target",xlabel="time (s)",ylabel="m")
        for ax in axes[row,1:]: ax.axvline(run["switch_time_s"],color="#7b3294",ls="--",lw=.8)
        for ax in axes[row]: ax.grid(alpha=.2)
    fig.suptitle("Deterministic median-time-effect dynamic examples (normal lane order)", fontsize=13)
    fig.savefig(output,dpi=180); plt.close(fig)
    return selected


def capture_lines(run, label, kind, target):
    prefix = label.replace("yopo-", "") + "_" + kind + "@"
    matches = [(abs(float(key.rsplit("@",1)[1])-target), value) for key,value in run.get("captures",{}).items() if key.startswith(prefix)]
    if not matches: return []
    return [line["points"] if isinstance(line,dict) else line for line in min(matches,key=lambda x:x[0])[1].get("lines",[])]


def plot_geometry_candidates(run, output):
    baseline, variant, _ = DEFINITIONS["geometry"]; target=float(run["switch_time_s"])+.8
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,model,title,color in ((axes[0],baseline,"single quintic / fixed T","#555"),(axes[1],variant,"2-piece MINCO / fixed T",COLORS["geometry"])):
        for line in capture_lines(run,model,"candidates",target):
            p=np.asarray(line); ax.plot(p[:,0],p[:,1],color="#87b5d8",alpha=.35,lw=.7)
        for line in capture_lines(run,model,"best",target):
            p=np.asarray(line); ax.plot(p[:,0],p[:,1],color=color,lw=3)
        odom=np.asarray(run["vehicles"][model]["odom"]); idx=np.argmin(abs(odom[:,0]-target)); c=odom[idx,1:3]
        ax.scatter(*c,marker="^",s=70,color=color); ax.set(title=title,xlabel="x (m)",ylabel="y (m)",xlim=(c[0]-2,c[0]+11),ylim=(c[1]-7,c[1]+7)); ax.set_aspect("equal"); ax.grid(alpha=.2)
    fig.suptitle(f"Same network scores and candidate IDs at t≈{target:.1f} s")
    fig.savefig(output,dpi=180); plt.close(fig)


def make_geometry_gif(run, output):
    baseline,variant,_=DEFINITIONS["geometry"]; styles=((baseline,"#555","single fixed"),(variant,COLORS["geometry"],"MINCO fixed"))
    fig,ax=plt.subplots(figsize=(7,5.2),constrained_layout=True); series={m:np.asarray(run["vehicles"][m]["odom"],dtype=float) for m,_,_ in styles}
    points=np.asarray(run.get("map_points",[]),dtype=float)
    if points.size: stride=max(1,len(points)//6000); ax.scatter(points[::stride,0],points[::stride,1],s=.5,c="#aaa",alpha=.2)
    lines={};dots={}
    for model,color,name in styles: lines[model],=ax.plot([],[],color=color,lw=2,label=name);dots[model],=ax.plot([],[],"o",color=color,ms=6)
    allp=np.vstack([v[:,1:3] for v in series.values()]); ax.set(xlim=(allp[:,0].min()-2,allp[:,0].max()+2),ylim=(allp[:,1].min()-2,allp[:,1].max()+2),xlabel="x (m)",ylabel="y (m)");ax.set_aspect("equal");ax.legend(frameon=False);ax.grid(alpha=.2);title=ax.set_title("")
    end=max(v[-1,0] for v in series.values())
    def update(t):
        for model,_,_ in styles:
            data=series[model];use=data[:,0]<=t;lines[model].set_data(data[use,1],data[use,2])
            if use.any(): dots[model].set_data([data[use][-1,1]],[data[use][-1,2]])
        title.set_text(f"t={t:.1f} s | {'new target' if t>=run['switch_time_s'] else 'initial target'}");return [*lines.values(),*dots.values(),title]
    animation.FuncAnimation(fig,update,frames=np.linspace(0,end,80),interval=80).save(output,writer=animation.PillowWriter(fps=12),dpi=100);plt.close(fig)


def validate(runs):
    condition_lanes={}; config_variants={}; invalid=0; max_delta=0.0
    for run in runs:
        exp=run["_experiment"];base,var,_=DEFINITIONS[exp];condition=(exp,run["maze_type"],run["map_seed"],run["scenario"])
        lane="normal" if run.get("lanes",{}).get("yopo_simple")==base else "swapped";condition_lanes.setdefault(condition,set()).add(lane)
        for label,config in run.get("vehicle_configs",{}).items():
            normalized=dict(config);normalized.pop("lane_order",None);config_variants.setdefault((exp,label),set()).add(json.dumps(normalized,sort_keys=True))
        a,b=run["vehicles"][base],run["vehicles"][var];invalid+=int(a["metrics"].get("start_in_collision") or b["metrics"].get("start_in_collision"));max_delta=max(max_delta,float(np.linalg.norm(np.asarray(a["odom"][0][1:4])-np.asarray(b["odom"][0][1:4]))))
    checks={"run_count":len(runs),"runs_per_experiment":{e:sum(r["_experiment"]==e for r in runs) for e in EXPERIMENTS},"condition_clusters":len(condition_lanes),"all_conditions_have_both_lane_orders":all(v=={"normal","swapped"} for v in condition_lanes.values()),"all_maps_identical_between_lanes":all(r.get("maps_identical",False) for r in runs),"invalid_start_runs":invalid,"max_initial_position_delta_m":max_delta,"unique_configs_per_experiment_role":{f"{k[0]}:{k[1]}":len(v) for k,v in config_variants.items()},"maze_types":sorted({int(r["maze_type"]) for r in runs}),"map_seeds":sorted({int(r["map_seed"]) for r in runs}),"scenarios":sorted({r["scenario"] for r in runs})}
    failures=[]
    if checks["run_count"]!=320 or set(checks["runs_per_experiment"].values())!={80}: failures.append("expected 320 runs, 80 per experiment")
    if checks["condition_clusters"]!=160 or not checks["all_conditions_have_both_lane_orders"]: failures.append("incomplete counterbalancing")
    if not checks["all_maps_identical_between_lanes"] or invalid: failures.append("map mismatch or invalid start")
    if max_delta>.01 or set(checks["unique_configs_per_experiment_role"].values())!={1}: failures.append("initial-state or config inconsistency")
    if failures: raise RuntimeError("; ".join(failures))
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--raw-root",default="results/controlled_ablation_raw");parser.add_argument("--output-dir",default="docs/report_assets");parser.add_argument("--skip-gif",action="store_true");args=parser.parse_args()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True);runs=load_runs(args.raw_root);rows=build_rows(runs);checks=validate(runs);paired=paired_statistics(rows)
    write_csv(output/"controlled_episodes.csv",rows);write_csv(output/"controlled_aggregate.csv",aggregate(rows))
    with open(output/"controlled_paired.json","w") as f:json.dump(paired,f,indent=2,sort_keys=True);f.write("\n")
    plot_effects(paired,output/"controlled_effects.png");plot_by_map(rows,output/"controlled_by_map.png");selected=plot_profiles(runs,output/"controlled_profiles.png")
    geometry_run=representative(runs,"geometry");plot_geometry_candidates(geometry_run,output/"controlled_geometry_candidates.png")
    if not args.skip_gif:make_geometry_gif(geometry_run,output/"controlled_geometry_dynamic.gif")
    metadata={"checks":checks,"definitions":{e:{"baseline":DEFINITIONS[e][0],"variant":DEFINITIONS[e][1]} for e in EXPERIMENTS},"representative_runs":selected,"bootstrap_seed":20260824,"bootstrap_resamples":10000}
    with open(output/"controlled_metadata.json","w") as f:json.dump(metadata,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps(metadata,indent=2,sort_keys=True))


if __name__=="__main__":main()
