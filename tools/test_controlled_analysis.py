#!/usr/bin/env python3
"""Synthetic 320-record design test for controlled-ablation aggregation and plots."""

import os
import tempfile

from analyze_controlled_ablation import (DEFINITIONS, EXPERIMENTS, METRICS,
                                         paired_statistics, plot_by_map,
                                         plot_effects, validate)


def main():
    rows, runs = [], []
    for experiment in EXPERIMENTS:
        baseline, variant, _label = DEFINITIONS[experiment]
        for maze in (1, 2, 5, 7):
            for seed in (1, 2, 3, 4, 5):
                for scenario in ("straight", "dynamic"):
                    for lane in ("normal", "swapped"):
                        suffix = "_swapped" if lane == "swapped" else ""
                        source = f"maze{maze}_seed{seed}_{scenario}{suffix}.json"
                        condition = source.replace("_swapped.json", ".json")
                        for role, model, value in (("baseline", baseline, 0.0),
                                                   ("variant", variant, 1.0)):
                            row = {"source": source, "condition": condition,
                                   "experiment": experiment, "role": role, "model": model,
                                   "valid_start": True, "maze_type": maze, "map_seed": seed,
                                   "scenario": scenario}
                            row.update({metric: value for metric in METRICS})
                            rows.append(row)
                        configs = {
                            baseline: {"lane_order": lane, "mode": "baseline"},
                            variant: {"lane_order": lane, "mode": "variant"},
                        }
                        vehicles = {
                            baseline: {"metrics": {"start_in_collision": False},
                                       "odom": [[0, 0, 0, 2]]},
                            variant: {"metrics": {"start_in_collision": False},
                                      "odom": [[0, 0, 0, 2]]},
                        }
                        runs.append({
                            "_experiment": experiment, "maze_type": maze, "map_seed": seed,
                            "scenario": scenario, "maps_identical": True,
                            "lanes": {"yopo_simple": baseline if lane == "normal" else variant},
                            "vehicle_configs": configs, "vehicles": vehicles,
                        })

    checks = validate(runs)
    assert checks["run_count"] == 320
    paired = paired_statistics(rows)
    for experiment in EXPERIMENTS:
        result = paired[experiment]["all"]["metrics"]["timeout_penalized_s"]
        assert result["paired_records"] == 80
        assert result["condition_clusters"] == 40
        assert result["mean_variant_minus_baseline"] == 1.0
    with tempfile.TemporaryDirectory() as directory:
        effect_path = os.path.join(directory, "effects.png")
        map_path = os.path.join(directory, "map.png")
        plot_effects(paired, effect_path)
        plot_by_map(rows, map_path)
        assert os.path.getsize(effect_path) > 1000 and os.path.getsize(map_path) > 1000
    print("controlled analysis synthetic design test: PASS")


if __name__ == "__main__":
    main()
