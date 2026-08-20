import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import benchmark_active_camera as benchmark
import verify_ablation_pair as verifier


class ActiveCameraBenchmarkTest(unittest.TestCase):
    def make_args(self, output_dir):
        parser = benchmark.make_parser()
        return parser.parse_args(
            [
                "--active-weight",
                "/tmp/active.pth",
                "--fixed-weight",
                "/tmp/fixed.pth",
                "--seeds",
                "101,102",
                "--dry-run",
                "--skip-weight-check",
                "--no-plot",
                "--output-dir",
                str(output_dir),
            ]
        )

    def test_default_seeds_are_holdout(self):
        args = benchmark.make_parser().parse_args([])
        self.assertEqual(args.seeds, [101, 102, 103, 104, 105])
        self.assertEqual(args.maze_type, 8)
        self.assertAlmostEqual(args.map_viz_resolution, 0.1)
        self.assertEqual(args.camera_state_stamped_topic, "/yopo/camera/orientation_stamped")
        self.assertAlmostEqual(args.camera_sync_slop, 0.03)
        self.assertAlmostEqual(args.min_depth_state_match_rate, 0.98)
        self.assertEqual(args.expected_checkpoint_epoch, 50)

    def test_trajectory_schema_contains_required_ablation_signals(self):
        source = (TOOLS / "benchmark_active_camera.py").read_text()
        for field in (
            '"t_s"',
            '"body_yaw_rad"',
            '"camera_pitch_rad"',
            '"camera_yaw_rad"',
            '"target_pitch_rad"',
            '"target_yaw_rad"',
            '"clearance_m"',
            '"collision"',
        ):
            self.assertIn(field, source)

    def test_manifest_pairs_only_change_treatment(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory)
            manifest = benchmark.build_manifest(args, Path(directory))
            self.assertEqual(benchmark.validate_manifest(manifest), [])
            self.assertEqual(len(manifest["runs"]), 4)
            by_pair = {}
            for run in manifest["runs"]:
                by_pair.setdefault(run["pair_id"], []).append(run)
                expected = run["arm"] == benchmark.ARM_ACTIVE
                self.assertIn(
                    "--active-camera {}".format(benchmark.bool_text(expected)),
                    run["commands"]["planner"],
                )
                self.assertIn(
                    "_active_camera:={}".format(benchmark.bool_text(expected)),
                    run["commands"]["sensor"],
                )
                self.assertIn(
                    "--camera-state-stamped-topic /yopo/camera/orientation_stamped",
                    run["commands"]["planner"],
                )
                self.assertIn("--camera-sync-slop 0.03", run["commands"]["planner"])
                self.assertIn("_maze_type:=8", run["commands"]["sensor"])
                self.assertIn("_map_viz_resolution:=0.1", run["commands"]["sensor"])
                self.assertIn("--start=-20.0,0.0,2.0", run["commands"]["monitor"])
            for runs in by_pair.values():
                self.assertEqual(len({run["controlled_hash"] for run in runs}), 1)

    def test_manifest_detects_controlled_change(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory)
            manifest = benchmark.build_manifest(args, Path(directory))
            manifest["runs"][1]["controlled_variables"]["flight"]["velocity_mps"] = 9.0
            manifest["runs"][1]["controlled_hash"] = benchmark.content_hash(
                manifest["runs"][1]["controlled_variables"]
            )
            errors = benchmark.validate_manifest(manifest)
            self.assertTrue(any("controlled variables" in error for error in errors))

    def test_aggregate_keeps_wilson_and_paired_raw_values(self):
        results = [
            {"seed": 101, "arm": "active", "status": "ok", "success": True,
             "collision_free_success": True, "camera_contract_ok": True,
             "camera_sync_contract_ok": True,
             "collision_detected": False, "collision_episode_count": 0, "min_clearance_m": 0.8},
            {"seed": 101, "arm": "fixed", "status": "collision", "success": True,
             "collision_free_success": False, "camera_contract_ok": True,
             "camera_sync_contract_ok": True,
             "collision_detected": True, "collision_episode_count": 1, "min_clearance_m": -0.1},
            {"seed": 102, "arm": "active", "status": "ok", "success": True,
             "collision_free_success": True, "camera_contract_ok": True,
             "camera_sync_contract_ok": True,
             "collision_detected": False, "collision_episode_count": 0, "min_clearance_m": 0.4},
            {"seed": 102, "arm": "fixed", "status": "incomplete", "success": False,
             "collision_free_success": False, "camera_contract_ok": True,
             "camera_sync_contract_ok": True,
             "collision_detected": True, "collision_episode_count": 2, "min_clearance_m": -0.2},
        ]
        aggregate = benchmark.aggregate_results(results)
        self.assertEqual(aggregate["active"]["collision_rate"], 0.0)
        self.assertEqual(aggregate["fixed"]["collision_rate"], 1.0)
        self.assertEqual(len(aggregate["active"]["collision_rate_wilson95"]), 2)
        paired = benchmark.paired_results(results)
        self.assertEqual(paired["complete_pairs"], 2)
        self.assertEqual(
            [item["collision_active_minus_fixed"] for item in paired["per_seed"]],
            [-1, -1],
        )
        self.assertEqual(
            [item["success_active_minus_fixed"] for item in paired["per_seed"]],
            [0, 1],
        )

    def test_verifier_rejects_fixed_camera_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_args(directory)
            manifest = benchmark.build_manifest(args, Path(directory))
            results = []
            for spec in manifest["runs"]:
                moved = spec["arm"] == "fixed" and spec["seed"] == 101
                results.append(
                    {
                        "run_id": spec["run_id"],
                        "seed": spec["seed"],
                        "arm": spec["arm"],
                        "status": "ok",
                        "success": True,
                        "collision_detected": False,
                        "camera_command_max_abs_rad": 0.1 if moved else (
                            0.2 if spec["arm"] == "active" else 0.0
                        ),
                        "camera_state_max_abs_rad": 0.0,
                        "camera_sync_contract_ok": True,
                        "camera_state_stamped_topic": "/yopo/camera/orientation_stamped",
                        "camera_state_stamped_samples": 100,
                        "depth_samples": 100,
                        "depth_state_matched_samples": 100,
                        "depth_state_unmatched_samples": 0,
                        "depth_state_match_rate": 1.0,
                        "depth_state_max_stamp_error_s": 0.0,
                        "depth_invalid_stamp_samples": 0,
                        "camera_state_invalid_stamp_samples": 0,
                        "depth_contract": {
                            "width": 160, "height": 192, "encoding": "32FC1", "step": 640,
                        },
                        "depth_observed_fps": 15.0,
                        "controlled_hash": spec["controlled_hash"],
                    }
                )
            summary = {
                "runs": results,
                "aggregate": benchmark.aggregate_results(results),
                "paired": benchmark.paired_results(results),
            }
            report = verifier.verify(manifest, summary=summary)
            self.assertFalse(report["ok"])
            self.assertTrue(any("fixed camera moved" in error for error in report["errors"]))

    def test_stamp_pair_audit_is_one_to_one_and_rejects_bad_stamps(self):
        audit = benchmark.StampPairAudit(0.03, max_pending=4)
        audit.add_state(10.0)
        audit.add_depth(10.0)
        audit.add_depth(11.0)
        audit.add_state(11.02)
        audit.add_depth(0.0)
        snapshot = audit.snapshot()
        self.assertEqual(snapshot["depth_state_matched_samples"], 2)
        self.assertEqual(snapshot["depth_state_unmatched_samples"], 1)
        self.assertAlmostEqual(snapshot["depth_state_match_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(snapshot["depth_state_max_stamp_error_s"], 0.02)
        self.assertEqual(snapshot["depth_invalid_stamp_samples"], 1)

    def test_valid_run_requires_camera_sync_contract(self):
        result = {
            "status": "ok",
            "collision_detected": False,
            "camera_contract_ok": True,
            "camera_sync_contract_ok": False,
        }
        self.assertFalse(benchmark.valid_evaluation_run(result))
        result["camera_sync_contract_ok"] = True
        self.assertTrue(benchmark.valid_evaluation_run(result))

    def test_plot_synthetic_artifacts(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib is not installed")
        import plot_active_camera_ablation as plotter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for arm in ("active", "fixed"):
                trajectory = root / "{}_trajectory.csv".format(arm)
                with trajectory.open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=["x", "y", "body_yaw_rad", "camera_yaw_rad"],
                    )
                    writer.writeheader()
                    writer.writerows([
                        {"x": 0, "y": 0, "body_yaw_rad": 0, "camera_yaw_rad": 0},
                        {"x": 1, "y": 0.2 if arm == "active" else -0.2,
                         "body_yaw_rad": 0.1, "camera_yaw_rad": 0.2 if arm == "active" else 0},
                    ])
                collision = root / "{}_collision.csv".format(arm)
                with collision.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["start_x", "start_y"])
                    writer.writeheader()
                map_path = root / "map.csv.gz"
                with gzip.open(str(map_path), "wt", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["x", "y", "z"])
                    writer.writerows([[0, 1, 1], [1, 1, 1]])
                runs.append(
                    {
                        "seed": 101,
                        "arm": arm,
                        "status": "ok",
                        "success": True,
                        "collision_detected": False,
                        "collision_free_success": True,
                        "camera_contract_ok": True,
                        "camera_sync_contract_ok": True,
                        "collision_episode_count": 0,
                        "min_clearance_m": 0.5,
                        "trajectory_csv": str(trajectory),
                        "collision_csv": str(collision),
                        "map_csv_gz": str(map_path),
                        "map_sha256": "same-map",
                    }
                )
            summary = {
                "aggregate": benchmark.aggregate_results(runs),
                "paired": benchmark.paired_results(runs),
                "runs": runs,
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary))
            generated = plotter.generate_plots(summary_path, root / "plots")
            self.assertEqual(len(generated), 2)
            self.assertTrue(all(path.stat().st_size > 0 for path in generated))


if __name__ == "__main__":
    unittest.main()
