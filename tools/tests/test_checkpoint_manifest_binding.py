import copy
import hashlib
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
YOPO_ROOT = ROOT / "YOPO"
for search_path in (TOOLS, YOPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import benchmark_active_camera as benchmark
import migrate_checkpoint_manifest as migration
import verify_ablation_pair as verifier
from policy.yopo_trainer import YOPOOmniTrainer


def json_sha(value):
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CheckpointManifestSnapshotTest(unittest.TestCase):
    def test_intermediate_checkpoint_keeps_immutable_manifest_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = YOPOOmniTrainer.__new__(YOPOOmniTrainer)
            trainer.tensorboard_path = str(root)
            trainer.training_manifest_path = root / "training_manifest.json"
            trainer.camera_mode = "active"
            trainer.training_manifest = {
                "optimization": {"completed_epochs": 10},
                "artifacts": {
                    "metrics_sha256": None,
                    "resolved_config_sha256": "resolved-config-sha",
                    "checkpoints": {},
                },
            }
            trainer.write_training_manifest()
            (root / "metrics.json").write_text("[]\n", encoding="utf-8")
            checkpoint = root / "epoch10.pth"
            checkpoint.write_bytes(b"checkpoint-epoch-10")

            trainer.register_checkpoint(checkpoint, 10)
            sidecar = json.loads((root / "epoch10.manifest.json").read_text())
            snapshot = root / sidecar["training_manifest"]
            snapshot_sha = YOPOOmniTrainer.sha256_file(snapshot)
            self.assertEqual(sidecar["schema"], "yopo.checkpoint.v2")
            self.assertTrue(sidecar["training_manifest_immutable"])
            self.assertEqual(sidecar["training_manifest_sha256"], snapshot_sha)

            # Simulate the same run progressing beyond the intermediate save.
            trainer.training_manifest["optimization"]["completed_epochs"] = 20
            trainer.write_training_manifest()
            self.assertEqual(YOPOOmniTrainer.sha256_file(snapshot), snapshot_sha)
            self.assertNotEqual(
                YOPOOmniTrainer.sha256_file(trainer.training_manifest_path), snapshot_sha
            )
            snapshot_record = json.loads(snapshot.read_text())["artifacts"]["checkpoints"][
                checkpoint.name
            ]
            self.assertEqual(snapshot_record, sidecar["checkpoint"])
            self.assertEqual(stat.S_IMODE(snapshot.stat().st_mode), 0o644)
            self.assertEqual(
                stat.S_IMODE((root / "epoch10.manifest.json").stat().st_mode), 0o644
            )

    def test_legacy_migration_is_validated_idempotent_and_tamper_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "epoch10.pth"
            checkpoint.write_bytes(b"legacy-checkpoint")
            checkpoint_sha = migration.file_sha256(checkpoint)
            record = {
                "path": checkpoint.name,
                "epoch": 10,
                "sha256": checkpoint_sha,
                "metrics_sha256": "metrics-sha",
            }
            manifest = {
                "optimization": {"completed_epochs": 10},
                "artifacts": {"checkpoints": {checkpoint.name: record}},
            }
            manifest_path = root / "training_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            sidecar_path = root / "epoch10.manifest.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "schema": "yopo.checkpoint.v1",
                        "checkpoint": record,
                        "training_manifest": manifest_path.name,
                        "training_manifest_sha256": migration.file_sha256(manifest_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            dry_run = migration.migrate_checkpoint(checkpoint, dry_run=True)
            self.assertEqual(dry_run["status"], "would_migrate")
            self.assertFalse((root / "epoch10.training_manifest.json").exists())
            migrated = migration.migrate_checkpoint(checkpoint)
            self.assertEqual(migrated["status"], "migrated")
            snapshot_path = root / "epoch10.training_manifest.json"
            self.assertEqual(stat.S_IMODE(snapshot_path.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE(sidecar_path.stat().st_mode), 0o644)
            self.assertEqual(
                migration.migrate_checkpoint(checkpoint)["status"], "already_migrated"
            )

            snapshot_path.chmod(0o600)
            sidecar_path.chmod(0o600)
            self.assertEqual(
                migration.migrate_checkpoint(checkpoint, dry_run=True)["status"],
                "would_repair_permissions",
            )
            self.assertEqual(
                migration.migrate_checkpoint(checkpoint)["status"],
                "repaired_permissions",
            )
            self.assertEqual(stat.S_IMODE(snapshot_path.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE(sidecar_path.stat().st_mode), 0o644)

            # A legacy sidecar whose bound manifest changed must be rejected
            # before either the snapshot or sidecar can be published.
            second = root / "epoch20.pth"
            second.write_bytes(b"legacy-checkpoint-20")
            second_record = {
                "path": second.name,
                "epoch": 20,
                "sha256": migration.file_sha256(second),
                "metrics_sha256": "metrics-sha-20",
            }
            manifest["optimization"]["completed_epochs"] = 20
            manifest["artifacts"]["checkpoints"][second.name] = second_record
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (root / "epoch20.manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "yopo.checkpoint.v1",
                        "checkpoint": second_record,
                        "training_manifest": manifest_path.name,
                        "training_manifest_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Training manifest SHA"):
                migration.migrate_checkpoint(second)
            self.assertFalse((root / "epoch20.training_manifest.json").exists())


class OfflineCheckpointBindingTest(unittest.TestCase):
    def build_inputs(self, root):
        args = benchmark.make_parser().parse_args(
            [
                "--active-weight", str(root / "active" / "epoch10.pth"),
                "--fixed-weight", str(root / "fixed" / "epoch10.pth"),
                "--seeds", "101,102",
                "--dry-run", "--skip-weight-check", "--no-plot",
                "--output-dir", str(root / "benchmark"),
            ]
        )
        manifest = benchmark.build_manifest(args, root / "benchmark")
        trainings = {}
        bindings = {}
        checkpoint_shas = {"active": "a" * 64, "fixed": "f" * 64}
        for arm, is_active in (("active", True), ("fixed", False)):
            checkpoint_name = "epoch10.pth"
            training = {
                "schema": "yopo.training-run.v1",
                "created_utc": arm,
                "treatment": {"active_camera": is_active, "camera_mode": arm},
                "optimization": {
                    "seed": 0,
                    "train_epoch": 10,
                    "completed_epochs": 10,
                    "batch_size": 16,
                    "learning_rate": 0.00015,
                },
                "model_contract": {
                    "sensor_model": "looper_insight_9",
                    "image_width": 160,
                    "image_height": 192,
                    "camera_loss_weight": 1.0,
                },
                "dataset": {
                    "path": "/dataset/{}".format(arm),
                    "metadata_sha256": arm,
                    "metadata": {
                        "active_camera": is_active,
                        "seed": 3,
                        "env_num": 10,
                        "dataset": {
                            "camera_pitch_range_deg": [-60, 60] if is_active else [0, 0],
                            "camera_yaw_range_deg": [-45, 45] if is_active else [0, 0],
                        },
                    },
                },
                "config": {
                    "active_camera": is_active,
                    "dataset_path": "/dataset/{}".format(arm),
                },
                "source": {"same": True},
                "artifacts": {
                    "output_dir": "/output/{}".format(arm),
                    "resolved_config": "/output/{}/resolved_config.yaml".format(arm),
                    "resolved_config_sha256": arm,
                    "metrics_sha256": arm,
                    "checkpoints": {
                        checkpoint_name: {
                            "path": checkpoint_name,
                            "epoch": 10,
                            "sha256": checkpoint_shas[arm],
                            "metrics_sha256": arm,
                        }
                    },
                },
            }
            trainings[arm] = training
            bindings[arm] = {
                "path": str(root / arm / "epoch10.training_manifest.json"),
                "sha256": json_sha(training),
            }

        for run in manifest["runs"]:
            arm = run["arm"]
            treatment = run["treatment"]
            treatment["checkpoint_sha256"] = checkpoint_shas[arm]
            treatment["checkpoint_artifact"] = {
                "checkpoint_sha256": checkpoint_shas[arm],
                "training_manifest_sha256": bindings[arm]["sha256"],
                "epoch": 10,
            }
            run["run_spec_hash"] = benchmark.content_hash(
                {
                    "seed": run["seed"],
                    "arm": run["arm"],
                    "treatment": treatment,
                    "controlled_variables": run["controlled_variables"],
                    "commands": run["commands"],
                }
            )
        return manifest, trainings, bindings

    def verify(self, manifest, trainings, bindings):
        return verifier.verify(
            manifest,
            training_pair=(trainings["active"], trainings["fixed"]),
            training_manifest_bindings=bindings,
            training_seeds=list(range(3, 13)),
            expected_training_epochs=10,
        )

    def test_accepts_closed_manifest_checkpoint_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, trainings, bindings = self.build_inputs(Path(directory))
            report = self.verify(manifest, trainings, bindings)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(
                report["training_checkpoint_bindings"]["active"]["epoch"], 10
            )

    def test_rejects_supplied_manifest_sha_and_checkpoint_record_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, trainings, bindings = self.build_inputs(Path(directory))
            bad_bindings = copy.deepcopy(bindings)
            bad_bindings["active"]["sha256"] = "0" * 64
            report = self.verify(manifest, trainings, bad_bindings)
            self.assertTrue(
                any("supplied training manifest SHA" in error for error in report["errors"])
            )

            bad_trainings = copy.deepcopy(trainings)
            bad_trainings["fixed"]["artifacts"]["checkpoints"]["epoch10.pth"][
                "epoch"
            ] = 9
            report = self.verify(manifest, bad_trainings, bindings)
            self.assertTrue(
                any("checkpoint record epoch" in error for error in report["errors"])
            )


if __name__ == "__main__":
    unittest.main()
