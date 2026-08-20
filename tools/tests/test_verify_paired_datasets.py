import csv
import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

import yaml


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import verify_paired_datasets as verifier


def png_chunk(kind, data):
    payload = kind + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))


def make_png(width=160, height=192, bit_depth=16, color_type=0):
    bytes_per_sample = 2 if bit_depth == 16 else 1
    raw = b"".join(b"\0" + b"\0" * (width * bytes_per_sample) for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    return (
        verifier.PNG_SIGNATURE
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(raw))
        + png_chunk(b"IEND", b"")
    )


class PairedDatasetVerifierTest(unittest.TestCase):
    def metadata(self, active):
        return {
            "schema_version": 1,
            "active_camera": active,
            "image_width": 160,
            "image_height": 192,
            "max_depth_m": 20,
            "depth_preprocess": "resize_nearest_full_fov_v1",
            "sensor_model": "looper_insight_9",
            "maze_type": 8,
            "seed": 3,
            "env_num": 1,
            "image_num": 1,
            "direction_num": 1,
            "schema": {"sample_columns": list(verifier.EXPECTED_SAMPLE_COLUMNS)},
            "sensor": {"network_width": 160, "network_height": 192},
            "dataset": {
                "environment_count": 1,
                "images_per_environment": 1,
                "directions_per_image": 1,
                "camera_pitch_range_deg": [-60, 60] if active else [0, 0],
                "camera_yaw_range_deg": [-45, 45] if active else [0, 0],
            },
            "pairing": {"control_variable": "active_camera"},
        }

    def build_pair(self, root):
        active = root / "active"
        fixed = root / "fixed"
        for directory, treatment in ((active, True), (fixed, False)):
            (directory / "0").mkdir(parents=True)
            (directory / "dataset_metadata.yaml").write_text(
                yaml.safe_dump(self.metadata(treatment), sort_keys=False)
            )
            (directory / "_SUCCESS").write_text(
                "schema_version: 1\nenvironment_count: 1\nimages_per_environment: 1\n"
            )
            for name, content in (
                ("pointcloud-0.ply", b"ply\npaired\n"),
                ("pose-0.csv", b"px,py,pz\n1,2,3\n"),
                ("guides-0.csv", b"sample_id,x,y,z\n0,1,2,3\n"),
            ):
                (directory / name).write_bytes(content)
            row = [str(index) for index in range(verifier.STRUCTURAL_COLUMN_COUNT)]
            row.extend(["0.1", "-0.2", "0.3", "-0.4"] if treatment else ["0"] * 4)
            with (directory / "samples-0.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(verifier.EXPECTED_SAMPLE_COLUMNS)
                writer.writerow(row)
            (directory / "0" / "img_0_depth.png").write_bytes(make_png())
        return active, fixed

    def test_accepts_strictly_paired_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            active, fixed = self.build_pair(Path(directory))
            report = verifier.verify_pair(active, fixed)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["training_seed_set"], [3])
            self.assertIsNotNone(report["structural_pair_hash"])
            self.assertEqual(report["counts"]["sample_rows"], 1)
            self.assertEqual(report["dimensions"]["width"], 160)

    def test_rejects_structural_and_disallowed_metadata_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            active, fixed = self.build_pair(Path(directory))
            (fixed / "pointcloud-0.ply").write_bytes(b"different")
            metadata = yaml.safe_load((fixed / "dataset_metadata.yaml").read_text())
            metadata["max_depth_m"] = 19
            (fixed / "dataset_metadata.yaml").write_text(
                yaml.safe_dump(metadata, sort_keys=False)
            )
            report = verifier.verify_pair(active, fixed)
            self.assertFalse(report["ok"])
            self.assertIsNone(report["structural_pair_hash"])
            self.assertTrue(any("max_depth_m" in error for error in report["errors"]))
            self.assertTrue(any("pointcloud-0.ply" in error for error in report["errors"]))

    def test_rejects_sample_prefix_fixed_motion_and_bad_png(self):
        with tempfile.TemporaryDirectory() as directory:
            active, fixed = self.build_pair(Path(directory))
            with (fixed / "samples-0.csv").open("r", newline="") as stream:
                rows = list(csv.reader(stream))
            rows[1][3] = "different"
            rows[1][21] = "0.01"
            with (fixed / "samples-0.csv").open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)
            (active / "0" / "img_0_depth.png").write_bytes(make_png(width=80))
            report = verifier.verify_pair(active, fixed)
            self.assertFalse(report["ok"])
            self.assertTrue(any("Structural sample field" in error for error in report["errors"]))
            self.assertTrue(any("not exact numeric zero" in error for error in report["errors"]))
            self.assertTrue(any("must be 160x192" in error for error in report["errors"]))

    def test_cli_exit_codes_and_output_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, fixed = self.build_pair(root)
            output = root / "audit.json"
            with contextlib.redirect_stdout(io.StringIO()):
                exit_ok = verifier.main(
                    [str(active), str(fixed), "--output-json", str(output)]
                )
            self.assertEqual(exit_ok, 0)
            self.assertTrue(json.loads(output.read_text())["ok"])
            (fixed / "pointcloud-0.ply").write_bytes(b"not paired")
            with contextlib.redirect_stdout(io.StringIO()):
                exit_mismatch = verifier.main([str(active), str(fixed)])
                exit_input = verifier.main([str(root / "missing"), str(fixed)])
            self.assertEqual(exit_mismatch, 1)
            self.assertEqual(exit_input, 2)


if __name__ == "__main__":
    unittest.main()
