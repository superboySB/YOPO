#!/usr/bin/env python3
"""Safely migrate a legacy YOPO checkpoint sidecar to an immutable manifest snapshot."""

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: {}".format(path))
    return value


def validate_record(checkpoint, checkpoint_sha, sidecar, manifest):
    sidecar_record = sidecar.get("checkpoint")
    if not isinstance(sidecar_record, dict):
        raise ValueError("Checkpoint sidecar has no checkpoint record")
    epoch = sidecar_record.get("epoch")
    if not isinstance(epoch, int) or epoch <= 0:
        raise ValueError("Checkpoint sidecar epoch must be a positive integer")
    expected = {
        "path": checkpoint.name,
        "epoch": epoch,
        "sha256": checkpoint_sha,
        "metrics_sha256": sidecar_record.get("metrics_sha256"),
    }
    if any(sidecar_record.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint sidecar record does not match the checkpoint artifact")

    manifest_record = (
        manifest.get("artifacts", {}).get("checkpoints", {}).get(checkpoint.name)
    )
    if not isinstance(manifest_record, dict) or any(
        manifest_record.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Training manifest checkpoint record does not match the sidecar")
    completed = manifest.get("optimization", {}).get("completed_epochs")
    if not isinstance(completed, int) or completed < epoch:
        raise ValueError("Training manifest predates the checkpoint epoch")
    return epoch


def write_new_snapshot(path, payload):
    """Atomically publish payload without ever replacing an existing snapshot."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                "Immutable snapshot already exists with different content: {}".format(path)
            )
        os.chmod(path, 0o644)
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    "Immutable snapshot was concurrently created with different content: {}".format(
                        path
                    )
                )
            os.chmod(path, 0o644)
            return False
        os.chmod(path, 0o644)
        return True
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_replace_json(path, value):
    path = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        os.chmod(path, 0o644)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def migrate_checkpoint(checkpoint_path, dry_run=False):
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint))
    sidecar_path = checkpoint.with_suffix(".manifest.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError("Checkpoint sidecar not found: {}".format(sidecar_path))

    sidecar = load_json(sidecar_path)
    checkpoint_sha = file_sha256(checkpoint)
    if sidecar.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise ValueError("Checkpoint SHA does not match its sidecar: {}".format(checkpoint))
    bound_manifest_path = checkpoint.parent / sidecar.get("training_manifest", "")
    if not bound_manifest_path.is_file():
        raise FileNotFoundError(
            "Bound training manifest not found: {}".format(bound_manifest_path)
        )
    bound_manifest_sha = file_sha256(bound_manifest_path)
    if sidecar.get("training_manifest_sha256") != bound_manifest_sha:
        raise ValueError("Training manifest SHA does not match the checkpoint sidecar")
    manifest = load_json(bound_manifest_path)
    epoch = validate_record(checkpoint, checkpoint_sha, sidecar, manifest)

    if sidecar.get("schema") == "yopo.checkpoint.v2" and sidecar.get(
        "training_manifest_immutable"
    ) is True:
        permission_repair_needed = any(
            artifact.stat().st_mode & 0o777 != 0o644
            for artifact in (bound_manifest_path, sidecar_path)
        )
        if not dry_run:
            os.chmod(bound_manifest_path, 0o644)
            os.chmod(sidecar_path, 0o644)
        return {
            "checkpoint": str(checkpoint),
            "epoch": epoch,
            "status": (
                "would_repair_permissions"
                if dry_run and permission_repair_needed
                else "repaired_permissions"
                if permission_repair_needed
                else "already_migrated"
            ),
            "snapshot": str(bound_manifest_path.resolve()),
            "snapshot_sha256": bound_manifest_sha,
        }

    snapshot_path = checkpoint.with_suffix(".training_manifest.json")
    source_payload = bound_manifest_path.read_bytes()
    migrated_sidecar = copy.deepcopy(sidecar)
    migrated_sidecar.update(
        {
            "schema": "yopo.checkpoint.v2",
            "training_manifest": snapshot_path.name,
            "training_manifest_sha256": hashlib.sha256(source_payload).hexdigest(),
            "training_manifest_immutable": True,
            "run_training_manifest": bound_manifest_path.name,
        }
    )
    result = {
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "status": "would_migrate" if dry_run else "migrated",
        "source_manifest": str(bound_manifest_path.resolve()),
        "source_manifest_sha256": bound_manifest_sha,
        "snapshot": str(snapshot_path.resolve()),
        "snapshot_sha256": migrated_sidecar["training_manifest_sha256"],
    }
    if dry_run:
        return result

    write_new_snapshot(snapshot_path, source_payload)
    atomic_replace_json(sidecar_path, migrated_sidecar)

    # Re-open the published artifacts and run the full chain once more.
    published_sidecar = load_json(sidecar_path)
    if file_sha256(snapshot_path) != published_sidecar.get("training_manifest_sha256"):
        raise RuntimeError("Published immutable snapshot failed SHA verification")
    validate_record(
        checkpoint, checkpoint_sha, published_sidecar, load_json(snapshot_path)
    )
    return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="One or more epochN.pth files.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report without writing files."
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        results = [
            migrate_checkpoint(checkpoint, dry_run=args.dry_run)
            for checkpoint in args.checkpoints
        ]
        print(json.dumps({"ok": True, "results": results}, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
