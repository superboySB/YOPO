import atexit
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from rich.progress import Progress
from ruamel.yaml import YAML
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.loss_function import YOPOOmniLoss
from policy.state_transform import state_body2world
from policy.yopo_dataset import YOPOOmniPoseDataset
from policy.yopo_network import YOPOOmniNetwork


def seed_dataloader_worker(_worker_id):
    """Make NumPy/Python sampling deterministic inside each torch worker."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class YOPOOmniTrainer:
    def __init__(
            self,
            learning_rate=1.5e-4,
            batch_size=16,
            tensorboard_path=None,
            checkpoint_path=None,
            run_name=None,
            save_on_exit=False,
            seed=0,
            train_epoch=50,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = int(seed)
        self.train_epoch = int(train_epoch)
        self.checkpoint_path = checkpoint_path or ""
        self.camera_mode = "active" if bool(cfg["active_camera"]) else "fixed"
        self.max_grad_norm = float(cfg["omni_max_grad_norm"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(cfg["omni_amp"]) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if save_on_exit:
            self._exit_func = atexit.register(self.save_model)

        self.progress_log = Progress()
        self.tensorboard_path = self.get_next_log_path(tensorboard_path, run_name=run_name)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)

        print(f"Loading YOPO {self.camera_mode}-camera network...")
        self.policy = YOPOOmniNetwork().to(self.device)
        if checkpoint_path:
            try:
                self.validate_checkpoint_config(checkpoint_path)
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.policy.load_state_dict(state_dict)
                print("Checkpoint ", checkpoint_path, " loaded successfully")
            except FileNotFoundError:
                print("Training from scratch")

        self.yopo_loss = YOPOOmniLoss()
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate, fused=torch.cuda.is_available())

        print(f"Loading YOPO {self.camera_mode}-camera dataset...")
        num_workers = int(cfg["omni_num_workers"])
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        train_generator = torch.Generator()
        train_generator.manual_seed(self.seed)
        val_generator = torch.Generator()
        val_generator.manual_seed(self.seed + 1)
        train_dataset = YOPOOmniPoseDataset(mode="train")
        val_dataset = YOPOOmniPoseDataset(mode="valid")
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=train_generator,
            worker_init_fn=seed_dataloader_worker,
            **loader_kwargs,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            generator=val_generator,
            worker_init_fn=seed_dataloader_worker,
            **loader_kwargs,
        )
        self.metrics_history = []
        self.write_resolved_config(train_dataset)
        print("Dataset Loaded!")
        print(
            f"Camera mode: {self.camera_mode}, "
            f"seed={self.seed}, preprocess={cfg['depth_preprocess']}"
        )

    def train(self, epoch, save_interval=None):
        with self.progress_log:
            total_progress = self.progress_log.add_task(f"Training YOPO {self.camera_mode.capitalize()}", total=epoch)
            for self.epoch_i in range(epoch):
                self.policy.train()
                train_metrics = self.train_one_epoch(self.epoch_i, total_progress)
                self.policy.eval()
                eval_metrics = self.eval_one_epoch(self.epoch_i)
                self.record_epoch_metrics(self.epoch_i + 1, train_metrics, eval_metrics)
                if save_interval is not None and (self.epoch_i + 1) % save_interval == 0:
                    self.save_model()
            if epoch > 0 and (save_interval is None or epoch % save_interval != 0):
                self.save_model()
            self.progress_log.console.log(f"Train YOPO {self.camera_mode.capitalize()} Finish!")
            self.progress_log.remove_task(total_progress)

    def train_one_epoch(self, epoch, total_progress):
        one_epoch_progress = self.progress_log.add_task(f"Epoch: {epoch}", total=len(self.train_dataloader))
        inspect_interval = max(1, len(self.train_dataloader) // 16)
        window_metrics = {}
        epoch_metrics = {}
        start_time = time.time()

        for step, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, detail = self.forward_and_compute_loss(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_weight = int(batch[3].shape[0])
            self.collect_metrics(window_metrics, loss, detail, batch_weight)
            self.collect_metrics(epoch_metrics, loss, detail, batch_weight)
            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                mean_metrics = self.mean_collected_metrics(window_metrics)
                self.progress_log.console.log(
                    f"Epoch: {epoch}, Loss: {mean_metrics['loss']:.3g}, "
                    f"Score: {mean_metrics['score']:.3g}, Camera: {mean_metrics['camera']:.3g}, "
                    f"Batch FPS: {batch_fps:.3g}"
                )
                global_step = epoch * len(self.train_dataloader) + step
                for name, value in mean_metrics.items():
                    self.tensorboard_log.add_scalar(f"Train/{name}", value, global_step)
                window_metrics = {}
                start_time = time.time()

            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / len(self.train_dataloader))

        self.progress_log.remove_task(one_epoch_progress)
        mean_epoch_metrics = self.mean_collected_metrics(epoch_metrics)
        for name, value in mean_epoch_metrics.items():
            self.tensorboard_log.add_scalar(f"TrainEpoch/{name}", value, epoch)
        return mean_epoch_metrics

    @torch.inference_mode()
    def eval_one_epoch(self, epoch):
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=len(self.val_dataloader))
        metrics = {}
        for batch in self.val_dataloader:
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, detail = self.forward_and_compute_loss(batch)
            self.collect_metrics(metrics, loss, detail, int(batch[3].shape[0]))
            self.progress_log.update(one_epoch_progress, advance=1)

        mean_metrics = self.mean_collected_metrics(metrics)
        self.progress_log.console.log(
            f"Eval: {epoch}, Loss: {mean_metrics['loss']:.3g}, "
            f"Score: {mean_metrics['score']:.3g}, Camera: {mean_metrics['camera']:.3g}"
        )
        for name, value in mean_metrics.items():
            self.tensorboard_log.add_scalar(f"Eval/{name}", value, epoch)
        self.progress_log.remove_task(one_epoch_progress)
        return {name: float(value) for name, value in mean_metrics.items()}

    def forward_and_compute_loss(self, batch):
        depth, pos, rot, state_b, _, guide_path_w, guide_mask, selected_topology, camera_target, map_id = [
            x.to(self.device) for x in batch
        ]

        endstate_b, score, pred_camera_target = self.policy(depth, state_b)
        (pos, rot, state_b, guide_path_w, guide_mask, selected_topology,
         camera_target, map_id, endstate_b, score, pred_camera_target) = (
            self.flatten_pose_batch(
                pos, rot, state_b, guide_path_w, guide_mask, selected_topology,
                camera_target, map_id, endstate_b, score, pred_camera_target
            )
        )
        B, K = endstate_b.shape[:2]

        zero_pos_b = torch.zeros_like(pos)
        start_pos_w, start_vel_w, start_acc_w = state_body2world(
            pos, rot, zero_pos_b, state_b[:, 0:3], state_b[:, 3:6]
        )
        start_state_w = torch.stack([start_pos_w, start_vel_w, start_acc_w], dim=1)

        endstate_flat = endstate_b.reshape(B * K, 9)
        pos_expanded = pos.repeat_interleave(K, dim=0)
        rot_expanded = rot.repeat_interleave(K, dim=0)
        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded,
            rot_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9],
        )
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1).reshape(B, K, 3, 3)

        return self.yopo_loss(
            start_state_w=start_state_w,
            end_state_w=end_state_w,
            endstate_b=endstate_b,
            state_b=state_b,
            guide_path_w=guide_path_w,
            guide_mask=guide_mask,
            selected_topology=selected_topology.long(),
            map_id=map_id.long(),
            pred_score=score,
            pred_camera_target=pred_camera_target,
            camera_target=camera_target,
            camera_orientation=state_b[:, 9:11],
        )

    @staticmethod
    def flatten_pose_batch(pos, rot, state_b, guide_path_w, guide_mask, selected_topology,
                           camera_target, map_id, endstate_b, score, pred_camera_target):
        if state_b.dim() == 2:
            return (pos, rot, state_b, guide_path_w, guide_mask, selected_topology,
                    camera_target, map_id, endstate_b, score, pred_camera_target)

        if state_b.dim() != 3:
            raise ValueError(f"Expected state_b shape [B,11] or [B,D,11], got {tuple(state_b.shape)}")

        B, D = state_b.shape[:2]
        K = endstate_b.shape[2]

        flat_pos = pos[:, None, :].expand(B, D, 3).reshape(B * D, 3)
        flat_rot = rot[:, None, :, :].expand(B, D, 3, 3).reshape(B * D, 3, 3)
        flat_state_b = state_b.reshape(B * D, 11)
        flat_guide_path_w = guide_path_w.reshape(B * D, guide_path_w.shape[-2], 3)
        flat_guide_mask = guide_mask.reshape(B * D)
        flat_selected_topology = selected_topology.reshape(B * D)
        flat_camera_target = camera_target.reshape(B * D, 2)
        flat_map_id = map_id[:, None].expand(B, D).reshape(B * D)
        flat_endstate_b = endstate_b.reshape(B * D, K, 9)
        flat_score = score.reshape(B * D, K)
        flat_pred_camera_target = pred_camera_target.reshape(B * D, K, 2)

        return (
            flat_pos,
            flat_rot,
            flat_state_b,
            flat_guide_path_w,
            flat_guide_mask,
            flat_selected_topology,
            flat_camera_target,
            flat_map_id,
            flat_endstate_b,
            flat_score,
            flat_pred_camera_target,
        )

    @staticmethod
    def collect_metrics(metrics, loss, detail, weight=1):
        metrics.setdefault("loss", []).append((loss.item(), int(weight)))
        for name, value in detail.items():
            metrics.setdefault(name, []).append((value.item(), int(weight)))

    @staticmethod
    def mean_collected_metrics(metrics):
        result = {}
        for name, values in metrics.items():
            total_weight = sum(weight for _, weight in values)
            result[name] = float(
                sum(value * weight for value, weight in values) / max(1, total_weight)
            )
        return result

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            policy_path = self.tensorboard_path + f"/epoch{self.epoch_i + 1}.pth"
            torch.save(self.policy.state_dict(), policy_path)
            self.register_checkpoint(policy_path, self.epoch_i + 1)
            if hasattr(self, "_exit_func"):
                atexit.unregister(self._exit_func)
                del self._exit_func

    @staticmethod
    def sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def git_head(repo_root):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def write_resolved_config(self, dataset):
        resolved = copy.deepcopy(cfg._data)
        resolved["seed"] = self.seed
        resolved["train_epoch"] = self.train_epoch
        resolved["batch_size"] = self.batch_size
        resolved["learning_rate"] = float(self.learning_rate)
        resolved["checkpoint_init"] = self.checkpoint_path
        resolved["device"] = str(self.device)
        resolved["amp_enabled"] = bool(self.use_amp)
        resolved["output_dir"] = str(Path(self.tensorboard_path).resolve())
        resolved["dataset_metadata"] = copy.deepcopy(dict(dataset.metadata))
        output_path = os.path.join(self.tensorboard_path, "resolved_config.yaml")
        yaml = YAML()
        yaml.default_flow_style = False
        with open(output_path, "w", encoding="utf-8") as stream:
            yaml.dump(resolved, stream)

        yopo_root = Path(__file__).resolve().parents[1]
        repo_root = yopo_root.parent
        source_paths = (
            "train_yopo.py",
            "policy/yopo_trainer.py",
            "policy/yopo_dataset.py",
            "policy/yopo_network.py",
            "policy/models/backbone.py",
            "loss/loss_function.py",
        )
        metadata_path = Path(dataset.data_dir) / dataset._METADATA_FILE
        self.training_manifest_path = Path(self.tensorboard_path) / "training_manifest.json"
        self.training_manifest = {
            "schema": "yopo.training-run.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "treatment": {
                "active_camera": bool(cfg["active_camera"]),
                "camera_mode": self.camera_mode,
            },
            "optimization": {
                "seed": self.seed,
                "train_epoch": self.train_epoch,
                "completed_epochs": 0,
                "batch_size": self.batch_size,
                "learning_rate": float(self.learning_rate),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
            },
            "model_contract": {
                "sensor_model": str(dataset.metadata["sensor_model"]),
                "image_width": int(cfg["image_width"]),
                "image_height": int(cfg["image_height"]),
                "max_depth_m": float(cfg["insight9_train_max_depth_m"]),
                "depth_preprocess": str(cfg["depth_preprocess"]),
                "state_dim": 11,
                "policy_head_dim": 12,
                "camera_loss_weight": float(cfg["w_camera"]),
                "camera_smooth_loss_weight": float(cfg["w_camera_smooth"]),
            },
            "dataset": {
                "path": str(Path(dataset.data_dir).resolve()),
                "metadata_sha256": self.sha256_file(metadata_path),
                "metadata": copy.deepcopy(dict(dataset.metadata)),
            },
            "config": copy.deepcopy(cfg._data),
            "runtime": {
                "device": str(self.device),
                "amp_enabled": bool(self.use_amp),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "source": {
                "git_head": self.git_head(repo_root),
                "files_sha256": {
                    relative: self.sha256_file(yopo_root / relative) for relative in source_paths
                },
            },
            "artifacts": {
                "output_dir": str(Path(self.tensorboard_path).resolve()),
                "resolved_config": str(Path(output_path).resolve()),
                "resolved_config_sha256": self.sha256_file(output_path),
                "metrics_sha256": None,
                "checkpoints": {},
            },
        }
        self.write_training_manifest()

    def write_training_manifest(self):
        with self.training_manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(self.training_manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")

    @staticmethod
    def write_immutable_json(path, value):
        """Write a content-addressed run artifact once, never silently replace it."""
        path = Path(path)
        content = json.dumps(value, indent=2, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise FileExistsError(
                    f"Immutable checkpoint artifact already exists with different content: {path}"
                )
            os.chmod(path, 0o644)
            return
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
            os.chmod(path, 0o644)
        finally:
            if temporary.exists():
                temporary.unlink()

    def register_checkpoint(self, checkpoint_path, epoch):
        checkpoint = Path(checkpoint_path).resolve()
        metrics_path = Path(self.tensorboard_path) / "metrics.json"
        metrics_sha = self.sha256_file(metrics_path) if metrics_path.is_file() else None
        record = {
            "epoch": int(epoch),
            "path": checkpoint.name,
            "sha256": self.sha256_file(checkpoint),
            "metrics_sha256": metrics_sha,
        }
        self.training_manifest["artifacts"]["metrics_sha256"] = metrics_sha
        self.training_manifest["artifacts"]["checkpoints"][checkpoint.name] = record
        self.write_training_manifest()

        # training_manifest.json remains the final, evolving run index.  A
        # checkpoint sidecar must not bind to it directly: later epochs mutate
        # that file and used to invalidate every intermediate checkpoint.  Bind
        # each checkpoint to an immutable point-in-time snapshot instead.
        snapshot_path = checkpoint.with_suffix(".training_manifest.json")
        self.write_immutable_json(snapshot_path, copy.deepcopy(self.training_manifest))
        sidecar = {
            "schema": "yopo.checkpoint.v2",
            "checkpoint": record,
            "training_manifest": snapshot_path.name,
            "training_manifest_sha256": self.sha256_file(snapshot_path),
            "training_manifest_immutable": True,
            "run_training_manifest": self.training_manifest_path.name,
            "resolved_config": "resolved_config.yaml",
            "resolved_config_sha256": self.training_manifest["artifacts"]["resolved_config_sha256"],
            "active_camera": bool(cfg["active_camera"]),
            "camera_mode": self.camera_mode,
        }
        sidecar_path = checkpoint.with_suffix(".manifest.json")
        with sidecar_path.open("w", encoding="utf-8") as stream:
            json.dump(sidecar, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(sidecar_path, 0o644)

    @staticmethod
    def validate_checkpoint_config(checkpoint_path):
        """Reject active/fixed or preprocessing mismatches when resuming."""
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            return
        sidecar_path = checkpoint.with_suffix(".manifest.json")
        if sidecar_path.is_file():
            with sidecar_path.open("r", encoding="utf-8") as stream:
                sidecar = json.load(stream)
            expected_hash = sidecar.get("checkpoint", {}).get("sha256")
            actual_hash = YOPOOmniTrainer.sha256_file(checkpoint)
            if expected_hash != actual_hash:
                raise ValueError(
                    f"Checkpoint SHA mismatch for {checkpoint}: sidecar={expected_hash}, actual={actual_hash}"
                )
            manifest_path = checkpoint.parent / sidecar.get("training_manifest", "")
            if not manifest_path.is_file() or YOPOOmniTrainer.sha256_file(manifest_path) != sidecar.get(
                "training_manifest_sha256"
            ):
                raise ValueError(f"Checkpoint training-manifest binding is invalid: {sidecar_path}")
            with manifest_path.open("r", encoding="utf-8") as stream:
                bound_manifest = json.load(stream)
            bound_record = (
                bound_manifest.get("artifacts", {})
                .get("checkpoints", {})
                .get(checkpoint.name)
            )
            sidecar_record = sidecar.get("checkpoint", {})
            if not isinstance(bound_record, dict) or any(
                bound_record.get(key) != sidecar_record.get(key)
                for key in ("path", "epoch", "sha256", "metrics_sha256")
            ):
                raise ValueError(
                    f"Checkpoint record is not bound by training manifest: {sidecar_path}"
                )
        else:
            print(f"WARNING: checkpoint has no hash-binding sidecar {sidecar_path.name}")
        resolved_path = checkpoint.parent / "resolved_config.yaml"
        if not resolved_path.is_file():
            print(
                f"WARNING: checkpoint has no {resolved_path.name}; resume contract cannot be audited "
                "(legacy checkpoint compatibility mode)."
            )
            return

        with resolved_path.open("r", encoding="utf-8") as stream:
            resolved = YAML(typ="safe").load(stream)
        if not isinstance(resolved, dict):
            raise ValueError(f"Checkpoint resolved config must be a YAML mapping: {resolved_path}")

        expected = {
            "active_camera": bool(cfg["active_camera"]),
            "image_width": int(cfg["image_width"]),
            "image_height": int(cfg["image_height"]),
            "insight9_train_max_depth_m": float(cfg["insight9_train_max_depth_m"]),
            "depth_preprocess": str(cfg["depth_preprocess"]),
        }
        missing = [key for key in expected if key not in resolved]
        if missing:
            raise ValueError(f"Checkpoint resolved config {resolved_path} is missing keys: {missing}")

        mismatches = []
        for key, expected_value in expected.items():
            actual_value = resolved[key]
            if key == "insight9_train_max_depth_m":
                try:
                    matches = math.isclose(float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-6)
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual_value == expected_value
            if not matches:
                mismatches.append(f"{key}: checkpoint={actual_value!r}, training={expected_value!r}")
        if mismatches:
            raise ValueError("Checkpoint/training contract mismatch: " + "; ".join(mismatches))
        print(f"Checkpoint contract: {resolved_path} (validated)")

    def record_epoch_metrics(self, epoch, train_metrics, eval_metrics):
        row = {"epoch": int(epoch)}
        row.update({f"train_{name}": float(value) for name, value in train_metrics.items()})
        row.update({f"eval_{name}": float(value) for name, value in eval_metrics.items()})
        self.metrics_history.append(row)

        json_path = os.path.join(self.tensorboard_path, "metrics.json")
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(self.metrics_history, stream, indent=2, sort_keys=True)
            stream.write("\n")

        csv_path = os.path.join(self.tensorboard_path, "metrics.csv")
        metric_names = sorted({key for item in self.metrics_history for key in item if key != "epoch"})
        fieldnames = ["epoch", *metric_names]
        with open(csv_path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)
        if hasattr(self, "training_manifest"):
            self.training_manifest["optimization"]["completed_epochs"] = int(epoch)
            self.training_manifest["artifacts"]["metrics_sha256"] = self.sha256_file(json_path)
            self.write_training_manifest()
        self.tensorboard_log.flush()

    def close(self):
        """Flush logs and terminate persistent DataLoader workers cleanly."""
        self.tensorboard_log.flush()
        self.tensorboard_log.close()
        for loader in (self.train_dataloader, self.val_dataloader):
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                loader._iterator = None

    @staticmethod
    def get_next_log_path(base_path, run_name=None):
        if run_name:
            base_name = run_name
            candidate = os.path.join(base_path, base_name)
            if os.path.exists(candidate):
                raise FileExistsError(
                    f"Training run directory already exists: {candidate}. "
                    "Choose a new --run-name so reports cannot silently reuse stale checkpoints."
                )
            os.makedirs(candidate, exist_ok=False)
            print("record tensorboard log to ", candidate)
            return candidate

        prefix = "YOPO_"
        nums = [
            int(name.split("_")[-1])
            for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name)) and name.startswith(prefix) and name.split("_")[-1].isdigit()
        ]
        next_n = max(nums, default=-1) + 1
        next_path = os.path.join(base_path, f"{prefix}{next_n}")
        os.makedirs(next_path, exist_ok=False)
        print("record tensorboard log to ", next_path)
        return next_path
