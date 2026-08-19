import atexit
import os
import time

import numpy as np
import torch
from rich.progress import Progress
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.loss_function import YOPOOmniLoss
from policy.state_transform import state_body2world
from policy.yopo_dataset import YOPOOmniPoseDataset
from policy.yopo_network import YOPOOmniNetwork


class YOPOOmniTrainer:
    def __init__(
            self,
            learning_rate=1.5e-4,
            batch_size=16,
            tensorboard_path=None,
            checkpoint_path=None,
            run_name=None,
            save_on_exit=False,
    ):
        self.batch_size = batch_size
        self.max_grad_norm = float(cfg["omni_max_grad_norm"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(cfg["omni_amp"]) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if save_on_exit:
            self._exit_func = atexit.register(self.save_model)

        self.progress_log = Progress()
        self.tensorboard_path = self.get_next_log_path(tensorboard_path, run_name=run_name)
        self.tensorboard_log = SummaryWriter(log_dir=self.tensorboard_path)

        print("Loading YOPO active-perception network...")
        self.policy = YOPOOmniNetwork().to(self.device)
        if checkpoint_path:
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.policy.load_state_dict(state_dict)
                print("Checkpoint ", checkpoint_path, " loaded successfully")
            except FileNotFoundError:
                print("Training from scratch")

        self.yopo_loss = YOPOOmniLoss()
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate, fused=torch.cuda.is_available())

        print("Loading YOPO active-perception dataset...")
        num_workers = int(cfg["omni_num_workers"])
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        self.train_dataloader = DataLoader(
            YOPOOmniPoseDataset(mode="train"),
            batch_size=self.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        self.val_dataloader = DataLoader(
            YOPOOmniPoseDataset(mode="valid"),
            batch_size=self.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        print("Dataset Loaded!")

    def train(self, epoch, save_interval=None):
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training YOPO Active", total=epoch)
            for self.epoch_i in range(epoch):
                self.policy.train()
                self.train_one_epoch(self.epoch_i, total_progress)
                self.policy.eval()
                self.eval_one_epoch(self.epoch_i)
                if save_interval is not None and (self.epoch_i + 1) % save_interval == 0:
                    self.save_model()
            self.progress_log.console.log("Train YOPO Active Finish!")
            self.progress_log.remove_task(total_progress)

    def train_one_epoch(self, epoch, total_progress):
        if hasattr(self.train_dataloader.dataset, "shuffle_groups"):
            self.train_dataloader.dataset.shuffle_groups(seed=epoch)
        one_epoch_progress = self.progress_log.add_task(f"Epoch: {epoch}", total=len(self.train_dataloader))
        inspect_interval = max(1, len(self.train_dataloader) // 16)
        metrics = {}
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

            self.collect_metrics(metrics, loss, detail)
            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                mean_metrics = {name: np.mean(values) for name, values in metrics.items()}
                self.progress_log.console.log(
                    f"Epoch: {epoch}, Loss: {mean_metrics['loss']:.3g}, "
                    f"Score: {mean_metrics['score']:.3g}, Camera: {mean_metrics['camera']:.3g}, "
                    f"Batch FPS: {batch_fps:.3g}"
                )
                global_step = epoch * len(self.train_dataloader) + step
                for name, value in mean_metrics.items():
                    self.tensorboard_log.add_scalar(f"Train/{name}", value, global_step)
                metrics = {}
                start_time = time.time()

            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / len(self.train_dataloader))

        self.progress_log.remove_task(one_epoch_progress)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch):
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=len(self.val_dataloader))
        metrics = {}
        for batch in self.val_dataloader:
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, detail = self.forward_and_compute_loss(batch)
            self.collect_metrics(metrics, loss, detail)
            self.progress_log.update(one_epoch_progress, advance=1)

        mean_metrics = {name: np.mean(values) for name, values in metrics.items()}
        self.progress_log.console.log(
            f"Eval: {epoch}, Loss: {mean_metrics['loss']:.3g}, "
            f"Score: {mean_metrics['score']:.3g}, Camera: {mean_metrics['camera']:.3g}"
        )
        for name, value in mean_metrics.items():
            self.tensorboard_log.add_scalar(f"Eval/{name}", value, epoch)
        self.progress_log.remove_task(one_epoch_progress)

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
    def collect_metrics(metrics, loss, detail):
        metrics.setdefault("loss", []).append(loss.item())
        for name, value in detail.items():
            metrics.setdefault(name, []).append(value.item())

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            policy_path = self.tensorboard_path + f"/epoch{self.epoch_i + 1}.pth"
            torch.save(self.policy.state_dict(), policy_path)
            if hasattr(self, "_exit_func"):
                atexit.unregister(self._exit_func)
                del self._exit_func

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
            suffix = 1
            while os.path.exists(candidate):
                candidate = os.path.join(base_path, f"{base_name}_{suffix}")
                suffix += 1
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
