"""
Training Strategy
supervised learning, imitation learning, testing, rollout
"""
import os
import time
import atexit
import numpy as np
import torch
from torch.nn import functional as F
from rich.progress import Progress
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config.config import cfg
from loss.loss_function import YOPOLoss
from policy.yopo_network import YopoNetwork
from policy.yopo_dataset import YOPODataset
from policy.state_transform import *


class YopoTrainer:
    def __init__(
            self,
            learning_rate=0.001,
            batch_size=32,
            loss_weight=None,
            tensorboard_path=None,
            checkpoint_path=None,
            num_workers=4,
            save_on_exit=False,
    ):
        self.batch_size = batch_size
        self.max_grad_norm = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        legacy_loss_weight = loss_weight or []
        self.trajectory_loss_weight = float(
            cfg.get("trajectory_loss_weight", legacy_loss_weight[0] if len(legacy_loss_weight) > 0 else 1.0)
        )
        self.score_loss_weight = float(
            cfg.get("score_loss_weight", legacy_loss_weight[1] if len(legacy_loss_weight) > 1 else 1.0)
        )
        self.loss_weight = [self.trajectory_loss_weight, self.score_loss_weight]
        self.num_workers = num_workers
        if save_on_exit: self._exit_func = atexit.register(self.save_model)
        # logger
        self.progress_log = Progress()
        self.run_dir = self.get_next_log_path(tensorboard_path)
        self.tensorboard_path = self.run_dir
        self.checkpoint_path = self.run_dir
        self.tensorboard_log = SummaryWriter(log_dir=self.run_dir)
        # params
        self.traj_num = cfg['traj_num']
        self.image_width = cfg["image_width"]
        self.image_height = cfg["image_height"]
        self.target_dynamic_max_count = int(cfg.get("target_dynamic_max_count", 3))
        self.target_separation_weight = float(cfg.get("target_separation_weight", 0.0))
        self.target_separation_distance = float(cfg["swarm_initial_spacing"])
        self.target_separation_eval_points = int(cfg.get("target_separation_eval_points", 5))
        if self.target_separation_distance <= 0.0:
            raise ValueError("swarm_initial_spacing must be positive.")
        if self.target_separation_eval_points <= 0:
            raise ValueError("target_separation_eval_points must be positive.")

        # network
        print("Loading network...")
        self.policy = YopoNetwork()
        self.policy = self.policy.to(self.device)
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            self.policy.load_state_dict(state_dict)
            print("Checkpoint ", checkpoint_path, " loaded successfully")
        except FileNotFoundError:
            print("Training from scratch")

        # loss
        self.yopo_loss = YOPOLoss()
        self._log_loss_weight_config()

        # optimizer
        fused_adamw = self.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate, fused=fused_adamw)
        print("Network Loaded! Loading Dataset...")

        # dataset (you can adjust num_workers according to your training speed)
        self.train_dataloader = DataLoader(YOPODataset(mode='train'), batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, pin_memory=True)
        self.val_dataloader = DataLoader(YOPODataset(mode='valid'), batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers, pin_memory=True)
        print("Dataset Loaded!")

    def train(self, epoch, save_interval=None):
        with self.progress_log:
            total_progress = self.progress_log.add_task("Training", total=epoch)
            for self.epoch_i in range(epoch):
                self.policy.train()
                self.train_one_epoch(self.epoch_i, total_progress)
                self.policy.eval()
                self.eval_one_epoch(self.epoch_i)
                if save_interval is not None and (self.epoch_i + 1) % save_interval == 0:
                    self.progress_log.console.log("Saving model...")
                    self.save_checkpoint(self.epoch_i + 1)
            self.progress_log.console.log("Train YOPO Finish!")
            self.progress_log.remove_task(total_progress)
            self._finish_exit_save(save_interval)
            self.tensorboard_log.flush()
            self.tensorboard_log.close()

    def train_one_epoch(self, epoch: int, total_progress):
        one_epoch_progress = self.progress_log.add_task(f"Epoch: {epoch}", total=len(self.train_dataloader))
        inspect_interval = max(1, len(self.train_dataloader) // 16)
        metric_buffer = {}
        start_time = time.time()
        for step, batch in enumerate(self.train_dataloader):  # obs: camera/body frame
            self.optimizer.zero_grad()

            metrics = self.forward_and_compute_loss(*batch)
            batch_weight = batch[0].shape[0]
            loss = (
                self.trajectory_loss_weight * metrics["trajectory_loss"]
                + self.score_loss_weight * metrics["score_loss"]
            )
            metrics["total_loss"] = loss.detach()
            metrics["weighted_trajectory_loss"] = self.trajectory_loss_weight * metrics["trajectory_loss"]
            metrics["weighted_score_loss"] = self.score_loss_weight * metrics["score_loss"]

            # Optimize the policy
            loss.backward()
            self.optimizer.step()

            self._append_metrics(metric_buffer, metrics, default_weight=batch_weight)

            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                avg_metrics = self._mean_metrics(metric_buffer)
                self.progress_log.console.log(
                    f"Epoch: {epoch}, Total Loss: {avg_metrics['total_loss']:.3g}, "
                    f"Traj Loss: {avg_metrics['trajectory_loss']:.3g}, "
                    f"Score Loss: {avg_metrics['score_loss']:.3g}, "
                    f"Static Safety: {avg_metrics['static_safety_loss']:.3g}, "
                    f"Dynamic Safety: {avg_metrics['dynamic_safety_loss']:.3g}, "
                    f"Target Sep: {avg_metrics['target_separation_loss']:.3g}, "
                    f"Batch FPS: {batch_fps:.3g}"
                )
                self._write_metrics("Train", avg_metrics, epoch * len(self.train_dataloader) + step)
                metric_buffer = {}
                start_time = time.time()

            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / len(self.train_dataloader))

        if metric_buffer:
            avg_metrics = self._mean_metrics(metric_buffer)
            self._write_metrics("Train", avg_metrics, epoch * len(self.train_dataloader) + len(self.train_dataloader) - 1)

        self.progress_log.remove_task(one_epoch_progress)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int):
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=len(self.val_dataloader))
        metric_buffer = {}
        for step, batch in enumerate(self.val_dataloader):  # obs: camera/body frame
            metrics = self.forward_and_compute_loss(*batch)
            batch_weight = batch[0].shape[0]
            metrics["total_loss"] = (
                self.trajectory_loss_weight * metrics["trajectory_loss"]
                + self.score_loss_weight * metrics["score_loss"]
            )
            metrics["weighted_trajectory_loss"] = self.trajectory_loss_weight * metrics["trajectory_loss"]
            metrics["weighted_score_loss"] = self.score_loss_weight * metrics["score_loss"]
            self._append_metrics(metric_buffer, metrics, default_weight=batch_weight)
            self.progress_log.update(one_epoch_progress, advance=1)

        avg_metrics = self._mean_metrics(metric_buffer)
        self.progress_log.console.log(
            f"Eval: {epoch}, Total Loss: {avg_metrics['total_loss']:.3g}, "
            f"Traj Loss: {avg_metrics['trajectory_loss']:.3g}, "
            f"Score Loss: {avg_metrics['score_loss']:.3g}, "
            f"Static Safety: {avg_metrics['static_safety_loss']:.3g}, "
            f"Dynamic Safety: {avg_metrics['dynamic_safety_loss']:.3g}, "
            f"Target Sep: {avg_metrics['target_separation_loss']:.3g} "
        )
        self._write_metrics("Eval", avg_metrics, epoch)
        self.progress_log.remove_task(one_epoch_progress)

    def forward_and_compute_loss(self, image, pos, rot, obs_b, target_w, target_visible, map_id):
        image, pos, rot, obs_b, target_w, target_visible, map_id = [
            x.to(self.device) for x in [image, pos, rot, obs_b, target_w, target_visible, map_id]
        ]
        batch_size = image.shape[0]

        # 1. pre-process
        goal_w, start_vel_w, start_acc_w = state_body2world(
            pos, rot, obs_b[:, 6:9], obs_b[:, 0:3], obs_b[:, 3:6]
        )
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

        # 2. forward propagation
        endstate, score = self.policy.inference(image, obs_b)

        # 3. post-process [B, V, H, 9] -> [B*V*H, 9]
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(batch_size * self.traj_num, 9)
        score_flat = score.reshape(batch_size * self.traj_num)

        pos_expanded = pos.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3]
        rot_expanded = rot.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3, 3]
        start_state_w = start_state_w.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3, 3]
        goal_w = goal_w.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3]
        # [B*V*H, 3] [B*V*H, 3] [B*V*H, 3]
        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded, rot_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9]
        )
        # [B*V*H, 3, 3]: [px, py, pz; vx, vy, vz; ax, ay, az]
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

        loss_components = self.yopo_loss(
            start_state_w, end_state_w, goal_w, map_id, target_w, target_visible
        )
        target_w_expanded = target_w.repeat_interleave(self.traj_num, dim=0)
        target_visible_expanded = target_visible.repeat_interleave(self.traj_num, dim=0)
        target_sep_raw_cost, target_sep_min_distance = self.compute_target_separation_cost(
            start_state_w, end_state_w, target_w_expanded, target_visible_expanded
        )
        target_sep_cost = self.target_separation_weight * target_sep_raw_cost

        smooth_cost = loss_components["smoothness"]
        static_safety_cost = loss_components["static_safety"]
        dynamic_safety_cost = loss_components["dynamic_safety"]
        safety_cost = static_safety_cost + dynamic_safety_cost
        goal_cost = loss_components["goal"]
        acc_cost = loss_components["acceleration"]
        trajectory_cost = smooth_cost + static_safety_cost + dynamic_safety_cost + goal_cost + acc_cost + target_sep_cost
        trajectory_loss = trajectory_cost.mean()

        score_label = trajectory_cost.clone().detach()
        score_loss = F.smooth_l1_loss(score_flat, score_label)

        static_min_distance = loss_components["static_min_distance"]
        dynamic_min_distance = loss_components["dynamic_min_distance"]
        dynamic_finite = torch.isfinite(dynamic_min_distance)
        dynamic_collision = (dynamic_min_distance < 0.0) & dynamic_finite
        dynamic_collision_rate, dynamic_collision_weight = self._finite_rate(dynamic_collision, dynamic_finite)
        target_sep_finite = torch.isfinite(target_sep_min_distance)
        target_sep_violation = (target_sep_raw_cost > 0.0) & target_sep_finite
        target_sep_violation_rate, target_sep_violation_weight = self._finite_rate(
            target_sep_violation, target_sep_finite
        )
        return {
            "trajectory_loss": trajectory_loss,
            "score_loss": score_loss,
            "smooth_loss": smooth_cost.mean(),
            "safety_loss": safety_cost.mean(),
            "static_safety_loss": static_safety_cost.mean(),
            "dynamic_safety_loss": dynamic_safety_cost.mean(),
            "goal_loss": goal_cost.mean(),
            "acceleration_loss": acc_cost.mean(),
            "target_separation_loss": target_sep_cost.mean(),
            "raw_smooth_cost": loss_components["raw_smoothness"].mean(),
            "raw_static_safety_cost": loss_components["raw_static_safety"].mean(),
            "raw_dynamic_safety_cost": loss_components["raw_dynamic_safety"].mean(),
            "raw_goal_cost": loss_components["raw_goal"].mean(),
            "raw_acceleration_cost": loss_components["raw_acceleration"].mean(),
            "raw_target_separation_cost": target_sep_raw_cost.mean(),
            "score_label_mean": score_label.mean(),
            "score_pred_mean": score_flat.mean(),
            "score_abs_error": (score_flat - score_label).abs().mean(),
            "visible_target_fraction": target_visible.float().mean(),
            "visible_target_sample_fraction": (target_visible > 0.5).any(dim=1).float().mean(),
            "static_min_distance": self._finite_mean(static_min_distance),
            "dynamic_min_distance": self._finite_mean(dynamic_min_distance),
            "target_separation_min_distance": self._finite_mean(target_sep_min_distance),
            "static_collision_rate": (static_min_distance < 0.0).float().mean(),
            "dynamic_collision_rate": dynamic_collision_rate,
            "dynamic_collision_rate_weight": dynamic_collision_weight,
            "dynamic_collision_rate_overall": dynamic_collision.float().mean(),
            "target_separation_violation_rate": target_sep_violation_rate,
            "target_separation_violation_rate_weight": target_sep_violation_weight,
            "target_separation_violation_rate_overall": target_sep_violation.float().mean(),
        }

    def compute_target_separation_cost(self, start_state_w, end_state_w, target_w, target_visible):
        batch_size = end_state_w.shape[0]
        Df = start_state_w.permute(0, 2, 1)
        Dp = end_state_w.permute(0, 2, 1)
        L = self.yopo_loss._L.unsqueeze(0).expand(batch_size, -1, -1)
        coeff = self.yopo_loss.safety_loss.get_coefficient_from_derivative(Dp, Df, L)

        dt = self.yopo_loss.sgm_time / float(self.target_separation_eval_points)
        t_list = torch.linspace(
            dt,
            self.yopo_loss.sgm_time,
            self.target_separation_eval_points,
            device=end_state_w.device,
            dtype=end_state_w.dtype,
        ).view(1, -1, 1).expand(batch_size, -1, -1)
        traj_pos_w = self.yopo_loss.safety_loss.get_position_from_coeff(coeff, t_list)

        visible = target_visible > 0.5
        distance = torch.linalg.norm(traj_pos_w[:, :, None, :] - target_w[:, None, :, :], dim=3)
        distance = distance.masked_fill(~visible[:, None, :], float("inf"))
        nearest_distance = distance.amin(dim=2)
        spacing_error = torch.relu(self.target_separation_distance - nearest_distance)
        worst_spacing_error = spacing_error.square().amax(dim=1)
        return worst_spacing_error, nearest_distance.amin(dim=1)

    @staticmethod
    def _finite_mean(value):
        finite = torch.isfinite(value)
        if finite.any():
            return value[finite].mean()
        return value.new_tensor(0.0)

    @staticmethod
    def _finite_rate(condition, finite):
        denominator = finite.float().sum()
        return condition.float().sum() / denominator.clamp_min(1.0), denominator.detach()

    def _append_metrics(self, metric_buffer, metrics, default_weight=1.0):
        for name, value in metrics.items():
            if name.endswith("_weight"):
                continue
            weight = metrics.get(f"{name}_weight", default_weight)
            if torch.is_tensor(value):
                value = value.detach()
                if value.numel() != 1:
                    value = value.mean()
                value = value.cpu().item()
            if torch.is_tensor(weight):
                weight = weight.detach().cpu().item()
            weight = float(weight)
            weighted_sum, weight_sum = metric_buffer.setdefault(name, [0.0, 0.0])
            if weight <= 0.0:
                continue
            metric_buffer[name] = [weighted_sum + float(value) * weight, weight_sum + weight]

    def _mean_metrics(self, metric_buffer):
        if not metric_buffer:
            raise RuntimeError("No metrics were collected. Check that the dataloader is not empty.")
        return {
            name: (weighted_sum / weight_sum if weight_sum > 0.0 else 0.0)
            for name, (weighted_sum, weight_sum) in metric_buffer.items()
        }

    def _write_metrics(self, split, metrics, step):
        loss_tags = {
            "total_loss": "TotalLoss",
            "weighted_trajectory_loss": "WeightedTrajLoss",
            "weighted_score_loss": "WeightedScoreLoss",
            "trajectory_loss": "TrajLoss",
            "score_loss": "ScoreLoss",
            "smooth_loss": "SmoothLoss",
            "safety_loss": "SafetyLoss",
            "static_safety_loss": "StaticSafetyLoss",
            "dynamic_safety_loss": "DynamicSafetyLoss",
            "goal_loss": "GoalLoss",
            "acceleration_loss": "AccelLoss",
            "target_separation_loss": "TargetSeparationLoss",
        }
        raw_tags = {
            "raw_smooth_cost": "SmoothCost",
            "raw_static_safety_cost": "StaticSafetyCost",
            "raw_dynamic_safety_cost": "DynamicSafetyCost",
            "raw_goal_cost": "GoalCost",
            "raw_acceleration_cost": "AccelCost",
            "raw_target_separation_cost": "TargetSeparationCost",
        }
        diagnostic_tags = {
            "score_label_mean": "ScoreLabelMean",
            "score_pred_mean": "ScorePredMean",
            "score_abs_error": "ScoreAbsError",
            "visible_target_fraction": "VisibleTargetFraction",
            "visible_target_sample_fraction": "VisibleTargetSampleFraction",
            "static_min_distance": "StaticMinDistance",
            "dynamic_min_distance": "DynamicMinDistance",
            "target_separation_min_distance": "TargetSeparationMinDistance",
            "static_collision_rate": "StaticCollisionRate",
            "dynamic_collision_rate": "DynamicCollisionRate",
            "dynamic_collision_rate_overall": "DynamicCollisionRateOverall",
            "target_separation_violation_rate": "TargetSeparationViolationRate",
            "target_separation_violation_rate_overall": "TargetSeparationViolationRateOverall",
        }
        for metric_name, tag_name in loss_tags.items():
            if metric_name in metrics:
                self.tensorboard_log.add_scalar(f"{split}/{tag_name}", metrics[metric_name], step)
        for metric_name, tag_name in raw_tags.items():
            if metric_name in metrics:
                self.tensorboard_log.add_scalar(f"{split}Raw/{tag_name}", metrics[metric_name], step)
        for metric_name, tag_name in diagnostic_tags.items():
            if metric_name in metrics:
                self.tensorboard_log.add_scalar(f"{split}Diagnostics/{tag_name}", metrics[metric_name], step)

    def _log_loss_weight_config(self):
        raw_weights = {
            "trajectory_loss_weight": self.trajectory_loss_weight,
            "score_loss_weight": self.score_loss_weight,
            "smoothness_weight": self.yopo_loss.raw_smoothness_weight,
            "acceleration_weight": self.yopo_loss.raw_acceleration_weight,
            "static_safety_weight": self.yopo_loss.static_safety_weight,
            "dynamic_safety_weight": self.yopo_loss.dynamic_safety_weight,
            "goal_weight": self.yopo_loss.goal_weight,
            "target_separation_weight": self.target_separation_weight,
            "guidance_perp_weight": float(cfg.get("guidance_perp_weight", 0.5)),
            "guidance_velocity_direction_weight": float(cfg.get("guidance_velocity_direction_weight", 0.0)),
        }
        effective_weights = {
            "smoothness_effective_weight": self.yopo_loss.smoothness_weight,
            "acceleration_effective_weight": self.yopo_loss.acceleration_weight,
        }
        for name, value in raw_weights.items():
            self.tensorboard_log.add_scalar(f"LossWeightsRaw/{name}", value, 0)
        for name, value in effective_weights.items():
            self.tensorboard_log.add_scalar(f"LossWeightsEffective/{name}", value, 0)

        lines = [
            f"config_path: `{cfg['config_path']}`",
            "",
            "| name | value |",
            "| --- | ---: |",
        ]
        for name, value in raw_weights.items():
            lines.append(f"| {name} | {value:.8g} |")
        for name, value in effective_weights.items():
            lines.append(f"| {name} | {value:.8g} |")
        self.tensorboard_log.add_text("Config/LossWeights", "\n".join(lines), 0)

    def save_model(self):
        if hasattr(self, "epoch_i"):
            self.progress_log.console.log("Saving model...")
            self.save_checkpoint(self.epoch_i + 1)
            self._unregister_exit_save()

    def save_checkpoint(self, epoch):
        policy_path = os.path.join(self.checkpoint_path, "epoch{}.pth".format(epoch))
        torch.save(self.policy.state_dict(), policy_path)
        self.tensorboard_log.flush()
        self.progress_log.console.log("Saved checkpoint to ", policy_path)

    def _finish_exit_save(self, save_interval):
        if not hasattr(self, "_exit_func"):
            return
        final_epoch = getattr(self, "epoch_i", -1) + 1
        saved_by_interval = save_interval is not None and final_epoch > 0 and final_epoch % save_interval == 0
        if saved_by_interval:
            self._unregister_exit_save()
        else:
            self.save_model()

    def _unregister_exit_save(self):
        if hasattr(self, "_exit_func"):
            try:
                atexit.unregister(self._exit_func)
            except ValueError:
                pass
            del self._exit_func

    def get_next_log_path(self, base_path):
        nums = [int(name.split("_")[1])
                for name in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, name)) and name.startswith("YOPO_") and name.split("_")[1].isdigit()]
        next_n = max(nums, default=-1) + 1
        next_path = os.path.join(base_path, f"YOPO_{next_n}")
        os.makedirs(next_path, exist_ok=False)
        print("record tensorboard logs and checkpoints to ", next_path)
        return next_path
