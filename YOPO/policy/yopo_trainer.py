"""
Training Strategy
supervised learning, imitation learning, testing, rollout
"""
import os
import time
import atexit
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
            loss_weight=[],
            tensorboard_path=None,
            checkpoint_path=None,
            num_workers=4,
            save_on_exit=False,
    ):
        self.batch_size = batch_size
        self.max_grad_norm = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_weight = loss_weight
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
        self.objectness_weight = cfg["objectness_weight"]
        self.target_position_weight = cfg["target_position_weight"]
        self.objectness_positive_weight = cfg["objectness_positive_weight"]
        self.objectness_negative_weight = cfg["objectness_negative_weight"]

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
        traj_losses, score_losses, obj_losses, target_losses, smooth_losses, safety_losses, goal_losses, acc_losses, start_time = [], [], [], [], [], [], [], [], time.time()
        for step, batch in enumerate(self.train_dataloader):  # obs: camera/body frame
            self.optimizer.zero_grad()

            trajectory_loss, score_loss, objectness_loss, target_loss, smooth_cost, safety_cost, goal_cost, acc_cost = self.forward_and_compute_loss(*batch)

            loss = (
                self.loss_weight[0] * trajectory_loss
                + self.loss_weight[1] * score_loss
                + self.objectness_weight * objectness_loss
                + self.target_position_weight * target_loss
            )

            # Optimize the policy
            loss.backward()
            self.optimizer.step()

            traj_losses.append(self.loss_weight[0] * trajectory_loss.item())
            score_losses.append(self.loss_weight[1] * score_loss.item())
            obj_losses.append(self.objectness_weight * objectness_loss.item())
            target_losses.append(self.target_position_weight * target_loss.item())
            smooth_losses.append(self.loss_weight[0] * smooth_cost.item())
            safety_losses.append(self.loss_weight[0] * safety_cost.item())
            goal_losses.append(self.loss_weight[0] * goal_cost.item())
            acc_losses.append(self.loss_weight[0] * acc_cost.item())

            if step % inspect_interval == inspect_interval - 1:
                batch_fps = inspect_interval / (time.time() - start_time)
                self.progress_log.console.log(f"Epoch: {epoch}, Traj Loss: {np.mean(traj_losses):.3g}, "
                                              f"Score Loss: {np.mean(score_losses):.3g}, "
                                              f"Obj Loss: {np.mean(obj_losses):.3g}, "
                                              f"Target Loss: {np.mean(target_losses):.3g} "
                                              f"Batch FPS: {batch_fps:.3g}")
                self.tensorboard_log.add_scalar("Train/TrajLoss", np.mean(traj_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Train/ScoreLoss", np.mean(score_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Train/ObjectnessLoss", np.mean(obj_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Train/TargetLoss", np.mean(target_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/SmoothLoss", np.mean(smooth_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/SafetyLoss", np.mean(safety_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/GoalLoss", np.mean(goal_losses), epoch * len(self.train_dataloader) + step)
                self.tensorboard_log.add_scalar("Detail/AccelLoss", np.mean(acc_losses), epoch * len(self.train_dataloader) + step)
                traj_losses, score_losses, obj_losses, target_losses, smooth_losses, safety_losses, goal_losses, acc_losses, start_time = [], [], [], [], [], [], [], [], time.time()

            self.progress_log.update(one_epoch_progress, advance=1)
            self.progress_log.update(total_progress, advance=1 / len(self.train_dataloader))

        self.progress_log.remove_task(one_epoch_progress)

    @torch.inference_mode()
    def eval_one_epoch(self, epoch: int):
        one_epoch_progress = self.progress_log.add_task(f"Eval: {epoch}", total=len(self.val_dataloader))
        traj_losses, score_losses, obj_losses, target_losses = [], [], [], []
        for step, batch in enumerate(self.val_dataloader):  # obs: camera/body frame
            trajectory_loss, score_loss, objectness_loss, target_loss, _, _, _, _ = self.forward_and_compute_loss(*batch)

            traj_losses.append(self.loss_weight[0] * trajectory_loss.item())
            score_losses.append(self.loss_weight[1] * score_loss.item())
            obj_losses.append(self.objectness_weight * objectness_loss.item())
            target_losses.append(self.target_position_weight * target_loss.item())
            self.progress_log.update(one_epoch_progress, advance=1)

        self.progress_log.console.log(
            f"Eval: {epoch}, Traj Loss: {np.mean(traj_losses):.3g}, "
            f"Score Loss: {np.mean(score_losses):.3g}, Obj Loss: {np.mean(obj_losses):.3g}, "
            f"Target Loss: {np.mean(target_losses):.3g} "
        )
        self.tensorboard_log.add_scalar("Eval/TrajLoss", np.mean(traj_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/ScoreLoss", np.mean(score_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/ObjectnessLoss", np.mean(obj_losses), epoch)
        self.tensorboard_log.add_scalar("Eval/TargetLoss", np.mean(target_losses), epoch)
        self.progress_log.remove_task(one_epoch_progress)

    def forward_and_compute_loss(self, image, pos, rot, obs_b, target_w, target_vel_w, target_b, target_visible, target_uv, map_id):
        image, pos, rot, obs_b, target_w, target_b, target_visible, target_uv, map_id = [
            x.to(self.device) for x in [image, pos, rot, obs_b, target_w, target_b, target_visible, target_uv, map_id]
        ]
        batch_size = image.shape[0]

        # 1. pre-process
        start_vel_w = rotate_body2world(rot, obs_b[:, 0:3])
        start_acc_w = rotate_body2world(rot, obs_b[:, 3:6])
        start_state_w = torch.stack([pos, start_vel_w, start_acc_w], dim=1)

        # 2. forward propagation
        endstate, score, objectness, target_pred = self.policy.inference(image, obs_b)

        # 3. post-process [B, V, H, 9] -> [B*V*H, 9]
        endstate_flat = endstate.permute(0, 2, 3, 1).reshape(batch_size * self.traj_num, 9)
        score_flat = score.reshape(batch_size * self.traj_num)

        pos_expanded = pos.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3]
        rot_expanded = rot.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3, 3]
        start_state_w = start_state_w.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3, 3]
        goal_w = target_w.repeat_interleave(self.traj_num, dim=0)  # [B*V*H, 3]

        # [B*V*H, 3] [B*V*H, 3] [B*V*H, 3]
        end_pos_w, end_vel_w, end_acc_w = state_body2world(
            pos_expanded, rot_expanded,
            endstate_flat[:, 0:3],
            endstate_flat[:, 3:6],
            endstate_flat[:, 6:9]
        )
        # [B*V*H, 3, 3]: [px, py, pz; vx, vy, vz; ax, ay, az]
        end_state_w = torch.stack([end_pos_w, end_vel_w, end_acc_w], dim=1)

        smooth_cost, safety_cost, goal_cost, acc_cost = self.yopo_loss(
            start_state_w, end_state_w, goal_w, map_id
        )
        trajectory_loss = (smooth_cost + safety_cost + goal_cost + acc_cost).mean()

        score_label = (smooth_cost + safety_cost + goal_cost + acc_cost).clone().detach()
        score_loss = F.smooth_l1_loss(score_flat, score_label)

        objectness_label, objectness_weight = self.build_objectness_labels(target_visible, target_uv)
        objectness_loss_map = F.binary_cross_entropy_with_logits(objectness, objectness_label, reduction='none')
        objectness_loss = (objectness_loss_map * objectness_weight).sum() / objectness_weight.sum().clamp(min=1.0)

        target_loss = self.compute_target_loss(target_pred, target_b, objectness_label)
        return trajectory_loss, score_loss, objectness_loss, target_loss, smooth_cost.mean(), safety_cost.mean(), goal_cost.mean(), acc_cost.mean()

    def build_objectness_labels(self, target_visible, target_uv):
        B = target_visible.shape[0]
        V = cfg["vertical_num"]
        H = cfg["horizon_num"]
        labels = torch.zeros((B, V, H), dtype=torch.float32, device=self.device)
        weights = torch.full(
            (B, V, H),
            float(self.objectness_negative_weight),
            dtype=torch.float32,
            device=self.device,
        )
        ignore_radius = int(cfg["ignore_grid_radius"])

        for b in range(B):
            if target_visible[b] < 0.5:
                continue
            u = torch.clamp(target_uv[b, 0], 0, self.image_width - 1)
            v = torch.clamp(target_uv[b, 1], 0, self.image_height - 1)
            h_idx = int(torch.clamp((u / self.image_width * H).long(), 0, H - 1).item())
            v_idx = int(torch.clamp((v / self.image_height * V).long(), 0, V - 1).item())
            labels[b, v_idx, h_idx] = 1.0
            weights[b, v_idx, h_idx] = float(self.objectness_positive_weight)
            if ignore_radius > 0:
                for vv in range(max(0, v_idx - ignore_radius), min(V, v_idx + ignore_radius + 1)):
                    for hh in range(max(0, h_idx - ignore_radius), min(H, h_idx + ignore_radius + 1)):
                        if vv == v_idx and hh == h_idx:
                            continue
                        weights[b, vv, hh] = 0.0
        return labels, weights

    def compute_target_loss(self, target_pred, target_b, objectness_label):
        positive_mask = objectness_label > 0.5
        if positive_mask.sum() == 0:
            return target_pred.sum() * 0.0
        target_label = target_b[:, :, None, None].expand_as(target_pred)
        loss_map = F.smooth_l1_loss(target_pred, target_label, reduction='none').sum(dim=1)
        return loss_map[positive_mask].mean()

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
