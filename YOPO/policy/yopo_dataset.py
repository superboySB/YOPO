import os
import sys
import cv2
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg


class YOPODataset(Dataset):
    def __init__(self, mode='train', val_ratio=0.1):
        super(YOPODataset, self).__init__()
        self.mode = mode
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.vel_max = cfg["vel_max_train"]
        self.acc_max = cfg["acc_max_train"]
        self.vx_lognorm_mean = np.log(1 - cfg["vx_mean_unit"])
        self.vx_logmorm_sigma = np.log(cfg["vx_std_unit"])
        self.v_mean = np.array([cfg["vx_mean_unit"], cfg["vy_mean_unit"], cfg["vz_mean_unit"]])
        self.v_std = np.array([cfg["vx_std_unit"], cfg["vy_std_unit"], cfg["vz_std_unit"]])
        self.a_mean = np.array([cfg["ax_mean_unit"], cfg["ay_mean_unit"], cfg["az_mean_unit"]])
        self.a_std = np.array([cfg["ax_std_unit"], cfg["ay_std_unit"], cfg["az_std_unit"]])
        self.goal_length = cfg["goal_length"]
        self.goal_pitch_std = cfg["goal_pitch_std"]
        self.goal_yaw_std = cfg["goal_yaw_std"]
        if mode == 'train':
            self.print_data()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "../", cfg["dataset_path"])
        self.depth_list, self.mask_list, self.map_idx = [], [], []
        self.positions = np.empty((0, 3), dtype=np.float32)
        self.quaternions = np.empty((0, 4), dtype=np.float32)
        self.target_dynamic_max_count = int(cfg.get("target_dynamic_max_count", 3))
        self.targets = np.empty((0, self.target_dynamic_max_count, 4), dtype=np.float32)

        datafolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        datafolders.sort(key=lambda x: int(os.path.basename(x)))
        if mode == 'train':
            print("Datafolders:")
            for folder in datafolders:
                print("    ", folder)

        print("Loading", mode, "dataset")
        for data_idx, datafolder in enumerate(datafolders):
            depth_file_names = [
                os.path.join(datafolder, filename)
                for filename in os.listdir(datafolder)
                if filename.startswith("depth_") and os.path.splitext(filename)[1] == ".png"
            ]
            if len(depth_file_names) == 0:
                depth_file_names = [
                    os.path.join(datafolder, filename)
                    for filename in os.listdir(datafolder)
                    if filename.startswith("img_") and os.path.splitext(filename)[1] == ".png"
                ]
            depth_file_names.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split("_")[1]))
            mask_file_names = [
                name.replace("/depth_", "/mask_").replace("/img_", "/mask_")
                for name in depth_file_names
            ]

            states = np.loadtxt(os.path.join(data_dir, f"pose-{data_idx}.csv"), delimiter=',', skiprows=1).astype(np.float32)
            if states.ndim == 1:
                states = states[None, :]
            positions = states[:, 0:3]
            quaternions = states[:, 3:7]
            if len(depth_file_names) != positions.shape[0]:
                raise RuntimeError(
                    f"{datafolder} has {len(depth_file_names)} depth images but "
                    f"{positions.shape[0]} poses. Regenerate the dataset."
                )

            target_path = os.path.join(data_dir, f"target-{data_idx}.csv")
            if not os.path.exists(target_path):
                raise FileNotFoundError(
                    f"{target_path} is required for YOPOv2-Tracker training. Regenerate the dataset with dataset_generator."
                )
            target_rows = np.loadtxt(target_path, delimiter=',', skiprows=1).astype(np.float32)
            if target_rows.ndim == 1:
                target_rows = target_rows[None, :]
            expected_target_cols = 1 + self.target_dynamic_max_count * 7
            if target_rows.shape[1] != expected_target_cols:
                raise RuntimeError(
                    f"{target_path} has {target_rows.shape[1]} columns; expected {expected_target_cols}. "
                    "Regenerate the dataset."
                )
            if target_rows.shape[0] != positions.shape[0]:
                raise RuntimeError(
                    f"{target_path} has {target_rows.shape[0]} rows but {positions.shape[0]} poses. "
                    "Regenerate the dataset."
                )
            targets = self._parse_targets(target_rows)

            split = train_test_split(
                depth_file_names, mask_file_names, positions, quaternions, targets,
                test_size=val_ratio, random_state=0
            )
            depth_train, depth_val, mask_train, mask_val, positions_train, positions_val, quaternions_train, quaternions_val, targets_train, targets_val = split

            if mode == 'train':
                self.depth_list.extend(depth_train)
                self.mask_list.extend(mask_train)
                self.positions = np.vstack((self.positions, positions_train.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_train.astype(np.float32)))
                self.targets = np.vstack((self.targets, targets_train.astype(np.float32)))
                self.map_idx.extend([data_idx] * len(depth_train))
            elif mode == 'valid':
                self.depth_list.extend(depth_val)
                self.mask_list.extend(mask_val)
                self.positions = np.vstack((self.positions, positions_val.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_val.astype(np.float32)))
                self.targets = np.vstack((self.targets, targets_val.astype(np.float32)))
                self.map_idx.extend([data_idx] * len(depth_val))
            else:
                raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")

        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Depth+Mask':<12} | Count: {len(self.depth_list):<3} | Shape: 2,{self.height},{self.width}")
        print(f"{'Positions':<12} | Count: {self.positions.shape[0]:<3} | Shape: {self.positions.shape[1]}")
        print(f"{'Quaternions':<12} | Count: {self.quaternions.shape[0]:<3} | Shape: {self.quaternions.shape[1]}")
        print(f"{'Targets':<12} | Count: {self.targets.shape[0]:<3} | Shape: {self.targets.shape[1:]}")
        print("==================================================")

    def __len__(self):
        return len(self.depth_list)

    def __getitem__(self, item):
        depth = cv2.imread(self.depth_list[item], -1).astype(np.float32)
        depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST) / 65535.0

        q_wxyz = self.quaternions[item, :]
        R_WC = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        rot_wc = R_WC.as_matrix().astype(np.float32)

        vel_c, acc_c = self._get_random_state()
        goal_c = self._get_random_goal()
        obs = np.hstack((vel_c, acc_c, goal_c)).astype(np.float32)

        target_row = self.targets[item]
        target_w = target_row[:, 0:3].astype(np.float32)
        target_visible = target_row[:, 3].astype(np.float32)

        mask = self._load_mask(self.mask_list[item])
        image = np.concatenate((depth[None, :, :], mask[None, :, :]), axis=0).astype(np.float32)

        return (
            image,
            self.positions[item],
            rot_wc,
            obs,
            target_w,
            target_visible,
            self.map_idx[item],
        )

    def _parse_targets(self, target_rows):
        targets = np.zeros(
            (target_rows.shape[0], self.target_dynamic_max_count, 4),
            dtype=np.float32,
        )
        counts = np.clip(target_rows[:, 0].astype(np.int32), 0, self.target_dynamic_max_count)
        for target_i in range(self.target_dynamic_max_count):
            base = 1 + target_i * 7
            targets[:, target_i, 0:3] = target_rows[:, base:base + 3]
            targets[:, target_i, 3] = (target_i < counts).astype(np.float32)
        return targets

    def _load_mask(self, mask_path):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"{mask_path} is required. Regenerate the dataset.")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read target mask {mask_path}.")
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32) / 255.0
        return mask.astype(np.float32)

    def _get_random_state(self):
        while True:
            vel = self.vel_max * (self.v_mean + self.v_std * np.random.randn(3))
            right_skewed_vx = -1
            while right_skewed_vx < 0:
                right_skewed_vx = self.vel_max * np.random.lognormal(
                    mean=self.vx_lognorm_mean,
                    sigma=self.vx_logmorm_sigma,
                    size=None,
                )
                right_skewed_vx = -right_skewed_vx + 1.2 * self.vel_max
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.vel_max:
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.randn(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_max:
                break
        return vel.astype(np.float32), acc.astype(np.float32)

    def _get_random_goal(self):
        goal_pitch_angle = np.radians(np.random.normal(0.0, self.goal_pitch_std))
        goal_yaw_angle = np.radians(np.random.normal(0.0, self.goal_yaw_std))
        goal_dir = np.array([
            np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
            np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle),
            np.sin(goal_pitch_angle),
        ], dtype=np.float32)

        random_near = np.random.rand()
        if random_near < 0.1:
            goal_dir = random_near * 10.0 * goal_dir
        return (self.goal_length * goal_dir).astype(np.float32)

    def print_data(self):
        import scipy.stats as stats
        p5 = self.vel_max * np.exp(stats.norm.ppf(0.05, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))
        p95 = self.vel_max * np.exp(stats.norm.ppf(0.95, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))

        v_lower = self.vel_max * (self.v_mean - 2 * self.v_std)
        v_upper = self.vel_max * (self.v_mean + 2 * self.v_std)
        v_lower[0] = max(-p95 + 1.2 * self.vel_max, 0)
        v_upper[0] = -p5 + 1.2 * self.vel_max
        a_lower = self.acc_max * (self.a_mean - 2 * self.a_std)
        a_upper = self.acc_max * (self.a_mean + 2 * self.a_std)

        print("----------------- Sampling State --------------------")
        print("| X-Y-Z | Vel 95% Range(m/s)  | Acc 95% Range(m/s2) |")
        print("|-------|---------------------|---------------------|")
        for i in range(3):
            print(f"|  {i:^4} | {v_lower[i]:^9.1f}~{v_upper[i]:^9.1f} |"
                  f" {a_lower[i]:^9.1f}~{a_upper[i]:^9.1f} |")
        print("-----------------------------------------------------")
        print(f"| Goal Pitch 90% (deg)        | {-self.goal_pitch_std * 2:^9.1f}~{self.goal_pitch_std * 2:^9.1f} |")
        print(f"| Goal Yaw   90% (deg)        | {-self.goal_yaw_std * 2:^9.1f}~{self.goal_yaw_std * 2:^9.1f} |")
        print("-----------------------------------------------------")

    def plot_sample_distribution(self):
        import matplotlib.pyplot as plt
        goals = np.array([self._get_random_goal() for _ in range(10000)])
        states = np.array([self._get_random_state() for _ in range(10000)])
        vels = np.stack([s[0] for s in states])
        accs = np.stack([s[1] for s in states])
        fig, axs = plt.subplots(3, 3, figsize=(15, 10))
        yaw = np.degrees(np.arctan2(goals[:, 1], goals[:, 0]))
        pitch = np.degrees(np.arctan2(goals[:, 2], np.linalg.norm(goals[:, 0:2], axis=1)))
        axs[0, 0].hist(yaw, bins=100)
        axs[0, 0].set_title("Goal Yaw")
        axs[0, 0].grid(True)
        axs[0, 1].hist(pitch, bins=100)
        axs[0, 1].set_title("Goal Pitch")
        axs[0, 1].grid(True)
        axs[0, 2].scatter(yaw, pitch, s=2, alpha=0.3)
        axs[0, 2].set_title("Goal Direction")
        axs[0, 2].grid(True)
        for i, name in enumerate(['Vx', 'Vy', 'Vz']):
            axs[1, i].hist(vels[:, i], bins=100)
            axs[1, i].set_title(f"Velocity {name}")
            axs[1, i].grid(True)
        for i, name in enumerate(['Ax', 'Ay', 'Az']):
            axs[2, i].hist(accs[:, i], bins=100)
            axs[2, i].set_title(f"Acceleration {name}")
            axs[2, i].grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    dataset = YOPODataset()
    dataset.plot_sample_distribution()

    max_workers = os.cpu_count()
    print(f"\ncpu_count = {max_workers}")
    results = []
    for nw in range(0, max_workers + 1):
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=nw)
        start = time.time()
        for i, _ in enumerate(data_loader):
            if i > 50:
                break
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start
        results.append((nw, elapsed))
        print(f"num_workers={nw}: {elapsed:.3f}s")

    best = min(results, key=lambda x: x[1])
    print(f"\nbest num_workers = {best[0]}, elapsed={best[1]:.3f}s")
