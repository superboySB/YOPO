import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg


class YOPOOmniDataset(Dataset):
    _DATA_CACHE = {}

    def __init__(self, mode="train", val_ratio=0.1, pose_level=False):
        super().__init__()
        self.pose_level = pose_level
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.guide_points = int(cfg["omni_guide_points"])
        self.vel_max = float(cfg["vel_max_train"])
        self.acc_max = float(cfg["acc_max_train"])
        self.vdes_min = float(cfg["omni_vdes_speed_min"])
        self.vdes_max = float(cfg["omni_vdes_speed_max"])
        self.vel_noise_std = float(cfg["omni_vel_noise_std"])
        self.acc_noise_std = float(cfg["omni_acc_noise_std"])
        self.view_names = ["front", "left", "right", "back"]
        self.depth_cache_size = int(cfg["omni_depth_cache_size"])
        self.depth_cache = OrderedDict()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.abspath(os.path.join(base_dir, "../", cfg["dataset_path"]))
        cache = self._load_or_get_cache(self.data_dir)
        self.arrays = cache["arrays"]
        self.guides = cache["guides"]

        group_key = self.arrays["map_id"].astype(np.int64) * 1_000_000_000 + self.arrays["pose_id"].astype(np.int64)
        unique_keys, group_starts, group_counts = np.unique(group_key, return_index=True, return_counts=True)
        train_keys, valid_keys = train_test_split(unique_keys, test_size=val_ratio, random_state=0)

        if mode == "train":
            selected_keys = train_keys
            self.shuffle_each_epoch = True
        elif mode == "valid":
            selected_keys = np.sort(valid_keys)
            self.shuffle_each_epoch = False
        else:
            raise ValueError(f"Invalid mode {mode}. Choose from 'train' or 'valid'.")

        self.group_keys = np.asarray(selected_keys, dtype=np.int64)
        self.start_by_key = dict(zip(unique_keys.tolist(), group_starts.tolist()))
        self.count_by_key = dict(zip(unique_keys.tolist(), group_counts.tolist()))
        self.direction_num = self._infer_direction_num()
        self.shuffle_groups(seed=0)

        unit_name = "Poses" if self.pose_level else "Samples"
        unit_count = len(self.indices)
        print(f"=============== YOPO-Omni {mode.capitalize()} Data Summary ===============")
        print(f"{unit_name:<12} | Count: {unit_count:<6} | Views: 4 | Shape: {self.width},{self.height}")
        if self.pose_level:
            print(f"{'Directions':<12} | Per pose: {self.direction_num:<3} | Effective samples: {unit_count * self.direction_num}")
        print(f"{'Guides':<12} | Points/sample: {self.guide_points:<3} | Depth cache: {self.depth_cache_size}")
        print("==================================================")

    @classmethod
    def _load_or_get_cache(cls, data_dir):
        if data_dir not in cls._DATA_CACHE:
            cls._DATA_CACHE[data_dir] = cls._load_dataset_arrays(data_dir)
        return cls._DATA_CACHE[data_dir]

    @classmethod
    def _load_dataset_arrays(cls, data_dir):
        map_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir() and os.path.basename(f.path).isdigit()]
        map_ids = sorted(int(os.path.basename(path)) for path in map_dirs)
        if not map_ids:
            raise FileNotFoundError(f"No map folders found in YOPO-Omni dataset: {data_dir}")

        arrays = {
            "sample_id": [],
            "pose_id": [],
            "dir_idx": [],
            "map_id": [],
            "pos": [],
            "quat_wxyz": [],
            "rot_wb": [],
            "vdes_unit": [],
            "goal_w": [],
            "guide_offset": [],
            "guide_len": [],
            "guide_mask": [],
            "selected_topology": [],
        }
        guides = {}

        for map_id in map_ids:
            sample_path = os.path.join(data_dir, f"samples-{map_id}.csv")
            guide_path = os.path.join(data_dir, f"guides-{map_id}.csv")
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"Missing YOPO-Omni label file: {sample_path}")
            if not os.path.exists(guide_path):
                raise FileNotFoundError(f"Missing YOPO-Omni guide file: {guide_path}")

            sample_data = np.loadtxt(sample_path, delimiter=",", skiprows=1, dtype=np.float32)
            if sample_data.ndim == 1:
                sample_data = sample_data[None, :]

            count = sample_data.shape[0]
            arrays["sample_id"].append(sample_data[:, 0].astype(np.int32))
            arrays["pose_id"].append(sample_data[:, 1].astype(np.int32))
            arrays["dir_idx"].append(sample_data[:, 2].astype(np.int16))
            arrays["map_id"].append(np.full(count, map_id, dtype=np.int16))
            arrays["pos"].append(sample_data[:, 3:6].astype(np.float32))
            quat = sample_data[:, 6:10].astype(np.float32)
            arrays["quat_wxyz"].append(quat)
            arrays["rot_wb"].append(cls._quat_wxyz_to_matrix(quat))
            arrays["vdes_unit"].append(sample_data[:, 10:13].astype(np.float32))
            arrays["goal_w"].append(sample_data[:, 13:16].astype(np.float32))
            arrays["guide_offset"].append(sample_data[:, 16].astype(np.int32))
            arrays["guide_len"].append(sample_data[:, 17].astype(np.int16))
            arrays["guide_mask"].append(sample_data[:, 18].astype(np.float32))
            arrays["selected_topology"].append(sample_data[:, 20].astype(np.int64))

            guides[map_id] = cls._load_guide_points(guide_path)

        for name, parts in arrays.items():
            arrays[name] = np.concatenate(parts, axis=0)
        arrays.pop("quat_wxyz")
        return {"arrays": arrays, "guides": guides}

    @staticmethod
    def _load_guide_points(guide_path):
        if os.path.getsize(guide_path) == 0:
            return np.empty((0, 3), dtype=np.float32)
        try:
            points = np.loadtxt(guide_path, delimiter=",", skiprows=1, usecols=(2, 3, 4), dtype=np.float32)
        except ValueError:
            return np.empty((0, 3), dtype=np.float32)
        if points.ndim == 1:
            points = points[None, :]
        return points.astype(np.float32, copy=False)

    @staticmethod
    def _quat_wxyz_to_matrix(q):
        q = q.astype(np.float32, copy=False)
        norm = np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-8)
        q = q / norm
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
        rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        rot[:, 0, 1] = 2.0 * (x * y - z * w)
        rot[:, 0, 2] = 2.0 * (x * z + y * w)
        rot[:, 1, 0] = 2.0 * (x * y + z * w)
        rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        rot[:, 1, 2] = 2.0 * (y * z - x * w)
        rot[:, 2, 0] = 2.0 * (x * z - y * w)
        rot[:, 2, 1] = 2.0 * (y * z + x * w)
        rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        return rot

    def __len__(self):
        return len(self.indices)

    def shuffle_groups(self, seed=None):
        keys = self.group_keys
        if self.shuffle_each_epoch:
            rng = np.random.default_rng(seed)
            keys = rng.permutation(keys)
        if self.pose_level:
            self.indices = np.asarray(keys, dtype=np.int64)
        else:
            self.indices = np.concatenate(
                [np.arange(self.start_by_key[int(key)],
                           self.start_by_key[int(key)] + self.count_by_key[int(key)],
                           dtype=np.int64)
                 for key in keys]
            )

    def __getitem__(self, item):
        if self.pose_level:
            return self._get_pose_item(item)
        idx = int(self.indices[item])
        map_id = int(self.arrays["map_id"][idx])
        pose_id = int(self.arrays["pose_id"][idx])

        depth = self._load_depth(map_id, pose_id)
        pos = self.arrays["pos"][idx]
        rot_wb = self.arrays["rot_wb"][idx]

        speed = np.random.uniform(self.vdes_min, self.vdes_max)
        vdes_b = speed * self.arrays["vdes_unit"][idx]
        vel_b = np.clip(vdes_b + np.random.randn(3).astype(np.float32) * self.vel_noise_std,
                        -self.vel_max, self.vel_max)
        acc_b = np.clip(np.random.randn(3).astype(np.float32) * self.acc_noise_std,
                        -self.acc_max, self.acc_max)
        state_b = np.concatenate((vel_b, acc_b, vdes_b)).astype(np.float32)

        guide_mask = np.float32(self.arrays["guide_mask"][idx])
        guide = self._get_guide(map_id, idx)
        selected_topology = np.int64(self.arrays["selected_topology"][idx])
        goal_w = self.arrays["goal_w"][idx]

        return (
            depth,
            pos,
            rot_wb,
            state_b,
            goal_w,
            guide,
            guide_mask,
            selected_topology,
            np.int64(map_id),
        )

    def _infer_direction_num(self):
        counts = np.asarray([self.count_by_key[int(key)] for key in self.group_keys], dtype=np.int64)
        if counts.size == 0:
            raise ValueError("YOPO-Omni dataset split is empty.")
        direction_num = int(counts[0])
        if self.pose_level and np.any(counts != direction_num):
            unique_counts = np.unique(counts).tolist()
            raise ValueError(f"Pose-level YOPO-Omni dataset requires fixed directions per pose, got {unique_counts}.")
        return direction_num

    def _get_pose_item(self, item):
        key = int(self.indices[item])
        start = self.start_by_key[key]
        count = self.count_by_key[key]
        idxs = np.arange(start, start + count, dtype=np.int64)

        map_id = int(self.arrays["map_id"][idxs[0]])
        pose_id = int(self.arrays["pose_id"][idxs[0]])

        depth = self._load_depth(map_id, pose_id)
        pos = self.arrays["pos"][idxs[0]]
        rot_wb = self.arrays["rot_wb"][idxs[0]]

        speed = np.random.uniform(self.vdes_min, self.vdes_max, size=(count, 1)).astype(np.float32)
        vdes_b = speed * self.arrays["vdes_unit"][idxs]
        vel_b = np.clip(vdes_b + np.random.randn(count, 3).astype(np.float32) * self.vel_noise_std,
                        -self.vel_max, self.vel_max)
        acc_b = np.clip(np.random.randn(count, 3).astype(np.float32) * self.acc_noise_std,
                        -self.acc_max, self.acc_max)
        state_b = np.concatenate((vel_b, acc_b, vdes_b), axis=1).astype(np.float32)

        guide = np.stack([self._get_guide(map_id, int(idx)) for idx in idxs], axis=0)
        guide_mask = self.arrays["guide_mask"][idxs].astype(np.float32, copy=False)
        selected_topology = self.arrays["selected_topology"][idxs].astype(np.int64, copy=False)
        goal_w = self.arrays["goal_w"][idxs]

        return (
            depth,
            pos,
            rot_wb,
            state_b,
            goal_w,
            guide,
            guide_mask,
            selected_topology,
            np.int64(map_id),
        )

    def _load_depth(self, map_id, pose_id):
        key = (map_id, pose_id)
        cached = self.depth_cache.get(key)
        if cached is not None:
            self.depth_cache.move_to_end(key)
            return cached

        depth = []
        datafolder = os.path.join(self.data_dir, str(map_id))
        for view in self.view_names:
            image_path = os.path.join(datafolder, f"img_{pose_id}_{view}.png")
            image = cv2.imread(image_path, -1)
            if image is None:
                raise FileNotFoundError(f"Missing depth image: {image_path}")
            if image.shape[0] != self.height or image.shape[1] != self.width:
                image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32) / 65535.0
            depth.append(image[None, ...])

        depth = np.stack(depth, axis=0).astype(np.float32)
        if self.depth_cache_size > 0:
            self.depth_cache[key] = depth
            if len(self.depth_cache) > self.depth_cache_size:
                self.depth_cache.popitem(last=False)
        return depth

    def _get_guide(self, map_id, idx):
        guide_len = int(self.arrays["guide_len"][idx])
        if guide_len <= 0:
            return np.zeros((self.guide_points, 3), dtype=np.float32)

        offset = int(self.arrays["guide_offset"][idx])
        points = self.guides[map_id][offset:offset + guide_len]
        return self._resample_guide(points, self.guide_points)

    @staticmethod
    def _resample_guide(points, target_count):
        if points.shape[0] == 0:
            return np.zeros((target_count, 3), dtype=np.float32)
        if points.shape[0] == 1:
            return np.repeat(points, target_count, axis=0).astype(np.float32)

        seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
        arc = np.concatenate(([0.0], np.cumsum(seg)))
        if arc[-1] < 1e-6:
            return np.repeat(points[:1], target_count, axis=0).astype(np.float32)

        target_arc = np.linspace(0.0, arc[-1], target_count)
        out = np.empty((target_count, 3), dtype=np.float32)
        for axis in range(3):
            out[:, axis] = np.interp(target_arc, arc, points[:, axis])
        return out


class YOPOOmniPoseDataset(YOPOOmniDataset):
    def __init__(self, mode="train", val_ratio=0.1):
        super().__init__(mode=mode, val_ratio=val_ratio, pose_level=True)
