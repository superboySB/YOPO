import os
from ruamel.yaml import YAML


# Global Configuration Management
class Config:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        yaml = YAML()
        self._data = yaml.load(open(os.path.join(base_dir, "traj_opt.yaml"), 'r'))
        tracker_config = os.environ.get(
            "YOPO_TRACKER_CONFIG_PATH",
            "/workspace/YOPO/YOPO/config/single_traj_opt.yaml",
        )
        if os.path.exists(tracker_config):
            tracker_data = yaml.load(open(tracker_config, "r"))
            if tracker_data:
                if "vel_max_train" in tracker_data:
                    self._data["vel_max_train"] = float(tracker_data["vel_max_train"])
                if "acc_max_train" in tracker_data:
                    self._data["acc_max_train"] = float(tracker_data["acc_max_train"])
                if "target_velocity" in tracker_data:
                    self._data["velocity"] = float(tracker_data["target_velocity"])
                elif "velocity" in tracker_data:
                    self._data["velocity"] = float(tracker_data["velocity"])
                if "target_acc_max" in tracker_data:
                    ratio = float(self._data["velocity"]) / float(self._data["vel_max_train"])
                    ratio = max(ratio, 1e-3)
                    self._data["acc_max_train"] = float(tracker_data["target_acc_max"]) / (ratio * ratio)
        self._data["train"] = True
        self._data["goal_length"] = 2.0 * self._data['radio_range']
        self._data["sgm_time"] = 2 * self._data["radio_range"] / self._data["vel_max_train"]
        self._data["traj_num"] = self._data['horizon_num'] * self._data['vertical_num'] * self._data["radio_num"]

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


cfg = Config()
