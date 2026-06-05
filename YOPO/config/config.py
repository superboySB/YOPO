import os
from ruamel.yaml import YAML


# Global Configuration Management
class Config:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.environ.get("YOPO_CONFIG_PATH", os.path.join(base_dir, "tracker_traj_opt.yaml"))
        if not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, config_path)
        with open(config_path, "r", encoding="utf-8") as config_file:
            self._data = YAML().load(config_file)
        self._data["config_path"] = config_path
        self._data["train"] = True
        self._data["goal_length"] = 2.0 * self._data['radio_range']
        self._data["sgm_time"] = 2 * self._data["radio_range"] / self._data["vel_max_train"]
        self._data["traj_num"] = self._data['horizon_num'] * self._data['vertical_num'] * self._data["radio_num"]

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


cfg = Config()
