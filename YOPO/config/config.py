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
        self._normalize_loss_weights()
        self._data["config_path"] = config_path
        self._data["train"] = True
        self._data["goal_length"] = 2.0 * self._data['radio_range']
        self._data["sgm_time"] = 2 * self._data["radio_range"] / self._data["vel_max_train"]
        self._data["traj_num"] = self._data['horizon_num'] * self._data['vertical_num'] * self._data["radio_num"]

    def _normalize_loss_weights(self):
        """
        Keep descriptive loss-weight names and the original YOPO shorthand aliases
        in sync. Descriptive names win when both forms are present.
        """
        alias_defaults = {
            "smoothness_weight": ("ws", 10.0),
            "acceleration_weight": ("wa", 1.0),
            "goal_weight": ("wg", 0.15),
        }
        for explicit_name, (legacy_name, default_value) in alias_defaults.items():
            if explicit_name not in self._data:
                self._data[explicit_name] = self._data.get(legacy_name, default_value)
            self._data[legacy_name] = self._data[explicit_name]

        legacy_safety_weight = self._data.get("safety_weight", self._data.get("wc", 1.0))
        if "static_safety_weight" not in self._data:
            self._data["static_safety_weight"] = legacy_safety_weight
        if "dynamic_safety_weight" not in self._data:
            self._data["dynamic_safety_weight"] = legacy_safety_weight
        self._data["wc"] = self._data["static_safety_weight"]

        self._data.setdefault("guidance_perp_weight", 0.5)
        self._data.setdefault("guidance_velocity_direction_weight", 0.0)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


cfg = Config()
