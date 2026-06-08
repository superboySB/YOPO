#!/usr/bin/env bash
set -Eeuo pipefail

TRIAL=0
EPOCH=50
SESSION="yopo-crossover"
DETACH=0
STOP_ONLY=0

UAV_NUM=""
CROSSOVER_DISTANCE="20.0"
CROSSOVER_CENTER_X="0.0"
CROSSOVER_CENTER_Y="0.0"
ALTITUDE=""
ARRIVE_RADIUS=""
COLLISION_RADIUS=""
SPAWN_CLEAR_RADIUS=""
MAZE_TYPE=""
ENV_NAME=""

YOPO_CONFIG="/workspace/YOPO/YOPO/config/tracker_traj_opt.yaml"
SIMULATOR_CONFIG="/workspace/YOPO/Simulator/src/config/swarm_config.yaml"
WEIGHTS_ROOT="saved"
VISUALIZE=1
RVIZ=0
VIS_PLY_PER_UAV=""
PLANNER_START_DELAY_STEP=0
DIAGNOSTIC_DIR=""
DIAGNOSTIC_STRIDE=1

usage() {
  cat <<'EOF'
Usage:
  tools/launch_crossover.sh [--trial N] [--epoch N] [--uav-num N] [--detach]
  tools/launch_crossover.sh --stop

Options:
  --trial N                 tracker checkpoint trial under YOPO/saved (default: 0)
  --epoch N                 tracker checkpoint epoch id (default: 50)
  --uav-num N               number of UAVs, must be >= 2 (default: 5)
  --crossover-distance M    each UAV flies this distance toward/across the circle center (default: 20.0)
  --crossover-center-x X    crossover circle center x (default: 0.0)
  --crossover-center-y Y    crossover circle center y (default: 0.0)
  --altitude Z              crossover altitude (default: simulator config)
  --arrive-radius R         per-UAV arrival radius override (default: swarm_arrive_radius from YOPO config)
  --collision-radius R      UAV-UAV collision counter radius (default: simulator config)
  --spawn-clear-radius R    tree clearing around starts/goals (default: simulator config)
  --env NAME                environment alias: forest, pillar, cave, wall (default: simulator config)
  --maze-type N             simulator maze_type override, e.g. 5=forest, 2=pillar, 1=cave
  --yopo-config PATH        tracker config yaml
  --sim-config PATH         simulator config yaml
  --weights-root DIR        tracker checkpoint root under YOPO/ (default: saved)
  --visualize 0|1           publish all candidate/lattice trajectory point clouds (default: 1)
  --rviz 0|1                open RViz (default: 0)
  --vis_ply_per_uav 0|1     publish lightweight per-UAV LiDAR point clouds in RViz (default: simulator config render_lidar)
  --planner-start-delay-step S  seconds of startup stagger per UAV index (default: 0)
  --diagnostic-dir DIR      write per-UAV tracker score/action CSV logs (default: disabled)
  --diagnostic-stride N     write every Nth tracker frame when diagnostics are enabled (default: 1)
  --session NAME            tmux session name (default: yopo-crossover)
  --detach                  create session only, do not attach
  --stop                    stop this session and related processes
  -h, --help                show help
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: command not found: $1" >&2
    exit 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --trial) TRIAL="${2:-}"; shift 2 ;;
      --epoch) EPOCH="${2:-}"; shift 2 ;;
      --uav-num) UAV_NUM="${2:-}"; shift 2 ;;
      --formation)
        echo "Error: crossover does not use --formation. Use --uav-num; UAVs are placed uniformly on a circle." >&2
        exit 1
        ;;
      --formation-start-x)
        echo "Error: crossover does not use --formation-start-x. Use --crossover-center-x/--crossover-center-y if you need to move the circle." >&2
        exit 1
        ;;
      --forward-distance)
        echo "Error: crossover uses --crossover-distance instead of --forward-distance." >&2
        exit 1
        ;;
      --crossover-distance) CROSSOVER_DISTANCE="${2:-}"; shift 2 ;;
      --crossover-center-x) CROSSOVER_CENTER_X="${2:-}"; shift 2 ;;
      --crossover-center-y) CROSSOVER_CENTER_Y="${2:-}"; shift 2 ;;
      --center-x|--center-y)
        echo "Error: use --crossover-center-x/--crossover-center-y for crossover-specific center settings." >&2
        exit 1
        ;;
      --altitude) ALTITUDE="${2:-}"; shift 2 ;;
      --arrive-radius) ARRIVE_RADIUS="${2:-}"; shift 2 ;;
      --collision-radius) COLLISION_RADIUS="${2:-}"; shift 2 ;;
      --spawn-clear-radius) SPAWN_CLEAR_RADIUS="${2:-}"; shift 2 ;;
      --env|--environment) ENV_NAME="${2:-}"; shift 2 ;;
      --maze-type|--maze_type) MAZE_TYPE="${2:-}"; shift 2 ;;
      --yopo-config) YOPO_CONFIG="${2:-}"; shift 2 ;;
      --sim-config) SIMULATOR_CONFIG="${2:-}"; shift 2 ;;
      --weights-root) WEIGHTS_ROOT="${2:-}"; shift 2 ;;
      --visualize) VISUALIZE="${2:-}"; shift 2 ;;
      --rviz) RVIZ="${2:-}"; shift 2 ;;
      --vis_ply_per_uav|--vis-ply-per-uav) VIS_PLY_PER_UAV="${2:-}"; shift 2 ;;
      --visualize-pointcloud) VIS_PLY_PER_UAV="${2:-}"; shift 2 ;;
      --enable-rviz-goal|--enable-rviz-direction-goal)
        echo "Error: crossover does not support resetting goals from RViz 2D Nav Goal." >&2
        exit 1
        ;;
      --planner-start-delay-step) PLANNER_START_DELAY_STEP="${2:-}"; shift 2 ;;
      --diagnostic-dir) DIAGNOSTIC_DIR="${2:-}"; shift 2 ;;
      --diagnostic-stride) DIAGNOSTIC_STRIDE="${2:-}"; shift 2 ;;
      --session) SESSION="${2:-}"; shift 2 ;;
      --detach) DETACH=1; shift ;;
      --stop) STOP_ONLY=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
  done
}

stop_all() {
  local patterns=(
    '/opt/ros/noetic/bin/roscore'
    '/opt/ros/noetic/bin/rosmaster'
    'roslaunch so3_quadrotor_simulator simulator_attitude_control_uav.launch'
    'quadrotor_simulator_so3'
    'network_control_node'
    'rosrun sensor_simulator sensor_simulator_cuda'
    'sensor_simulator_cuda'
    'python3 test_yopo_ros_swarm_tracker.py'
    'rviz -d'
  )

  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  for pattern in "${patterns[@]}"; do
    pkill -f "${pattern}" >/dev/null 2>&1 || true
  done
  sleep 1
  for pattern in "${patterns[@]}"; do
    pkill -9 -f "${pattern}" >/dev/null 2>&1 || true
  done
  sleep 2
  rosparam delete /sensor_simulator_node >/dev/null 2>&1 || true
  timeout 5 bash -lc 'yes y | rosnode cleanup' >/dev/null 2>&1 || true
}

ensure_inside_container() {
  if [[ ! -f "/.dockerenv" ]]; then
    echo "Error: run this script inside the YOPO container." >&2
    exit 1
  fi
}

validate_binary_flag() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "Error: ${name} must be 0 or 1." >&2
    exit 1
  fi
}

validate_crossover() {
  local uav_num="$1"
  python3 - "$uav_num" "$YOPO_CONFIG" <<'PY'
import math
import sys

import yaml

try:
    uav_num = int(sys.argv[1])
except ValueError:
    raise SystemExit("Error: --uav-num must be an integer >= 2.")
if uav_num < 2:
    raise SystemExit("Error: crossover requires --uav-num >= 2; single-UAV crossover is not supported.")

with open(sys.argv[2], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

def read_positive_float(key):
    if key not in cfg:
        raise SystemExit(f"Error: {key} is required in YOPO config.")
    value = float(cfg[key])
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"Error: {key} in YOPO config must be a positive number.")
    return value

initial_spacing = read_positive_float("swarm_initial_spacing")
arrive_radius = read_positive_float("swarm_arrive_radius")
velocity = read_positive_float("velocity")
# Regular polygon radius that makes the adjacent chord length equal to spacing.
ring_radius = initial_spacing / (2.0 * math.sin(math.pi / uav_num))

print(f"{initial_spacing:.6f}\t{ring_radius:.6f}\t{arrive_radius:.6f}\t{velocity:.6f}")
PY
}

build_layout() {
  local uav_num="$1"
  local center_x="$2"
  local center_y="$3"
  local ring_radius="$4"
  local crossover_distance="$5"
  local altitude="$6"
  python3 - "$uav_num" "$center_x" "$center_y" "$ring_radius" "$crossover_distance" "$altitude" <<'PY'
import math
import sys

uav_num = int(sys.argv[1])
center_x = float(sys.argv[2])
center_y = float(sys.argv[3])
ring_radius = float(sys.argv[4])
crossover_distance = float(sys.argv[5])
altitude = float(sys.argv[6])

for idx in range(uav_num):
    angle = 2.0 * math.pi * idx / uav_num
    init_x = center_x + ring_radius * math.cos(angle)
    init_y = center_y + ring_radius * math.sin(angle)
    yaw = math.atan2(center_y - init_y, center_x - init_x)
    goal_x = init_x + crossover_distance * math.cos(yaw)
    goal_y = init_y + crossover_distance * math.sin(yaw)
    print(
        f"{idx}\t"
        f"{init_x:.3f}\t{init_y:.3f}\t{altitude:.3f}\t"
        f"{goal_x:.3f}\t{goal_y:.3f}\t{altitude:.3f}\t"
        f"{yaw:.6f}"
    )
PY
}

check_config_consistency() {
  python3 - "$YOPO_CONFIG" "$SIMULATOR_CONFIG" <<'PY'
import math
import sys

import yaml

yopo_path, sim_path = sys.argv[1:3]
with open(yopo_path, "r", encoding="utf-8") as f:
    yopo = yaml.safe_load(f) or {}
with open(sim_path, "r", encoding="utf-8") as f:
    sim = yaml.safe_load(f) or {}

camera = sim.get("camera") or {}
target = sim.get("target") or {}

checks = [
    ("image_width", yopo.get("image_width"), camera.get("image_width")),
    ("image_height", yopo.get("image_height"), camera.get("image_height")),
    ("camera_fx", yopo.get("camera_fx"), camera.get("fx")),
    ("camera_fy", yopo.get("camera_fy"), camera.get("fy")),
    ("camera_cx", yopo.get("camera_cx"), camera.get("cx")),
    ("camera_cy", yopo.get("camera_cy"), camera.get("cy")),
    ("target_dynamic_max_count", yopo.get("target_dynamic_max_count"), target.get("mask_dynamic_max_count")),
    ("target_mask_min_pixels", yopo.get("target_mask_min_pixels"), target.get("mask_min_visible_pixels")),
    ("swarm_initial_spacing", yopo.get("swarm_initial_spacing"), target.get("min_center_distance")),
]
for name, left, right in checks:
    if left is None or right is None:
        raise SystemExit(f"Error: missing shared config key for {name}.")
    if abs(float(left) - float(right)) > 1e-6:
        raise SystemExit(f"Error: shared config mismatch for {name}: YOPO={left}, simulator={right}.")

yopo_size = yopo.get("target_ellipsoid_size")
sim_size = target.get("ellipsoid_size")
if not isinstance(yopo_size, list) or not isinstance(sim_size, list) or len(yopo_size) != len(sim_size):
    raise SystemExit("Error: target ellipsoid size must be a list with matching length in both configs.")
for idx, (left, right) in enumerate(zip(yopo_size, sim_size)):
    if abs(float(left) - float(right)) > 1e-6:
        raise SystemExit(
            f"Error: shared config mismatch for target_ellipsoid_size[{idx}]: YOPO={left}, simulator={right}."
        )

for name in ("swarm_initial_spacing", "swarm_arrive_radius", "velocity"):
    value = float(yopo.get(name, float("nan")))
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"Error: YOPO config {name} must be a positive number.")
PY
}

read_simulator_defaults() {
  python3 - "$SIMULATOR_CONFIG" <<'PY'
import math
import sys

import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
swarm = config.get("swarm") or {}
camera = config.get("camera") or {}
namespace_prefix = swarm.get("namespace_prefix", "uav")
if namespace_prefix != "uav":
    raise SystemExit("Error: tools/launch_crossover.sh currently requires simulator swarm.namespace_prefix to be 'uav'.")

def number(name, value, positive=False, non_negative=False):
    if value is None:
        raise SystemExit(f"Error: simulator config {name} is required.")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"Error: simulator config {name} must be a number.")
    if not math.isfinite(parsed) or (positive and parsed <= 0.0) or (non_negative and parsed < 0.0):
        if positive:
            kind = "a positive"
        elif non_negative:
            kind = "a non-negative"
        else:
            kind = "a finite"
        raise SystemExit(f"Error: simulator config {name} must be {kind} number.")
    return parsed

def integer(name, value):
    if value is None:
        raise SystemExit(f"Error: simulator config {name} is required.")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"Error: simulator config {name} must be an integer.")
    if parsed <= 0:
        raise SystemExit(f"Error: simulator config {name} must be a positive integer.")
    return parsed

values = [
    str(integer("swarm.uav_num", swarm.get("uav_num", 5))),
    f"{number('swarm.altitude', swarm.get('altitude', 1.5), positive=True):.6f}",
    f"{number('swarm.collision_radius', swarm.get('collision_radius', 0.155), positive=True):.6f}",
    f"{number('swarm.spawn_clear_radius', swarm.get('spawn_clear_radius', 0.0), non_negative=True):.6f}",
    f"{number('depth_fps', config.get('depth_fps'), positive=True):.6f}",
    f"{number('camera.max_depth_dist', camera.get('max_depth_dist'), positive=True):.6f}",
    "1" if bool(config.get("render_lidar", False)) else "0",
    str(integer("maze_type", config.get("maze_type", 5))),
]
print("\t".join(values))
PY
}

resolve_maze_type() {
  local default_maze_type="$1"
  python3 - "$default_maze_type" "$MAZE_TYPE" "$ENV_NAME" <<'PY'
import sys

default_raw, maze_raw, env_raw = sys.argv[1:4]
labels = {
    1: "cave",
    2: "pillar",
    3: "maze",
    4: "maze3d",
    5: "forest",
    6: "room",
    7: "wall",
}
aliases = {
    "cave": 1,
    "caves": 1,
    "cavern": 1,
    "perlin": 1,
    "perlin3d": 1,
    "pillar": 2,
    "pillars": 2,
    "column": 2,
    "columns": 2,
    "forest": 5,
    "tree": 5,
    "trees": 5,
    "woods": 5,
    "wall": 7,
    "walls": 7,
    "maze": 3,
    "maze2d": 3,
    "maze3d": 4,
    "room": 6,
    "rooms": 6,
}

if maze_raw and env_raw:
    raise SystemExit("Error: use only one of --env or --maze-type.")

if env_raw:
    key = env_raw.strip().lower().replace("_", "").replace("-", "")
    if key not in aliases:
        raise SystemExit("Error: --env must be one of forest, pillar, cave, wall, maze, maze3d, room.")
    maze_type = aliases[key]
elif maze_raw:
    try:
        maze_type = int(maze_raw)
    except ValueError:
        raise SystemExit("Error: --maze-type must be an integer.")
else:
    maze_type = int(default_raw)

if maze_type not in labels:
    raise SystemExit("Error: --maze-type must be between 1 and 7.")

print(f"{maze_type}\t{labels[maze_type]}")
PY
}

build_runtime_simulator_config() {
  local maze_type="$1"
  local uav_num="$2"
  local center_x="$3"
  local center_y="$4"
  local ring_radius="$5"
  local crossover_distance="$6"
  local spawn_clear_radius="$7"
  python3 - "$SIMULATOR_CONFIG" "$maze_type" "$uav_num" "$center_x" "$center_y" "$ring_radius" "$crossover_distance" "$spawn_clear_radius" <<'PY'
import math
import os
import sys
import tempfile

import yaml

src_path = sys.argv[1]
maze_type = int(sys.argv[2])
uav_num = int(sys.argv[3])
center_x = float(sys.argv[4])
center_y = float(sys.argv[5])
ring_radius = float(sys.argv[6])
crossover_distance = float(sys.argv[7])
spawn_clear_radius = float(sys.argv[8])

with open(src_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

clear_positions = []
for idx in range(uav_num):
    angle = 2.0 * math.pi * idx / uav_num
    init_x = center_x + ring_radius * math.cos(angle)
    init_y = center_y + ring_radius * math.sin(angle)
    yaw = math.atan2(center_y - init_y, center_x - init_x)
    goal_x = init_x + crossover_distance * math.cos(yaw)
    goal_y = init_y + crossover_distance * math.sin(yaw)
    clear_positions.append([round(init_x, 3), round(init_y, 3)])
    clear_positions.append([round(goal_x, 3), round(goal_y, 3)])

config["maze_type"] = maze_type
max_abs_x = max(abs(position[0]) for position in clear_positions)
max_abs_y = max(abs(position[1]) for position in clear_positions)
map_margin = max(5.0, spawn_clear_radius + 2.0)
config["x_length"] = max(int(config.get("x_length", 0)), int(math.ceil(2.0 * (max_abs_x + map_margin))))
config["y_length"] = max(int(config.get("y_length", 0)), int(math.ceil(2.0 * (max_abs_y + map_margin))))
swarm = config.setdefault("swarm", {})
swarm["uav_num"] = uav_num
swarm["ring_radius"] = float(ring_radius)
swarm["forward_distance"] = float(crossover_distance)
swarm["clear_positions"] = clear_positions
swarm.pop("formation_rows", None)

fd, out_path = tempfile.mkstemp(prefix="yopo_crossover_config_", suffix=".yaml")
os.close(fd)
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

print(out_path)
PY
}

build_wait_lib() {
  cat <<'EOF'
wait_for_master() {
  local deadline=$((SECONDS + 60))
  until rosnode list >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[launch_crossover] timed out waiting for roscore." >&2
      exit 1
    fi
    sleep 1
  done
}

wait_for_topic() {
  local topic="$1"
  local deadline=$((SECONDS + 90))
  until timeout 2 rostopic echo -n 1 "$topic" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[launch_crossover] timed out waiting for topic ${topic}." >&2
      exit 1
    fi
    sleep 1
  done
}
EOF
}

build_rviz_config() {
  local uav_num="$1"
  local vis_ply_per_uav="$2"
  python3 - "/workspace/YOPO/YOPO/swarm_tracker.rviz" "${uav_num}" "${vis_ply_per_uav}" <<'PY'
import copy
import os
import sys
import tempfile
from pathlib import Path

import yaml

src_path = Path(sys.argv[1])
uav_num = int(sys.argv[2])
lidar_enabled = sys.argv[3] == "1"
config = yaml.safe_load(src_path.read_text(encoding="utf-8"))
displays = config.get("Visualization Manager", {}).get("Displays", [])

depth_template = next((d for d in displays if d.get("Class") == "rviz/Image" and (d.get("Name") == "Depth" or d.get("Image Topic") in ("/depth_image", "/uav0/depth_image"))), None)
mask_template = next((d for d in displays if d.get("Class") == "rviz/Image" and (d.get("Name") == "Target_Mask" or d.get("Image Topic") in ("/target_mask_image", "/uav0/target_mask_image"))), None)
traj_template = next((d for d in displays if d.get("Class") == "rviz/Group" and d.get("Name") == "Trajectory"), None)
drone_template = next((d for d in displays if d.get("Class") == "rviz/Marker" and d.get("Name") == "Drone"), None)
label_template = next((d for d in displays if d.get("Class") == "rviz/Marker" and d.get("Name") == "Status_Label"), None)
lidar_template = next((d for d in displays if d.get("Class") == "rviz/PointCloud2" and (d.get("Name") == "Lidar" or d.get("Topic") in ("/lidar_points", "/uav0/lidar_points"))), None)

palette = [
    "231; 76; 60", "52; 152; 219", "46; 204; 113", "241; 196; 15", "155; 89; 182",
    "230; 126; 34", "26; 188; 156", "149; 165; 166", "236; 240; 241", "52; 73; 94",
]

def clone_depth(idx):
    item = copy.deepcopy(depth_template)
    item["Image Topic"] = f"/uav{idx}/depth_image"
    item["Name"] = f"Depth_uav{idx}"
    item["Enabled"] = idx < 2
    item["Value"] = idx < 2
    return item

def clone_mask(idx):
    item = copy.deepcopy(mask_template)
    item["Image Topic"] = f"/uav{idx}/target_mask_image"
    item["Name"] = f"Mask_uav{idx}"
    item["Enabled"] = idx == 0
    item["Value"] = idx == 0
    return item

def clone_traj(idx):
    item = copy.deepcopy(traj_template)
    item["Name"] = f"Trajectory_uav{idx}"
    color = palette[idx % len(palette)]
    for child in item.get("Displays", []):
        name = child.get("Name")
        if name == "Best_Traj":
            child["Topic"] = f"/uav{idx}/yopo_tracker/best_traj_visual"
            child["Color"] = color
        elif name == "All_traj":
            child["Topic"] = f"/uav{idx}/yopo_tracker/trajs_visual"
            child["Color"] = color
        elif name == "Primitive_Traj":
            child["Topic"] = f"/uav{idx}/yopo_tracker/lattice_trajs_visual"
            child["Color"] = color
            child["Enabled"] = False
            child["Value"] = False
    return item

def clone_lidar(idx):
    item = copy.deepcopy(lidar_template)
    item["Name"] = f"Lidar_uav{idx}"
    item["Topic"] = f"/uav{idx}/lidar_points"
    item["Enabled"] = lidar_enabled
    item["Value"] = lidar_enabled
    return item

def clone_drone(idx):
    item = copy.deepcopy(drone_template)
    item["Marker Topic"] = f"/uav{idx}_simulator/uav"
    item["Name"] = f"Drone_uav{idx}"
    return item

def clone_label(idx):
    item = copy.deepcopy(label_template)
    item["Marker Topic"] = f"/uav{idx}/yopo_tracker/status_text"
    item["Name"] = f"Status_Label_uav{idx}"
    return item

new_displays = []
inserted_images = False
inserted_trajs = False
inserted_drones = False
inserted_labels = False
inserted_lidars = False

for item in displays:
    cls = item.get("Class")
    name = item.get("Name")
    if cls == "rviz/Image" and (name in ("Depth", "Target_Mask") or item.get("Image Topic") in ("/depth_image", "/target_mask_image", "/uav0/depth_image", "/uav0/target_mask_image")):
        if not inserted_images and depth_template and mask_template:
            for idx in range(uav_num):
                new_displays.append(clone_depth(idx))
            for idx in range(uav_num):
                new_displays.append(clone_mask(idx))
            inserted_images = True
        continue
    if cls == "rviz/Group" and name == "Trajectory":
        if not inserted_trajs and traj_template:
            for idx in range(uav_num):
                new_displays.append(clone_traj(idx))
            inserted_trajs = True
        continue
    if cls == "rviz/PointCloud2" and (name == "Lidar" or item.get("Topic") in ("/lidar_points", "/uav0/lidar_points")):
        if not inserted_lidars and lidar_template:
            for idx in range(uav_num):
                new_displays.append(clone_lidar(idx))
            inserted_lidars = True
        continue
    if cls == "rviz/Marker" and name in ("Drone", "Target_Drone"):
        if not inserted_drones and drone_template:
            for idx in range(uav_num):
                new_displays.append(clone_drone(idx))
            inserted_drones = True
        continue
    if cls == "rviz/Marker" and name == "Status_Label":
        if not inserted_labels and label_template:
            for idx in range(uav_num):
                new_displays.append(clone_label(idx))
            inserted_labels = True
        continue
    new_displays.append(copy.deepcopy(item))

config["Visualization Manager"]["Displays"] = new_displays
fd, out_path = tempfile.mkstemp(prefix="launch_crossover_", suffix=".rviz")
os.close(fd)
Path(out_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(out_path)
PY
}

main() {
  parse_args "$@"
  ensure_inside_container
  require_cmd tmux
  require_cmd roscore
  require_cmd roslaunch
  require_cmd rosrun
  require_cmd python3

  validate_binary_flag --visualize "${VISUALIZE}"
  validate_binary_flag --rviz "${RVIZ}"
  if [[ "${RVIZ}" == "1" ]]; then
    require_cmd rviz
  fi

  if [[ "${STOP_ONLY}" -eq 1 ]]; then
    stop_all
    echo "[launch_crossover] stopped session and related processes."
    exit 0
  fi

  if [[ ! -f "${YOPO_CONFIG}" ]]; then
    echo "Error: YOPO config not found: ${YOPO_CONFIG}" >&2
    exit 1
  fi
  if [[ ! -f "${SIMULATOR_CONFIG}" ]]; then
    echo "Error: simulator config not found: ${SIMULATOR_CONFIG}" >&2
    exit 1
  fi

  check_config_consistency

  local simulator_defaults
  local simulator_uav_num
  local simulator_altitude simulator_collision_radius simulator_spawn_clear_radius simulator_depth_fps simulator_max_depth_dist simulator_render_lidar simulator_maze_type
  simulator_defaults="$(read_simulator_defaults)"
  IFS=$'\t' read -r \
    simulator_uav_num \
    simulator_altitude \
    simulator_collision_radius \
    simulator_spawn_clear_radius \
    simulator_depth_fps \
    simulator_max_depth_dist \
    simulator_render_lidar \
    simulator_maze_type <<<"${simulator_defaults}"

  local maze_meta maze_type maze_label
  maze_meta="$(resolve_maze_type "${simulator_maze_type}")"
  IFS=$'\t' read -r maze_type maze_label <<<"${maze_meta}"

  local uav_num="${UAV_NUM:-5}"
  local crossover_distance="${CROSSOVER_DISTANCE}"
  local center_x="${CROSSOVER_CENTER_X}"
  local center_y="${CROSSOVER_CENTER_Y}"
  local altitude="${ALTITUDE:-${simulator_altitude}}"
  local collision_radius="${COLLISION_RADIUS:-${simulator_collision_radius}}"
  local spawn_clear_radius="${SPAWN_CLEAR_RADIUS:-${simulator_spawn_clear_radius}}"
  local vis_ply_per_uav="${VIS_PLY_PER_UAV:-${simulator_render_lidar}}"
  validate_binary_flag --vis_ply_per_uav "${vis_ply_per_uav}"

  local crossover_meta initial_spacing ring_radius config_arrive_radius yopo_velocity
  crossover_meta="$(validate_crossover "${uav_num}")"
  IFS=$'\t' read -r initial_spacing ring_radius config_arrive_radius yopo_velocity <<<"${crossover_meta}"
  local arrive_radius="${ARRIVE_RADIUS:-${config_arrive_radius}}"

  python3 - "${center_x}" "${center_y}" "${crossover_distance}" "${altitude}" "${collision_radius}" "${spawn_clear_radius}" "${arrive_radius}" "${PLANNER_START_DELAY_STEP}" <<'PY'
import math
import sys

checks = [
    ("--crossover-center-x", "finite"),
    ("--crossover-center-y", "finite"),
    ("--crossover-distance", "positive"),
    ("--altitude", "positive"),
    ("--collision-radius", "positive"),
    ("--spawn-clear-radius", "non-negative"),
    ("--arrive-radius", "positive"),
    ("--planner-start-delay-step", "non-negative"),
]
for (name, mode), raw in zip(checks, sys.argv[1:]):
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"Error: {name} must be a {mode} number.")
    if (
        not math.isfinite(value)
        or (mode == "positive" and value <= 0.0)
        or (mode == "non-negative" and value < 0.0)
    ):
        raise SystemExit(f"Error: {name} must be a {mode} number.")
PY

  local weights_root_abs
  if [[ "${WEIGHTS_ROOT}" = /* ]]; then
    weights_root_abs="${WEIGHTS_ROOT}"
  else
    weights_root_abs="/workspace/YOPO/YOPO/${WEIGHTS_ROOT}"
  fi
  local weight_path="${weights_root_abs}/YOPO_${TRIAL}/epoch${EPOCH}.pth"
  if [[ ! -f "${weight_path}" ]]; then
    echo "Error: tracker checkpoint not found: ${weight_path}" >&2
    exit 1
  fi

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Error: tmux session '${SESSION}' already exists." >&2
    echo "Use: tools/launch_crossover.sh --session ${SESSION} --stop" >&2
    exit 1
  fi

  local env_setup='source /opt/ros/noetic/setup.bash'
  local wait_lib
  wait_lib="$(build_wait_lib)"
  local runtime_simulator_config
  runtime_simulator_config="$(build_runtime_simulator_config "${maze_type}" "${uav_num}" "${center_x}" "${center_y}" "${ring_radius}" "${crossover_distance}" "${spawn_clear_radius}")"
  local yopo_env="export YOPO_CONFIG_PATH=${YOPO_CONFIG}; "
  mapfile -t layout_lines < <(
    build_layout \
      "${uav_num}" \
      "${center_x}" \
      "${center_y}" \
      "${ring_radius}" \
      "${crossover_distance}" \
      "${altitude}"
  )

  local wait_for_all_odom_topics=""
  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r idx _init_x _init_y _init_z _goal_x _goal_y _goal_z _init_yaw <<<"${line}"
    wait_for_all_odom_topics+="wait_for_topic /uav${idx}/sim/odom; "
  done

  tmux new-session -d -s "${SESSION}" -n roscore "bash -lc '${env_setup}; roscore'"

  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r idx init_x init_y init_z _goal_x _goal_y _goal_z init_yaw <<<"${line}"
    local uav_name="uav${idx}"
    local cmd_controller="${env_setup}; ${wait_lib}; wait_for_master; cd /workspace/YOPO/Controller; source devel/setup.bash; roslaunch so3_quadrotor_simulator simulator_attitude_control_uav.launch uav_name:=${uav_name} init_x:=${init_x} init_y:=${init_y} init_z:=${init_z} init_yaw:=${init_yaw} odom_topic:=/${uav_name}/sim/odom imu_topic:=/${uav_name}/sim/imu ctrl_topic:=/${uav_name}/so3_control/pos_cmd so3_cmd_topic:=/${uav_name}/so3_cmd force_disturbance_topic:=/${uav_name}/force_disturbance moment_disturbance_topic:=/${uav_name}/moment_disturbance simulator_node_name:=${uav_name}_simulator controller_node_name:=${uav_name}_controller"
    tmux new-window -t "${SESSION}:" -n "ctrl_${uav_name}" "bash -lc '${cmd_controller}'"
  done

  local render_lidar_param="false"
  if [[ "${vis_ply_per_uav}" == "1" ]]; then
    render_lidar_param="true"
  fi
  local cmd_simulator="${env_setup}; ${wait_lib}; wait_for_master; ${wait_for_all_odom_topics} rosparam delete /sensor_simulator_node >/dev/null 2>&1 || true; cd /workspace/YOPO/Simulator; source devel/setup.bash; rosrun sensor_simulator sensor_simulator_cuda _config_path:=${runtime_simulator_config} _swarm_enabled:=true _swarm_uav_num:=${uav_num} _swarm_namespace_prefix:=uav _swarm_altitude:=${altitude} _swarm_collision_radius:=${collision_radius} _swarm_spawn_clear_radius:=${spawn_clear_radius} _swarm_ring_radius:=${ring_radius} _swarm_forward_distance:=${crossover_distance} _render_lidar:=${render_lidar_param}"
  tmux new-window -t "${SESSION}:" -n simulator "bash -lc '${cmd_simulator}'"

  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r idx _init_x _init_y _init_z goal_x goal_y goal_z _init_yaw <<<"${line}"
    local uav_name="uav${idx}"
    local rviz_goal_args="--enable_rviz_goal=0 --reject_rviz_goal=0"
    if [[ "${idx}" == "0" ]]; then
      rviz_goal_args="--enable_rviz_goal=0 --reject_rviz_goal=1 --rviz_goal_topic=/move_base_simple/goal"
    fi
    local planner_delay
    planner_delay="$(python3 - "${idx}" "${PLANNER_START_DELAY_STEP}" <<'PY'
import sys
print(f"{int(sys.argv[1]) * float(sys.argv[2]):.2f}")
PY
)"
    local cmd_planner="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /${uav_name}/sim/odom; wait_for_topic /${uav_name}/depth_image; wait_for_topic /${uav_name}/target_mask_image; sleep ${planner_delay}; cd /workspace/YOPO/YOPO; ${yopo_env}python3 test_yopo_ros_swarm_tracker.py --trial=${TRIAL} --epoch=${EPOCH} --weights_root=${weights_root_abs} --agent_name=${uav_name} --node_name=yopo_tracker_${uav_name} --odom_topic=/${uav_name}/sim/odom --depth_topic=/${uav_name}/depth_image --target_mask_topic=/${uav_name}/target_mask_image --ctrl_topic=/${uav_name}/so3_control/pos_cmd --visual_prefix=/${uav_name}/yopo_tracker --status_prefix=/${uav_name}/yopo --goal_x=${goal_x} --goal_y=${goal_y} --goal_z=${goal_z} --arrive_radius=${arrive_radius} --max_depth_dist=${simulator_max_depth_dist} --depth_fps=${simulator_depth_fps} --visualize=${VISUALIZE} --diagnostic_dir=${DIAGNOSTIC_DIR} --diagnostic_stride=${DIAGNOSTIC_STRIDE} ${rviz_goal_args}"
    tmux new-window -t "${SESSION}:" -n "plan_${uav_name}" "bash -lc '${cmd_planner}'"
  done

  if [[ "${RVIZ}" == "1" ]]; then
    local rviz_config
    rviz_config="$(build_rviz_config "${uav_num}" "${vis_ply_per_uav}")"
    local rviz_wait_topics="wait_for_topic /uav0/depth_image; wait_for_topic /uav0/target_mask_image; "
    if [[ "${vis_ply_per_uav}" == "1" ]]; then
      rviz_wait_topics="wait_for_topic /uav0/lidar_points; ${rviz_wait_topics}"
    fi
    local cmd_rviz="${env_setup}; ${wait_lib}; wait_for_master; ${rviz_wait_topics}cd /workspace/YOPO/YOPO; rviz -d ${rviz_config}"
    tmux new-window -t "${SESSION}:" -n rviz "bash -lc '${cmd_rviz}'"
  fi

  tmux set-option -t "${SESSION}" remain-on-exit on
  tmux bind-key -T root C-c if-shell -F "#{==:#{session_name},${SESSION}}" "kill-session -t ${SESSION}" "send-keys C-c"
  tmux set-hook -t "${SESSION}" session-closed "unbind-key -T root C-c"

  echo "[launch_crossover] started tmux session='${SESSION}', env=${maze_label}, maze_type=${maze_type}, uav_num=${uav_num}, spacing=${initial_spacing}m, ring_radius=${ring_radius}m, crossover_distance=${crossover_distance}m, center=(${center_x},${center_y}), arrive_radius=${arrive_radius}m, speed=${yopo_velocity}m/s, altitude=${altitude}m, collision_radius=${collision_radius}m, spawn_clear_radius=${spawn_clear_radius}m, rviz_goal=unsupported, vis_ply_per_uav=${vis_ply_per_uav}"
  echo "[launch_crossover] RViz 2D Nav Goal clicks are ignored for crossover; uav0 will print a warning if a click is received."
  echo "[launch_crossover] tracker checkpoint: ${weight_path}"
  echo "[launch_crossover] simulator config: ${runtime_simulator_config}"
  echo "[launch_crossover] stop with: tools/launch_crossover.sh --session ${SESSION} --stop"

  if [[ "${DETACH}" -eq 1 ]]; then
    echo "[launch_crossover] detached; attach with: tmux attach -t ${SESSION}"
    exit 0
  fi

  exec tmux attach -t "${SESSION}"
}

main "$@"
