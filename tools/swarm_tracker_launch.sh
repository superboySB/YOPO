#!/usr/bin/env bash
set -Eeuo pipefail

TRIAL=0
EPOCH=50
SESSION="yopo-swarm-tracker"
DETACH=0
STOP_ONLY=0

UAV_NUM=1
FORMATION="1"
FORMATION_START_X=-30.0
FORWARD_DISTANCE=50.0
ALTITUDE=1.5
ARRIVE_RADIUS=""
COLLISION_RADIUS=0.155
SPAWN_CLEAR_RADIUS=2.6

YOPO_CONFIG="/workspace/YOPO/YOPO/config/tracker_traj_opt.yaml"
SIMULATOR_CONFIG="/workspace/YOPO/Simulator/src/config/swarm_config.yaml"
WEIGHTS_ROOT="saved"
VISUALIZE=1
RVIZ=0
VISUALIZE_POINTCLOUD=0
MIN_ALTITUDE=1.5
ENABLE_RVIZ_GOAL=1
PLANNER_START_DELAY_STEP=0.5

usage() {
  cat <<'EOF'
Usage:
  tools/swarm_tracker_launch.sh [--trial N] [--epoch N] [--uav-num N] [--detach]
  tools/swarm_tracker_launch.sh --stop

Options:
  --trial N                 tracker checkpoint trial under YOPO/saved (default: 0)
  --epoch N                 tracker checkpoint epoch id (default: 50)
  --uav-num N               number of UAVs (default: 1)
  --formation ROWS          pipe-separated back-to-front rows, e.g. 1 or 1|2|3|4 (default: 1)
  --formation-start-x X     x position of the back row (default: -30.0)
  --forward-distance M      each UAV goal is init_x + this distance (default: 50.0)
  --altitude Z              formation altitude (default: 1.5)
  --arrive-radius R         per-UAV arrival radius override (default: swarm_arrive_radius from YOPO config)
  --collision-radius R      UAV-UAV collision counter radius (default: 0.155)
  --spawn-clear-radius R    tree clearing around starts/goals (default: 2.6)
  --yopo-config PATH        tracker config yaml
  --sim-config PATH         simulator config yaml
  --weights-root DIR        tracker checkpoint root under YOPO/ (default: saved)
  --visualize 0|1           publish all candidate/lattice trajectory point clouds (default: 1)
  --rviz 0|1                open RViz (default: 0)
  --visualize-pointcloud 0|1 publish aggregated local lidar map; only meaningful with lidar on (default: 0)
  --min-altitude Z          clamp planned primitive endpoint z above this height (default: 1.5)
  --enable-rviz-goal 0|1    let RViz 2D Nav Goal set uav0's target and translate the formation (default: 1)
  --planner-start-delay-step S  seconds of startup stagger per UAV index (default: 0.5)
  --session NAME            tmux session name (default: yopo-swarm-tracker)
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
      --formation) FORMATION="${2:-}"; shift 2 ;;
      --formation-start-x) FORMATION_START_X="${2:-}"; shift 2 ;;
      --forward-distance) FORWARD_DISTANCE="${2:-}"; shift 2 ;;
      --altitude) ALTITUDE="${2:-}"; shift 2 ;;
      --arrive-radius) ARRIVE_RADIUS="${2:-}"; shift 2 ;;
      --collision-radius) COLLISION_RADIUS="${2:-}"; shift 2 ;;
      --spawn-clear-radius) SPAWN_CLEAR_RADIUS="${2:-}"; shift 2 ;;
      --yopo-config) YOPO_CONFIG="${2:-}"; shift 2 ;;
      --sim-config) SIMULATOR_CONFIG="${2:-}"; shift 2 ;;
      --weights-root) WEIGHTS_ROOT="${2:-}"; shift 2 ;;
      --visualize) VISUALIZE="${2:-}"; shift 2 ;;
      --rviz) RVIZ="${2:-}"; shift 2 ;;
      --visualize-pointcloud) VISUALIZE_POINTCLOUD="${2:-}"; shift 2 ;;
      --min-altitude) MIN_ALTITUDE="${2:-}"; shift 2 ;;
      --enable-rviz-goal) ENABLE_RVIZ_GOAL="${2:-}"; shift 2 ;;
      --enable-rviz-direction-goal) ENABLE_RVIZ_GOAL="${2:-}"; shift 2 ;;
      --planner-start-delay-step) PLANNER_START_DELAY_STEP="${2:-}"; shift 2 ;;
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

validate_formation() {
  python3 - "$UAV_NUM" "$FORMATION" "$YOPO_CONFIG" <<'PY'
import math
import re
import sys

import yaml

try:
    uav_num = int(sys.argv[1])
except ValueError:
    raise SystemExit("Error: --uav-num must be a positive integer.")
if uav_num <= 0:
    raise SystemExit("Error: --uav-num must be a positive integer.")

formation = sys.argv[2]
if not re.fullmatch(r"[1-9][0-9]*(\|[1-9][0-9]*)*", formation):
    raise SystemExit(
        "Error: --formation must use positive integers separated by '|', e.g. --formation '1|2|3|4'."
    )

rows = [int(item) for item in formation.split("|")]
row_sum = sum(rows)
if row_sum != uav_num:
    raise SystemExit(
        f"Error: --formation '{formation}' sums to {row_sum}, but --uav-num={uav_num}. "
        "Set them consistently, e.g. --uav-num 10 --formation '1|2|3|4'."
    )

with open(sys.argv[3], "r", encoding="utf-8") as f:
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
clearance_distance = read_positive_float("target_clearance_distance")

# With centered rows, adjacent rows with different parity are staggered by distance/2,
# so sqrt(3)/2 * distance gives an equilateral spacing. Same-parity rows need a full
# distance in x to keep the nearest inter-row distance from dropping below the target.
has_same_parity_neighbors = any((rows[i] % 2) == (rows[i + 1] % 2) for i in range(len(rows) - 1))
row_spacing = initial_spacing if has_same_parity_neighbors else (math.sqrt(3.0) * 0.5 * initial_spacing)
rows_csv = ",".join(str(row) for row in rows)
print(f"{rows_csv}\t{row_spacing:.6f}\t{initial_spacing:.6f}\t{arrive_radius:.6f}\t{clearance_distance:.6f}")
PY
}

build_layout() {
  python3 - "$UAV_NUM" "$1" "$FORMATION_START_X" "$2" "$3" "$FORWARD_DISTANCE" "$ALTITUDE" <<'PY'
import sys

uav_num = int(sys.argv[1])
rows = [int(item) for item in sys.argv[2].split(",") if item.strip()]
start_x = float(sys.argv[3])
row_spacing = float(sys.argv[4])
lateral_spacing = float(sys.argv[5])
forward_distance = float(sys.argv[6])
altitude = float(sys.argv[7])

if sum(rows) != uav_num:
    raise SystemExit(f"formation rows hold {sum(rows)} UAVs, but --uav-num={uav_num}")

idx = 0
for row_idx, row_count in enumerate(rows):
    remaining = uav_num - idx
    effective_count = min(row_count, remaining)
    if effective_count <= 0:
        continue
    x = start_x + row_idx * row_spacing
    y0 = -0.5 * (effective_count - 1) * lateral_spacing
    for col in range(effective_count):
        y = y0 + col * lateral_spacing
        print(f"{idx}\t{x:.3f}\t{y:.3f}\t{altitude:.3f}\t{x + forward_distance:.3f}\t{y:.3f}\t{altitude:.3f}")
        idx += 1
PY
}

build_wait_lib() {
  cat <<'EOF'
wait_for_master() {
  local deadline=$((SECONDS + 60))
  until rosnode list >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[swarm_tracker] timed out waiting for roscore." >&2
      exit 1
    fi
    sleep 1
  done
}

wait_for_topic() {
  local topic="$1"
  local deadline=$((SECONDS + 90))
  until rostopic info "$topic" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[swarm_tracker] timed out waiting for topic ${topic}." >&2
      exit 1
    fi
    sleep 1
  done
}
EOF
}

build_rviz_config() {
  python3 - "/workspace/YOPO/YOPO/swarm_tracker.rviz" "${UAV_NUM}" <<'PY'
import copy
import os
import sys
import tempfile
from pathlib import Path

import yaml

src_path = Path(sys.argv[1])
uav_num = int(sys.argv[2])
config = yaml.safe_load(src_path.read_text(encoding="utf-8"))
displays = config.get("Visualization Manager", {}).get("Displays", [])

depth_template = next((d for d in displays if d.get("Class") == "rviz/Image" and (d.get("Name") == "Depth" or d.get("Image Topic") in ("/depth_image", "/uav0/depth_image"))), None)
mask_template = next((d for d in displays if d.get("Class") == "rviz/Image" and (d.get("Name") == "Target_Mask" or d.get("Image Topic") in ("/target_mask_image", "/uav0/target_mask_image"))), None)
traj_template = next((d for d in displays if d.get("Class") == "rviz/Group" and d.get("Name") == "Trajectory"), None)
drone_template = next((d for d in displays if d.get("Class") == "rviz/Marker" and d.get("Name") == "Drone"), None)
label_template = next((d for d in displays if d.get("Class") == "rviz/Marker" and d.get("Name") == "Status_Label"), None)

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
fd, out_path = tempfile.mkstemp(prefix="swarm_tracker_", suffix=".rviz")
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
  validate_binary_flag --visualize-pointcloud "${VISUALIZE_POINTCLOUD}"
  validate_binary_flag --enable-rviz-goal "${ENABLE_RVIZ_GOAL}"
  if [[ "${RVIZ}" == "1" ]]; then
    require_cmd rviz
  fi

  if [[ "${STOP_ONLY}" -eq 1 ]]; then
    stop_all
    echo "[swarm_tracker] stopped session and related processes."
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

  local formation_meta formation_rows_csv formation_row_spacing formation_lateral_spacing config_arrive_radius target_clearance_distance
  formation_meta="$(validate_formation)"
  IFS=$'\t' read -r formation_rows_csv formation_row_spacing formation_lateral_spacing config_arrive_radius target_clearance_distance <<<"${formation_meta}"
  local arrive_radius="${ARRIVE_RADIUS:-${config_arrive_radius}}"

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
    echo "Use: tools/swarm_tracker_launch.sh --session ${SESSION} --stop" >&2
    exit 1
  fi

  local simulator_depth_fps simulator_max_depth_dist
  IFS=$'\t' read -r simulator_depth_fps simulator_max_depth_dist < <(
    python3 - "${SIMULATOR_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
print(f"{float(config['depth_fps'])}\t{float(config['camera']['max_depth_dist'])}")
PY
  )

  local env_setup='source /opt/ros/noetic/setup.bash'
  local wait_lib
  wait_lib="$(build_wait_lib)"
  local yopo_env="export YOPO_CONFIG_PATH=${YOPO_CONFIG}; "
  mapfile -t layout_lines < <(build_layout "${formation_rows_csv}" "${formation_row_spacing}" "${formation_lateral_spacing}")

  local wait_for_all_odom_topics=""
  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r idx _init_x _init_y _init_z _goal_x _goal_y _goal_z <<<"${line}"
    wait_for_all_odom_topics+="wait_for_topic /uav${idx}/sim/odom; "
  done

  tmux new-session -d -s "${SESSION}" -n roscore "bash -lc '${env_setup}; roscore'"

  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r idx init_x init_y init_z _goal_x _goal_y _goal_z <<<"${line}"
    local uav_name="uav${idx}"
    local cmd_controller="${env_setup}; ${wait_lib}; wait_for_master; cd /workspace/YOPO/Controller; source devel/setup.bash; roslaunch so3_quadrotor_simulator simulator_attitude_control_uav.launch uav_name:=${uav_name} init_x:=${init_x} init_y:=${init_y} init_z:=${init_z} odom_topic:=/${uav_name}/sim/odom imu_topic:=/${uav_name}/sim/imu ctrl_topic:=/${uav_name}/so3_control/pos_cmd so3_cmd_topic:=/${uav_name}/so3_cmd force_disturbance_topic:=/${uav_name}/force_disturbance moment_disturbance_topic:=/${uav_name}/moment_disturbance simulator_node_name:=${uav_name}_simulator controller_node_name:=${uav_name}_controller"
    tmux new-window -t "${SESSION}:" -n "ctrl_${uav_name}" "bash -lc '${cmd_controller}'"
  done

  local cmd_simulator="${env_setup}; ${wait_lib}; wait_for_master; ${wait_for_all_odom_topics} cd /workspace/YOPO/Simulator; source devel/setup.bash; rosrun sensor_simulator sensor_simulator_cuda _config_path:=${SIMULATOR_CONFIG} _swarm_enabled:=true _swarm_uav_num:=${UAV_NUM} _swarm_namespace_prefix:=uav _swarm_altitude:=${ALTITUDE} _swarm_collision_radius:=${COLLISION_RADIUS} _swarm_spawn_clear_radius:=${SPAWN_CLEAR_RADIUS} _swarm_forward_distance:=${FORWARD_DISTANCE} _swarm_formation_start_x:=${FORMATION_START_X} _swarm_formation_row_spacing:=${formation_row_spacing} _swarm_formation_lateral_spacing:=${formation_lateral_spacing} _swarm_formation_rows:=${formation_rows_csv} _visualize_local_map:=${VISUALIZE_POINTCLOUD}"
  tmux new-window -t "${SESSION}:" -n simulator "bash -lc '${cmd_simulator}'"

  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r idx _init_x _init_y _init_z goal_x goal_y goal_z <<<"${line}"
    local uav_name="uav${idx}"
    local planner_delay
    planner_delay="$(python3 - "${idx}" "${PLANNER_START_DELAY_STEP}" <<'PY'
import sys
print(f"{int(sys.argv[1]) * float(sys.argv[2]):.2f}")
PY
)"
    local cmd_planner="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /${uav_name}/sim/odom; wait_for_topic /${uav_name}/depth_image; wait_for_topic /${uav_name}/target_mask_image; sleep ${planner_delay}; cd /workspace/YOPO/YOPO; ${yopo_env}python3 test_yopo_ros_swarm_tracker.py --trial=${TRIAL} --epoch=${EPOCH} --weights_root=${weights_root_abs} --agent_name=${uav_name} --node_name=yopo_tracker_${uav_name} --odom_topic=/${uav_name}/sim/odom --depth_topic=/${uav_name}/depth_image --target_mask_topic=/${uav_name}/target_mask_image --ctrl_topic=/${uav_name}/so3_control/pos_cmd --visual_prefix=/${uav_name}/yopo_tracker --status_prefix=/${uav_name}/yopo --rviz_goal_topic=/move_base_simple/goal --rviz_goal_reference_odom_topic=/uav0/sim/odom --goal_x=${goal_x} --goal_y=${goal_y} --goal_z=${goal_z} --arrive_radius=${arrive_radius} --max_depth_dist=${simulator_max_depth_dist} --depth_fps=${simulator_depth_fps} --visualize=${VISUALIZE} --min_altitude=${MIN_ALTITUDE} --enable_rviz_goal=${ENABLE_RVIZ_GOAL}"
    tmux new-window -t "${SESSION}:" -n "plan_${uav_name}" "bash -lc '${cmd_planner}'"
  done

  if [[ "${RVIZ}" == "1" ]]; then
    local rviz_config
    rviz_config="$(build_rviz_config)"
    local cmd_rviz="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /mock_map; wait_for_topic /uav0/depth_image; wait_for_topic /uav0/target_mask_image; cd /workspace/YOPO/YOPO; rviz -d ${rviz_config}"
    tmux new-window -t "${SESSION}:" -n rviz "bash -lc '${cmd_rviz}'"
  fi

  tmux set-option -t "${SESSION}" remain-on-exit on
  tmux bind-key -T root C-c if-shell -F "#{==:#{session_name},${SESSION}}" "kill-session -t ${SESSION}" "send-keys C-c"
  tmux set-hook -t "${SESSION}" session-closed "unbind-key -T root C-c"

  echo "[swarm_tracker] started tmux session='${SESSION}', uav_num=${UAV_NUM}, formation=${FORMATION}, row_spacing=${formation_row_spacing}m, lateral_spacing=${formation_lateral_spacing}m, arrive_radius=${arrive_radius}m, keep_distance=${target_clearance_distance}m, forward=${FORWARD_DISTANCE}m, speed=5m/s, rviz_goal=${ENABLE_RVIZ_GOAL}"
  echo "[swarm_tracker] tracker checkpoint: ${weight_path}"
  echo "[swarm_tracker] stop with: tools/swarm_tracker_launch.sh --session ${SESSION} --stop"

  if [[ "${DETACH}" -eq 1 ]]; then
    echo "[swarm_tracker] detached; attach with: tmux attach -t ${SESSION}"
    exit 0
  fi

  exec tmux attach -t "${SESSION}"
}

main "$@"
