#!/usr/bin/env bash
set -Eeuo pipefail

SESSION="yopo-compare"
SIMPLE_WEIGHT="/workspace/YOPO/YOPO/saved/yopo-simple/epoch50.pth"
MINCO_WEIGHT="/workspace/YOPO/YOPO/saved/yopo-minco/epoch50.pth"
VELOCITY="6.0"
SAFE_RADIUS="0.05"
MINCO_CORRIDOR_SIGMA="1.0"
MINCO_TOPK="1"
MINCO_CONTINUITY_MODE="none"
SIMPLE_POLICY="simple"
SIMPLE_SAFE_RADIUS="0.05"
SIMPLE_CORRIDOR_SIGMA="1.0"
SIMPLE_TOPK="1"
SIMPLE_CONTINUITY_MODE="none"
ARRIVAL_RADIUS="1.0"
MAZE_TYPE="5"
MAP_SEED="3"
YAW_GOAL_WEIGHT="6.0"
SIMPLE_Y="-0.4"
MINCO_Y="0.4"
DETACH=0
HEADLESS=0
RVIZ_SOFTWARE_GL=0
STOP_ONLY=0

usage() {
  cat <<'EOF'
Usage (inside the retained YOPO container):
  tools/launch_compare.sh [options]

Options:
  --simple-weight PATH   YOPO-Simple checkpoint
  --minco-weight PATH    YOPO-MINCO checkpoint
  --velocity MPS         both planners' test velocity (default: 6.0)
  --safe-radius M        MINCO corridor admission radius (default: 0.05)
  --minco-sigma K        right MINCO lower bound uses mu-K*b (default: 1.0)
  --minco-topk N         right MINCO score shortlist size (default: 1)
  --minco-continuity M   right selection: none, inner, or jerk
  --simple-policy P      left planner implementation: simple or minco
  --simple-safe-radius M left MINCO admission radius (only with --simple-policy minco)
  --simple-sigma K       left MINCO lower-bound multiplier
  --simple-topk N        left MINCO score shortlist size
  --simple-continuity M  left MINCO selection: none, inner, or jerk
  --arrival-radius M     common goal radius (default: 1.0)
  --maze-type N          random map type (supported policy: 1,2,5,7)
  --map-seed N           deterministic random map seed
  --yaw-goal-weight X    common yaw-to-goal blend scale (default: 6.0)
  --simple-y M           simple initial lateral position (default: -0.4)
  --minco-y M            MINCO initial lateral position (default: 0.4)
  --session NAME         tmux session name (default: yopo-compare)
  --detach               start without attaching
  --headless             do not start RViz
  --rviz-software-gl     set LIBGL_ALWAYS_SOFTWARE=1 for RViz
  --stop                 stop this tmux session
  -h, --help             show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --simple-weight) SIMPLE_WEIGHT="${2:?missing value}"; shift 2 ;;
    --minco-weight) MINCO_WEIGHT="${2:?missing value}"; shift 2 ;;
    --velocity) VELOCITY="${2:?missing value}"; shift 2 ;;
    --safe-radius) SAFE_RADIUS="${2:?missing value}"; shift 2 ;;
    --minco-sigma) MINCO_CORRIDOR_SIGMA="${2:?missing value}"; shift 2 ;;
    --minco-topk) MINCO_TOPK="${2:?missing value}"; shift 2 ;;
    --minco-continuity) MINCO_CONTINUITY_MODE="${2:?missing value}"; shift 2 ;;
    --simple-policy) SIMPLE_POLICY="${2:?missing value}"; shift 2 ;;
    --simple-safe-radius) SIMPLE_SAFE_RADIUS="${2:?missing value}"; shift 2 ;;
    --simple-sigma) SIMPLE_CORRIDOR_SIGMA="${2:?missing value}"; shift 2 ;;
    --simple-topk) SIMPLE_TOPK="${2:?missing value}"; shift 2 ;;
    --simple-continuity) SIMPLE_CONTINUITY_MODE="${2:?missing value}"; shift 2 ;;
    --arrival-radius) ARRIVAL_RADIUS="${2:?missing value}"; shift 2 ;;
    --maze-type) MAZE_TYPE="${2:?missing value}"; shift 2 ;;
    --map-seed) MAP_SEED="${2:?missing value}"; shift 2 ;;
    --yaw-goal-weight) YAW_GOAL_WEIGHT="${2:?missing value}"; shift 2 ;;
    --simple-y) SIMPLE_Y="${2:?missing value}"; shift 2 ;;
    --minco-y) MINCO_Y="${2:?missing value}"; shift 2 ;;
    --session) SESSION="${2:?missing value}"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --headless) HEADLESS=1; shift ;;
    --rviz-software-gl) RVIZ_SOFTWARE_GL=1; shift ;;
    --stop) STOP_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ "$SIMPLE_POLICY" == "simple" || "$SIMPLE_POLICY" == "minco" ]] || {
  echo "--simple-policy must be simple or minco" >&2; exit 2;
}
for mode in "$SIMPLE_CONTINUITY_MODE" "$MINCO_CONTINUITY_MODE"; do
  [[ "$mode" == "none" || "$mode" == "inner" || "$mode" == "jerk" ]] || {
    echo "continuity mode must be none, inner, or jerk" >&2; exit 2;
  }
done

if [[ ! -f /.dockerenv ]]; then
  echo "Error: run this script inside the YOPO Docker container." >&2
  exit 1
fi

# A fresh container does not put ROS tools on PATH until its setup file is
# sourced.  Do this before checking roslaunch/rosrun so the script works from
# both interactive and non-interactive shells.
set +u
source /opt/ros/noetic/setup.bash
set -u

if [[ "$STOP_ONLY" -eq 1 ]]; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! rosnode list >/dev/null 2>&1; then
      break
    fi
    sleep 0.25
  done
  echo "[launch_compare] stopped session '$SESSION'."
  exit 0
fi

for command in tmux roslaunch rosrun python3; do
  command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done
[[ -f "$SIMPLE_WEIGHT" ]] || { echo "Missing simple weight: $SIMPLE_WEIGHT" >&2; exit 1; }
[[ -f "$MINCO_WEIGHT" ]] || { echo "Missing MINCO weight: $MINCO_WEIGHT" >&2; exit 1; }
tmux has-session -t "$SESSION" 2>/dev/null && {
  echo "Session '$SESSION' already exists; use --session '$SESSION' --stop first." >&2
  exit 1
}

setup='source /opt/ros/noetic/setup.bash'
master_cmd="$setup; roscore"
controller_cmd="$setup; sleep 2; cd /workspace/YOPO/Controller; source devel/setup.bash; roslaunch so3_quadrotor_simulator simulator_dual_compare.launch simple_init_y:=$SIMPLE_Y minco_init_y:=$MINCO_Y"
simple_sensor_cmd="$setup; sleep 4; cd /workspace/YOPO/Simulator; source devel/setup.bash; rosrun sensor_simulator sensor_simulator_cuda __name:=yopo_simple_sensor _maze_type:=$MAZE_TYPE _map_seed:=$MAP_SEED _odom_topic:=/yopo_simple/odom _depth_topic:=/yopo_simple/depth_image _lidar_topic:=/yopo_simple/lidar_points _stereo_topic:=/yopo_simple/stereo_depth _camera_info_topic:=/yopo_simple/camera_info _map_topic:=/yopo_simple/mock_map _collision_topic:=/yopo_simple/collision_count _collision_samples_topic:=/yopo_simple/collision_samples _collision_state_topic:=/yopo_simple/collision_state _clearance_topic:=/yopo_simple/clearance _body_frame:=yopo_simple/base_link _camera_frame:=yopo_simple/camera_link"
minco_sensor_cmd="$setup; sleep 8; cd /workspace/YOPO/Simulator; source devel/setup.bash; rosrun sensor_simulator sensor_simulator_cuda __name:=yopo_minco_sensor _maze_type:=$MAZE_TYPE _map_seed:=$MAP_SEED _odom_topic:=/yopo_minco/odom _depth_topic:=/yopo_minco/depth_image _lidar_topic:=/yopo_minco/lidar_points _stereo_topic:=/yopo_minco/stereo_depth _camera_info_topic:=/yopo_minco/camera_info _map_topic:=/yopo_minco/mock_map _collision_topic:=/yopo_minco/collision_count _collision_samples_topic:=/yopo_minco/collision_samples _collision_state_topic:=/yopo_minco/collision_state _clearance_topic:=/yopo_minco/clearance _body_frame:=yopo_minco/base_link _camera_frame:=yopo_minco/camera_link"
if [[ "$SIMPLE_POLICY" == "simple" ]]; then
  simple_planner_cmd="$setup; sleep 8; cd /workspace/YOPO/YOPO/simple_runtime; python3 test_yopo_ros.py --wait-for-goal --weight '$SIMPLE_WEIGHT' --velocity '$VELOCITY' --arrival-radius '$ARRIVAL_RADIUS' --yaw-goal-weight '$YAW_GOAL_WEIGHT' --node-name yopo_simple_planner --odom-topic /yopo_simple/odom --depth-topic /yopo_simple/depth_image --ctrl-topic /yopo_simple/pos_cmd --viz-prefix /yopo_simple"
else
  simple_planner_cmd="$setup; sleep 8; cd /workspace/YOPO/YOPO; python3 test_yopo_ros.py --wait-for-goal --weight '$SIMPLE_WEIGHT' --velocity '$VELOCITY' --safe-radius '$SIMPLE_SAFE_RADIUS' --corridor-sigma '$SIMPLE_CORRIDOR_SIGMA' --topk '$SIMPLE_TOPK' --continuity-mode '$SIMPLE_CONTINUITY_MODE' --arrival-radius '$ARRIVAL_RADIUS' --yaw-goal-weight '$YAW_GOAL_WEIGHT' --node-name yopo_simple_planner --odom-topic /yopo_simple/odom --depth-topic /yopo_simple/depth_image --ctrl-topic /yopo_simple/pos_cmd --viz-prefix /yopo_simple"
fi
minco_planner_cmd="$setup; sleep 8; cd /workspace/YOPO/YOPO; python3 test_yopo_ros.py --wait-for-goal --weight '$MINCO_WEIGHT' --velocity '$VELOCITY' --safe-radius '$SAFE_RADIUS' --corridor-sigma '$MINCO_CORRIDOR_SIGMA' --topk '$MINCO_TOPK' --continuity-mode '$MINCO_CONTINUITY_MODE' --arrival-radius '$ARRIVAL_RADIUS' --yaw-goal-weight '$YAW_GOAL_WEIGHT' --node-name yopo_minco_planner --odom-topic /yopo_minco/odom --depth-topic /yopo_minco/depth_image --ctrl-topic /yopo_minco/pos_cmd --viz-prefix /yopo_minco"
monitor_cmd="$setup; sleep 5; cd /workspace/YOPO; python3 tools/compare_monitor.py --arrival-radius '$ARRIVAL_RADIUS'"
rviz_prefix=""
[[ "$RVIZ_SOFTWARE_GL" -eq 1 ]] && rviz_prefix="LIBGL_ALWAYS_SOFTWARE=1 "
rviz_cmd="$setup; sleep 22; cd /workspace/YOPO/YOPO; ${rviz_prefix}rviz -d yopo_compare.rviz"

tmux new-session -d -s "$SESSION" -n master "bash -lc '$master_cmd'"
tmux new-window -t "$SESSION:" -n controllers "bash -lc '$controller_cmd'"
tmux new-window -t "$SESSION:" -n sensor-simple "bash -lc '$simple_sensor_cmd'"
tmux new-window -t "$SESSION:" -n sensor-minco "bash -lc '$minco_sensor_cmd'"
tmux new-window -t "$SESSION:" -n planner-simple "bash -lc '$simple_planner_cmd'"
tmux new-window -t "$SESSION:" -n planner-minco "bash -lc '$minco_planner_cmd'"
tmux new-window -t "$SESSION:" -n metrics "bash -lc '$monitor_cmd'"
if [[ "$HEADLESS" -eq 0 ]]; then
  command -v rviz >/dev/null || { echo "Missing command: rviz" >&2; exit 1; }
  tmux new-window -t "$SESSION:" -n rviz "bash -lc '$rviz_cmd'"
fi
tmux set-option -t "$SESSION" remain-on-exit on

echo "[launch_compare] session='$SESSION' velocity=$VELOCITY m/s map=$MAZE_TYPE seed=$MAP_SEED"
echo "[launch_compare] left=$SIMPLE_POLICY weight=$SIMPLE_WEIGHT topk=$SIMPLE_TOPK continuity=$SIMPLE_CONTINUITY_MODE sigma=$SIMPLE_CORRIDOR_SIGMA"
echo "[launch_compare] right=minco weight=$MINCO_WEIGHT topk=$MINCO_TOPK continuity=$MINCO_CONTINUITY_MODE sigma=$MINCO_CORRIDOR_SIGMA"
echo "[launch_compare] publish one /move_base_simple/goal; both aircraft react together."
echo "[launch_compare] metrics: rostopic echo /yopo_compare/metrics"
if [[ "$DETACH" -eq 1 ]]; then
  echo "[launch_compare] attach with: tmux attach -t '$SESSION'"
else
  exec tmux attach -t "$SESSION"
fi
