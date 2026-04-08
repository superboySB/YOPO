#!/usr/bin/env bash
set -Eeuo pipefail

TRIAL=1
EPOCH=50
SESSION="yopo-swarm"
DETACH=0
STOP_ONLY=0

UAV_NUM=4
RADIUS=16.0
ALTITUDE=2.0
SPAWN_CLEAR_RADIUS=2.6
COLLISION_RADIUS=0.25
ARRIVE_RADIUS=1.5
SWARM_TANGENT_BIAS=0.0
SWARM_BIAS_RADIUS=8.0
YOPO_CONFIG="/workspace/YOPO/YOPO/config/swarm_traj_opt.yaml"
WEIGHTS_ROOT="saved"

usage() {
  cat <<'EOF'
Usage (inside container):
  tools/swarm_launch.sh [--trial N] [--epoch N] [--uav-num N] [--radius R] [--altitude Z] [--swarm-tangent-bias B] [--swarm-bias-radius R] [--yopo-config PATH] [--weights-root DIR] [--detach] [--stop]

Options:
  --trial N                YOPO checkpoint trial id (default: 1)
  --epoch N                YOPO checkpoint epoch id (default: 50)
  --uav-num N              number of UAVs on the ring (default: 4)
  --radius R               ring radius in meters (default: 16.0)
  --altitude Z             swarm flight altitude (default: 2.0)
  --swarm-tangent-bias B   tangential goal bias near center (default: 0.0)
  --swarm-bias-radius R    distance-to-center activation range for tangential bias (default: 8.0)
  --yopo-config PATH       YOPO config yaml (default: /workspace/YOPO/YOPO/config/swarm_traj_opt.yaml)
  --weights-root DIR       checkpoint root under YOPO/ (default: saved)
  --session NAME           tmux session name (default: yopo-swarm)
  --detach                 create session only, do not auto-attach
  --stop                   stop existing swarm session and related processes
  -h, --help               show help
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
      --trial)
        TRIAL="${2:-}"
        shift 2
        ;;
      --epoch)
        EPOCH="${2:-}"
        shift 2
        ;;
      --uav-num)
        UAV_NUM="${2:-}"
        shift 2
        ;;
      --radius)
        RADIUS="${2:-}"
        shift 2
        ;;
      --altitude)
        ALTITUDE="${2:-}"
        shift 2
        ;;
      --swarm-tangent-bias)
        SWARM_TANGENT_BIAS="${2:-}"
        shift 2
        ;;
      --swarm-bias-radius)
        SWARM_BIAS_RADIUS="${2:-}"
        shift 2
        ;;
      --yopo-config)
        YOPO_CONFIG="${2:-}"
        shift 2
        ;;
      --weights-root)
        WEIGHTS_ROOT="${2:-}"
        shift 2
        ;;
      --session)
        SESSION="${2:-}"
        shift 2
        ;;
      --detach)
        DETACH=1
        shift
        ;;
      --stop)
        STOP_ONLY=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown arg: $1" >&2
        usage
        exit 1
        ;;
    esac
  done
}

stop_all() {
  local patterns=(
    '/opt/ros/noetic/bin/roscore'
    '/opt/ros/noetic/bin/rosmaster'
    'roslaunch so3_quadrotor_simulator single_attitude_control.launch'
    'roslaunch so3_quadrotor_simulator simulator_attitude_control_uav.launch'
    'quadrotor_simulator_so3'
    'so3_control_nodelet'
    'rosrun sensor_simulator sensor_simulator_cuda'
    'sensor_simulator_cuda'
    'python3 test_yopo_ros.py --trial='
    'python3 test_yopo_ros.py --agent_name='
    'python3 test_yopo_ros.py'
    'rviz -d swarm_yopo.rviz'
    'rviz -d yopo_swarm.rviz'
    'rviz'
  )

  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  for pattern in "${patterns[@]}"; do
    pkill -f "${pattern}" >/dev/null 2>&1 || true
  done
  sleep 1
  for pattern in "${patterns[@]}"; do
    pkill -9 -f "${pattern}" >/dev/null 2>&1 || true
  done
}

ensure_inside_container() {
  if [[ ! -f "/.dockerenv" ]]; then
    echo "Error: run this script inside the YOPO container." >&2
    echo "Hint: docker exec -it dzp-yopo /bin/bash" >&2
    exit 1
  fi
}

build_layout() {
  python3 - "$UAV_NUM" "$RADIUS" "$ALTITUDE" <<'PY'
import math
import sys

uav_num = int(sys.argv[1])
radius = float(sys.argv[2])
altitude = float(sys.argv[3])

for idx in range(uav_num):
    angle = 2.0 * math.pi * idx / uav_num
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    print(f"{idx}\t{x:.3f}\t{y:.3f}\t{altitude:.3f}\t{-x:.3f}\t{-y:.3f}\t{altitude:.3f}")
PY
}

build_wait_lib() {
  cat <<'EOF'
wait_for_master() {
  local deadline=$((SECONDS + 60))
  until rosnode list >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[launch_swarm] timed out waiting for roscore." >&2
      exit 1
    fi
    echo "[launch_swarm] waiting for roscore..."
    sleep 1
  done
}

wait_for_topic() {
  local topic="$1"
  local deadline=$((SECONDS + 60))
  until rostopic info "$topic" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[launch_swarm] timed out waiting for topic ${topic}." >&2
      exit 1
    fi
    echo "[launch_swarm] waiting for topic ${topic}..."
    sleep 1
  done
}
EOF
}

main() {
  parse_args "$@"

  ensure_inside_container
  require_cmd tmux
  require_cmd roscore
  require_cmd roslaunch
  require_cmd rosrun
  require_cmd python3
  require_cmd rviz

  if [[ ! -f "${YOPO_CONFIG}" ]]; then
    echo "Error: YOPO config not found: ${YOPO_CONFIG}" >&2
    exit 1
  fi

  if [[ "${STOP_ONLY}" -eq 1 ]]; then
    stop_all
    echo "[launch_swarm] stopped session and related processes."
    exit 0
  fi

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Error: tmux session '${SESSION}' already exists." >&2
    echo "Use: tools/swarm_launch.sh --session ${SESSION} --stop" >&2
    exit 1
  fi

  local env_setup='source /opt/ros/noetic/setup.bash'
  mapfile -t layout_lines < <(build_layout)
  local wait_lib
  wait_lib="$(build_wait_lib)"
  local yopo_env="export YOPO_CONFIG_PATH='${YOPO_CONFIG}'; "
  local wait_for_all_odom_topics=""
  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r idx _init_x _init_y _init_z _goal_x _goal_y _goal_z <<<"${line}"
    local uav_name="uav${idx}"
    wait_for_all_odom_topics+="wait_for_topic /${uav_name}/sim/odom; "
  done

  tmux new-session -d -s "${SESSION}" -n roscore "bash -lc '${env_setup}; roscore'"

  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r idx init_x init_y init_z goal_x goal_y goal_z <<<"${line}"
    local uav_name="uav${idx}"
    local cmd_controller="${env_setup}; ${wait_lib}; wait_for_master; cd /workspace/YOPO/Controller; source devel/setup.bash; roslaunch so3_quadrotor_simulator simulator_attitude_control_uav.launch uav_name:=${uav_name} init_x:=${init_x} init_y:=${init_y} init_z:=${init_z} odom_topic:=/${uav_name}/sim/odom imu_topic:=/${uav_name}/sim/imu ctrl_topic:=/${uav_name}/so3_control/pos_cmd so3_cmd_topic:=/${uav_name}/so3_cmd force_disturbance_topic:=/${uav_name}/force_disturbance moment_disturbance_topic:=/${uav_name}/moment_disturbance simulator_node_name:=${uav_name}_simulator controller_node_name:=${uav_name}_controller"
    tmux new-window -t "${SESSION}:" -n "ctrl_${uav_name}" "bash -lc '${cmd_controller}'"
  done

  local cmd_simulator="${env_setup}; ${wait_lib}; wait_for_master; ${wait_for_all_odom_topics} cd /workspace/YOPO/Simulator; source devel/setup.bash; rosrun sensor_simulator sensor_simulator_cuda _swarm_enabled:=true _swarm_uav_num:=${UAV_NUM} _swarm_namespace_prefix:=uav _swarm_ring_radius:=${RADIUS} _swarm_altitude:=${ALTITUDE} _swarm_spawn_clear_radius:=${SPAWN_CLEAR_RADIUS} _swarm_collision_radius:=${COLLISION_RADIUS}"
  tmux new-window -t "${SESSION}:" -n simulator "bash -lc '${cmd_simulator}'"

  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r idx init_x init_y init_z goal_x goal_y goal_z <<<"${line}"
    local uav_name="uav${idx}"
    local cmd_planner="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /${uav_name}/sim/odom; wait_for_topic /${uav_name}/depth_image; cd /workspace/YOPO/YOPO; ${yopo_env}python3 test_yopo_ros.py --trial=${TRIAL} --epoch=${EPOCH} --weights_root=${WEIGHTS_ROOT} --agent_name=${uav_name} --node_name=yopo_net_${uav_name} --odom_topic=/${uav_name}/sim/odom --depth_topic=/${uav_name}/depth_image --ctrl_topic=/${uav_name}/so3_control/pos_cmd --goal_topic='' --visual_prefix=/${uav_name}/yopo_net --status_prefix=/${uav_name}/yopo --goal_x=${goal_x} --goal_y=${goal_y} --goal_z=${goal_z} --arrive_radius=${ARRIVE_RADIUS} --swarm_center_x=0.0 --swarm_center_y=0.0 --swarm_center_z=${ALTITUDE} --swarm_tangent_bias=${SWARM_TANGENT_BIAS} --swarm_bias_radius=${SWARM_BIAS_RADIUS} --visualize=1"
    tmux new-window -t "${SESSION}:" -n "plan_${uav_name}" "bash -lc '${cmd_planner}'"
  done

  local cmd_rviz="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /mock_map; wait_for_topic /uav0/depth_image; wait_for_topic /uav0/yopo_net/trajs_visual; cd /workspace/YOPO/YOPO; rviz -d swarm_yopo.rviz"
  tmux new-window -t "${SESSION}:" -n rviz "bash -lc '${cmd_rviz}'"
  tmux set-option -t "${SESSION}" remain-on-exit on

  tmux bind-key -T root C-c if-shell -F "#{==:#{session_name},${SESSION}}" "kill-session -t ${SESSION}" "send-keys C-c"
  tmux set-hook -t "${SESSION}" session-closed "unbind-key -T root C-c"

  echo "[launch_swarm] started tmux session='${SESSION}', uav_num=${UAV_NUM}, radius=${RADIUS}, altitude=${ALTITUDE}, tangent_bias=${SWARM_TANGENT_BIAS}, trial=${TRIAL}, epoch=${EPOCH}"
  echo "[launch_swarm] Ctrl+C in this tmux session will stop all windows."
  echo "[launch_swarm] manual stop: tools/swarm_launch.sh --session ${SESSION} --stop"

  if [[ "${DETACH}" -eq 1 ]]; then
    echo "[launch_swarm] detached mode; attach with: tmux attach -t ${SESSION}"
    exit 0
  fi

  exec tmux attach -t "${SESSION}"
}

main "$@"
