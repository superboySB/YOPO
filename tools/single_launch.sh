#!/usr/bin/env bash
set -Eeuo pipefail

TRIAL=1
EPOCH=50
SESSION="yopo-sim"
DETACH=0
STOP_ONLY=0
YOPO_CONFIG="/workspace/YOPO/YOPO/config/single_traj_opt.yaml"
WEIGHTS_ROOT="saved"

usage() {
  cat <<'EOF'
Usage (inside container):
  tools/single_launch.sh [--trial N] [--epoch N] [--session NAME] [--yopo-config PATH] [--weights-root DIR] [--detach] [--stop]

Options:
  --trial N        YOPO checkpoint trial id (default: 1)
  --epoch N        YOPO checkpoint epoch id (default: 50)
  --session NAME   tmux session name (default: yopo-sim)
  --yopo-config PATH  YOPO config yaml (default: /workspace/YOPO/YOPO/config/single_traj_opt.yaml)
  --weights-root DIR  checkpoint root under YOPO/ (default: saved)
  --detach         Create session only, do not auto-attach
  --stop           Stop existing simulation session and related processes
  -h, --help       Show help
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
      --session)
        SESSION="${2:-}"
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
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  pkill -f '/opt/ros/noetic/bin/roscore' >/dev/null 2>&1 || true
  pkill -f '/opt/ros/noetic/bin/rosmaster' >/dev/null 2>&1 || true
  pkill -f 'so3_quadrotor_simulator single_attitude_control.launch' >/dev/null 2>&1 || true
  pkill -f 'so3_quadrotor_simulator simulator_attitude_control.launch' >/dev/null 2>&1 || true
  pkill -f 'rosrun sensor_simulator sensor_simulator_cuda' >/dev/null 2>&1 || true
  pkill -f 'python3 test_yopo_ros.py --trial=' >/dev/null 2>&1 || true
  pkill -f 'rviz -d single_yopo.rviz' >/dev/null 2>&1 || true
  pkill -f 'rviz -d yopo.rviz' >/dev/null 2>&1 || true
}

ensure_inside_container() {
  if [[ ! -f "/.dockerenv" ]]; then
    echo "Error: run this script inside the YOPO container." >&2
    echo "Hint: docker exec -it dzp-yopo /bin/bash" >&2
    exit 1
  fi
}

build_wait_lib() {
  cat <<'EOF'
wait_for_master() {
  local deadline=$((SECONDS + 60))
  until rosnode list >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[launch_sim] timed out waiting for roscore." >&2
      exit 1
    fi
    echo "[launch_sim] waiting for roscore..."
    sleep 1
  done
}

wait_for_topic() {
  local topic="$1"
  local deadline=$((SECONDS + 60))
  until rostopic info "$topic" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[launch_sim] timed out waiting for topic ${topic}." >&2
      exit 1
    fi
    echo "[launch_sim] waiting for topic ${topic}..."
    sleep 1
  done
}
EOF
}

main() {
  parse_args "$@"

  ensure_inside_container
  require_cmd tmux
  require_cmd roslaunch
  require_cmd rosrun
  require_cmd rviz

  if [[ ! -f "${YOPO_CONFIG}" ]]; then
    echo "Error: YOPO config not found: ${YOPO_CONFIG}" >&2
    exit 1
  fi

  if [[ "${STOP_ONLY}" -eq 1 ]]; then
    stop_all
    echo "[launch_sim] stopped session and related processes."
    exit 0
  fi

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Error: tmux session '${SESSION}' already exists." >&2
    echo "Use: tools/single_launch.sh --session ${SESSION} --stop" >&2
    exit 1
  fi

  local env_setup='source /opt/ros/noetic/setup.bash'
  local wait_lib
  wait_lib="$(build_wait_lib)"
  local yopo_env="export YOPO_CONFIG_PATH='${YOPO_CONFIG}'; "
  local cmd_controller="${env_setup}; ${wait_lib}; wait_for_master; cd /workspace/YOPO/Controller; source devel/setup.bash; roslaunch so3_quadrotor_simulator single_attitude_control.launch"
  local cmd_simulator="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /sim/odom; cd /workspace/YOPO/Simulator; source devel/setup.bash; rosrun sensor_simulator sensor_simulator_cuda"
  local cmd_planner="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /sim/odom; wait_for_topic /depth_image; cd /workspace/YOPO/YOPO; ${yopo_env}python3 test_yopo_ros.py --trial=${TRIAL} --epoch=${EPOCH} --weights_root=${WEIGHTS_ROOT}"
  local cmd_rviz="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /mock_map; wait_for_topic /local_map_visual; wait_for_topic /depth_image; wait_for_topic /yopo_net/trajs_visual; cd /workspace/YOPO/YOPO; rviz -d single_yopo.rviz"

  tmux new-session -d -s "${SESSION}" -n roscore "bash -lc '${env_setup}; roscore'"
  tmux new-window -t "${SESSION}:" -n controller "bash -lc '${cmd_controller}'"
  tmux new-window -t "${SESSION}:" -n simulator "bash -lc '${cmd_simulator}'"
  tmux new-window -t "${SESSION}:" -n planner "bash -lc '${cmd_planner}'"
  tmux new-window -t "${SESSION}:" -n rviz "bash -lc '${cmd_rviz}'"
  tmux set-option -t "${SESSION}" remain-on-exit on

  # In this session, Ctrl+C stops all 4 windows at once.
  tmux bind-key -T root C-c if-shell -F "#{==:#{session_name},${SESSION}}" "kill-session -t ${SESSION}" "send-keys C-c"
  tmux set-hook -t "${SESSION}" session-closed "unbind-key -T root C-c"

  echo "[launch_sim] started tmux session='${SESSION}', trial=${TRIAL}, epoch=${EPOCH}"
  echo "[launch_sim] Ctrl+C in this tmux session will stop all windows."
  echo "[launch_sim] manual stop: tools/single_launch.sh --session ${SESSION} --stop"

  if [[ "${DETACH}" -eq 1 ]]; then
    echo "[launch_sim] detached mode; attach with: tmux attach -t ${SESSION}"
    exit 0
  fi

  exec tmux attach -t "${SESSION}"
}

main "$@"
