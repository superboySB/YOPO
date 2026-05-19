#!/usr/bin/env bash
set -Eeuo pipefail

TRIAL=0
EPOCH=50
SESSION="yopo-sim"
DETACH=0
STOP_ONLY=0

usage() {
  cat <<'EOF'
Usage (inside container):
  tools/launch_sim.sh [--trial N] [--epoch N] [--session NAME] [--detach] [--stop]

Options:
  --trial N        YOPO checkpoint trial id (default: 1)
  --epoch N        YOPO checkpoint epoch id (default: 50)
  --session NAME   tmux session name (default: yopo-sim)
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
  pkill -f 'se3_quadrotor_simulator simulator_attitude_control.launch' >/dev/null 2>&1 || true
  pkill -f 'rosrun sensor_simulator sensor_simulator_cuda' >/dev/null 2>&1 || true
  pkill -f 'python3 test_yopo_ros.py --trial=' >/dev/null 2>&1 || true
  pkill -f 'rviz -d yopo.rviz' >/dev/null 2>&1 || true
}

ensure_inside_container() {
  if [[ ! -f "/.dockerenv" ]]; then
    echo "Error: run this script inside the YOPO container." >&2
    echo "Hint: docker exec -it dzp-yopo /bin/bash" >&2
    exit 1
  fi
}

main() {
  parse_args "$@"

  ensure_inside_container
  require_cmd tmux
  require_cmd roslaunch
  require_cmd rosrun
  require_cmd rviz

  if [[ "${STOP_ONLY}" -eq 1 ]]; then
    stop_all
    echo "[launch_sim] stopped session and related processes."
    exit 0
  fi

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Error: tmux session '${SESSION}' already exists." >&2
    echo "Use: tools/launch_sim.sh --session ${SESSION} --stop" >&2
    exit 1
  fi

  local env_setup='source /opt/ros/noetic/setup.bash'
  local cmd_controller="${env_setup}; cd /workspace/YOPO/Controller; source devel/setup.bash; roslaunch se3_quadrotor_simulator simulator_attitude_control.launch"
  local cmd_simulator="${env_setup}; cd /workspace/YOPO/Simulator; source devel/setup.bash; rosrun sensor_simulator sensor_simulator_cuda"
  local cmd_planner="${env_setup}; cd /workspace/YOPO/YOPO; python3 test_yopo_ros.py --trial=${TRIAL} --epoch=${EPOCH}"
  local cmd_rviz="${env_setup}; cd /workspace/YOPO/YOPO; rviz -d yopo.rviz"

  tmux new-session -d -s "${SESSION}" -n controller "bash -lc '${cmd_controller}'"
  tmux new-window -t "${SESSION}:" -n simulator "bash -lc '${cmd_simulator}'"
  tmux new-window -t "${SESSION}:" -n planner "bash -lc '${cmd_planner}'"
  tmux new-window -t "${SESSION}:" -n rviz "bash -lc '${cmd_rviz}'"
  tmux set-option -t "${SESSION}" remain-on-exit on

  # In this session, Ctrl+C stops all 4 windows at once.
  tmux bind-key -T root C-c if-shell -F "#{==:#{session_name},${SESSION}}" "kill-session -t ${SESSION}" "send-keys C-c"
  tmux set-hook -t "${SESSION}" session-closed "unbind-key -T root C-c"

  echo "[launch_sim] started tmux session='${SESSION}', trial=${TRIAL}, epoch=${EPOCH}"
  echo "[launch_sim] Ctrl+C in this tmux session will stop all 4 windows."
  echo "[launch_sim] manual stop: tools/launch_sim.sh --session ${SESSION} --stop"

  if [[ "${DETACH}" -eq 1 ]]; then
    echo "[launch_sim] detached mode; attach with: tmux attach -t ${SESSION}"
    exit 0
  fi

  exec tmux attach -t "${SESSION}"
}

main "$@"
