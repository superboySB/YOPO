#!/usr/bin/env bash
set -Eeuo pipefail

TRIAL=""
EPOCH=50
SESSION="yopo-sim"
DETACH=0
STOP_ONLY=0
YOPO_CONFIG="/workspace/YOPO/YOPO/config/single_traj_opt.yaml"
WEIGHTS_ROOT="saved/with_tracker"
TARGET_TRIAL=1
TARGET_EPOCH=50
TARGET_WEIGHTS_ROOT="saved/no_tracker"

usage() {
  cat <<'EOF'
Usage (inside container):
  tools/single_launch.sh --trial N [--epoch N] [--session NAME] [--yopo-config PATH] [--weights-root DIR] [--target-trial N] [--target-epoch N] [--target-weights-root DIR] [--detach]
  tools/single_launch.sh [--session NAME] --stop

Options:
  --trial N        follower YOPOv2-Tracker trial under YOPO/saved/with_tracker (required unless --stop)
  --epoch N        follower YOPOv2-Tracker epoch id (default: 50)
  --session NAME   tmux session name (default: yopo-sim)
  --yopo-config PATH  YOPO config yaml (default: /workspace/YOPO/YOPO/config/single_traj_opt.yaml)
  --weights-root DIR  follower tracker checkpoint root under YOPO/ (default: saved/with_tracker)
  --target-trial N    front no-tracker YOPO avoidance trial under YOPO/saved/no_tracker (default: 1)
  --target-epoch N    front no-tracker YOPO avoidance epoch id (default: 50)
  --target-weights-root DIR  front no-tracker checkpoint root under YOPO/ (default: saved/no_tracker)
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
      --target-trial)
        TARGET_TRIAL="${2:-}"
        shift 2
        ;;
      --target-epoch)
        TARGET_EPOCH="${2:-}"
        shift 2
        ;;
      --target-weights-root)
        TARGET_WEIGHTS_ROOT="${2:-}"
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
  pkill -f 'so3_quadrotor_simulator tracking_attitude_control.launch' >/dev/null 2>&1 || true
  pkill -f 'so3_quadrotor_simulator simulator_attitude_control.launch' >/dev/null 2>&1 || true
  pkill -f 'rosrun sensor_simulator sensor_simulator_cuda' >/dev/null 2>&1 || true
  pkill -f 'rosrun sensor_simulator target_motion_node.py' >/dev/null 2>&1 || true
  pkill -f 'python3 test_yopo_ros_single.py --trial=' >/dev/null 2>&1 || true
  pkill -f 'python3 test_target_yopo_ros.py --trial=' >/dev/null 2>&1 || true
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

  if [[ -z "${TRIAL}" ]]; then
    echo "Error: --trial is required because legacy avoidance checkpoints are incompatible with YOPOv2-Tracker." >&2
    echo "Train first into YOPO/saved/with_tracker, then launch with the printed YOPO_N id, for example: tools/single_launch.sh --trial 3 --epoch 50" >&2
    exit 1
  fi

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
  local target_weights_root_abs
  if [[ "${TARGET_WEIGHTS_ROOT}" = /* ]]; then
    target_weights_root_abs="${TARGET_WEIGHTS_ROOT}"
  else
    target_weights_root_abs="/workspace/YOPO/YOPO/${TARGET_WEIGHTS_ROOT}"
  fi
  local target_weight_path="${target_weights_root_abs}/YOPO_${TARGET_TRIAL}/epoch${TARGET_EPOCH}.pth"
  if [[ ! -f "${target_weight_path}" ]]; then
    echo "Error: target avoidance checkpoint not found: ${target_weight_path}" >&2
    echo "Copy it from /root/YOPO-YOPO-Simple/YOPO/saved/YOPO_1/epoch50.pth into YOPO/saved/no_tracker/ or pass matching --target-weights-root/--target-trial/--target-epoch." >&2
    exit 1
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
  local target_env="export YOPO_TRACKER_CONFIG_PATH='${YOPO_CONFIG}'; export PYTHONPATH='/workspace/YOPO/YOPO:/workspace/YOPO/YOPO/target_avoidance':\${PYTHONPATH:-}; "
  local cmd_controller="${env_setup}; ${wait_lib}; wait_for_master; cd /workspace/YOPO/Controller; source devel/setup.bash; roslaunch so3_quadrotor_simulator tracking_attitude_control.launch"
  local cmd_target="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /target/odom; wait_for_topic /target_depth_image; cd /workspace/YOPO/YOPO/target_avoidance; ${target_env}python3 test_target_yopo_ros.py --trial=${TARGET_TRIAL} --epoch=${TARGET_EPOCH} --weights_root=${target_weights_root_abs}"
  local cmd_simulator="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /sim/odom; wait_for_topic /target/odom; cd /workspace/YOPO/Simulator; source devel/setup.bash; rosrun sensor_simulator sensor_simulator_cuda"
  local cmd_planner="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /sim/odom; wait_for_topic /target/odom; wait_for_topic /depth_image; wait_for_topic /target_mask_image; cd /workspace/YOPO/YOPO; ${yopo_env}python3 test_yopo_ros_single.py --trial=${TRIAL} --epoch=${EPOCH} --weights_root=${weights_root_abs}"
  local cmd_rviz="${env_setup}; ${wait_lib}; wait_for_master; wait_for_topic /mock_map; wait_for_topic /local_map_visual; wait_for_topic /depth_image; wait_for_topic /target_mask_image; wait_for_topic /target/marker; wait_for_topic /yopo_tracker/trajs_visual; cd /workspace/YOPO/YOPO; rviz -d single_yopo.rviz"

  tmux new-session -d -s "${SESSION}" -n roscore "bash -lc '${env_setup}; roscore'"
  tmux new-window -t "${SESSION}:" -n controller "bash -lc '${cmd_controller}'"
  tmux new-window -t "${SESSION}:" -n target_yopo "bash -lc '${cmd_target}'"
  tmux new-window -t "${SESSION}:" -n simulator "bash -lc '${cmd_simulator}'"
  tmux new-window -t "${SESSION}:" -n planner "bash -lc '${cmd_planner}'"
  tmux new-window -t "${SESSION}:" -n rviz "bash -lc '${cmd_rviz}'"
  tmux set-option -t "${SESSION}" remain-on-exit on

  # In this session, Ctrl+C stops all 4 windows at once.
  tmux bind-key -T root C-c if-shell -F "#{==:#{session_name},${SESSION}}" "kill-session -t ${SESSION}" "send-keys C-c"
  tmux set-hook -t "${SESSION}" session-closed "unbind-key -T root C-c"

  echo "[launch_sim] started tmux session='${SESSION}'"
  echo "[launch_sim] follower tracker: ${weight_path}"
  echo "[launch_sim] front no-tracker target: ${target_weight_path}"
  echo "[launch_sim] Ctrl+C in this tmux session will stop all windows."
  echo "[launch_sim] manual stop: tools/single_launch.sh --session ${SESSION} --stop"

  if [[ "${DETACH}" -eq 1 ]]; then
    echo "[launch_sim] detached mode; attach with: tmux attach -t ${SESSION}"
    exit 0
  fi

  exec tmux attach -t "${SESSION}"
}

main "$@"
