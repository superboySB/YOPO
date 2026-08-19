#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="yopo_sim"
WEIGHT="${ROOT_DIR}/YOPO/saved/YOPO_Active_Final/epoch50.pth"
PYTHON_BIN="python3"
GPU_ID="0"
VELOCITY="3.0"
MAX_DEPTH="20"
ARRIVE_DIST="2.0"
GOAL_HEIGHT="2.0"
RADIUS_MIN=""
RADIUS_MAX=""
RADIO_RANGE=""
SGM_TIME=""
VISUALIZE="1"
START_RVIZ="1"
RVIZ_SOFTWARE_GL="0"
RVIZ_OPENGL="210"
FIXED_YAW="0"
FIXED_YAW_VALUE=""

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --weight PATH       YOPO active-perception checkpoint path.
  --python PATH       Python executable. Default: ${PYTHON_BIN}
  --gpu ID            CUDA_VISIBLE_DEVICES value. Default: ${GPU_ID}
  --velocity VALUE    Desired speed for planner. Default: ${VELOCITY}
  --max-depth VALUE   Insight 9 depth normalization max range. Default: ${MAX_DEPTH}
  --arrive-dist VALUE Braking trigger distance in meters. Default: ${ARRIVE_DIST}
  --goal-height VALUE Fixed target altitude in meters. Default: ${GOAL_HEIGHT}
  --radius-min VALUE  Override omni_radius_min for checkpoint-consistent decoding.
  --radius-max VALUE  Override omni_radius_max for checkpoint-consistent decoding.
  --radio-range VALUE Override radio_range and recompute sgm_time unless --sgm-time is also set.
  --sgm-time VALUE    Override trajectory segment time. Must match training for fair tests.
  --fixed-yaw         Keep yaw fixed instead of turning toward the goal.
  --fixed-yaw-value R Fixed yaw in radians. If omitted with --fixed-yaw, lock to initial odometry yaw.
  --no-rviz           Do not start RViz.
  --rviz-software-gl  Start RViz with Mesa llvmpipe software OpenGL.
  --rviz-opengl VER   RViz OpenGL version. Default: ${RVIZ_OPENGL}; use 120 or 210 for compatibility.
  --session NAME      tmux session name. Default: ${SESSION}
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weight)
      WEIGHT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
      ;;
    --velocity)
      VELOCITY="$2"
      shift 2
      ;;
    --max-depth)
      MAX_DEPTH="$2"
      shift 2
      ;;
    --arrive-dist)
      ARRIVE_DIST="$2"
      shift 2
      ;;
    --goal-height)
      GOAL_HEIGHT="$2"
      shift 2
      ;;
    --radius-min)
      RADIUS_MIN="$2"
      shift 2
      ;;
    --radius-max)
      RADIUS_MAX="$2"
      shift 2
      ;;
    --radio-range)
      RADIO_RANGE="$2"
      shift 2
      ;;
    --sgm-time)
      SGM_TIME="$2"
      shift 2
      ;;
    --fixed-yaw)
      FIXED_YAW="1"
      shift
      ;;
    --fixed-yaw-value)
      FIXED_YAW="1"
      FIXED_YAW_VALUE="$2"
      shift 2
      ;;
    --no-rviz)
      START_RVIZ="0"
      shift
      ;;
    --rviz-software-gl)
      RVIZ_SOFTWARE_GL="1"
      shift
      ;;
    --rviz-opengl)
      RVIZ_OPENGL="$2"
      shift 2
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required. Install it or run the commands manually." >&2
  exit 1
fi

if [[ "${PYTHON_BIN}" == */* ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
  fi
elif ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found in PATH: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${WEIGHT}" ]]; then
  echo "Checkpoint not found: ${WEIGHT}" >&2
  exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  echo "Attach with: tmux attach -t ${SESSION}" >&2
  echo "Stop it with: tmux kill-session -t ${SESSION}" >&2
  exit 1
fi

ROS_SETUP="source /opt/ros/noetic/setup.bash"
CTRL_SETUP="source ${ROOT_DIR}/Controller/devel/setup.bash"
SIM_SETUP="source ${ROOT_DIR}/Simulator/devel/setup.bash"
YOPO_PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:${ROOT_DIR}/Controller/devel/lib/python3/dist-packages:${ROOT_DIR}/Simulator/devel/lib/python3/dist-packages:${ROOT_DIR}/YOPO"

ROSCORE_CMD="${ROS_SETUP} && roscore"
CTRL_CMD="cd ${ROOT_DIR}/Controller && ${ROS_SETUP} && ${CTRL_SETUP} && roslaunch so3_quadrotor_simulator simulator_attitude_control.launch"
SIM_CMD="cd ${ROOT_DIR}/Simulator && ${ROS_SETUP} && ${SIM_SETUP} && rosrun sensor_simulator sensor_simulator_cuda"
PLANNER_CMD="cd ${ROOT_DIR} && ${ROS_SETUP} && ${CTRL_SETUP} && ${SIM_SETUP} && PYTHONPATH=${YOPO_PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} YOPO/test_yopo_ros.py --weight ${WEIGHT} --velocity ${VELOCITY} --max-depth ${MAX_DEPTH} --arrive-dist ${ARRIVE_DIST} --goal-height ${GOAL_HEIGHT} --visualize ${VISUALIZE}"
if [[ -n "${RADIUS_MIN}" ]]; then
  PLANNER_CMD="${PLANNER_CMD} --radius-min ${RADIUS_MIN}"
fi
if [[ -n "${RADIUS_MAX}" ]]; then
  PLANNER_CMD="${PLANNER_CMD} --radius-max ${RADIUS_MAX}"
fi
if [[ -n "${RADIO_RANGE}" ]]; then
  PLANNER_CMD="${PLANNER_CMD} --radio-range ${RADIO_RANGE}"
fi
if [[ -n "${SGM_TIME}" ]]; then
  PLANNER_CMD="${PLANNER_CMD} --sgm-time ${SGM_TIME}"
fi
if [[ "${FIXED_YAW}" == "1" ]]; then
  PLANNER_CMD="${PLANNER_CMD} --fixed-yaw"
  if [[ -n "${FIXED_YAW_VALUE}" ]]; then
    PLANNER_CMD="${PLANNER_CMD} --fixed-yaw-value ${FIXED_YAW_VALUE}"
  fi
fi
RVIZ_ENV="QT_X11_NO_MITSHM=1"
if [[ "${RVIZ_SOFTWARE_GL}" == "1" ]]; then
  RVIZ_ENV="${RVIZ_ENV} __GLX_VENDOR_LIBRARY_NAME=mesa MESA_LOADER_DRIVER_OVERRIDE=llvmpipe GALLIUM_DRIVER=llvmpipe LIBGL_ALWAYS_SOFTWARE=1"
fi
RVIZ_CMD="cd ${ROOT_DIR} && ${ROS_SETUP} && export DISPLAY=${DISPLAY:-:0} && ${RVIZ_ENV} rviz --opengl ${RVIZ_OPENGL} --disable-anti-aliasing -d YOPO/yopo.rviz"

tmux new-session -d -s "${SESSION}" -n roscore "bash -lc '${ROSCORE_CMD}'"
tmux new-window -t "${SESSION}" -n controller "bash -lc 'sleep 3; ${CTRL_CMD}'"
tmux new-window -t "${SESSION}" -n sensor "bash -lc 'sleep 7; ${SIM_CMD}'"
tmux new-window -t "${SESSION}" -n planner "bash -lc 'sleep 11; ${PLANNER_CMD}'"

if [[ "${START_RVIZ}" == "1" ]]; then
  tmux new-window -t "${SESSION}" -n rviz "bash -lc 'sleep 14; ${RVIZ_CMD}; rc=\$?; echo; echo \"RViz exited with code \$rc\"; echo \"If this says could not connect to display, run: xhost +SI:localuser:\$(id -un)\"; exec bash'"
fi

tmux select-window -t "${SESSION}:planner"

cat <<EOF
Started tmux session: ${SESSION}

Attach:
  tmux attach -t ${SESSION}

Stop all:
  tmux kill-session -t ${SESSION}

Planner checkpoint:
  ${WEIGHT}

Insight 9 topics:
  /depth_image
  /yopo/camera/command
  /yopo/camera/orientation

Lidar point cloud:
  /lidar_points
EOF

if [[ "${START_RVIZ}" == "1" ]]; then
  cat <<EOF
RViz:
  RViz is started as an X11 GUI window on DISPLAY=${DISPLAY:-:0}.
  To inspect RViz logs: tmux attach -t ${SESSION}, then switch to the rviz window.
  If no GUI window appears, run xhost on the host machine and start this script from a graphical/VNC session.
EOF
fi
