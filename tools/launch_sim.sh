#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="yopo_sim"
WEIGHT="${ROOT_DIR}/YOPO/saved/YOPO_0/epoch200.pth"
PYTHON_BIN="python3"
GPU_ID="0"
VELOCITY="6.0"
MAX_DEPTH="4"
DEPTH_NORMALIZED="0"
ARRIVE_DIST="1.0"
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
JOYSTICK_DEVICE="/dev/input/js0"
JOYSTICK_AUTO="1"
JOYSTICK_AXIS_X="0"
JOYSTICK_AXIS_Y="1"
JOYSTICK_AXIS_Z="2"
JOYSTICK_AXIS_YAW="3"
JOYSTICK_AXIS_MAX="32767"
JOYSTICK_DEADZONE="0.08"
JOYSTICK_INVERT_X="1"
JOYSTICK_INVERT_Y="0"
JOYSTICK_INVERT_Z="0"
JOYSTICK_INVERT_YAW="1"
JOYSTICK_SWAP_XY="1"
JOYSTICK_VERTICAL_VELOCITY="2.0"
JOYSTICK_YAW_RATE="1.0"
JOYSTICK_CALIBRATE="0"
CONTROL_MODE="nav_goal"
CONTROL_HINT=""
START_SENSOR="1"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --weight PATH       YOPO-Omni checkpoint path. Default: ${WEIGHT}
  --python PATH       Python executable. Default: ${PYTHON_BIN}
  --gpu ID            CUDA_VISIBLE_DEVICES value. Default: ${GPU_ID}
  --velocity VALUE    Desired speed for planner. Default: ${VELOCITY}
  --max-depth VALUE   ToF depth normalization max range. Default: ${MAX_DEPTH}
  --depth-normalized  Treat 32FC1 depth as already normalized [0,1] instead of meters.
  --arrive-dist VALUE Goal arrival distance threshold in meters. Default: ${ARRIVE_DIST}
  --radius-min VALUE  Override omni_radius_min for checkpoint-consistent decoding.
  --radius-max VALUE  Override omni_radius_max for checkpoint-consistent decoding.
  --radio-range VALUE Override radio_range and recompute sgm_time unless --sgm-time is also set.
  --sgm-time VALUE    Override trajectory segment time. Must match training for fair tests.
  --fixed-yaw         Keep yaw fixed instead of turning toward the goal.
  --fixed-yaw-value R Fixed yaw in radians. If omitted with --fixed-yaw, lock to initial odometry yaw.
  --joystick-device P Linux joystick device for auto manual-assist mode. Default: ${JOYSTICK_DEVICE}
  --joystick-axis-x N Right-stick horizontal axis. Default: ${JOYSTICK_AXIS_X}
  --joystick-axis-y N Right-stick vertical axis. Default: ${JOYSTICK_AXIS_Y}
  --joystick-axis-z N Left-stick vertical climb/descent axis. Default: ${JOYSTICK_AXIS_Z}
  --joystick-axis-yaw N Left-stick horizontal yaw axis. Default: ${JOYSTICK_AXIS_YAW}
  --joystick-axis-max V Absolute raw axis maximum. Default: ${JOYSTICK_AXIS_MAX}
  --joystick-deadzone V Radial deadzone in [0,1). Default: ${JOYSTICK_DEADZONE}
  --joystick-invert-x 0|1 Invert horizontal axis. Default: ${JOYSTICK_INVERT_X}
  --joystick-invert-y 0|1 Invert vertical axis. Default: ${JOYSTICK_INVERT_Y}
  --joystick-invert-z 0|1 Invert climb/descent axis. Default: ${JOYSTICK_INVERT_Z}
  --joystick-invert-yaw 0|1 Invert yaw axis. Default: ${JOYSTICK_INVERT_YAW}
  --joystick-swap-xy 0|1 Map vertical/horizontal to body x/y. Default: ${JOYSTICK_SWAP_XY}
  --joystick-vertical-velocity V Maximum climb/descent speed in m/s. Default: ${JOYSTICK_VERTICAL_VELOCITY}
  --joystick-yaw-rate V Maximum yaw rate in rad/s. Default: ${JOYSTICK_YAW_RATE}
  --joystick-calibrate Print raw axes and mapped body velocity while running.
  --no-joystick       Disable joystick auto-detection and keep RViz 2D Nav Goal control.
  --no-sensor         Do not start sensor_simulator_cuda (when using an external depth source).
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
    --depth-normalized)
      DEPTH_NORMALIZED="1"
      shift
      ;;
    --arrive-dist)
      ARRIVE_DIST="$2"
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
    --joystick-device)
      JOYSTICK_DEVICE="$2"
      shift 2
      ;;
    --joystick-axis-x)
      JOYSTICK_AXIS_X="$2"
      shift 2
      ;;
    --joystick-axis-y)
      JOYSTICK_AXIS_Y="$2"
      shift 2
      ;;
    --joystick-axis-z)
      JOYSTICK_AXIS_Z="$2"
      shift 2
      ;;
    --joystick-axis-yaw)
      JOYSTICK_AXIS_YAW="$2"
      shift 2
      ;;
    --joystick-axis-max)
      JOYSTICK_AXIS_MAX="$2"
      shift 2
      ;;
    --joystick-deadzone)
      JOYSTICK_DEADZONE="$2"
      shift 2
      ;;
    --joystick-invert-x)
      JOYSTICK_INVERT_X="$2"
      shift 2
      ;;
    --joystick-invert-y)
      JOYSTICK_INVERT_Y="$2"
      shift 2
      ;;
    --joystick-invert-z)
      JOYSTICK_INVERT_Z="$2"
      shift 2
      ;;
    --joystick-invert-yaw)
      JOYSTICK_INVERT_YAW="$2"
      shift 2
      ;;
    --joystick-swap-xy)
      JOYSTICK_SWAP_XY="$2"
      shift 2
      ;;
    --joystick-vertical-velocity)
      JOYSTICK_VERTICAL_VELOCITY="$2"
      shift 2
      ;;
    --joystick-yaw-rate)
      JOYSTICK_YAW_RATE="$2"
      shift 2
      ;;
    --joystick-calibrate)
      JOYSTICK_CALIBRATE="1"
      shift
      ;;
    --no-joystick)
      JOYSTICK_AUTO="0"
      shift
      ;;
    --no-sensor)
      START_SENSOR="0"
      shift
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

if [[ "${JOYSTICK_AUTO}" == "1" ]]; then
  if [[ ! -e "${JOYSTICK_DEVICE}" ]]; then
    echo "Joystick is required but was not found: ${JOYSTICK_DEVICE}" >&2
    echo "Connect the joystick, choose another path with --joystick-device, or explicitly use --no-joystick for RViz goal mode." >&2
    exit 1
  fi
  CONTROL_MODE="joystick"
  CONTROL_HINT="[control] ${JOYSTICK_DEVICE}: right axes ${JOYSTICK_AXIS_X}/${JOYSTICK_AXIS_Y}=YOPO forward/left; left axis ${JOYSTICK_AXIS_Z}=closed-loop climb, ${JOYSTICK_AXIS_YAW}=yaw. Center switches READY to a continuous EMPTY braking reference, then hover."
else
  CONTROL_MODE="nav_goal"
  CONTROL_HINT="[control] joystick disabled explicitly: using original RViz 2D Nav Goal control."
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
PLANNER_CMD="cd ${ROOT_DIR} && ${ROS_SETUP} && ${CTRL_SETUP} && ${SIM_SETUP} && PYTHONPATH=${YOPO_PYTHONPATH} CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} YOPO/test_yopo_ros.py --weight ${WEIGHT} --velocity ${VELOCITY} --max-depth ${MAX_DEPTH} --arrive-dist ${ARRIVE_DIST} --visualize ${VISUALIZE}"
if [[ "${DEPTH_NORMALIZED}" == "1" ]]; then
  PLANNER_CMD="${PLANNER_CMD} --depth-normalized"
fi
if [[ "${CONTROL_MODE}" == "joystick" ]]; then
  PLANNER_CMD="${PLANNER_CMD} --control-mode joystick --joystick-device ${JOYSTICK_DEVICE} --joystick-axis-x ${JOYSTICK_AXIS_X} --joystick-axis-y ${JOYSTICK_AXIS_Y} --joystick-axis-z ${JOYSTICK_AXIS_Z} --joystick-axis-yaw ${JOYSTICK_AXIS_YAW} --joystick-axis-max ${JOYSTICK_AXIS_MAX} --joystick-deadzone ${JOYSTICK_DEADZONE} --joystick-invert-x ${JOYSTICK_INVERT_X} --joystick-invert-y ${JOYSTICK_INVERT_Y} --joystick-invert-z ${JOYSTICK_INVERT_Z} --joystick-invert-yaw ${JOYSTICK_INVERT_YAW} --joystick-swap-xy ${JOYSTICK_SWAP_XY} --joystick-vertical-velocity ${JOYSTICK_VERTICAL_VELOCITY} --joystick-yaw-rate ${JOYSTICK_YAW_RATE} --joystick-calibrate ${JOYSTICK_CALIBRATE}"
fi
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
if [[ "${START_SENSOR}" == "1" ]]; then
  tmux new-window -t "${SESSION}" -n sensor "bash -lc 'sleep 7; ${SIM_CMD}'"
fi
tmux new-window -t "${SESSION}" -n planner "bash -lc 'sleep 11; echo \"${CONTROL_HINT}\"; ${PLANNER_CMD}'"

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

Depth topics:
  /depth_image_front
  /depth_image_left
  /depth_image_right
  /depth_image_back

Sensor simulator:
  $([[ "${START_SENSOR}" == "1" ]] && echo "enabled" || echo "disabled; an external/fixture depth publisher is required")

Lidar point cloud:
  /lidar_points

Control:
  ${CONTROL_HINT}
  horizontal_max=${VELOCITY}m/s vertical_max=${JOYSTICK_VERTICAL_VELOCITY}m/s yaw_rate_max=${JOYSTICK_YAW_RATE}rad/s
  direct_model=1 planner_candidate_veto=0 altitude_rewrite=0
EOF

if [[ "${START_RVIZ}" == "1" ]]; then
  cat <<EOF
RViz:
  RViz is started as an X11 GUI window on DISPLAY=${DISPLAY:-:0}.
  To inspect RViz logs: tmux attach -t ${SESSION}, then switch to the rviz window.
  If no GUI window appears, run xhost on the host machine and start this script from a graphical/VNC session.
EOF
fi
