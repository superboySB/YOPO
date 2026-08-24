#!/usr/bin/env bash
set -Eeuo pipefail

MAZE_TYPES="1,2,5,7"
SEEDS="1,2,3,4,5"
SCENARIOS="straight,dynamic"
LANE_ORDERS="normal,swapped"
EXPERIMENTS="inner_topk3,jerk_topk3,corridor_calibrated"
OUTPUT_DIR="/workspace/YOPO/results/minco_ablation_raw"
SESSION="yopo-ablation"
WEIGHT="/workspace/YOPO/YOPO/saved/yopo-minco/epoch50.pth"
CALIBRATION_JSON="/workspace/YOPO/docs/report_assets/corridor_calibration.json"
VELOCITY="6.0"
SAFE_RADIUS="0.05"
ARRIVAL_RADIUS="1.0"
YAW_GOAL_WEIGHT="6.0"
TIMEOUT="20.0"

usage() {
  cat <<'EOF'
Usage (inside the retained YOPO container): tools/run_minco_ablation.sh [options]

Options:
  --experiments CSV     inner_topk3,jerk_topk3,corridor_calibrated
  --maze-types CSV      default: 1,2,5,7
  --seeds CSV           default: 1,2,3,4,5
  --scenarios CSV       default: straight,dynamic
  --lane-orders CSV     normal,swapped (default: both; counterbalances ROS lanes)
  --output-dir PATH     raw JSON root (ignored by git)
  --weight PATH         one checkpoint used by both lanes
  --calibration-json P  output of tools/calibrate_corridor.py
  --velocity MPS        common requested speed (default: 6.0)
  --safe-radius M       common corridor threshold (default: 0.05)
  --arrival-radius M    common goal radius (default: 1.0)
  --yaw-goal-weight X   common yaw blend scale (default: 6.0)
  --timeout SEC         episode timeout (default: 20)
  --session NAME        tmux session reused across episodes
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiments) EXPERIMENTS="${2:?missing value}"; shift 2 ;;
    --maze-types) MAZE_TYPES="${2:?missing value}"; shift 2 ;;
    --seeds) SEEDS="${2:?missing value}"; shift 2 ;;
    --scenarios) SCENARIOS="${2:?missing value}"; shift 2 ;;
    --lane-orders) LANE_ORDERS="${2:?missing value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?missing value}"; shift 2 ;;
    --weight) WEIGHT="${2:?missing value}"; shift 2 ;;
    --calibration-json) CALIBRATION_JSON="${2:?missing value}"; shift 2 ;;
    --velocity) VELOCITY="${2:?missing value}"; shift 2 ;;
    --safe-radius) SAFE_RADIUS="${2:?missing value}"; shift 2 ;;
    --arrival-radius) ARRIVAL_RADIUS="${2:?missing value}"; shift 2 ;;
    --yaw-goal-weight) YAW_GOAL_WEIGHT="${2:?missing value}"; shift 2 ;;
    --timeout) TIMEOUT="${2:?missing value}"; shift 2 ;;
    --session) SESSION="${2:?missing value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -f /.dockerenv ]] || { echo "Run this benchmark inside the YOPO container." >&2; exit 1; }
[[ -f "$WEIGHT" ]] || { echo "Missing checkpoint: $WEIGHT" >&2; exit 1; }
cd /workspace/YOPO
mkdir -p "$OUTPUT_DIR"

set +u
source /opt/ros/noetic/setup.bash
source /workspace/YOPO/Controller/devel/setup.bash
set -u

stop_session() {
  tools/launch_compare.sh --session "$SESSION" --stop >/dev/null 2>&1 || true
}
stop_on_signal() {
  stop_session
  exit 130
}
trap stop_session EXIT
trap stop_on_signal INT TERM

IFS=',' read -r -a experiment_values <<< "$EXPERIMENTS"
IFS=',' read -r -a maze_values <<< "$MAZE_TYPES"
IFS=',' read -r -a seed_values <<< "$SEEDS"
IFS=',' read -r -a scenario_values <<< "$SCENARIOS"
IFS=',' read -r -a lane_order_values <<< "$LANE_ORDERS"

calibrated_sigma=""
if [[ ",$EXPERIMENTS," == *",corridor_calibrated,"* ]]; then
  [[ -f "$CALIBRATION_JSON" ]] || {
    echo "Missing calibration JSON; run tools/calibrate_corridor.py first: $CALIBRATION_JSON" >&2
    exit 1
  }
  calibrated_sigma=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_sigma"])' "$CALIBRATION_JSON")
fi

total=$((${#experiment_values[@]} * ${#maze_values[@]} * ${#seed_values[@]} * ${#scenario_values[@]} * ${#lane_order_values[@]}))
index=0
for experiment in "${experiment_values[@]}"; do
  case "$experiment" in
    inner_topk3) right_topk=3; right_mode=inner; right_sigma=1.0 ;;
    jerk_topk3) right_topk=3; right_mode=jerk; right_sigma=1.0 ;;
    corridor_calibrated) right_topk=1; right_mode=none; right_sigma="$calibrated_sigma" ;;
    *) echo "Unsupported experiment: $experiment" >&2; exit 2 ;;
  esac
  experiment_dir="$OUTPUT_DIR/$experiment"
  mkdir -p "$experiment_dir"
  for maze_type in "${maze_values[@]}"; do
    for map_seed in "${seed_values[@]}"; do
      for scenario in "${scenario_values[@]}"; do
        for lane_order in "${lane_order_values[@]}"; do
          index=$((index + 1))
          case "$lane_order" in
            normal) suffix=""; left_label=minco-baseline; right_label="$experiment"
                    left_topk=1; left_mode=none; left_sigma=1.0
                    lane_right_topk="$right_topk"; lane_right_mode="$right_mode"; lane_right_sigma="$right_sigma" ;;
            swapped) suffix="_swapped"; left_label="$experiment"; right_label=minco-baseline
                     left_topk="$right_topk"; left_mode="$right_mode"; left_sigma="$right_sigma"
                     lane_right_topk=1; lane_right_mode=none; lane_right_sigma=1.0 ;;
            *) echo "Unsupported lane order: $lane_order" >&2; exit 2 ;;
          esac
          output="$experiment_dir/maze${maze_type}_seed${map_seed}_${scenario}${suffix}.json"
          if [[ -s "$output" ]]; then
            echo "[ablation $index/$total] skip existing: $output"
            continue
          fi
          echo "[ablation $index/$total] experiment=$experiment maze=$maze_type seed=$map_seed scenario=$scenario lane=$lane_order"
          stop_session
          tools/launch_compare.sh --headless --detach --session "$SESSION" \
            --maze-type "$maze_type" --map-seed "$map_seed" --simple-y 0 --minco-y 0 \
            --simple-policy minco --simple-weight "$WEIGHT" --minco-weight "$WEIGHT" \
            --simple-safe-radius "$SAFE_RADIUS" --safe-radius "$SAFE_RADIUS" \
            --simple-sigma "$left_sigma" --simple-topk "$left_topk" --simple-continuity "$left_mode" \
            --minco-sigma "$lane_right_sigma" --minco-topk "$lane_right_topk" --minco-continuity "$lane_right_mode" \
            --velocity "$VELOCITY" --arrival-radius "$ARRIVAL_RADIUS" \
            --yaw-goal-weight "$YAW_GOAL_WEIGHT"

          left_config=$(printf '{"policy":"minco","topk":%s,"continuity_mode":"%s","corridor_sigma":%s,"checkpoint":"%s","lane_order":"%s"}' "$left_topk" "$left_mode" "$left_sigma" "$WEIGHT" "$lane_order")
          right_config=$(printf '{"policy":"minco","topk":%s,"continuity_mode":"%s","corridor_sigma":%s,"checkpoint":"%s","lane_order":"%s"}' "$lane_right_topk" "$lane_right_mode" "$lane_right_sigma" "$WEIGHT" "$lane_order")
          set +e
          python3 tools/benchmark_episode.py \
            --output "$output" --maze-type "$maze_type" --map-seed "$map_seed" \
            --scenario "$scenario" --velocity "$VELOCITY" --safe-radius "$SAFE_RADIUS" \
            --arrival-radius "$ARRIVAL_RADIUS" --yaw-goal-weight "$YAW_GOAL_WEIGHT" --timeout "$TIMEOUT" \
            --left-label "$left_label" --right-label "$right_label" \
            --left-config "$left_config" --right-config "$right_config" --left-best-type marker
          status=$?
          set -e
          stop_session
          if [[ "$status" -ne 0 ]]; then
            echo "[ablation $index/$total] failed with status $status" >&2
          fi
        done
      done
    done
  done
done

trap - EXIT INT TERM
echo "[ablation] complete: $OUTPUT_DIR"
