#!/usr/bin/env bash
set -Eeuo pipefail

MAZE_TYPES="1,2,5,7"
# std::default_random_engine maps seed 0 to the same initial state as seed 1
# in this simulator, so use five distinct maps by default.
SEEDS="1,2,3,4,5"
SCENARIOS="straight,dynamic"
OUTPUT_DIR="/workspace/YOPO/results/benchmark_raw"
SESSION="yopo-benchmark"
VELOCITY="6.0"
SAFE_RADIUS="0.05"
ARRIVAL_RADIUS="1.0"
YAW_GOAL_WEIGHT="6.0"
TIMEOUT="20.0"

usage() {
  cat <<'EOF'
Usage (inside the retained YOPO container): tools/run_comparison_benchmark.sh [options]

Options:
  --maze-types CSV       default: 1,2,5,7
  --seeds CSV            default: 1,2,3,4,5 (five distinct generated maps)
  --scenarios CSV        straight,dynamic (default: both)
  --output-dir PATH      raw JSON directory
  --velocity MPS         common requested speed (default: 6.0)
  --safe-radius M        MINCO conservative corridor threshold (default: 0.05)
  --arrival-radius M     common goal radius (default: 1.0)
  --yaw-goal-weight X    common yaw-to-goal blend scale (default: 6.0)
  --timeout SEC          episode timeout including pre-switch flight (default: 20)
  --session NAME         tmux session reused for each episode
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --maze-types) MAZE_TYPES="${2:?missing value}"; shift 2 ;;
    --seeds) SEEDS="${2:?missing value}"; shift 2 ;;
    --scenarios) SCENARIOS="${2:?missing value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?missing value}"; shift 2 ;;
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

IFS=',' read -r -a maze_values <<< "$MAZE_TYPES"
IFS=',' read -r -a seed_values <<< "$SEEDS"
IFS=',' read -r -a scenario_values <<< "$SCENARIOS"

total=$((${#maze_values[@]} * ${#seed_values[@]} * ${#scenario_values[@]}))
index=0
for maze_type in "${maze_values[@]}"; do
  for map_seed in "${seed_values[@]}"; do
    for scenario in "${scenario_values[@]}"; do
      index=$((index + 1))
      output="$OUTPUT_DIR/maze${maze_type}_seed${map_seed}_${scenario}.json"
      if [[ -s "$output" ]]; then
        echo "[benchmark $index/$total] skip existing: $output"
        continue
      fi

      echo "[benchmark $index/$total] maze=$maze_type seed=$map_seed scenario=$scenario"
      stop_session
      tools/launch_compare.sh --headless --detach --session "$SESSION" \
        --maze-type "$maze_type" --map-seed "$map_seed" \
        --simple-y 0 --minco-y 0 \
        --velocity "$VELOCITY" --safe-radius "$SAFE_RADIUS" \
        --arrival-radius "$ARRIVAL_RADIUS" --yaw-goal-weight "$YAW_GOAL_WEIGHT"

      set +e
      python3 tools/benchmark_episode.py \
        --output "$output" --maze-type "$maze_type" --map-seed "$map_seed" \
        --scenario "$scenario" --velocity "$VELOCITY" --safe-radius "$SAFE_RADIUS" \
        --arrival-radius "$ARRIVAL_RADIUS" --yaw-goal-weight "$YAW_GOAL_WEIGHT" --timeout "$TIMEOUT"
      status=$?
      set -e
      stop_session
      if [[ "$status" -ne 0 ]]; then
        echo "[benchmark $index/$total] failed with status $status" >&2
      fi
    done
  done
done

trap - EXIT INT TERM
echo "[benchmark] complete: $OUTPUT_DIR"
