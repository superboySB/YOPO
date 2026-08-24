#!/usr/bin/env bash
set -Eeuo pipefail

MAZE_TYPES="1,2,5,7"
SEEDS="1,2,3,4,5"
SCENARIOS="straight,dynamic"
LANE_ORDERS="normal,swapped"
EXPERIMENTS="geometry,duration,corridor,temporal"
OUTPUT_DIR="/workspace/YOPO/results/controlled_ablation_raw"
SESSION="yopo-controlled"
WEIGHT="/workspace/YOPO/YOPO/saved/yopo-minco/epoch50.pth"
VELOCITY="6.0"
SAFE_RADIUS="0.05"
ARRIVAL_RADIUS="1.0"
YAW_GOAL_WEIGHT="6.0"
TIMEOUT="20.0"

usage() {
  cat <<'EOF'
Usage (inside the retained YOPO container): tools/run_controlled_ablation.sh [options]

Experiments (variant minus baseline):
  geometry   minco_fixed minus single_fixed; corridor off, score top-1
  duration   minco_variable minus minco_fixed; corridor off, score top-1
  corridor   learned filter minus off; minco_variable, score top-1
  temporal   jerk top-3 minus score top-1; minco_variable, corridor off

Options:
  --experiments CSV     default: geometry,duration,corridor,temporal
  --maze-types CSV      default: 1,2,5,7
  --seeds CSV           default: 1,2,3,4,5
  --scenarios CSV       default: straight,dynamic
  --lane-orders CSV     default: normal,swapped
  --output-dir PATH     raw JSON root (ignored by git)
  --weight PATH         one checkpoint used by both lanes
  --velocity MPS        default: 6.0
  --safe-radius M       filter experiment threshold (default: 0.05)
  --arrival-radius M    default: 1.0
  --yaw-goal-weight X   default: 6.0
  --timeout SEC         default: 20
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

total=$((${#experiment_values[@]} * ${#maze_values[@]} * ${#seed_values[@]} * ${#scenario_values[@]} * ${#lane_order_values[@]}))
index=0
for experiment in "${experiment_values[@]}"; do
  case "$experiment" in
    geometry)
      baseline_label=single_fixed; baseline_traj=single_fixed; baseline_corridor=off; baseline_topk=1; baseline_continuity=none
      variant_label=minco_fixed; variant_traj=minco_fixed; variant_corridor=off; variant_topk=1; variant_continuity=none ;;
    duration)
      baseline_label=minco_fixed; baseline_traj=minco_fixed; baseline_corridor=off; baseline_topk=1; baseline_continuity=none
      variant_label=minco_variable; variant_traj=minco_variable; variant_corridor=off; variant_topk=1; variant_continuity=none ;;
    corridor)
      baseline_label=corridor_off; baseline_traj=minco_variable; baseline_corridor=off; baseline_topk=1; baseline_continuity=none
      variant_label=corridor_filter; variant_traj=minco_variable; variant_corridor=filter; variant_topk=1; variant_continuity=none ;;
    temporal)
      baseline_label=score_top1; baseline_traj=minco_variable; baseline_corridor=off; baseline_topk=1; baseline_continuity=none
      variant_label=jerk_top3; variant_traj=minco_variable; variant_corridor=off; variant_topk=3; variant_continuity=jerk ;;
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
            normal)
              suffix=""; left_label="$baseline_label"; right_label="$variant_label"
              left_traj="$baseline_traj"; right_traj="$variant_traj"
              left_corridor="$baseline_corridor"; right_corridor="$variant_corridor"
              left_topk="$baseline_topk"; right_topk="$variant_topk"
              left_continuity="$baseline_continuity"; right_continuity="$variant_continuity" ;;
            swapped)
              suffix="_swapped"; left_label="$variant_label"; right_label="$baseline_label"
              left_traj="$variant_traj"; right_traj="$baseline_traj"
              left_corridor="$variant_corridor"; right_corridor="$baseline_corridor"
              left_topk="$variant_topk"; right_topk="$baseline_topk"
              left_continuity="$variant_continuity"; right_continuity="$baseline_continuity" ;;
            *) echo "Unsupported lane order: $lane_order" >&2; exit 2 ;;
          esac
          output="$experiment_dir/maze${maze_type}_seed${map_seed}_${scenario}${suffix}.json"
          if [[ -s "$output" ]]; then
            echo "[controlled $index/$total] skip existing: $output"
            continue
          fi
          echo "[controlled $index/$total] experiment=$experiment maze=$maze_type seed=$map_seed scenario=$scenario lane=$lane_order"
          stop_session
          tools/launch_compare.sh --headless --detach --session "$SESSION" \
            --maze-type "$maze_type" --map-seed "$map_seed" --simple-y 0 --minco-y 0 \
            --simple-policy minco --simple-weight "$WEIGHT" --minco-weight "$WEIGHT" \
            --simple-trajectory "$left_traj" --minco-trajectory "$right_traj" \
            --simple-corridor "$left_corridor" --minco-corridor "$right_corridor" \
            --simple-safe-radius "$SAFE_RADIUS" --safe-radius "$SAFE_RADIUS" \
            --simple-sigma 1.0 --minco-sigma 1.0 \
            --simple-topk "$left_topk" --minco-topk "$right_topk" \
            --simple-continuity "$left_continuity" --minco-continuity "$right_continuity" \
            --velocity "$VELOCITY" --arrival-radius "$ARRIVAL_RADIUS" --yaw-goal-weight "$YAW_GOAL_WEIGHT"

          left_config=$(printf '{"policy":"minco","trajectory_mode":"%s","corridor_mode":"%s","topk":%s,"continuity_mode":"%s","checkpoint":"%s","lane_order":"%s"}' "$left_traj" "$left_corridor" "$left_topk" "$left_continuity" "$WEIGHT" "$lane_order")
          right_config=$(printf '{"policy":"minco","trajectory_mode":"%s","corridor_mode":"%s","topk":%s,"continuity_mode":"%s","checkpoint":"%s","lane_order":"%s"}' "$right_traj" "$right_corridor" "$right_topk" "$right_continuity" "$WEIGHT" "$lane_order")
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
            echo "[controlled $index/$total] failed with status $status" >&2
          fi
        done
      done
    done
  done
done

trap - EXIT INT TERM
echo "[controlled] complete: $OUTPUT_DIR"
