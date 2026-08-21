#!/usr/bin/env bash
set -Eeuo pipefail

MODE="full"
DATASET="/workspace/YOPO/dataset_minco"
ENV_NUM=10
IMAGE_NUM=10000
EPOCHS=50
BATCH_SIZE=16
NUM_WORKERS=8
SEED=0
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage (inside container): tools/run_minco_pipeline.sh [options]

Options:
  --mode generate|train|promote|evaluate|full  default: full
  --dataset PATH       default: /workspace/YOPO/dataset_minco
  --env-num N          default: 10
  --image-num N        images per environment, default: 10000
  --epochs N           default: 50
  --batch-size N       default: 16
  --num-workers N      default: 8
  --seed N             default: 0
  --overwrite          allow dataset generation to replace PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:?missing value}"; shift 2 ;;
    --dataset) DATASET="${2:?missing value}"; shift 2 ;;
    --env-num) ENV_NUM="${2:?missing value}"; shift 2 ;;
    --image-num) IMAGE_NUM="${2:?missing value}"; shift 2 ;;
    --epochs) EPOCHS="${2:?missing value}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?missing value}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:?missing value}"; shift 2 ;;
    --seed) SEED="${2:?missing value}"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -f /.dockerenv ]] || { echo "Run this pipeline inside the YOPO container." >&2; exit 1; }
case "$MODE" in generate|train|promote|evaluate|full) ;; *) echo "Invalid mode: $MODE" >&2; exit 2 ;; esac

generate() {
  set +u
  source /opt/ros/noetic/setup.bash
  cd /workspace/YOPO/Simulator
  source devel/setup.bash
  set -u
  args=(--save-path "$DATASET" --env-num "$ENV_NUM" --image-num "$IMAGE_NUM" --seed "$SEED")
  [[ "$OVERWRITE" -eq 1 ]] && args+=(--overwrite)
  rosrun sensor_simulator dataset_generator "${args[@]}"
}

train() {
  source /opt/ros/noetic/setup.bash
  cd /workspace/YOPO/YOPO
  python3 train_yopo.py \
    --dataset-path "$DATASET" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --save-interval 10 \
    --seed "$SEED"
}

latest_run() {
  find /workspace/YOPO/YOPO/saved -maxdepth 1 -mindepth 1 -type d -name 'YOPO_[0-9]*' \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
}

promote() {
  run_dir=$(latest_run)
  [[ -n "$run_dir" ]] || { echo "No YOPO_<n> training run found." >&2; exit 1; }
  checkpoint="$run_dir/epoch${EPOCHS}.pth"
  [[ -f "$checkpoint" ]] || { echo "Missing final checkpoint: $checkpoint" >&2; exit 1; }
  event=$(find "$run_dir" -maxdepth 1 -type f -name '*tfevents*' -printf '%T@ %p\n' |
    sort -nr | head -1 | cut -d' ' -f2-)
  [[ -n "$event" ]] || { echo "No TensorBoard event in $run_dir" >&2; exit 1; }
  install -m 0644 "$checkpoint" "/workspace/YOPO/YOPO/saved/yopo-minco/epoch${EPOCHS}-retrained.pth"
  install -m 0644 "$event" "/workspace/YOPO/YOPO/saved/yopo-minco/events.out.tfevents.minco.latest"
  echo "Promoted checkpoint and event from $run_dir"
}

evaluate() {
  cd /workspace/YOPO/YOPO
  retrained="saved/yopo-minco/epoch${EPOCHS}-retrained.pth"
  [[ -f "$retrained" ]] || { echo "Missing promoted checkpoint: $retrained" >&2; exit 1; }
  python3 evaluate_minco.py \
    "official=saved/yopo-minco/epoch50.pth" \
    "retrained=$retrained" \
    --dataset-path "$DATASET" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --output ../results/minco_checkpoint_eval.json
}

case "$MODE" in
  generate) generate ;;
  train) train ;;
  promote) promote ;;
  evaluate) evaluate ;;
  full) generate; train; promote; evaluate ;;
esac
