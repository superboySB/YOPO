#!/usr/bin/env bash
set -Eeuo pipefail

TRIALS=3
YOPO_TRIAL=1
YOPO_EPOCH=50
UAV_NUM=4
TIMEOUT=140
RADIUS=16
ALTITUDE=2
SWARM_TANGENT_BIAS=0
SWARM_BIAS_RADIUS=8
OUT_DIR=""
YOPO_CONFIG="/workspace/YOPO/YOPO/config/swarm_traj_opt.yaml"
WEIGHTS_ROOT="saved"

usage() {
  cat <<'EOF'
Usage:
  tools/swarm_run_trials.sh [--trials N] [--uav-num N] [--timeout SEC] [--trial N] [--epoch N] [--swarm-tangent-bias B] [--swarm-bias-radius R] [--yopo-config PATH] [--weights-root DIR]

Options:
  --trials N     number of repeated swarm runs (default: 3)
  --uav-num N    number of UAVs (default: 4)
  --timeout SEC  monitor timeout per run (default: 140)
  --trial N      YOPO weight trial id (default: 1)
  --epoch N      YOPO weight epoch id (default: 50)
  --radius R     ring radius in meters (default: 16)
  --altitude Z   flight altitude in meters (default: 2)
  --swarm-tangent-bias B  tangential goal bias near center (default: 0)
  --swarm-bias-radius R   distance-to-center activation range for tangential bias (default: 8)
  --yopo-config PATH      YOPO config yaml (default: /workspace/YOPO/YOPO/config/swarm_traj_opt.yaml)
  --weights-root DIR      checkpoint root under YOPO/ (default: saved)
  --out-dir DIR  output directory for json results
  -h, --help     show help
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --trials)
        TRIALS="${2:-}"
        shift 2
        ;;
      --uav-num)
        UAV_NUM="${2:-}"
        shift 2
        ;;
      --timeout)
        TIMEOUT="${2:-}"
        shift 2
        ;;
      --trial)
        YOPO_TRIAL="${2:-}"
        shift 2
        ;;
      --epoch)
        YOPO_EPOCH="${2:-}"
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
      --out-dir)
        OUT_DIR="${2:-}"
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

wait_for_ros_master() {
  local deadline=$((SECONDS + 30))
  while true; do
    if source /opt/ros/noetic/setup.bash >/dev/null 2>&1 && rosnode list >/dev/null 2>&1; then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "[run_swarm_trials] ros master did not become ready within 30s" >&2
      return 1
    fi
    sleep 1
  done
}

main() {
  parse_args "$@"

  if [[ ! -f "${YOPO_CONFIG}" ]]; then
    echo "[run_swarm_trials] YOPO config not found: ${YOPO_CONFIG}" >&2
    exit 1
  fi

  if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="/tmp/yopo-swarm-results-$(date +%Y%m%d-%H%M%S)"
  fi
  mkdir -p "${OUT_DIR}"

  for run_idx in $(seq 1 "${TRIALS}"); do
    session="yopo-swarm-${run_idx}"
    result_path="${OUT_DIR}/run_${run_idx}.json"

    echo "[run_swarm_trials] starting run ${run_idx}/${TRIALS}"
    ./tools/swarm_launch.sh --session "${session}" --stop >/dev/null 2>&1 || true
    sleep 1
    ./tools/swarm_launch.sh \
      --trial "${YOPO_TRIAL}" \
      --epoch "${YOPO_EPOCH}" \
      --uav-num "${UAV_NUM}" \
      --radius "${RADIUS}" \
      --altitude "${ALTITUDE}" \
      --swarm-tangent-bias "${SWARM_TANGENT_BIAS}" \
      --swarm-bias-radius "${SWARM_BIAS_RADIUS}" \
      --yopo-config "${YOPO_CONFIG}" \
      --weights-root "${WEIGHTS_ROOT}" \
      --session "${session}" \
      --detach

    wait_for_ros_master

    set +e
    python3 ./tools/swarm_run_monitor.py --uav-num "${UAV_NUM}" --timeout "${TIMEOUT}" > "${result_path}"
    monitor_status=$?
    set -e

    python3 - "${result_path}" "${run_idx}" "${monitor_status}" <<'PY'
import json
import sys

path = sys.argv[1]
run_idx = sys.argv[2]
status = int(sys.argv[3])
with open(path, "r", encoding="utf-8") as f:
    raw = f.read()
json_start = raw.find("{")
if json_start < 0:
    raise SystemExit(f"[run_swarm_trials] run {run_idx}: missing json payload in {path}")
data = json.loads(raw[json_start:])
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
print(f"[run_swarm_trials] run {run_idx}: success={status == 0}, all_arrived={data['all_arrived']}, "
      f"uav_collision_total={data['uav_collision_total']}, occupied_collision_total={data['occupied_collision_total']}, "
      f"elapsed_sec={data['elapsed_sec']}")
PY

    ./tools/swarm_launch.sh --session "${session}" --stop
    sleep 2
  done

  python3 - "${OUT_DIR}" <<'PY'
import glob
import json
import os
import statistics
import sys

out_dir = sys.argv[1]
paths = sorted(glob.glob(os.path.join(out_dir, "run_*.json")))
results = []
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        results.append(json.load(f))

successes = [r for r in results if r["all_arrived"] and r["uav_collision_total"] == 0]
elapsed = [r["elapsed_sec"] for r in successes]
summary = {
    "out_dir": out_dir,
    "runs": len(results),
    "successful_runs": len(successes),
    "success_rate": 0.0 if not results else len(successes) / len(results),
    "mean_success_elapsed_sec": None if not elapsed else round(statistics.mean(elapsed), 3),
}
print("[run_swarm_trials] summary")
print(json.dumps(summary, indent=2))
with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
PY
}

main "$@"
