#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: apt-install-retry <pkg1> [pkg2 ...]" >&2
  exit 2
fi

max_retries="${APT_MAX_RETRIES:-12}"
sleep_base="${APT_RETRY_SLEEP:-15}"
for i in $(seq 1 "$max_retries"); do
  echo "[apt-install-retry] attempt ${i}/${max_retries}: $*"
  if apt-get -o Acquire::Retries=10 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 update && \
     DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=10 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 \
     install -y --no-install-recommends --fix-missing "$@"; then
    rm -rf /var/lib/apt/lists/*
    exit 0
  else
    rc=$?
  fi
  echo "[apt-install-retry] failed with exit=${rc}, retrying..." >&2
  rm -rf /var/lib/apt/lists/*
  sleep "$(( sleep_base * i ))"
done
echo "[apt-install-retry] exhausted retries for: $*" >&2
exit 1
