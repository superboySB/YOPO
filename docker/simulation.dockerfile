FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Keep proxy config (override by --build-arg if needed)
ARG HTTP_PROXY=http://127.0.0.1:8889
ARG HTTPS_PROXY=http://127.0.0.1:8889
ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV no_proxy=localhost,127.0.0.1
ENV NO_PROXY=localhost,127.0.0.1
ARG APT_MAX_RETRIES=12
ARG APT_RETRY_SLEEP=15
ENV APT_MAX_RETRIES=${APT_MAX_RETRIES}
ENV APT_RETRY_SLEEP=${APT_RETRY_SLEEP}

# Setup basic packages
RUN printf 'Acquire::Retries "8";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' > /etc/apt/apt.conf.d/99-network-retry
RUN cat >/usr/local/bin/apt-install-retry <<'EOF' && chmod +x /usr/local/bin/apt-install-retry
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
  fi
  rc=$?
  echo "[apt-install-retry] failed with exit=${rc}, retrying..." >&2
  rm -rf /var/lib/apt/lists/*
  sleep "$(( sleep_base * i ))"
done
echo "[apt-install-retry] exhausted retries for: $*" >&2
exit 1
EOF

RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo ${TZ} > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    tzdata ca-certificates curl wget gnupg2 lsb-release software-properties-common

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential git git-lfs vim tmux unzip zip sudo pkg-config gedit

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    python3.8 python3.8-dev python3-pip python3-setuptools python3-wheel python3-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 100 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 100 && \
    python3 --version && python --version && \
    python3 -m pip install --no-cache-dir --upgrade \
    "pip<25" \
    "setuptools<75" \
    wheel \
    "importlib-metadata>=6.0,<8" \
    "zipp>=3.20"

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    libjpeg-dev libpng-dev libx11-dev libegl1-mesa-dev libglfw3-dev libglm-dev libomp-dev

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    libyaml-cpp-dev libopencv-dev libcgal-dev libompl-dev

# Install cmake
ARG CMAKE_VERSION=3.26.4
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh -O /tmp/cmake.sh && \
    mkdir -p /opt/cmake && \
    sh /tmp/cmake.sh --prefix=/opt/cmake --skip-license && \
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    cmake --version && \
    rm -f /tmp/cmake.sh

# -----------------------------------------------------
# ROS and relevant infra
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-install-retry \
    ros-noetic-desktop-full
RUN apt-install-retry \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool python3-catkin-tools && \
    (rosdep init || true) && \
    (rosdep update || true) && \
    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

# -----------------------------------------------------
# Simulation image keeps ROS + YOPO only (no PX4)
WORKDIR /workspace

# -----------------------------------------------------
# YOPO python dependencies (no conda/mamba/uv)
COPY docker/requirements.txt /tmp/yopo-requirements.txt
RUN python3 -m pip install --no-cache-dir --ignore-installed "PyYAML>=6.0.1,<7" && \
    python3 -m pip install --no-cache-dir --ignore-installed -r /tmp/yopo-requirements.txt && \
    python3 -m pip install --no-cache-dir --ignore-installed "empy==3.3.4" && \
    rm -f /tmp/yopo-requirements.txt

# RUN rm -rf /var/lib/apt/lists/* && apt-get clean
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG=quiet

# If proxy is not needed in your target machine, unset in runtime:
#   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

WORKDIR /workspace/YOPO
CMD ["/bin/bash"]
