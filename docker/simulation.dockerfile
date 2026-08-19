FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Build and runtime proxy settings are injected by the Docker client config.
# Do not bake a host-specific loopback proxy into the image.
ARG APT_MAX_RETRIES=12
ARG APT_RETRY_SLEEP=15
ENV APT_MAX_RETRIES=${APT_MAX_RETRIES}
ENV APT_RETRY_SLEEP=${APT_RETRY_SLEEP}

# Setup basic packages
RUN printf 'Acquire::Retries "8";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' > /etc/apt/apt.conf.d/99-network-retry
COPY docker/apt-install-retry.sh /usr/local/bin/apt-install-retry
RUN chmod +x /usr/local/bin/apt-install-retry

RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo ${TZ} > /etc/timezone && \
    apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    tzdata ca-certificates curl wget gnupg2 lsb-release software-properties-common v4l-utils

RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential git git-lfs vim tmux unzip zip sudo pkg-config

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

# Official Looper Robotics Insight 9 Linux SDK, pinned for reproducibility.
# The ROS bridge is conditionally built when these headers/library are present.
ARG INSIGHT9_SDK_COMMIT=afddfacde54323eb3484136e82ced189c9ee90ab
RUN git clone --filter=blob:none --no-checkout https://github.com/LooperRobotics/insight-sdk.git /tmp/insight-sdk && \
    git -C /tmp/insight-sdk sparse-checkout init --cone && \
    git -C /tmp/insight-sdk sparse-checkout set Linux-SDK && \
    git -C /tmp/insight-sdk checkout "${INSIGHT9_SDK_COMMIT}" && \
    cmake -S /tmp/insight-sdk/Linux-SDK -B /tmp/insight-sdk/build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/insight-sdk/build --target insight9 -j"$(nproc)" && \
    cp -a /tmp/insight-sdk/build/libinsight9.so* /usr/local/lib/ && \
    install -m 0644 /tmp/insight-sdk/Linux-SDK/Insight_9_receive.h /usr/local/include/ && \
    install -m 0644 /tmp/insight-sdk/Linux-SDK/UvcExtensionUnit.hpp /usr/local/include/ && \
    ldconfig && \
    rm -rf /tmp/insight-sdk

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
COPY docker/download-pinned-wheels.py /usr/local/bin/download-pinned-wheels
ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/cu118/torch-2.4.1%2Bcu118-cp38-cp38-linux_x86_64.whl
RUN apt-install-retry aria2
RUN aria2c --console-log-level=warn --summary-interval=30 \
      --max-connection-per-server=16 --split=16 --min-split-size=1M \
      --file-allocation=none --dir=/tmp \
      --out=torch-2.4.1+cu118-cp38-cp38-linux_x86_64.whl \
      "${TORCH_WHEEL_URL}" && \
    python3 -m pip install --no-cache-dir --no-deps \
      /tmp/torch-2.4.1+cu118-cp38-cp38-linux_x86_64.whl && \
    rm -f /tmp/torch-2.4.1+cu118-cp38-cp38-linux_x86_64.whl
RUN python3 /usr/local/bin/download-pinned-wheels \
      --output-dir /tmp/torch-cuda-wheels \
      --aria-input /tmp/torch-cuda-wheels.txt && \
    aria2c --console-log-level=warn --summary-interval=30 \
      --max-concurrent-downloads=12 --max-connection-per-server=16 \
      --split=16 --min-split-size=1M --file-allocation=none \
      --input-file=/tmp/torch-cuda-wheels.txt && \
    python3 -m pip install --no-cache-dir --no-deps /tmp/torch-cuda-wheels/*.whl && \
    rm -rf /tmp/torch-cuda-wheels /tmp/torch-cuda-wheels.txt
RUN python3 -m pip install --no-cache-dir --ignore-installed "PyYAML>=6.0.1,<7" && \
    python3 -m pip install --no-cache-dir --ignore-installed -r /tmp/yopo-requirements.txt && \
    python3 -m pip install --no-cache-dir --ignore-installed "empy==3.3.4" && \
    rm -f /tmp/yopo-requirements.txt && \
    python3 -m pip check && \
    python3 -c "import torch; print(torch.__version__, torch.version.cuda)"

# RUN rm -rf /var/lib/apt/lists/* && apt-get clean
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG=quiet

WORKDIR /workspace/YOPO
CMD ["/bin/bash"]
