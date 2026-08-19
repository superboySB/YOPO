# YOPO Active Perception 复现手册

本文对应分支 `active-perception`。系统保留 YOPO-Simple 的“给目标点后自主飞行”方式，删除手柄辅助驾驶，并让策略用一幅可转动 Looper Robotics Insight 9 深度图同时输出：

1. 8 条候选五次多项式轨迹的末端状态与代价；
2. 与选中轨迹绑定的二自由度相机目标角（pitch/yaw）。

规划器在收到 `/move_base_simple/goal` 前只悬停，不会再沿历史默认目标自行起飞。

## 1. 已实现架构

```text
Insight 9 Z16/仿真 32FC1 depth
          │  单帧 544×640，运行前缩放为 160×192
          ▼
历史 ResNet-18 backbone（恢复自 ToFSense-M 切换前代码）
          │  带二维位置编码的空间 token
          ▼
Transformer decoder ◄── UAV v/a/v_des + 当前相机 pitch/yaw（11维）
          │
          ├── 8 × [末端 p/v/a + score]
          └── 8 × [camera pitch/yaw]
                         │
                         ▼
             一阶舵机 + 限速模型 / 实际云台
```

恢复的历史 backbone 位于 `YOPO/policy/models/`。原 ResNet 在提交 `79899a3`（`switch to tofsense-m`）中因 8×8 ToF 输入而被移除；当前高维深度图重新使用 ResNet-18，再将 CNN 空间特征送入 Transformer。

相机模型参考 `/home/dzp/projects/active-perception-RL-navigation/note.md`：pitch/yaw 两轴、一阶时间常数 0.25 s、最大角速度 120 deg/s、pitch ±60 deg、yaw ±45 deg。区别是这里没有 RL action/rollout；监督来自 YOPO 地图、深度图、ESDF 代价、A* 引导路径和解析式 gaze target。

YOPO-Simple 固定高度任务继续使用 RViz `2D Nav Goal`。网络仍学习三维候选轨迹，但在线控制把选中轨迹的终端高度约束到目标高度，以消除高频重规划的垂直漂移；训练中也加入了世界系高度稳定损失。

## 2. Insight 9 参数依据

官方资料：

- 产品页：<https://looper-robotics.com/home/product/insight-9/>
- 产品手册 v3.0：<https://prod-us-sv-alicloud-looper-robotics-deepmirror-s3.oss-us-west-1.aliyuncs.com/web/products/pdf/v3.0_Insight%209_EN_20260715.pdf>
- Linux SDK：<https://github.com/LooperRobotics/insight-sdk>
- Docker 固定 SDK commit：`afddfacde54323eb3484136e82ced189c9ee90ab`

代码采用的参数如下：

| 项目 | 值 |
|---|---:|
| 深度输出 | Z16，最大 544×640 |
| 深度帧率 | 最大 15 Hz |
| 对角/水平/垂直 FoV | 157.2° / 96.8° / 115.6° |
| 最近深度 | 约 0.19 m |
| 理想量程 | 0.3–30 m |
| 训练/在线截断 | 20 m |
| 3 m 深度精度 | 小于 2% |
| 尺寸 | 129×33.9×35 mm |
| 重量 | 176 g |
| 接口 | USB-C 3.1 |
| VIO / IMU | 最高 100 Hz / 400 Hz，BMI088，±24 g |

仿真深度会应用最近/最远量程、随距离增长且不超过 2% 的高斯误差以及毫米量化。网络分辨率为 160×192，保持 544×640 的宽高比与光学 FoV，显著高于旧 8×8 ToF。

## 3. 分支、Docker 与编译

宿主机：

```bash
cd /home/dzp/projects/YOPO
git switch active-perception

docker build --network=host \
  -f docker/simulation.dockerfile \
  -t dzp_yopo:active-perception-u2004-noetic-py38 .
```

允许 RViz 使用 X11，并启动容器：

```bash
xhost +SI:localuser:root
HOST_XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
test -f "$HOST_XAUTHORITY"

docker run --name dzp-yopo-active -itd \
  --privileged \
  --gpus all \
  --network host \
  --entrypoint bash \
  -e DISPLAY \
  -e XAUTHORITY=/root/.Xauthority \
  -e QT_X11_NO_MITSHM=1 \
  -v "$HOST_XAUTHORITY:/root/.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev:/dev \
  --shm-size=4g \
  -v /home/dzp/projects/YOPO:/workspace/YOPO \
  dzp_yopo:active-perception-u2004-noetic-py38

docker exec -it dzp-yopo-active bash
```

容器内编译：

```bash
cd /workspace/YOPO/Controller
source /opt/ros/noetic/setup.bash
catkin_make -j2

cd /workspace/YOPO/Simulator
source /opt/ros/noetic/setup.bash
catkin_make -j2

# 新镜像内应同时存在仿真器和真实相机桥
test -x devel/lib/sensor_simulator/sensor_simulator_cuda
test -x devel/lib/sensor_simulator/dataset_generator
test -x devel/lib/sensor_simulator/insight9_ros_bridge
```

## 4. 数据采集

默认正式配置是 10 张地图、每图 10,000 个相机位姿、每个位姿 8 个 desired direction，即 100,000 幅深度图和 800,000 条方向监督。

注意：生成器会重建 `--save-path` 指向的数据目录；若同名数据需要保留，先备份。

快速冒烟采集（不会覆盖正式目录）：

```bash
cd /workspace/YOPO
source /opt/ros/noetic/setup.bash
python3 tools/run_yopo_pipeline.py \
  --mode generate \
  --env-num 2 \
  --image-num 10 \
  --save-path ../dataset_active_smoke
```

正式采集：

```bash
cd /workspace/YOPO
source /opt/ros/noetic/setup.bash
python3 tools/run_yopo_pipeline.py \
  --mode generate \
  --env-num 10 \
  --image-num 10000 \
  --save-path ../dataset_active_v2
```

生成结构：

```text
dataset_active_v2/
  0/img_0_depth.png       # 单个可转动 Insight 9 视角，16-bit 归一化深度
  pose-0.csv              # UAV 世界位姿
  samples-0.csv           # 方向、目标、guide 索引、相机当前角/目标角
  guides-0.csv            # A* 专家路径点
  pointcloud-0.ply        # 训练安全损失使用的地图
  ...
```

`samples-*.csv` 为 25 列 active-perception schema。生成器会把期望速度和 A* 近场方向正确变换到包含 roll/pitch/yaw 的完整机体系；相机目标指向引导路径上约 4 m 的前视点，并受云台角度限制。

本次交付验收使用了较省时但仍覆盖 10 张独立地图的数据集：每图 500 位姿，共 5,000 幅深度图、40,000 条方向监督。生成目录为 `dataset_active_v2/`，约 229 MB；数据目录被 `.gitignore` 排除，但保留在本机工作区。

## 5. 训练至少 50 epoch

从零训练 50 epoch：

```bash
cd /workspace/YOPO
PYTHONPATH=/workspace/YOPO/YOPO \
python3 YOPO/train_yopo.py \
  --train-epoch 50 \
  --batch-size 16 \
  --run-name YOPO_Active_Repro
```

也可由统一流水线启动：

```bash
python3 tools/run_yopo_pipeline.py \
  --mode train \
  --dataset-path ../dataset_active_v2 \
  --train-epoch 50 \
  --batch-size 16 \
  --run-name YOPO_Active_Repro
```

从某个 checkpoint 继续训练：

```bash
python3 YOPO/train_yopo.py \
  --checkpoint YOPO/saved/YOPO_Active_Final/epoch50.pth \
  --train-epoch 50 \
  --run-name YOPO_Active_Finetune
```

TensorBoard：

```bash
tensorboard --logdir=/workspace/YOPO/YOPO/saved --bind_all
```

最终交付 checkpoint：

```text
YOPO/saved/YOPO_Active_Final/epoch50.pth
SHA-256: 9f5deb248a67e047edb1870a7b7a491f06a7e17d29eabf446a1635e9aaca8bf9
```

本次实际训练集/验证集为 4,500/500 个 pose（有效方向样本 36,000/4,000）。第 50 轮验证结果：总 loss 5.04、score loss 0.45、camera loss 0.0152。训练正常退出并保存 epoch10/20/30/40/50。

## 6. 离线模型验收

网络 contract（含 CUDA backward）：

```bash
cd /workspace/YOPO
PYTHONPATH=/workspace/YOPO/YOPO \
python3 tools/test_active_perception_contract.py --backward
```

预期关键输出：

```text
PASS device=cuda parameters=11,857,100
single=(2, 8, 9), pose=(2, 8, 8, 9), backward=True
```

checkpoint、数据读取和归一化：

```bash
PYTHONPATH=/workspace/YOPO/YOPO \
python3 tools/test_yopo_checkpoint.py \
  --weight YOPO/saved/YOPO_Active_Final/epoch50.pth \
  --num-batches 4 \
  --strict-depth-range
```

本次实测 32 个 pose / 256 个方向样本全部通过，输出形状分别为 `[B,D,8,9]`、`[B,D,8]`、`[B,D,8,2]`；RTX 4070 Ti SUPER 上每个 8-pose batch 平均约 28.85 ms。

## 7. 仿真、目标飞行和 RViz

一条命令启动 roscore、控制器、Insight 9 云台深度仿真、规划器和 RViz：

```bash
cd /workspace/YOPO
tools/launch_sim.sh \
  --weight /workspace/YOPO/YOPO/saved/YOPO_Active_Final/epoch50.pth \
  --velocity 3.0 \
  --max-depth 20 \
  --goal-height 2.0 \
  --rviz-software-gl
```

若只做无界面 CI：

```bash
tools/launch_sim.sh --no-rviz --session yopo_active_ci
```

查看/停止 tmux：

```bash
tmux attach -t yopo_sim
tmux kill-session -t yopo_sim
```

等待控制器输出 `TakeOff Done! Ready to Flight`。此时规划器日志应显示 `Waiting for /move_base_simple/goal`，无人机稳定悬停而不自行前飞。然后在 RViz 使用 `2D Nav Goal`，或发布一次目标：

```bash
source /opt/ros/noetic/setup.bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped "{
  header: {frame_id: world},
  pose: {
    position: {x: 0.0, y: 8.0, z: 2.0},
    orientation: {w: 1.0}
  }
}"
```

`z` 由 `--goal-height` 决定；RViz 的 2D goal 只使用 x/y。

核心话题：

```text
/depth_image                    sensor_msgs/Image，15 Hz
/sim/odom                       nav_msgs/Odometry
/move_base_simple/goal          geometry_msgs/PoseStamped
/yopo/camera/command            geometry_msgs/Vector3，目标 pitch/yaw(rad)
/yopo/camera/orientation        geometry_msgs/Vector3，实际 pitch/yaw(rad)
/yopo/best_traj_visual          sensor_msgs/PointCloud2
/yopo/trajs_visual              sensor_msgs/PointCloud2
/yopo/topology_endstates_visual sensor_msgs/PointCloud2
/yopo/active_camera_visual      visualization_msgs/MarkerArray
/yopo/collision_counter_total   std_msgs/Int32
```

检查频率和角度：

```bash
rostopic hz /depth_image /yopo/best_traj_visual
rostopic echo -n 1 /yopo/camera/command
rostopic echo -n 1 /yopo/camera/orientation
rostopic echo -n 1 /yopo/collision_counter_total
```

RViz 中：

- 蓝色扁盒是无人机；
- 橙色 129×33.9×35 mm 盒是 Insight 9；
- 橙色线框是 96.8°×115.6° FoV；
- 相机盒/FoV 使用云台实际 pitch/yaw，因此会与机体朝向实时不同；
- 彩色点云显示所有候选轨迹、选中轨迹及拓扑末端。

本次无界面闭环验收中，深度与轨迹均稳定约 15 Hz，单帧网络前向约 1.76–1.99 ms；发布 `(0,8)` 后在 2 m 制动触发距离切换悬停，最终停在 `(0.122,8.718,2.003)`，距目标约 0.73 m、速度收敛到约 `1.5e-8 m/s` 且碰撞计数保持 0。相机命令与实际角曾分别为 `(0.0461,-0.0589)` 和 `(0.0383,-0.0737)` rad，证明一阶/限速云台状态不是直接复制机体姿态。

## 8. 真实 Insight 9 与实体云台

新 Docker 镜像会编译安装固定版本的官方 Linux SDK。连接相机时必须将 `/dev/video*` 和 `/dev/hidraw*` 暴露给容器；上面的 `--privileged -v /dev:/dev` 已覆盖。先检查设备：

```bash
v4l2-ctl --list-devices
ls -l /dev/video* /dev/hidraw*
```

启动真实相机 ROS bridge：

```bash
cd /workspace/YOPO/Simulator
source /opt/ros/noetic/setup.bash
source devel/setup.bash
# 若整套 ROS 系统尚未启动 master，先在另一终端运行 roscore
rosrun sensor_simulator insight9_ros_bridge
```

它复制 SDK callback 缓冲区并发布：

```text
/depth_image   16UC1 Z16（毫米）
/insight9/imu  sensor_msgs/Imu
/insight9/vio  nav_msgs/Odometry
```

SDK 当前 header 定义 depth `cam_id=2`，部分 README 版本写作 3；bridge 同时接受 2/3，但强制要求像素格式为 Z16，避免把灰度流误当深度。

实体二轴云台适配器：

```bash
cd /workspace/YOPO
source /opt/ros/noetic/setup.bash
PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages \
python3 YOPO/insight9_gimbal_bridge.py \
  --pitch-topic /gimbal/pitch_position_controller/command \
  --yaw-topic /gimbal/yaw_position_controller/command \
  --joint-states-topic /joint_states \
  --pitch-joint insight9_pitch_joint \
  --yaw-joint insight9_yaw_joint
```

输出为 `std_msgs/Float64` 弧度，可 remap 到 ros_control 或飞控板的舵机驱动。若 `/joint_states` 有反馈，`/yopo/camera/orientation` 使用实测关节角；没有反馈时使用与仿真相同的 0.25 s 一阶、120 deg/s 限速估计。

真实飞行规划器可显式指定机上话题：

```bash
cd /workspace/YOPO
source /opt/ros/noetic/setup.bash
source Controller/devel/setup.bash
source Simulator/devel/setup.bash
PYTHONPATH=/workspace/YOPO/YOPO \
python3 YOPO/test_yopo_ros.py \
  --weight YOPO/saved/YOPO_Active_Final/epoch50.pth \
  --odom-topic /your_vehicle/odom \
  --depth-topic /depth_image \
  --ctrl-topic /your_controller/pos_cmd \
  --velocity 3.0 \
  --goal-height 2.0
```

真实飞行前必须确认 `quadrotor_msgs/PositionCommand` 与飞控桥的坐标系、单位、急停和 geofence。没有连接 Insight 9 与实体云台时，只能验证 SDK 编译/API、仿真闭环和话题协议，不能把无硬件环境的测试等同于实机标定或安全飞行认证。

## 9. 配置位置与常见问题

主要配置：

```text
Simulator/src/config/config.yaml  # 传感器、云台、地图与采集
YOPO/config/traj_opt.yaml         # 网络、损失、训练与轨迹
YOPO/yopo.rviz                    # 深度、无人机、相机/FoV、轨迹显示
```

常见问题：

- `Checkpoint not found`：显式传 `--weight`，或确认 `YOPO/saved/YOPO_Active_Final/epoch50.pth` 在本机；权重目录不提交 Git。
- RViz 无窗口：确认宿主机 `DISPLAY`、Xauthority 和 `xhost`；无桌面会话时用 `--no-rviz`。
- CUDA architecture 检测失败：按 CMake 输出在 `Simulator/src/CMakeLists.txt` 设置本机 `-gencode`。
- SDK 初始化失败：官方 SDK 需要足够的 UVC/HID 设备，检查 USB 3.1、供电、`/dev/video*`、`/dev/hidraw*` 和容器权限。
- 深度异常：真实相机必须发布 `16UC1` 毫米 Z16；仿真为 `32FC1` 米。规划器会统一缩放并截断到 20 m。
- 启动后无人机不动：这是预期安全行为；先等待起飞完成，再发布 `/move_base_simple/goal`。
- 相机不动：检查 command/orientation 两个话题，以及实体 bridge 的 pitch/yaw controller topic 和 joint 名称。

## 10. 本次交付验收摘要（2026-08-19）

| 项目 | 结果 |
|---|---|
| active-perception 分支 | 通过 |
| Docker 完整构建、`pip check`、CUDA 运行 | 通过；镜像 `dzp_yopo:active-perception-u2004-noetic-py38` |
| 历史 ResNet-18 恢复并接 Transformer | 通过 |
| 单 Insight 9 深度仿真、FoV/量程/噪声 | 通过 |
| 轨迹 + 相机联合输出与 backward | 通过 |
| 10 地图/5,000 pose/40,000 方向样本 | 通过 |
| 从零训练 50 epoch | 通过 |
| epoch50 离线 checkpoint 检查 | 通过 |
| 起飞等待目标、目标飞行、到达后定高 | 通过 |
| 15 Hz 深度/轨迹、动态相机角与 RViz marker | 通过 |
| Controller/Simulator 编译 | 通过 |
| 官方 SDK 安装、API 链接与 bridge 编译 | 通过；无设备启动按设计以 exit 2 给出明确诊断 |
| 实体 Insight 9/云台数据与实飞 | 当前主机无对应硬件，未冒充实机验证 |
