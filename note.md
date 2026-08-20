# YOPO 主动/固定相机复现手册

本文对应 `active-perception` 分支，说明 Insight 9 深度输入、`active_camera` 开关、
`maze_type=8` 盲角场景、配对数据生成、训练、推理和公平 A/B 评测的用法。实验数值、
图和逐次轨迹单独记录在
[`reports/active_camera_ablation/epoch100_chicane_20260820/REPORT.md`](reports/active_camera_ablation/epoch100_chicane_20260820/REPORT.md)。

系统在收到 `/move_base_simple/goal` 前保持悬停。策略使用一幅 Insight 9 深度图和
11 维状态，输出 8 组候选轨迹、代价及对应的相机 pitch/yaw 目标。

## 1. 深度图并非原图直喂

审计结论如下。

| 链路 | 传感器/原始尺寸 | 仿真或网络输入 | 处理 |
|---|---|---|---|
| `YOPO-Simple` | README 仅写通用 RealSense `480×270` | 仿真 `160×90`，网络 `160×96` | Dataset 和 ROS 均 `INTER_NEAREST` resize |
| 本分支 Insight 9 仿真/数据 | 名义 native `544×640` 只作传感器合同 | 直接 raycast `160(W)×192(H)` | 不先渲染 native 图，避免约 11.33 倍额外像素开销 |
| 本分支 Insight 9 实机 | SDK 实际发布的 Z16 尺寸，名义最大 `544×640` | `[B,1,192,160]` | 推理前 `INTER_NEAREST` resize 到 `160×192` |

`YOPO-Simple` 源码没有 D455 标识，只有旧的 `env: 435` 和通用 RealSense 说明，
因此不能把该分支断言为 D455。D455 的 `640×480` 参数和显式降维出现在参考的
`/home/dzp/projects/active-perception-RL-navigation` 实机代码中，而不是
`YOPO-Simple`。

Insight 9 的 `544/640=0.85`，网络的 `160/192=0.8333`；当前预处理保留完整 FoV，
但存在约 1.96% 的比例拉伸，不是严格等比例 letterbox。像素数由 348,160 降到
30,720，即保留 8.82%（约 11.33 倍降维）。仿真、离线 Dataset 和 ROS 推理统一使用
`resize_nearest_full_fov_v1` 合同、20 m 截断和相同 H/W；旧 checkpoint 若预处理合同
不匹配会被拒绝。

## 2. 场景与相机开关

主要开关位于 `Simulator/src/config/config.yaml` 和 `YOPO/config/traj_opt.yaml`：

```yaml
active_camera: true   # false 时相机状态、目标和实际云台均强制为 0
maze_type: 2          # 默认保持原柱子图；8 为主动感知盲角赛道
```

现有 `maze_type=1..7` 均保留，森林仍是 `maze_type=5`。新增的 `maze_type=8` 包含：

- 两个 90° S 形盲角和一个封闭 T 支路；
- 落地隔板、细杆和墙间横梁，而不是无支撑的随机浮空物体；
- 封闭起终点 chamber 和 5 m 顶棚，阻止从墙外或墙顶抄近路；
- seed 控制镜像和小幅几何扰动；训练位姿沿可通行主路线按弧长采样。

参考主动感知项目的默认环境不是树林，而是 3 个 panel、35 个随机 object 和边界墙；
大量物体会暂时位于固定相机视锥外。其策略还使用历史占据图、`21×21×21` local grid
和时序网络。当前 YOPO 仍是单帧策略，所以“相机可转”本身不保证碰撞率一定下降；
这一点必须通过 A/B 数据而不是预设结论。

`active_camera` 已贯穿四个阶段：

- 数据生成：active 采样当前/目标云台角；fixed 消耗同一 RNG 后把四列精确置零；
- 训练：两臂都使用相同 11 维 state、12 维 head 和 camera loss 权重；
- 仿真：active 执行一阶/限速云台；fixed 忽略命令并持续发布零状态；
- 推理：checkpoint、Dataset、预处理和运行时模式必须一致。

`--fixed-yaw` 只锁机体 yaw，不等于固定相机；它会同时施加于 A/B 两臂，适合作为
“独立云台是否有价值”的诊断条件，但不能冒充 YOPO-Simple 默认机体控制。

## 3. Docker 与编译

宿主机：

```bash
cd /home/dzp/projects/YOPO
git switch active-perception

docker build --network=host \
  -f docker/simulation.dockerfile \
  -t dzp_yopo:active-perception-u2004-noetic-py38 .
```

启动容器（需要 RViz 时保留 X11 参数）：

```bash
xhost +SI:localuser:root
HOST_XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
test -f "$HOST_XAUTHORITY"

docker run --name dzp-yopo-active -itd \
  --privileged --gpus all --network host --entrypoint bash \
  -e DISPLAY -e XAUTHORITY=/root/.Xauthority -e QT_X11_NO_MITSHM=1 \
  -v "$HOST_XAUTHORITY:/root/.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev --shm-size=4g \
  -v /home/dzp/projects/YOPO:/workspace/YOPO \
  dzp_yopo:active-perception-u2004-noetic-py38
```

容器内：

```bash
cd /workspace/YOPO/Controller
source /opt/ros/noetic/setup.bash
catkin_make -j2

cd /workspace/YOPO/Simulator
source /opt/ros/noetic/setup.bash
catkin_make -j2

test -x devel/lib/sensor_simulator/sensor_simulator_cuda
test -x devel/lib/sensor_simulator/dataset_generator
test -x devel/lib/sensor_simulator/insight9_ros_bridge
```

## 4. 生成严格配对的数据

本次正式实验使用项目完整规模：10 张地图、每图 10,000 个位姿。生成器会重建
`--save-path`，不要指向需要保留的目录。

```bash
cd /workspace/YOPO
source /opt/ros/noetic/setup.bash

# Active：训练地图 seed 3..12
python3 tools/run_yopo_pipeline.py \
  --mode generate --maze-type 8 --seed 3 \
  --active-camera true --env-num 10 --image-num 10000 \
  --save-path ../dataset_chicane_active_100ep

# Fixed：除相机 treatment 外完全相同
python3 tools/run_yopo_pipeline.py \
  --mode generate --maze-type 8 --seed 3 \
  --active-camera false --env-num 10 --image-num 10000 \
  --save-path ../dataset_chicane_fixed_100ep
```

`run_yopo_pipeline.py` 默认在退出时恢复两份 YAML；只有显式传 `--keep-config` 才保留
临时覆盖。一个完整 Dataset 必须含 `dataset_metadata.yaml` 和 `_SUCCESS`。

严格审计配对关系：

```bash
python3 tools/verify_paired_datasets.py \
  dataset_chicane_active_100ep dataset_chicane_fixed_100ep \
  --output-json \
  reports/active_camera_ablation/epoch100_chicane_20260820/dataset_pair_audit.json
```

审计会检查 PLY、pose、guide、样本前 21 个结构字段、fixed 四个零相机字段、PNG
尺寸/位深和文件集合；active/fixed 图像本来就来自不同视角，不要求图像哈希相同。

## 5. 训练：项目默认 50 epoch，本次正式实验显式 100 epoch

项目入口 `YOPO/train_yopo.py` 和统一流水线的通用默认值保持 50 epoch；本次正式 A/B
两臂均显式训练 100 epoch：

```bash
cd /workspace/YOPO

python3 tools/run_yopo_pipeline.py \
  --mode train --seed 0 --active-camera true \
  --dataset-path ../dataset_chicane_active_100ep \
  --train-epoch 100 --batch-size 16 --num-workers 4 \
  --run-name active_camera_chicane_100ep_seed0

python3 tools/run_yopo_pipeline.py \
  --mode train --seed 0 --active-camera false \
  --dataset-path ../dataset_chicane_fixed_100ep \
  --train-epoch 100 --batch-size 16 --num-workers 4 \
  --run-name fixed_camera_chicane_100ep_seed0
```

复现实验不得省略 `--train-epoch 100`。已有同名目录会直接报错，避免静默读写旧
checkpoint；如果只是使用项目通用默认训练，则应另取 run name，不能与本报告模型混用。

每个 checkpoint 都有不可变的 manifest 绑定，记录 epoch、SHA-256、模式、数据元信息、
预处理合同和源码哈希；不要只复制 `.pth` 而丢失旁边的 manifest/resolved config。

Git 默认忽略 `YOPO/saved` 中的训练输出，只对白名单中的两套 epoch100 最终模型、sidecar、
不可变 manifest、resolved config、metrics 和 model card 例外。两份 `.pth` 各约 47.5 MB，
直接随分支保存；训练数据与中间 checkpoint 不上传，均可用本手册命令重新生成。

## 6. 离线合同与 checkpoint 检查

```bash
cd /workspace/YOPO
PYTHONPATH=/workspace/YOPO/YOPO \
python3 tools/test_active_perception_contract.py --backward

PYTHONPATH=/workspace/YOPO/YOPO \
python3 tools/test_active_perception_contract.py \
  --backward --active-camera false

PYTHONPATH=/workspace/YOPO/YOPO \
python3 tools/test_yopo_checkpoint.py \
  --weight YOPO/saved/active_camera_chicane_100ep_seed0/epoch100.pth \
  --dataset-path ../dataset_chicane_active_100ep \
  --active-camera true --num-batches 4 --strict-depth-range

PYTHONPATH=/workspace/YOPO/YOPO \
python3 tools/test_yopo_checkpoint.py \
  --weight YOPO/saved/fixed_camera_chicane_100ep_seed0/epoch100.pth \
  --dataset-path ../dataset_chicane_fixed_100ep \
  --active-camera false --num-batches 4 --strict-depth-range
```

训练固定 seed、DataLoader 和 cuDNN 设置。当前 CUDA 的
`grid_sampler_3d_backward`/Flash Attention 没有完全确定性实现，因此采用 warn-only
deterministic mode；重新训练不承诺 checkpoint 逐 bit 相同，比较时应使用保存的 manifest。

## 7. 交互式仿真与推理开关

`launch_sim.sh` 没有修改地图/seed 的参数。若需交互查看 maze 8，先在
`Simulator/src/config/config.yaml` 设置 `maze_type: 8`；正式 A/B 使用下一节工具，避免
手工配置漂移。

```bash
cd /workspace/YOPO

# 主动相机
tools/launch_sim.sh \
  --weight /workspace/YOPO/YOPO/saved/active_camera_chicane_100ep_seed0/epoch100.pth \
  --active-camera true --velocity 3.0 --max-depth 20 --goal-height 2.0

# 固定相机（需使用 fixed checkpoint）
tools/launch_sim.sh \
  --weight /workspace/YOPO/YOPO/saved/fixed_camera_chicane_100ep_seed0/epoch100.pth \
  --active-camera false --velocity 3.0 --max-depth 20 --goal-height 2.0
```

无界面运行增加 `--no-rviz`。发布目标：

```bash
rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped "{
  header: {frame_id: world},
  pose: {position: {x: 20.0, y: 0.0, z: 2.0}, orientation: {w: 1.0}}
}"
```

规划器现在采用 `PoseStamped.position.z`；`--goal-height` 只是初始/默认目标高度，不会
覆盖一个有限的消息 z。

关键话题：

```text
/depth_image                         sensor_msgs/Image，Insight 9 深度
/sim/odom                            nav_msgs/Odometry
/yopo/camera/command                 geometry_msgs/Vector3，相机目标角
/yopo/camera/orientation             geometry_msgs/Vector3，兼容用实际角
/yopo/camera/orientation_stamped     geometry_msgs/Vector3Stamped，与深度时间同步
/yopo/collision_counter_total        std_msgs/Int32，碰撞进入事件总数
```

Active 推理会将带时间戳的相机实际角与深度帧同步后才送入网络，避免把当前图像配到上一帧
云台角。Fixed 推理不依赖该同步，状态始终为精确零。

## 8. 公平 A/B benchmark

完整命令、权重哈希、训练 seed、评测 seed、结果、Wilson 区间、轨迹 CSV 和最终图均在
独立实验报告中。工具的关键约束是：

- 两臂必须是对应模式的 checkpoint；benchmark 通用默认期望 50 epoch，本报告的正式
  复现命令必须显式传 `--expected-checkpoint-epoch 100`；
- 相同 seed 的两臂必须具有相同 runtime map SHA、点数和 bounds；
- 深度必须为 `160×192`、`32FC1`、约 15 Hz，并与 stamped 相机状态配对；
- active 必须观测到实际云台运动，fixed 必须持续收到命令/状态精确零；
- 只有完整且合同有效的 seed pair 才进入主统计；碰撞率使用 UAV 球半径 0.45 m；
- 训练 seed 与 holdout seed 不得重叠。

本次 100-epoch 预注册主实验（normal body yaw，seeds 401..410）的实测结果是：Active
0/10 碰撞、5/10 到达、5/10 无碰撞且到达；Fixed 5/10 碰撞、8/10 到达、5/10
无碰撞且到达。Active 降低了碰撞，但以更多超时为代价，综合的无碰撞到达率没有提升。
预声明的 fixed-body-yaw 诊断（seeds 501..510）中，两组都 0/10 碰撞，但 Active
0/10 到达、Fixed 5/10 到达。完整 Wilson 区间、配对差、每个 seed 的原始轨迹和图见
实验报告；不得把“碰撞少”单独改写成“整体性能更好”，也不得把超时算成碰撞。

结果目录已存在时，完整复跑应换新目录；只有同一 spec 的中断恢复才使用 `--resume`。
正式验收不要给 verifier 传 `--allow-incomplete`。

## 9. 真实 Insight 9 与实体云台

连接相机时将 `/dev/video*` 和 `/dev/hidraw*` 暴露给容器。启动 SDK bridge：

```bash
cd /workspace/YOPO/Simulator
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun sensor_simulator insight9_ros_bridge
```

bridge 原样发布 SDK 实际分辨率的 `16UC1` Z16（毫米），不在 bridge 内 resize；规划器
统一 nearest-resize 到 160×192。bridge 当前不发布 CameraInfo，实物部署前必须确认 Z16
已 rectified，并使用实机标定内参与畸变模型验证仿真 FoV。

启动实体二轴云台适配器：

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

真实主动推理必须同时有 `/depth_image` 和带时间戳的云台状态：

```bash
PYTHONPATH=/workspace/YOPO/YOPO \
python3 YOPO/test_yopo_ros.py \
  --weight YOPO/saved/active_camera_chicane_100ep_seed0/epoch100.pth \
  --active-camera true \
  --odom-topic /your_vehicle/odom \
  --depth-topic /depth_image \
  --ctrl-topic /your_controller/pos_cmd \
  --velocity 3.0 --goal-height 2.0
```

真实飞行前仍需独立验证坐标系、深度单位、云台时间戳、急停、geofence 和控制器接口。
没有对应硬件时，仿真和 SDK 编译通过不等于实机标定或飞行安全认证。

## 10. 常见问题

- `Checkpoint contract mismatch`：active/fixed、预处理、尺寸、max depth 或 sidecar 不匹配；
  不要绕过检查混用权重。
- `Training run directory already exists`：换一个新 `--run-name`，不要覆盖旧证据。
- Dataset 缺 `_SUCCESS`：上次生成中断或写盘失败，不能用于训练。
- 相机不动：检查 command、orientation 和 orientation_stamped 三个话题；active 推理要求
  stamped 状态能与 depth 时间匹配。
- 深度异常：真实相机应为 `16UC1` 毫米，仿真为 `32FC1` 米；两者均截断到 20 m。
- 无人机不动：先等待起飞完成并发布 `/move_base_simple/goal`。
- benchmark 启动失败：确认所选 ROS master 端口空闲，且 Controller/Simulator 已编译。
- RViz 无窗口：检查 `DISPLAY`、Xauthority；CI 使用 `--no-rviz`。

## 11. 报告边界

本次两条 pipeline 均使用完整的每图 10,000 位姿数据训练 100 epoch；项目通用默认仍为
50 epoch。主实验虽有 50 个百分点的描述性碰撞率下降，但无碰撞到达率差为 0；诊断组
主动模型的无碰撞到达率反而低 50 个百分点。因此当前证据不支持“主动感知 pipeline
整体优于固定相机基线”的结论，不能用单条好看的轨迹替代完整有效 pair 的统计结果。
