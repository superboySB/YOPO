# YOPO-swarm 改进版

## Docker 构建与启动

```bash
cd /home/dzp/projects/YOPO

docker build -f docker/simulation.dockerfile \
  -t dzp_yopo:sim-u2004-noetic-py38 \
  --network=host --progress=plain .

xhost +local:root

docker run --name dzp-yopo -itd --privileged --gpus all --network host \
  --entrypoint bash \
  -e DISPLAY -e QT_X11_NO_MITSHM=1 \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --shm-size=4g \
  -v /home/dzp/projects/YOPO:/workspace/YOPO \
  dzp_yopo:sim-u2004-noetic-py38

docker exec -it dzp-yopo /bin/bash
```

## 编译

```bash
source /opt/ros/noetic/setup.bash && \
cd /workspace/YOPO/Controller && \
catkin_make && \
cd /workspace/YOPO/Simulator && \
catkin_make
```

## 数据采集

仿真配置文件：`/workspace/YOPO/Simulator/src/config/swarm_config.yaml`。

```bash
cd /workspace/YOPO/Simulator && \
source /opt/ros/noetic/setup.bash && \
source devel/setup.bash && \
rosrun sensor_simulator dataset_generator \
  --config /workspace/YOPO/Simulator/src/config/swarm_config.yaml \
  --save-path /workspace/YOPO/dataset/ \
  --env-num 10 \
  --image-num 10000
```

数据集 depth 包含静态地图和当前帧动态目标；动态目标按 250 级四旋翼近似为 `0.31 x 0.31 x 0.14 m` 椭球，并同步写入 mask。每帧动态目标数量按 `50%,25%,12.5%,12.5%` 采样为 `0,1,2,3` 个，距离本机 `1-5m`，目标间真值 3D 距离不小于 `2m`。CSV 按真值 3D 距离排序记录所有可见目标。

## 正式训练

网络配置文件：`/workspace/YOPO/YOPO/config/tracker_traj_opt.yaml`。

```bash
cd /workspace/YOPO/YOPO

python3 train_yopo.py \
  --config /workspace/YOPO/YOPO/config/tracker_traj_opt.yaml \
  --save-root saved \
  --epochs 400 \
  --batch-size 32 \
  --num-workers 16
```

训练配置使用水平 `120 deg`、垂直 `90 deg` 相机模型。状态输入包含本机导航目标向量。训练 score label 由 smoothness、static safety、dynamic safety、goal guidance、acceleration 和 visible-target separation 组成；static safety 保持原版 ESDF 30 点采样，dynamic safety 和 separation 对当前帧所有可见目标用 5 个采样点，不做速度外推。单机或无可见目标时 dynamic/separation 项为 0。各项权重在 `YOPO/config/tracker_traj_opt.yaml` 的 `Loss weights` 和 `Guidance sub-loss knobs` 中设置；旧配置里的 `wg/ws/wa/wc` 仍会被兼容映射。

TensorBoard 会同时记录 `Train/*` 和 `Eval/*` 的加权 loss 组件，并在 `TrainRaw/*`、`EvalRaw/*` 下记录未加权 cost 量级。`TrainDiagnostics/*` 和 `EvalDiagnostics/*` 包含 static/dynamic 最小距离、碰撞率、目标可见比例和 separation 违反率，用来辅助判断静态碰撞、动态目标避让和编队间距设置是否需要调参。

## Tracker Swarm 测试

### 任务 Traversal
Traversal 启动脚本：`/workspace/YOPO/tools/launch_traversal.sh`。它明确支持单机和多机；默认 `1` 机、`--formation '1'`。多机时用 `--formation '1|2|3|4'` 这种从后到前的行数描述，脚本会检查各行求和是否等于 `--uav-num`。初始编队最近邻距离和 separation loss 距离都读 `swarm_initial_spacing`，到达成功半径读 `swarm_arrive_radius`；这些量都在 `tracker_traj_opt.yaml` 的 Swarm distance knobs 注释块里设置。

Traversal 单机、少机和多机使用同一入口：

```bash
cd /workspace/YOPO

./tools/launch_traversal.sh --trial 0 --epoch 400 --rviz 1
./tools/launch_traversal.sh --uav-num 5 --formation '2|1|2' --trial 0 --epoch 400 --rviz 1
./tools/launch_traversal.sh --uav-num 10 --formation '4|3|2|1' --trial 0 --epoch 400 --rviz 1

./tools/launch_traversal.sh --stop
```

### 任务2 Crossover

Crossover 启动脚本：`/workspace/YOPO/tools/launch_crossover.sh`。这是新的多机相向穿越场景，不支持单机，也不使用 `--formation`。脚本仍然运行同一个 YOPOv2-Tracker 网络；每架飞机均匀放在圆上，相邻飞机的直线距离由 `swarm_initial_spacing` 决定，初始 yaw 指向圆心，默认沿圆心方向飞行 `20m`，也可以用 `--crossover-distance` 临时覆盖。该场景不绑定某一种障碍物环境，`--env`/`--maze-type`、`--spawn-clear-radius`、RViz 和 diagnostic 选项继续复用 traversal 的启动逻辑；起点和终点清障、必要时的地图边界扩展只写入临时 runtime config，不影响数据采集和训练配置。

Crossover 5 机和 10 机示例：

```bash
cd /workspace/YOPO

./tools/launch_crossover.sh --uav-num 5 --trial 0 --epoch 400 --rviz 1
./tools/launch_crossover.sh --uav-num 10 --trial 0 --epoch 400 --rviz 1

./tools/launch_crossover.sh --stop
```

### 注意事项

1. 默认环境是树林，也就是 `Simulator/src/config/swarm_config.yaml` 里的 `maze_type: 5`。Traversal 不传 `--env`/`--maze-type` 时直接使用原始 `swarm_config.yaml`；Crossover 会从同一份 YAML 派生临时 runtime config，只额外写入本次圆形起点/终点清障点。临时切换环境时不需要手动改 YAML，直接在启动命令后追加参数即可：`--env forest`/`--maze-type 5` 是树林，`--env pillar`/`--maze-type 2` 是柱子，`--env cave`/`--maze-type 1` 是溶洞。例如：

```bash
./tools/launch_traversal.sh --env cave --uav-num 5 --formation '2|1|2' --trial 0 --epoch 400 --rviz 1
./tools/launch_traversal.sh --env pillar --uav-num 10 --formation '4|3|2|1' --trial 0 --epoch 400 --rviz 1
./tools/launch_crossover.sh --env cave --uav-num 10 --trial 0 --epoch 400 --rviz 1
```

2. 周围点云可视化默认关闭，以免多机 RViz 太卡。需要打开时，在任一启动命令后追加 `--vis_ply_per_uav 1`，
该功能只发布轻量 per-UAV LiDAR 点云到 `/uavN/lidar_points` 供 RViz decay 累积显示，不使用之前的全局格子/盒子地图，不改变训练数据、深度图、mask、YOPO 网络输入输出或控制指令。

3. 起点和终点附近不生成障碍物的逻辑会随 swarm 启动参数一起复用到这些环境；清障半径默认读取 `swarm.spawn_clear_radius`，也可以用 `--spawn-clear-radius 2.5` 临时调整。

4. Traversal 中 RViz `2D Nav Goal` 的点击位置表示 `uav0` 的目标位置。所有飞机根据 `uav0` 实时位置到点击位置的位移更新各自目标，因此队形按同一位移平移，不会聚集到同一绝对坐标。Crossover 不支持 RViz 重新设置 goal；如果点击 `2D Nav Goal`，`uav0` planner 会提示该任务不支持并忽略该目标。

5. ROS planner 统一入口为 `/workspace/YOPO/YOPO/test_yopo_ros_swarm_tracker.py`,`--uav-num 1` 时没有其它飞机 mask，输入第二通道为空，行为退化为单机 YOPO navigation。，***不论什么修改都要保持单机静态避障能力**

## 任务定义，输入各个维度，输出意义，网络结构

任务：无通信高速 swarm navigation。每架飞机运行同一个 YOPOv2-Tracker 网络，以本机独立 goal 导航为主，以相机内所有可见动态目标 mask 为辅，在静态障碍和队友之间规划轨迹。

输入：

- 图像：`(2, 96, 160)`。
- 第 0 通道：静态障碍和动态目标共同渲染的深度图。
- 第 1 通道：动态目标椭球投影 mask。swarm 模式下包含本机相机内所有可见其它飞机；无可见飞机或单机模式下为空 mask。
- 状态：`(9,)`，相机/body 坐标系速度 `(vx, vy, vz)`、加速度 `(ax, ay, az)`、本机导航目标向量 `(gx, gy, gz)`。
- 相机：`fx=46.1880215`、`fy=48.0`、`cx=80.0`、`cy=48.0`，对应水平 `120 deg`、垂直 `90 deg`。

输出：

- `5 x 3 = 15` 个 primitives，每个 primitive 输出 `10` 维。
- `0:9`：终端位置、速度、加速度参数。
- `9`：轨迹 cost，越小越优。

网络结构：2 通道 ResNet-18 backbone + 9 维状态 broadcast + 三层 `1x1 Conv` head。推理时直接选择网络 score 最小的 primitive，不叠加测试端规则或虚拟目标。节点将选定 primitive 转换为五次多项式，并发布 `PositionCommand`。
