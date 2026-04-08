# YOPO Docker 速记

## 配置
```bash
cd /home/dzp/projects/YOPO

docker build -f docker/simulation.dockerfile \
  -t dzp_yopo:sim-u2004-noetic-py38 \
  --network=host --progress=plain .

xhost +local:root

docker run --name dzp-yopo -itd --privileged --gpus all --network host \
  --entrypoint bash \
  -e DISPLAY -e QT_X11_NO_MITSHM=1 \
  -e http_proxy=http://127.0.0.1:8889 \
  -e https_proxy=http://127.0.0.1:8889 \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --shm-size=4g \
  -v /home/dzp/projects/YOPO:/workspace/YOPO \
  dzp_yopo:sim-u2004-noetic-py38

docker exec -it dzp-yopo /bin/bash

cd /workspace/YOPO
```

## 代码用法（容器内）
```bash
source /opt/ros/noetic/setup.bash
python3 --version   # 期望 3.8.x
```

首次编译（只需一次，代码有 C++ 变更时重编）：
```bash
cd /workspace/YOPO/Controller
catkin_make

cd /workspace/YOPO/Simulator
catkin_make
```

运行仿真与 YOPO（建议 4 个终端）：
```bash
# 终端 1: 控制器/动力学
cd /workspace/YOPO/Controller
source devel/setup.bash
roslaunch so3_quadrotor_simulator simulator_attitude_control.launch

# 终端 2: 传感器与环境仿真
cd /workspace/YOPO/Simulator
source devel/setup.bash
rosrun sensor_simulator sensor_simulator_cuda

# 终端 3: YOPO 规划器
cd /workspace/YOPO/YOPO
python3 test_yopo_ros.py --trial=1 --epoch=50

# 终端 4: 可视化
cd /workspace/YOPO/YOPO
rviz -d yopo.rviz
```

## 一键启动（容器内）
先进入容器：
```bash
docker exec -it dzp-yopo /bin/bash
```

然后在容器内执行：
```bash
cd /workspace/YOPO
./tools/launch_sim.sh --trial 1 --epoch 50
```

说明：
- 该脚本必须在容器内运行（会检查 `/.dockerenv`）。
- 该脚本基于 `tmux` 自动开 4 个窗口（controller/simulator/planner/rviz）。
- 在该 `tmux` 会话里按 `Ctrl+C` 会同时停止 4 个窗口里的进程。
- 常用参数：
```bash
# 查看帮助
./tools/launch_sim.sh -h

# 后台启动（不自动 attach）
./tools/launch_sim.sh --trial 1 --epoch 50 --detach

# 停止会话及相关进程
./tools/launch_sim.sh --stop

# 累计碰撞步数（整个运行过程累计）
rostopic echo /yopo/collision_counter_total
```

## YOPO-SWARM
这个分支现在支持一个可重复运行的 YOPO-SWARM 任务：
- 多架控制器/动力学实例同时运行。
- 单个 `sensor_simulator_cuda` 同时为多架机渲染各自的深度图和激光。
- 深度图里会把其他无人机当成动态球形障碍渲染进去。
- 额外提供机间碰撞计数话题，方便区分“静态地图占据”与“无人机互撞”。
- 不需要手点 `2D Nav Goal`；脚本会把无人机放到圆环上，并自动把目标设成对径点。

如果改过 `Controller/` 或 `Simulator/` 里的 C++/CUDA 代码，先重新编译：
```bash
cd /workspace/YOPO/Controller
catkin_make

cd /workspace/YOPO/Simulator
catkin_make
```

默认启动 4 机 swarm（容器内）：
```bash
cd /workspace/YOPO
./tools/launch_swarm_sim.sh --trial 1 --epoch 50
```

常用形式：
```bash
# 后台启动
./tools/launch_swarm_sim.sh --trial 1 --epoch 50 --detach

# 做多轮回归
./tools/run_swarm_trials.sh --trial 1 --epoch 50

# 指定无人机数量或圆环半径
./tools/launch_swarm_sim.sh --trial 1 --epoch 50 --uav-num 2 --radius 12

# 停止 swarm 并释放资源
./tools/launch_swarm_sim.sh --stop
```

默认行为（当前建议的稳定演示配置）：
- `YOPO/config/traj_opt.yaml` 已把测试速度下调到 `1.2 m/s`。
- 默认 `4` 架机、圆环半径 `16m`、高度 `2m`。
- 无人机命名为 `uav0 ~ uavN`，初始位置在圆环上，目标点仍然是各自显式设置的对径点位置。
- 默认 `swarm_tangent_bias = 0`，也就是不额外给“人为绕圈”的切向引导；这套任务本身就是 head-on 冲突场景，更能体现 YOPO 对树林和其它无人机的实时避障。
- 启动脚本会显式先起 `roscore`，再起控制器、传感器、YOPO 和 `rviz`。

关键话题：
```bash
# 每架机的深度图 / 里程计 / 控制命令
/uav0/depth_image
/uav0/sim/odom
/uav0/so3_control/pos_cmd

# 每架机的轨迹可视化（每架机都会独立发布）
/uav0/yopo_net/best_traj_visual
/uav0/yopo_net/trajs_visual
/uav0/yopo_net/lattice_trajs_visual

# 地图与无人机 mesh
/mock_map
/uav0_simulator/uav

# 静态占据近似碰撞计数
/yopo/collision_counter_total
/uav0/yopo/collision_counter

# 机间碰撞计数（新增，专门看无人机是否互相进入碰撞体）
rostopic echo /yopo/collision_counter_total
rostopic echo /yopo/uav_collision_counter_total

# 每架机是否到达目标
/uav0/yopo/arrived
/uav0/yopo/goal_distance
```

`rviz` 说明：
- swarm 脚本现在会默认加载 `YOPO/yopo_swarm.rviz`。
- 该布局默认预载入 `/mock_map`、`/uav0~uav3` 的无人机 Marker、`/uav0~uav3` 的 `best_traj_visual` 和 `trajs_visual`，以及 `/uav0/depth_image`。
- RViz 里默认只放第一个无人机的深度图，避免 4 张深度图同时显示过于拥挤；但 `/uav0~uav3/depth_image` 话题都会正常发布。
- 如果你把 `--uav-num` 改成别的值，超出 `uav0~uav3` 的部分可以在 RViz 里按同样格式手动补 topic。

当前验证结果：
- 多机实例、自动对径目标、多份 YOPO 推理已跑通。
- 在静止对视测试里，`/uav1/depth_image` 正前方中心区域能稳定看到另一架机的球形遮挡，说明深度图里已经能看到其他无人机。
- 在人工把 `/uav0/sim/odom` 瞬时改到 `/uav1` 同位置的测试里，`/yopo/uav_collision_counter_total` 与 `/<uav>/yopo/uav_collision_counter` 会增长，说明无人机已经作为碰撞体参与统计。
- `./tools/launch_swarm_sim.sh --trial 1 --epoch 50 --uav-num 4 --radius 16 --detach` 可以完整拉起 `roscore + 4 控制器 + 1 传感器节点 + 4 个 YOPO + rviz`。
- `rosnode list` 中可以同时看到 4 个独立的 YOPO 节点：`/yopo_net_uav0 ~ /yopo_net_uav3`，说明 4 个网络实例是独立实时推理的，而不是共享一个 ROS 节点。
- 4 个 `/uav*/depth_image` 话题都已实际收帧成功；实测为 `90x160`、`32FC1`，并且深度统计值不是“全 20m 空场”，说明树林障碍确实在深度图里。
- `test_yopo_ros.py` 启动时会先做一次 CUDA cache 清理；`tools/run_swarm_trials.sh` 在每轮启动前也会先执行一次 `./tools/launch_swarm_sim.sh --stop`，避免旧进程残留占着显存。
- 在 `RTX 4070 Ti SUPER 16GB` 上，4 个 YOPO 进程各自大约占用 `394 MB` 显存，`sensor_simulator_cuda` 约 `420 MB`；整轮运行时 GPU 总占用约 `3.2 / 16.4 GB`，4 机是可以正常放下的。

运行结果（2026-04-08）：
- `./tools/run_swarm_trials.sh --trials 2 --uav-num 4 --timeout 170 --trial 1 --epoch 50 --radius 16 --swarm-tangent-bias 0`
  2/2 成功，全部到达对径点，`uav_collision_total = 0`，平均耗时约 `59.3s`。
- `./tools/run_swarm_trials.sh --trials 1 --uav-num 4 --timeout 170 --trial 1 --epoch 50 --radius 16 --swarm-tangent-bias 0`
  在默认 `visualize=1`、RViz 同时订阅 4 架机 `trajs_visual` 的情况下也已跑通，`uav_collision_total = 0`，耗时约 `49.8s`。
- `./tools/run_swarm_trials.sh --trials 3 --uav-num 2 --timeout 70 --trial 1 --epoch 50 --radius 12`
  3/3 成功，全部到达对径点，`uav_collision_total = 0`，平均耗时约 `45.7s`。
- 更紧的 4 机任务 `radius=12` 会把交汇区压得很紧；我实际多轮回归时出现过稳定的邻机擦碰，典型结果是 `uav_collision_total = 2`。因此当前默认演示改成了更稳的 `4 机 + 16m`。

补充说明：
- `collision_counter_total` 统计的是“机体中心进入占据体素”的近似碰撞事件，在森林图里会比肉眼看到的实际碰撞更敏感。
- `uav_collision_counter_total` 统计的是“无人机之间进入碰撞半径”的事件计数，现在按“进入碰撞状态的一次事件”计数，不会在持续接触时每帧累加。
- 这套多机链路已经能直接用现有 YOPO 做推理和避障；当前默认任务是“树林 + 4 机对径冲突目标 + 无切向人为引导”，更适合作为集群避障演示。
- 若要再次运行，建议始终先执行一次 `./tools/launch_swarm_sim.sh --stop`，再重新启动，避免旧 ROS/YOPO 进程残留。

## 2D Nav Goal 与“是否撞障”判断
- README 里的 `2D Nav Goal` 只是给目标点（`/move_base_simple/goal`），对应 `README.md` 的测试说明。
- 在代码中，`YOPO/test_yopo_ros.py` 仅在 `callback_set_goal` 把目标写成 `[x, y, 2]`，并打印 `New Goal`；到达条件是 `距离目标 < 5m` 后打印 `Arrive!`，没有任何“撞障”状态位或回调。
- 动力学仿真 `Controller/src/so3_quadrotor_simulator` 不读取障碍物地图，`Quadrotor.cpp` 里仅处理“地面约束”（`z<0` 时把 `z,vz` 置零），没有障碍物接触模型。
- `Simulator/src/src/sensor_simulator.cu` 的障碍占据信息只用于深度/激光射线查询（`mapQuery`），用于生成传感器观测，不会反作用到无人机动力学状态。

结论：
- 当前这套默认仿真里，没有内置“撞上障碍物就触发 crash”的自动判定。
- 实操上只能用可视化判断：在 RViz 中看 `Drone`（`/quadrotor_simulator_so3/uav`）是否穿入 `Map`（`/lidar_points`）点云。
- 已在 `sensor_simulator_cuda` 增加基于原生 `GridMap::mapQuery` 的累计碰撞步数统计（“在占据体素内一步就 +1”）。


训练流程：
```bash
# 1) 采集数据
cd /workspace/YOPO/Simulator
source devel/setup.bash
rosrun sensor_simulator dataset_generator

# 2) 训练策略
cd /workspace/YOPO/YOPO
python3 train_yopo.py

# 3) 查看日志
cd /workspace/YOPO/YOPO/saved
tensorboard --logdir=./
```

## 代码功能总结（全面但简洁）
- `Controller/`：SO3 控制器与无人机动力学仿真，接收 `PositionCommand` 并输出姿态/速度响应；支持位置控制和姿态控制两种模式。
- `Simulator/`：CUDA 深度/点云传感器仿真与随机环境生成。`sensor_simulator_cuda` 用于在线仿真，`dataset_generator` 用于离线生成训练数据（深度图+位姿+点云地图）。
- `YOPO/`：学习式一阶段规划器主模块（PyTorch）。输入深度图与状态观测，输出每个运动基元的终点状态偏移和代价分数。
- `YOPO/policy/primitive.py`：生成离散运动基元（lattice anchors），定义规划搜索空间（水平/垂直方向和规划半径）。
- `YOPO/policy/state_transform.py`：完成 body/world/primitive 坐标转换，负责训练和推理时输入归一化、输出反变换。
- `YOPO/policy/yopo_network.py`：网络本体（图像 backbone + head），预测终点状态 `p/v/a` 与 `score`。
- `YOPO/loss/`：损失函数由四部分组成：平滑性（jerk/acc）、安全性（ESDF 距离）、目标引导（朝向目标）和分数监督。
- `YOPO/loss/safety_loss.py`：从 `dataset/pointcloud-*.ply` 构建 ESDF，训练时对多条候选轨迹做可微距离查询与碰撞惩罚。
- `YOPO/policy/yopo_dataset.py`：读取深度图与位姿，随机采样速度/加速度/目标方向，构造训练 observation。
- `YOPO/policy/yopo_trainer.py`：训练主循环。前向后将预测轨迹变换到世界系，按损失计算梯度并写 TensorBoard。
- `YOPO/test_yopo_ros.py`：在线 ROS 节点。订阅深度与里程计，网络推理后选择最低代价基元，并用五次多项式生成可执行轨迹发布到控制器。
- `YOPO/yopo_trt_transfer.py`：将 PyTorch 权重导出为 TensorRT，加速机载部署。
- 关键配置集中在 `YOPO/config/traj_opt.yaml`：飞行速度、基元数量、相机参数、训练采样分布、各项损失权重。

## 说明
- 当前 Docker 方案默认 Python=3.8、ROS=noetic，不使用 conda/mamba/uv。
- 该镜像不包含 PX4，仅用于 YOPO 仿真与训练。
- 依赖文件已统一放在 `docker/`：
  - `docker/requirements.txt`（YOPO Python 依赖）
- 代理配置默认保留在镜像与运行命令中。
