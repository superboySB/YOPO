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
roslaunch se3_quadrotor_simulator simulator_attitude_control.launch

# 终端 2: 传感器与环境仿真
cd /workspace/YOPO/Simulator
source devel/setup.bash
rosrun sensor_simulator sensor_simulator_cuda

# 终端 3: YOPO 规划器
cd /workspace/YOPO/YOPO
python3 test_yopo_ros.py --trial=0 --epoch=50

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
./tools/launch_sim.sh --trial 0 --epoch 50
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
./tools/launch_sim.sh --trial 0 --epoch 50 --detach

# 停止会话及相关进程
./tools/launch_sim.sh --stop

# 累计碰撞步数（整个运行过程累计）
rostopic echo /yopo/collision_counter_total
```

## 2D Nav Goal 与“是否撞障”判断
- README 里的 `2D Nav Goal` 只是给目标点（`/move_base_simple/goal`），对应 `README.md` 的测试说明。
- 在代码中，`YOPO/test_yopo_ros.py` 仅在 `callback_set_goal` 把目标写成 `[x, y, 2]`，并打印 `New Goal`；到达条件是 `距离目标 < 5m` 后打印 `Arrive!`，没有任何“撞障”状态位或回调。
- 动力学仿真 `Controller/src/se3_quadrotor_simulator` 不读取障碍物地图，`Quadrotor.cpp` 里仅处理“地面约束”（`z<0` 时把 `z,vz` 置零），没有障碍物接触模型。
- `Simulator/src/src/sensor_simulator.cu` 的障碍占据信息只用于深度/激光射线查询（`mapQuery`），用于生成传感器观测，不会反作用到无人机动力学状态。

结论：
- 当前这套默认仿真里，没有内置“撞上障碍物就触发 crash”的自动判定。
- 实操上只能用可视化判断：在 RViz 中看 `Drone`（`/quadrotor_simulator_se3/uav`）是否穿入 `Map`（`/lidar_points`）点云。
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
- `Controller/`：SE3 控制器与无人机动力学仿真，接收带 `jerk` 的 `PositionCommand` 并输出姿态/力响应；默认在线链路走 SE3 平坦映射控制。
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
