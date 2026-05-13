# YOPO 运行说明

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

说明：
- 构建镜像时如果速度慢，可以自行修改 `docker/simulation.dockerfile` 里的代理或软件源。
- 进入容器后，代码目录默认挂载到 `/workspace/YOPO`。

## 单机窄缝 Gate

### 测试自带模型

代码有 C++ / CUDA 改动时先重编：
```bash
source /opt/ros/noetic/setup.bash

cd /workspace/YOPO/Controller
catkin_make

cd /workspace/YOPO/Simulator
catkin_make
```

停止 / 启动：
```bash
cd /workspace/YOPO

./tools/launch_sim.sh --stop

./tools/launch_sim.sh --trial 1 --epoch 50
```

如果不指定权重，会自动选择最新：
```bash
./tools/launch_sim.sh
```

碰撞统计：
```bash
source /opt/ros/noetic/setup.bash && rostopic echo /yopo/collision_counter_total
```

常用话题：
```bash
/sim/odom
/depth_image
/lidar_points
/mock_map
/so3_control/pos_cmd
/yopo_net/gate_marker
/yopo_net/best_traj_visual
/yopo_net/trajs_visual
```

RViz：
- 直接用启动脚本默认拉起的 RViz 即可。
- 可以在 RViz 里用 `2D Nav Goal` 点 gate 另一侧目标。

补充：
- 当前默认任务不是原始 YOPO 的随机障碍避障，而是单 gate 窄缝穿越。
- 当前测试默认锁定 `yaw=0`，允许大 roll / pitch 机动。
- 当前场景有地面，中间是一整面薄墙，只留一个 slit 开口。

### 训练

先生成数据集：
```bash
cd /workspace/YOPO/Simulator
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun sensor_simulator dataset_generator
```

再训练：
```bash
cd /workspace/YOPO/YOPO
python3 train_yopo.py --train_epoch 50
```

TensorBoard：
```bash
cd /workspace/YOPO/YOPO
tensorboard --logdir=./saved
```

## 什么时候需要重采和重训

以下情况建议重新采数据并重新训练：
- 改了 gate 姿态，比如 `gate_pose`、`gate_slit_roll_deg`
- 改了 gate 尺寸，比如 `gate_inner_*`、`gate_outer_*`、`gate_depth`
- 改了墙体或开口几何，比如 `gate_wall_*`
- 改了机体碰撞尺寸，比如 `uav_collision_box_*`
- 改了采样范围，比如 `gate_sample_*`
- 改了损失权重或训练目标
- 改了是否允许大姿态机动的约束

一句话判断：
- 场景几何变了，要重采。
- 训练目标变了，要重训。

## 当前默认配置

主要配置文件：
```bash
/workspace/YOPO/Simulator/src/config/config.yaml
/workspace/YOPO/YOPO/config/traj_opt.yaml
```

当前默认场景要点：
- `maze_type: 8`
- 有地面
- 一整面薄墙，只留一个矩形 slit
- 默认单 gate
- 目标是从 gate 一侧穿到另一侧

当前默认 gate 设置：
```yaml
gate_pose: [85.0, 0.0, 0.0, 0.0, 0.0, 1.20]
gate_slit_roll_deg: 85.0
gate_inner_width: 0.70
gate_inner_length: 0.30
gate_goal_z: 1.20
gate_test_goal: [3.2, 0.0, 1.20]
```

当前默认姿态相关设置：
```yaml
gate_objective_lateral_velocity_weight: 0.00
gate_loss_roll_align_weight: 0.00
gate_lock_yaw: true
gate_lock_yaw_deg: 0.0
```

含义：
- 不再额外约束姿态必须“端正”穿缝。
- 不再额外压制横向出缝速度。
- 测试时默认不主动转 yaw。

## 场景检查

启动后如果正常，应该看到：
- 有地面
- 中间是一整面薄墙，不是孤立门框
- 墙中央只有一个 slit 开口
- `gate_slit_roll_deg=85` 时，缝接近竖直

如果飞机绕开缝飞，优先检查：
- 现在用的是否是新采集、新训练的模型
- 场景是否真的是整面墙而不是门框

如果飞机姿态太保守，优先检查：
- `gate_loss_roll_align_weight` 是否还是 `0.00`
- `gate_objective_lateral_velocity_weight` 是否还是 `0.00`
- 控制器 `max_tilt_deg` 是否足够大

如果飞机又开始钻地，优先检查：
- `add_ground_points: true`
- `occupy_below_ground: true`
- `gate_floor_min_z`
- `gate_abort_min_z`

## 主要代码位置

地图与仿真：
- [Simulator/src/src/maps.cpp](/workspace/YOPO/Simulator/src/src/maps.cpp)
- [Simulator/src/src/sensor_simulator.cu](/workspace/YOPO/Simulator/src/src/sensor_simulator.cu)
- [Simulator/src/src/test_simulator_cuda.cpp](/workspace/YOPO/Simulator/src/src/test_simulator_cuda.cpp)
- [Simulator/src/src/dataset_generator.cpp](/workspace/YOPO/Simulator/src/src/dataset_generator.cpp)

训练与测试：
- [YOPO/policy/yopo_dataset.py](/workspace/YOPO/YOPO/policy/yopo_dataset.py)
- [YOPO/loss/safety_loss.py](/workspace/YOPO/YOPO/loss/safety_loss.py)
- [YOPO/loss/gate_loss.py](/workspace/YOPO/YOPO/loss/gate_loss.py)
- [YOPO/test_yopo_ros.py](/workspace/YOPO/YOPO/test_yopo_ros.py)

控制：
- [Controller/src/so3_control/src/SO3Control.cpp](/workspace/YOPO/Controller/src/so3_control/src/SO3Control.cpp)
- [Controller/src/so3_control/src/NetworkControl.cpp](/workspace/YOPO/Controller/src/so3_control/src/NetworkControl.cpp)
