# YOPOv2-Tracker 简明用法

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
  --epochs 50 \
  --batch-size 16 \
  --num-workers 4
```

训练配置使用水平 `120 deg`、垂直 `90 deg` 相机模型。状态输入包含本机导航目标向量。训练 score label 由 smoothness、static safety、dynamic safety、goal guidance、acceleration 和 visible-target separation 组成；static safety 保持原版 ESDF 30 点采样，dynamic safety 和 separation 对当前帧所有可见目标用 5 个采样点，不做速度外推。单机或无可见目标时 dynamic/separation 项为 0。

## Tracker Swarm 测试

统一启动脚本：`/workspace/YOPO/tools/swarm_tracker_launch.sh`。它同时覆盖单机和多机；默认 `1` 机、`--formation '1'`。多机时用 `--formation '1|2|3|4'` 这种从后到前的行数描述，脚本会检查各行求和是否等于 `--uav-num`。初始编队最近邻距离和 separation loss 距离都读 `swarm_initial_spacing`，到达成功半径读 `swarm_arrive_radius`；这些量都在 `tracker_traj_opt.yaml` 的 Swarm distance knobs 注释块里设置。

单机和少机使用同一入口：

```bash
cd /workspace/YOPO

./tools/swarm_tracker_launch.sh --trial 0 --epoch 50 --rviz 1
./tools/swarm_tracker_launch.sh --uav-num 5 --formation '2|1|2' --trial 0 --epoch 50 --rviz 1
./tools/swarm_tracker_launch.sh --uav-num 10 --formation '4|3|2|1' --trial 0 --epoch 50 --rviz 1

./tools/swarm_tracker_launch.sh --stop
```

RViz `2D Nav Goal` 的点击位置表示 `uav0` 的目标位置。所有飞机根据 `uav0` 实时位置到点击位置的位移更新各自目标，因此队形按同一位移平移，不会聚集到同一绝对坐标。

ROS planner 统一入口为 `/workspace/YOPO/YOPO/test_yopo_ros_swarm_tracker.py`,`--uav-num 1` 时没有其它飞机 mask，输入第二通道为空，行为退化为单机 YOPO navigation。

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
