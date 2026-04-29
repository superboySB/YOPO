# YOPOv2-Tracker 运行速记

这份 note 记录当前 YOPOv2-Tracker 版本的实际跑通流程：编译、追踪测试、数据采集、训练和 RViz 交互。README 保持原版说明，tracking 相关操作以后看这里。

当前输入不是伪造 RGB，而是 `depth + target bbox mask` 两通道。第二通道模拟实物中 YOLO/检测器给出的目标 bounding box：仿真在线发布 `/target_mask_image`，数据集保存 `mask_*.png`。

注意：旧版避障权重不能直接用于现在的 tracker 网络。现在的权重必须满足 `input_channels=2`、`output_dim=14`，也就是首层卷积输入为 depth+mask，head 输出包含 primitive、cost、objectness 和 target 位置。

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

实际构建时注意 dockerfile 里的代理地址可以按网络情况调整。

## 1. 编译

C++ / CUDA simulator 有改动时先重编：

```bash
source /opt/ros/noetic/setup.bash && \
cd /workspace/YOPO/Controller && \
catkin_make && \
cd /workspace/YOPO/Simulator && \
catkin_make
```

## 2. 数据采集

正式训练默认读取 `/workspace/YOPO/dataset/`。下面命令会删除并重建该目录。

```bash
cd /workspace/YOPO/Simulator && \
source /opt/ros/noetic/setup.bash && \
source devel/setup.bash && \
rosrun sensor_simulator dataset_generator \
  --config /workspace/YOPO/Simulator/src/config/single_config.yaml \
  --save-path /workspace/YOPO/dataset/ \
  --env-num 10 \
  --image-num 10000
```

数据结构：

```text
/workspace/YOPO/dataset/
├── 0/
│   ├── depth_0.png
│   ├── mask_0.png
│   └── ...
├── pose-0.csv
├── target-0.csv
├── pointcloud-0.ply
└── ...
```

## 3. 正式训练

```bash
cd /workspace/YOPO/YOPO

python3 train_yopo.py \
  --config /workspace/YOPO/YOPO/config/single_traj_opt.yaml \
  --save-root saved \
  --epochs 50 \
  --batch-size 16 \
  --num-workers 4 \
  --save-interval 10
```

训练阶段会对第二通道 target mask 自动做检测器误差增强，开关和强度在 `YOPO/config/single_traj_opt.yaml` 里：

- `target_mask_jitter_px`：bbox 中心抖动
- `target_mask_scale_jitter`：bbox 尺寸误差
- `target_mask_dropout_prob`：随机漏检
- `target_mask_false_positive_prob` / `target_mask_false_positive_max`：假阳性框

## 4. 交互式追踪测试

有图形界面时启动完整仿真：

```bash
cd /workspace/YOPO
./tools/single_launch.sh --trial N --epoch X
```

当前已有 `YOPO/saved/YOPO_0/epoch30.pth` 时：

```bash
./tools/single_launch.sh --trial 0 --epoch 30
```

如果训练得到 `YOPO/saved/YOPO_3/epoch50.pth`，就改成：

```bash
./tools/single_launch.sh --trial 3 --epoch 50
```

启动脚本会打开 tmux，并启动：

- `roscore`
- follower dynamics 和 SO3/network controller
- target motion node
- CUDA depth + target-mask sensor simulator
- YOPOv2-Tracker policy node
- RViz

RViz 中应该能看到：

- target 无人机 marker：`/target/marker`
- follower 轨迹：`/yopo_tracker/trajs_visual`
- 最优轨迹：`/yopo_tracker/best_traj_visual`
- target 估计：`/yopo_tracker/target_estimate`
- 深度和目标框 mask：`/depth_image`、`/target_mask_image`

交互方式：

1. 启动后 target 默认按脚本轨迹飞，follower 会追踪它。
2. 在 RViz 工具栏选择 `2D Nav Goal`。
3. 在地图上点击一个新位置。
4. target 会飞向这个 waypoint；同一次点击也会重置 tracker 的 target estimate；follower 会继续追踪新的 target 位置。

近距离行为：

- tracker 距离 target 大于 `follow_distance + follow_deadband` 时，继续使用网络输出的追踪轨迹。
- 进入观察距离带后，会保持约 `follow_distance` 的短距离观察。
- target 停止且 tracker 已在距离带内时，policy 会发布零期望速度的 hover/brake 指令，避免追到以后继续围着目标盘旋。

相关参数在 `YOPO/config/single_traj_opt.yaml`：

- `follow_distance`
- `follow_deadband`
- `follow_target_speed_threshold`
- `target_velocity_ema_alpha`

常用检查：

```bash
source /opt/ros/noetic/setup.bash
rostopic echo /target/odom
rostopic echo /yopo/collision_counter_total
rostopic hz /target_mask_image
rostopic hz /depth_image
```

停止完整仿真：

```bash
cd /workspace/YOPO
./tools/single_launch.sh --stop
```

## 5. TensorRT 可选部署

只有安装了 `torch2trt` / TensorRT 后才运行：

```bash
cd /workspace/YOPO/YOPO
python3 yopo_trt_transfer.py --trial N --epoch X
python3 test_yopo_ros_single.py --use_tensorrt=1
```

## 6. 关键文件

- Simulator 配置：`Simulator/src/config/single_config.yaml`
- YOPO tracker 训练配置：`YOPO/config/single_traj_opt.yaml`
- RViz 配置：`YOPO/single_yopo.rviz`
- 一键启动脚本：`tools/single_launch.sh`
- Target 运动和 RViz waypoint 交互：`Simulator/src/scripts/target_motion_node.py`

实现要点：

- 网络接口按当前 tracker 复现：depth + target bbox mask + 6D state 输入，14D per-primitive 输出。
- target head 输出 target uv/depth logits，解码为 camera/body frame 下的目标 3D 位置。
- policy node 发布 `PositionCommand`，沿用现有 SO3 controller 执行追踪轨迹。
