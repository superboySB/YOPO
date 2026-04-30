# YOPOv2-Tracker 运行速记

这份 note 记录当前 YOPOv2-Tracker 版本的实际跑通流程：编译、追踪测试、数据采集、训练和 RViz 交互。README 保持原版说明，tracking 相关操作以后看这里。

当前输入不是伪造 RGB，而是 `depth + target bbox mask` 两通道。第二通道模拟实物中 YOLO/检测器给出的目标 bounding box：仿真在线发布 `/target_mask_image`，数据集保存 `mask_*.png`。

注意：旧版避障权重不能直接用于现在的 tracker 网络。现在的权重必须满足 `input_channels=2`、`output_dim=14`，也就是首层卷积输入为 depth+mask，head 输出包含 primitive、cost、objectness 和 target 位置。

模型目录分成两组：

- `YOPO/saved/no_tracker/YOPO_N/epochX.pth`：原版 YOPO 单机避障模型，给前方被追的 target 无人机用。
- `YOPO/saved/with_tracker/YOPO_N/epochX.pth`：YOPOv2-Tracker 追踪+避障模型，给后方 follower 无人机用。

交互测试时，RViz 的 `2D Nav Goal` 是发给前机 no-tracker YOPO 的 waypoint。前机先用单机避障模型飞过去，后机用 with-tracker 模型观察 `/target_mask_image` 并继续追踪前机。

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
  --save-root saved/with_tracker \
  --epochs 50 \
  --batch-size 16 \
  --num-workers 4 \
  --save-interval 10
```

每次训练会自动创建新的 `YOPO/saved/with_tracker/YOPO_N/` 目录。TensorBoard 的 `events.out.tfevents...` 和 checkpoint `epochX.pth` 会写在这个同一个 `YOPO_N` 目录里，方便按 `--trial N --epoch X` 直接测试。

训练阶段会对第二通道 target mask 自动做检测器误差增强，开关和强度在 `YOPO/config/single_traj_opt.yaml` 里：

- `target_mask_jitter_px`：bbox 中心抖动
- `target_mask_scale_jitter`：bbox 尺寸误差
- `target_mask_dropout_prob`：随机漏检
- `target_mask_false_positive_prob` / `target_mask_false_positive_max`：假阳性框

## 4. 交互式追踪测试

有图形界面时启动完整仿真：

```bash
cd /workspace/YOPO && \
./tools/single_launch.sh \
  --target-trial T --target-epoch E \
  --trial N --epoch X
```

其中：

- `--target-trial T --target-epoch E` 选择前机 no-tracker 模型：`YOPO/saved/no_tracker/YOPO_T/epochE.pth`
- `--trial N --epoch X` 选择后机 with-tracker 模型：`YOPO/saved/with_tracker/YOPO_N/epochX.pth`


如果训练得到新的 tracker `YOPO/saved/with_tracker/YOPO_3/epoch50.pth`，前机仍用 `YOPO/saved/no_tracker/YOPO_1/epoch50.pth`，就改成：

```bash
cd /workspace/YOPO && \
./tools/single_launch.sh \
  --target-trial 1 --target-epoch 50 \
  --trial 0 --epoch 50
```

启动脚本会打开 tmux，并启动：

- `roscore`
- follower dynamics 和 SO3/network controller
- target dynamics 和 SO3/network controller
- target YOPO-Simple avoidance policy node
- CUDA depth + target-mask sensor simulator，其中 `/target_depth_image` 给 target 避障 policy 使用
- YOPOv2-Tracker policy node
- RViz

RViz 中应该能看到：

- target 无人机 marker：`/target/marker`
- follower 轨迹：`/yopo_tracker/trajs_visual`
- 最优轨迹：`/yopo_tracker/best_traj_visual`
- target 估计：`/yopo_tracker/target_estimate`
- follower 深度和目标框 mask：`/depth_image`、`/target_mask_image`
- target 自己的避障深度：`/target_depth_image`

交互方式：

1. 启动后 target 是一架真实仿真无人机，默认使用 `YOPO/saved/no_tracker/YOPO_1/epoch50.pth` 的单机避障模型飞向前方目标点。
2. 在 RViz 工具栏选择 `2D Nav Goal`。
3. 在地图上点击一个新位置。
4. target 会用 no-tracker 避障模型飞向这个 waypoint；同一次点击也会重置 tracker 的 target estimate；follower 会用 with-tracker 模型继续追踪新的 target 位置。

近距离行为：

- policy 优先用 `/target_mask_image` 的 bbox 加 `/depth_image` 反投影估计 target 位置；网络 target head 作为 fallback。
- target 运动或距离较远时，follower 使用 tracker 网络候选轨迹，保留避障+tracking 的联合选择。
- target 停止且 tracker 进入 `follow_capture_distance` 后，policy 会锁住一个固定 hover setpoint，并让 SO3 controller 走 position-control 分支，避免追到以后继续围着目标盘旋。
- target 再次移动或 RViz 收到新的 `2D Nav Goal` 后，会清掉旧 hover setpoint 并重新追踪。
- target no-tracker 到达 waypoint 容差范围后，会悬停在自己已经由避障网络飞到的安全到达位置，而不是强行吸附到点击的精确点；这样 RViz 点到树或占据 voxel 附近时，不会在最后一步绕过避障逻辑。

target 行为：

- target 不是匀速脚本，也不会再直接发布虚拟 `/target/odom`。
- target 由第二套 quadrotor dynamics、SO3 controller 和 YOPO-Simple 避障 policy 驱动。
- target policy 的 `velocity / vel_max_train / acc_max_train` 会从 `YOPO/config/single_traj_opt.yaml` 继承，和 follower 测试速度保持一致。
- target 碰撞计数在 `/yopo/target_collision_counter_total`，follower 碰撞计数在 `/yopo/collision_counter_total`。

相关参数在 `YOPO/config/single_traj_opt.yaml`：

- `follow_distance`
- `follow_deadband`
- `follow_capture_distance`
- `follow_target_speed_threshold`
- `follow_max_step`
- `target_velocity_ema_alpha`
- `target_measurement_timeout`
- `target_hold_timeout`
- `use_mask_target_estimate`

常用检查：

```bash
source /opt/ros/noetic/setup.bash
rostopic echo /target/odom
rostopic echo /yopo/collision_counter_total
rostopic echo /yopo/target_collision_counter_total
rostopic hz /target_mask_image
rostopic hz /depth_image
rostopic hz /target_depth_image
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
python3 yopo_trt_transfer.py --weights-root saved/with_tracker --trial N --epoch X
python3 test_yopo_ros_single.py --use_tensorrt=1
```

## 6. 关键文件

- Simulator 配置：`Simulator/src/config/single_config.yaml`
- YOPO tracker 训练配置：`YOPO/config/single_traj_opt.yaml`
- RViz 配置：`YOPO/single_yopo.rviz`
- 一键启动脚本：`tools/single_launch.sh`
- Target no-tracker 避障 policy：`YOPO/target_avoidance/test_target_yopo_ros.py`
- 双无人机动力学 launch：`Controller/src/so3_quadrotor_simulator/launch/tracking_attitude_control.launch`

实现要点：

- 网络接口按当前 tracker 复现：depth + target bbox mask + 6D state 输入，14D per-primitive 输出。
- target head 输出 target uv/depth logits，解码为 camera/body frame 下的目标 3D 位置。
- policy node 发布 `PositionCommand`，沿用现有 SO3 controller 执行追踪轨迹。
