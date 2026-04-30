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


如果训练得到新的 tracker `YOPO/saved/with_tracker/YOPO_0/epoch50.pth`，前机仍用 `YOPO/saved/no_tracker/YOPO_1/epoch50.pth`，就改成：

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

行为树要点：

- target 在视野内且距离较远时，follower 使用 tracker 网络候选轨迹，保留避障+tracking 的联合选择。
- target 停止且 follower 进入 `follow_capture_distance` 后，follower 锁住固定观察点，避免追到后绕圈。
- target mask 为空或过期超过 `target_lost_timeout` 时，follower 进入 lost-target brake：锁定当前安全位置刹停，并朝最后一次目标估计方向转头等待重捕获，避免目标被障碍遮挡或飞出视野后继续无目标前冲。
- target 再次移动或 RViz 收到新的 `2D Nav Goal` 后，会清掉旧 hover / lost 状态并重新追踪。

相关参数在 `YOPO/config/single_traj_opt.yaml`：

- `velocity`：后机 follower / YOPOv2-Tracker 测试推理速度，单位是 m/s。它控制 tracker primitive 的速度尺度。
- `target_velocity`：前机 target / no-tracker YOPO 测试推理速度，单位是 m/s。只影响被追的那架无人机；不影响后机。
- `target_acc_max`：前机 target / no-tracker YOPO 测试推理加速度上限，单位是 m/s^2。只影响被追的那架无人机；不影响后机。
- `follow_distance`：follower 追到 target 后希望保持的水平观察距离，单位是米。当前用于生成 target 后方的 standoff/hover 点；距离设得稍大，可以给 target 横向移动留出更多视野余量。
- `follow_deadband`：观察距离的死区宽度，单位是米。距离在 `follow_distance ± follow_deadband` 内时，follower 不会因为微小误差反复前后修正，从而减少追到后绕圈或抖动。
- `follow_capture_distance`：进入近距离保持逻辑的最大水平距离，单位是米。target 停止且 follower 小于这个距离后，才会从网络轨迹切到固定观察点 position-control；它需要大于 `follow_distance + follow_deadband`。
- `follow_target_speed_threshold`：判断 target 是否已经停止的速度阈值，单位是 m/s。target 速度低于这个值时，follower 可以进入 hover/保持距离逻辑；高于这个值时继续用 tracker 网络追。
- `follow_max_step`：一次保持距离修正允许移动的最大步长，单位是米。防止 standoff 点突然跳得太远，导致控制指令过激。
- `target_velocity_ema_alpha`：target 估计速度的指数滑动平均系数。值越大越跟随最新测量，值越小越平滑。
- `target_measurement_timeout`：target 测量在运动状态下允许过期的时间，单位是秒。超过后，如果没有可靠 hover 状态，就不再信任旧估计。
- `target_hold_timeout`：target 静止时允许保留旧 target 估计的时间，单位是秒。静止目标可以比运动目标保留更久，避免检测短暂丢帧就立刻退出保持逻辑。
- `use_mask_target_estimate`：是否优先用 `/target_mask_image` 加 `/depth_image` 反投影估计 target 三维位置。开启时，检测框得到的位置优先于网络 target head。
- `target_mask_timeout`：检测框消息和深度帧允许的最大时间差，单位是秒。超过后认为 mask 过期，不再用于当前帧。
- `target_lost_timeout`：target mask 丢失后进入 lost-target brake 的等待时间，单位是秒。超过这个时间还没重新看到 target，follower 会刹停等待重捕获。
- `target_mask_min_pixels`：判断检测框有效所需的最少前景像素数。低于这个数量时认为 mask 为空或噪声。

常用指标检查：
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
