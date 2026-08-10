# YOPO-Omni Docker 速记

## 配置
在宿主机进入项目根目录：
```bash
cd /home/dzp/projects/YOPO
git switch omni-transformer
```

构建镜像：
```bash
docker build -f docker/simulation.dockerfile \
  -t dzp_yopo:omni-u2004-noetic-py38 \
  --progress=plain .
```

启动容器：
```bash
docker run --name dzp-yopo-omni -itd \
  --privileged \
  --gpus all \
  --network host \
  --entrypoint bash \
  -e DISPLAY \
  -e XAUTHORITY=/root/.Xauthority \
  -e QT_X11_NO_MITSHM=1 \
  -v "$HOST_XAUTHORITY:/root/.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/input:/dev/input \
  --shm-size=4g \
  -v /home/dzp/projects/YOPO:/workspace/YOPO \
  dzp_yopo:omni-u2004-noetic-py38
```

进入容器：
```bash
docker exec -it dzp-yopo-omni /bin/bash
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

## 采集 YOPO-Omni 数据

> 当前手柄辅助驾驶直接使用 `YOPO/saved/YOPO_0/epoch200.pth`，不需要重新采集或训练。不要为运行辅助驾驶执行下面的删除命令。

默认采集 10 张地图，每张地图 10000 个位姿；每个位姿渲染 front/left/right/back 四个 TOFSense-M 等效 ToF 深度图，并展开 8 个 desired direction 样本。

ToF 输入为 8x8 pixels、水平/垂直 45 度 FoV、65 度对角 FoV、1.5cm 到 4m 量程；前向高清 debug 图只用于观察，不参与训练。

只有明确要从零训练、且已备份模型时，才清理旧数据和旧模型：
```bash
cd /workspace/YOPO
rm -rf dataset_omni
rm -rf YOPO/saved/YOPO_0
```

采集正式数据集：
```bash
cd /workspace/YOPO
python3 tools/run_yopo_pipeline.py \
  --mode generate \
  --env-num 10 \
  --image-num 10000 \
  --save-path ../dataset_omni
```

生成结果：
```text
dataset_omni/
  0/img_0_front.png
  0/img_0_left.png
  0/img_0_right.png
  0/img_0_back.png
  0/img_0_debug_front.png
  pose-0.csv
  samples-0.csv
  guides-0.csv
  pointcloud-0.ply
```

`img_*_front/left/right/back.png` 是 8x8 TOFSense-M 深度图。每个 pixel 内部用多条子射线做小视锥聚合，再按 4m 量程截断并保存为 16-bit PNG，读取时归一化到 `[0, 1]`。

`img_*_debug_front.png` 是 160x90 前向高清深度图，只用于人工检查采集场景，训练代码不会读取它。

## 训练 YOPO-Omni
训练 200 epoch：
```bash
cd /workspace/YOPO
python3 tools/run_yopo_pipeline.py \
  --mode train \
  --python python3 \
  --dataset-path ../dataset_omni \
  --train-epoch 200 \
  --batch-size 16 \
  --num-workers 4
```

查看 TensorBoard：
```bash
cd /workspace/YOPO/YOPO/saved
tensorboard --logdir=./
```

## 仿真测试
启动 roscore、控制器、四向深度传感器、Omni 规划器和 RViz。脚本使用固定启动延时（controller 3 秒、sensor 7 秒、planner 11 秒、RViz 14 秒）；看到控制器输出 `TakeOff Done! Ready to Flight` 后再拨杆：
```bash
cd /workspace/YOPO

bash tools/launch_sim.sh \
  --weight /workspace/YOPO/YOPO/saved/YOPO_0/epoch200.pth \
  --python python3 \
  --velocity 3.0 \
  --max-depth 4.0 \
  --rviz-software-gl
```

进入 tmux：
```bash
tmux attach -t yopo_sim
```

停止仿真：
```bash
tmux kill-session -t yopo_sim
```

注意，`tools/launch_sim.sh` 默认要求 `/dev/input/js0` 存在；设备缺失会退出，不会静默改成 Nav Goal 自行前飞。确实要用原来的 RViz `2D Nav Goal` 时必须显式传 `--no-joystick`。

当前 `NATIONS RADIOMASTER SIM` 的实测映射如下。这里的数值是 `jstest --event` 显示的 Linux 原始轴值，不是 ROS Joy 消息编号：

- 右杆 axis 1：`+32767` 前进，`-32767` 后退；
- 右杆 axis 0：`-32767` 左移，`+32767` 右移；
- 左杆 axis 2：`+32767` 上升，`-32767` 下降；
- 左杆 axis 3：`-32767` 左转，`+32767` 右转。

对应的默认启动参数是：

| 操作 | Linux 轴 | 原始方向 | 启动参数 | 默认反向参数 |
|---|---:|---|---|---:|
| 右杆左右（左移/右移） | 0 | 左负、右正 | `--joystick-axis-x 0` | `--joystick-invert-x 1` |
| 右杆上下（前进/后退） | 1 | 上正、下负 | `--joystick-axis-y 1` | `--joystick-invert-y 0` |
| 左杆上下（上升/下降） | 2 | 上正、下负 | `--joystick-axis-z 2` | `--joystick-invert-z 0` |
| 左杆左右（左转/右转） | 3 | 左负、右正 | `--joystick-axis-yaw 3` | `--joystick-invert-yaw 1` |

`--joystick-axis-x/y/z/yaw` 表示“控制功能”，不要求新手柄也使用相同的物理 axis 编号。`invert=1` 表示把该 Linux 原始值乘以 `-1`。右杆默认还使用 `--joystick-swap-xy 1`，即物理上下轴映射前后速度、物理左右轴映射侧向速度。

注意：右杆有输入时，`epoch200.pth` 持续根据四向 ToF 规划水平避障轨迹；左杆升降和偏航由外层控制器同时执行，不会覆盖模型的水平 `x/y` 轨迹。但升降方向没有独立的 YOPO 垂直避障保证；只有左杆、右杆回中时，不会产生水平规划运动。


## RViz 与状态检查
`YOPO/yopo.rviz` 已配置四个 ToF 深度图面板和一个前向高清 debug 面板：
```text
/depth_image_front   # 8x8 ToF
/depth_image_left    # 8x8 ToF
/depth_image_right   # 8x8 ToF
/depth_image_back    # 8x8 ToF
/depth_image         # 160x90 front debug，只用于观察
```

检查四向 ToF 频率：
```bash
source /opt/ros/noetic/setup.bash
rostopic hz /depth_image_front /depth_image_left /depth_image_right /depth_image_back
```

检查图像尺寸：
```bash
rostopic echo -n 1 /depth_image_front | grep -E "height|width|encoding"
rostopic echo -n 1 /depth_image | grep -E "height|width|encoding"
```

检查碰撞计数：
```bash
rostopic echo /yopo/collision_counter_total
```

正常运行时可看到：
```text
/depth_image_front 约 15 Hz
/depth_image_left  约 15 Hz
/depth_image_right 约 15 Hz
/depth_image_back  约 15 Hz
/depth_image_front height=8,width=8
/depth_image       height=90,width=160
/yopo/collision_counter_total: 0
```

仅在 `tools/launch_sim.sh --no-joystick` 模式下发布新目标点：
```bash
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped "{
  header: {frame_id: 'world'},
  pose: {
    position: {x: 10.0, y: 0.0, z: 2.0},
    orientation: {w: 1.0}
  }
}"
```

## 常用配置
### 1. 配置容器代理
需要提前配置宿主机的代理 `~/.docker/config.json` 如下
```json
{
  "proxies": {
    "default": {
      "httpProxy": "http://172.17.0.1:7897",
      "httpsProxy": "http://172.17.0.1:7897",
      "noProxy": "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
    }
  }
}
```
其中，`proxies.default` 自动注入当前用户经 Docker CLI 发起的 build 内置代理参数和新建容器环境，因此构建和运行命令都不需要再手写 `-e http_proxy=...`。修改该配置后要重建容器，已有容器不会自动刷新环境。可用下面两条命令核对实际注入值和容器内联网：
```bash
LATEST_BUILD_REF="$(docker buildx history ls --format '{{.Ref}}' | head -n 1)"
docker buildx history inspect "$LATEST_BUILD_REF" | grep -i -E 'BUILD ARG|_PROXY'
docker exec dzp-yopo-omni bash -lc 'env | grep -i _proxy; curl -I --max-time 15 https://www.google.com'
```

### 2. 配置图形界面
允许容器使用图形界面：
```bash
xhost +local:root
HOST_XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
test -f "$HOST_XAUTHORITY"
```

### 3. 数据相关config
数据生成配置：
```text
Simulator/src/config/config.yaml
```

关键项：
```yaml
save_path: "../dataset_omni/"
env_num: 10
image_num: 10000
depth_fps: 15
tof:
  model: "tofsense_m"
  image_width: 8
  image_height: 8
  horizontal_fov_deg: 45.0
  vertical_fov_deg: 45.0
  diagonal_fov_deg: 65.0
  fx: 9.656854
  fy: 9.656854
  cx: 3.5
  cy: 3.5
  zone_subsample: 4
  depth_quantile: 0.35
  noise_std: 0.015
  far_noise_std: 0.08
  signal_floor: 0.08
  max_depth_dist: 4.0
  min_depth_dist: 0.015
omni:
  direction_num: 8
  goal_length: 10.0
  dijkstra_resolution: 0.5
  dijkstra_inflation: 0.5
  astar_local_radius: 16.0
```

### 4. 训练相关config
训练配置：
```text
YOPO/config/traj_opt.yaml
```

关键项：
```yaml
dataset_path: "../dataset_omni"
image_height: 8
image_width: 8
tof_horizontal_fov_deg: 45.0
tof_vertical_fov_deg: 45.0
omni_topology_num: 8
omni_d_model: 128
omni_num_heads: 4
omni_decoder_layers: 2
omni_amp: true
omni_num_workers: 4
sgm_time: 1.4
```

说明：
- `tools/run_yopo_pipeline.py` 的 `--save-path ../dataset_omni` 会同时覆盖训练用的 `dataset_path`。
- 不加 `--keep-config` 时，脚本运行结束会恢复原始配置文件。
- 在线测试订阅四向深度图，输出 8 个 topology 的候选轨迹和 score。

### 5. 更换手柄配置


更换手柄时按下面步骤配置，不需要修改模型或重新训练：

1. 查找稳定设备路径，并确认容器中也能看到设备：

   ```bash
   ls -l /dev/input/by-id/*joystick* /dev/input/js* 2>/dev/null
   docker exec dzp-yopo-omni bash -lc 'ls -l /dev/input/by-id/*joystick* /dev/input/js* 2>/dev/null'
   ```

2. 停止正在运行的飞行会话，在宿主机或容器里执行 `jstest`，依次只推动一个方向，记录右左右上下、左上下左右各自变化的 axis 编号、中心值和正负号：

   ```bash
   tmux kill-session -t yopo_sim 2>/dev/null || true
   jstest --event /dev/input/js0
   ```

   理想的自回中轴中心应接近 0，满行程通常接近 `-32767/+32767`。如果中心长期偏离零，先做系统手柄校准，或适当增大 `--joystick-deadzone`。按 `Ctrl-C` 结束 `jstest` 后再启动仿真。

3. 优先通过启动参数覆盖映射，不要为了换一只手柄直接修改 Python。下面是假设新手柄右杆为 axis 3/4、左杆为 axis 1/0，且右杆上下方向需要反转的示例：

   ```bash
   bash tools/launch_sim.sh \
     --joystick-device /dev/input/by-id/你的手柄-joystick \
     --joystick-axis-x 3 \
     --joystick-axis-y 4 \
     --joystick-axis-z 1 \
     --joystick-axis-yaw 0 \
     --joystick-invert-x 1 \
     --joystick-invert-y 1 \
     --joystick-invert-z 0 \
     --joystick-invert-yaw 1 \
     --joystick-swap-xy 1 \
     --joystick-deadzone 0.08 \
     --velocity 2.0 \
     --joystick-vertical-velocity 2.0 \
     --joystick-yaw-rate 1.0 \
     --weight /workspace/YOPO/YOPO/saved/YOPO_0/epoch200.pth
   ```

   如果某个方向相反，只切换对应的 `--joystick-invert-* 0/1`；如果右杆上下变成侧移、左右变成前后，切换 `--joystick-swap-xy 0/1`。四个 `--joystick-axis-*` 必须是互不相同的非负整数。如果设备满行程并非约 32767，还要给正式启动传入对应的 `--joystick-axis-max`，给映射工具传入相同的 `--axis-max`。

4. 启动飞行前可用同一组参数做纯映射检查。这个工具只读取遥控器，不启动 ROS、模型或电机：

   ```bash
   python3 tools/test_joystick_mapping.py \
     --device /dev/input/js0 \
     --axis-x 0 --axis-y 1 --axis-z 2 --axis-yaw 3 \
     --invert-x 1 --invert-y 0 --invert-z 0 --invert-yaw 1 \
     --swap-xy 1 --speed 2 --vertical-speed 2 --yaw-rate 1
   ```

   输出中的 `vdes_heading=(forward,left,up)` 应满足：前/左/上为正，后/右/下为负；`yaw_rate` 左转为正、右转为负。完成后按 `Ctrl-C` 退出。

5. 正式启动或设备重新连接后，必须让四个配置轴同时回中并保持约 0.1 秒。日志出现 `Joystick unlocked after all four configured axes initialized...` 后才会接受运动输入。可以用 `rostopic echo /yopo/vdes_body` 再确认回中为零、各方向符号正确。

两根摇杆默认有 8% 中心死区，越过死区后的杆量连续重映射到满量程。右杆最大水平期望速度由 `--velocity` 设置，左杆最大升降速度由 `--joystick-vertical-velocity` 设置，最大偏航角速度由 `--joystick-yaw-rate` 设置。

若确认某套映射要成为项目的新默认值，再修改以下位置：

- `tools/launch_sim.sh` 顶部的 `JOYSTICK_AXIS_*`、`JOYSTICK_INVERT_*`、`JOYSTICK_SWAP_XY` 和速度默认值；
- `YOPO/test_yopo_ros.py` 中 `--joystick-axis-*`、`--joystick-invert-*` 等 argparse 默认值，保证直接运行 Python 时一致；
- `tools/test_joystick_mapping.py` 的 argparse 默认值，保证校准工具与正式启动一致；
- `YOPO/joystick_control.py` 的 `map_dual_sticks()` 是通用映射公式，普通换手柄不应修改；只有确实改变控制语义时才改这里，并重新执行纯映射检查和低速仿真确认。

建议始终优先使用 `/dev/input/by-id/...-joystick`，因为 `/dev/input/js0` 在插拔多个输入设备后可能变成 `js1`。当前 Docker 启动命令已经映射整个 `/dev/input`，所以无需再修改镜像。

右杆的水平速度向量直接进入 `epoch200.pth`：规划器将世界速度、加速度和操作者期望速度转换到机体系，拼成 `[v_body, a_body, vdes_body]`，网络对八条轨迹输出 endpoint 和 score，直接选择原始最低 score，并按训练时的 1.4 秒时域发布轨迹。正常手柄路径不再运行外层速度伺服、端点缩放、锁高改写、候选 veto 或 depth-plan hold；不会因为这些保底条件把有效模型输出替换成悬停。

`epoch200.pth` 的训练数据把期望速度幅值采在 3–6m/s，而且 intent loss 会归一化 `vdes`，所以它可靠表达的是水平意图方向，不保证飞机实际速度严格正比于杆量。杆量到 `/yopo/vdes_body` 的映射是精确连续的，但低杆和满杆可能得到相近的物理速度；若必须精确比例调速，需要模型本身覆盖这个监督目标，本实现不会再用模型外的“速度保底层”伪造它。

离线检查还确认该 checkpoint 的八个拓扑主要是水平轨迹：纯 `+z/-z` 意图仍会产生明显水平 endpoint。因此左杆升降作为核心人工控制量直接进入飞控的世界系垂直速度闭环，而不是强塞进不具备竖直拓扑的网络；右杆水平仍由 YOPO 避障轨迹控制。左杆左右直接积分为 yaw 参考和 yaw rate。`/yopo/vdes_body` 始终发布操作者完整的三维期望速度，左右杆也可以同时使用。

平移杆回中时，控制从 YOPO 的 `READY` 模式立即切到与 YOPO-Simple 相同的 `EMPTY` 位置/速度闭环，但不会把速度字段突然清零。规划器从最后一帧 p/v/a 接一段连续五次多项式制动参考，平滑减速到零后保持最终位置；这不是 Nav Goal，也不会重新引入 target 规划。
