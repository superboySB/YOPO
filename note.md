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

首次构建会下载 PyTorch、cuDNN、cuBLAS、Open3D 等数 GB 的 wheel；大包下载期间日志可能数分钟只更新一条进度，不等于构建卡死。只要 build/pip 进程和代理连接仍有流量，就不要中断当前层。

`docker/simulation.dockerfile` 不写机器相关的 `ARG HTTP_PROXY` / `ENV http_proxy`。当前代理由宿主机 `~/.docker/config.json` 的 `proxies.default` 自动注入当前用户经 Docker CLI 发起的 build 内置代理参数和新建容器环境，因此构建和运行命令都不需要再手写 `-e http_proxy=...`。修改该配置后要重建容器，已有容器不会自动刷新环境。可用下面两条命令核对实际注入值和容器内联网：
```bash
LATEST_BUILD_REF="$(docker buildx history ls --format '{{.Ref}}' | head -n 1)"
docker buildx history inspect "$LATEST_BUILD_REF" | grep -i -E 'BUILD ARG|_PROXY'
docker exec dzp-yopo-omni bash -lc 'env | grep -i _proxy; curl -I --max-time 15 https://www.google.com'
```

允许容器使用图形界面：
```bash
xhost +local:root
HOST_XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
test -f "$HOST_XAUTHORITY"
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
  --velocity 6.0 \
  --max-depth 4.0 \
  --rviz-software-gl
```

`tools/launch_sim.sh` 默认要求 `/dev/input/js0` 存在：检测到手柄时，规划器用右摇杆生成 heading-frame 水平期望速度；设备缺失会安全退出，不会静默切到默认目标并自行前飞。确实要使用原来的 RViz `2D Nav Goal` / `/move_base_simple/goal` 控制时，必须显式传 `--no-joystick`。当前 RadioMaster 实测通道为 axis 0（左右）和 axis 1（上下）；上推为 `+x`，下推为 `-x`，左推为 `+y`，右推为 `-y`。摇杆幅度经径向死区重映射到 `0..--velocity`，回中/断连后持续发布闭环刹停与位置保持命令，并锁定 yaw。每次启动或重新连接后，要让右摇杆保持回中约 0.1 秒完成安全解锁，才会接受运动输入。

当前实现不把低杆量直接当成 checkpoint 的低速训练样本。`epoch200.pth` 始终接收训练域内的 6m/s 方向意图；操作者杆量单独作为 `0..--velocity` 的实际速度目标。网络原始端点和 score 不改，8 个候选分别经过有界速度伺服：`a*=clip(1.5*(v_cmd-v_actual), ±4m/s²)`，端点尺度严格限制在 `[0,1]`、绝不外扩；空间收缩仍不够时，单计划时域才从训练的 1.4 秒缩短到最低 1.0 秒，并对缩短方向做低通和每帧限速。候选还要通过下次重规划前的加速度检查和锁高改写深度 veto，最后只在安全候选中按 checkpoint 原 score 选最小；全拒就持续 `EMPTY` hold。因此无需重新采集数据或训练。`/yopo/vdes_body` 发布操作者实际命令；飞机倾斜时，为保持世界水平运动，瞬时 body-frame 向量允许出现非零 z 分量。

右杆只控制 heading-frame 水平速度；世界 z 在第一次有效拨杆时锁定，并在运动、回中、断连和重连后保持同一高度。默认锁高 veto 使用未做 resize/nan-to-num 的 8×8 米制深度：近场可见碰撞、坏像素、球心落入相机盲区都会拒绝。默认 `--joystick-strict-footprint 0` 允许球心仍在 FoV 内、但 0.45m 检查球边缘被 FoV 裁切的候选继续使用原 YOPO score，并在日志中明确 residual risk；需要整颗检查球都落入 FoV 时传 `--joystick-strict-footprint 1`，极端姿态下可能保守地全候选 hold。

仿真器发布的 `32FC1` 深度默认按米处理，并由 `--max-depth`（默认 4m）归一化；`16UC1` 默认按毫米处理。只有接入本来就是 `[0,1]` 的 `32FC1` 数据源时才给 planner 显式加 `--depth-normalized`，不要再依赖图像最大值猜测单位。

安全链路包括：

- 回中、断连或显著换向时推进 intent epoch，旧推理不能重新接管，并持续发布闭环 hold；
- 只有“当前意图的计划成功提交”才刷新默认 0.20 秒 depth/plan watchdog，原始深度到达但推理失败不能续命；
- 单计划超过自己的 1.0–1.4 秒时域仍未更新也会 hold；
- 默认开启锁高改写深度 veto，全候选拒绝时 fail-closed；
- `network_control_node` 另有默认 0.30 秒 `PositionCommand` watchdog，planner/GPU 整体停更时以 50Hz 捕获并保持当前位置，新命令到达后恢复。

回中命令的目标速度会立即变为 0，但物理速度需要控制器完成减速，不能把“杆为 0”理解成速度瞬间跳零。

宿主机或容器内可用下面的命令重新校准原始通道：
```bash
jstest --event /dev/input/js0
```

若同时接入多个输入设备，可使用稳定的 by-id 路径：
```bash
bash tools/launch_sim.sh \
  --joystick-device /dev/input/by-id/usb-NATIONS_RADIOMASTER_SIM_N32G45x-joystick \
  --weight /workspace/YOPO/YOPO/saved/YOPO_0/epoch200.pth
```

不启动 ROS 和飞行控制，仅检查右摇杆的二维 heading-frame 速度映射：
```bash
python3 tools/test_joystick_mapping.py --device /dev/input/js0 --speed 6
```
该命令会持续读取真实设备，完成上下左右检查后按 `Ctrl-C` 退出。

需要在仿真日志中同时查看原始轴值和映射后的速度比例时：
```bash
bash tools/launch_sim.sh --joystick-calibrate \
  --weight /workspace/YOPO/YOPO/saved/YOPO_0/epoch200.pth
```

仿真启动后可直接查看手柄产生的实际 body-frame 期望速度向量（网络内部会按上文所述使用训练域内的方向意图）：
```bash
rostopic echo /yopo/vdes_body
```

需要可重复地验收 axis 事件解析、四方向、半/满杆比例、回中停稳以及完整模型/控制器/动力学闭环时，可用 FIFO 代替人工拨杆；它不会写真实 `/dev/input/js0`：
```bash
test -p /tmp/yopo-js-test || mkfifo /tmp/yopo-js-test
bash tools/launch_sim.sh --no-rviz --no-sensor \
  --joystick-device /tmp/yopo-js-test \
  --weight /workspace/YOPO/YOPO/saved/YOPO_0/epoch200.pth

source /opt/ros/noetic/setup.bash
source /workspace/YOPO/Controller/devel/setup.bash
python3 tools/joystick_e2e_test.py \
  --fifo /tmp/yopo-js-test \
  --publish-clear-depth \
  --summary-only \
  --output /tmp/joystick_e2e.json
```

验收器返回码为 0 才表示通过，并会检查 `/yopo/vdes_body`、`/so3_control/pos_cmd` 和 `/sim/odom`，而不只是输入映射。它在每个方向后立即检查实际运动、比例、锁高、固定 yaw 和回中停稳，并持续监控高度、速度、位置范围、odom 新鲜度及非有限值，越界会先归中再中止。`--no-sensor` 与 `--publish-clear-depth` 必须配套，不能同时混入真实 sensor。

单独验证“满杆保持不变，只停四路深度后 0.20 秒 hold，再恢复深度自动恢复飞行”：
```bash
python3 tools/depth_watchdog_e2e_test.py \
  --fifo /tmp/yopo-js-test \
  --output /tmp/depth_watchdog_e2e.json
```

真实 `sensor_simulator_cuda` 地图测试不要传 `--no-sensor` / `--publish-clear-depth`，并把碰撞累计值变成硬门禁。复杂地图中某方向被障碍封住而安全 hold 是正确行为；可用 `--directions right` 只测试当前可通方向：
```bash
test -p /tmp/yopo-map-js || mkfifo /tmp/yopo-map-js
bash tools/launch_sim.sh --session yopo_map --no-rviz \
  --velocity 3.0 \
  --joystick-device /tmp/yopo-map-js \
  --weight /workspace/YOPO/YOPO/saved/YOPO_0/epoch200.pth

python3 tools/joystick_e2e_test.py \
  --fifo /tmp/yopo-map-js \
  --max-speed 3.0 \
  --directions right \
  --max-actual-projection-slope 0.5 \
  --require-collision-topic \
  --collision-topic /yopo/collision_counter_total \
  --summary-only \
  --output /tmp/joystick_map_e2e.json
```

本次最终代码的闭环结果：6m/s clear-depth 四向 4 秒和 8 秒压力测试均通过；8 秒测试半杆投影 2.47–2.53m/s、满杆 5.20–5.22m/s、锁高最大漂移 1.26cm、9218 次安全检查无违规。深度断流专测在 0.201 秒出现首个零命令、0.280 秒确认连续 hold，恢复深度后 0.209 秒恢复飞行。seed=3 森林地图以 3m/s 做真实渲染深度测试，right 半/满杆分别达到 1.290/2.614m/s，1813 个碰撞计数样本保持 0。

安全边界：当前“真实 sensor”仍是地图渲染的 `sensor_simulator_cuda`，不是物理 ToF。物理接入必须匹配四路 8×8、45° FoV、固定朝向、30ms 内时间同步、`32FC1` 米或 `16UC1` 毫米以及 odom/控制话题。锁高 veto 只处理“把 YOPO 三维轨迹压到固定高度”新增的可见风险，不重做全局规划；偏差不超过 0.20m、球边缘裁切部分及 4m 量程外仍依赖原 YOPO score。4m/15Hz 传感器存在盲区；6m/s 在 4m/s² 理想恒减速下仅刹车距离就是 4.5m，已超过量程，因此满速障碍飞行未获硬安全保证。碰撞计数只检查离散 odom 时刻的机体中心，也不等同于连续体积碰撞认证。

进入 tmux：
```bash
tmux attach -t yopo_sim
```

停止仿真：
```bash
tmux kill-session -t yopo_sim
```

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
