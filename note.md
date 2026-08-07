# YOPO-Omni Docker 速记

## 配置
在宿主机进入项目根目录：
```bash
cd /workspace/YOPO
git switch omni-transformer
```

构建镜像：
```bash
docker build -f docker/simulation.dockerfile \
  -t dzp_yopo:omni-u2004-noetic-py38 \
  --network=host --progress=plain .
```

允许容器使用图形界面：
```bash
xhost +local:root
```

启动容器：
```bash
docker run --name dzp-yopo-omni -itd --privileged --gpus all --network host \
  --entrypoint bash \
  -e DISPLAY -e QT_X11_NO_MITSHM=1 \
  -e http_proxy=http://127.0.0.1:8889 \
  -e https_proxy=http://127.0.0.1:8889 \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --shm-size=4g \
  -v /workspace/YOPO:/workspace/YOPO \
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
默认采集 10 张地图，每张地图 10000 个位姿；每个位姿渲染 front/left/right/back 四个 TOFSense-M 等效 ToF 深度图，并展开 8 个 desired direction 样本。

ToF 输入为 8x8 pixels、水平/垂直 45 度 FoV、65 度对角 FoV、1.5cm 到 4m 量程；前向高清 debug 图只用于观察，不参与训练。

重新采集前删除旧数据和旧模型：
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
训练 50 epoch：
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
启动 roscore、控制器、四向深度传感器、Omni 规划器和 RViz。脚本会按顺序等待各 ROS 节点启动，RViz 图形窗口通常会在命令执行后约 14 秒弹出：
```bash
cd /workspace/YOPO

bash tools/launch_sim.sh \
  --weight /workspace/YOPO/YOPO/saved/YOPO_0/epoch200.pth \
  --python python3 \
  --velocity 6.0 \
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

发布新目标点：
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
