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
默认采集 10 张地图，每张地图 10000 个位姿；每个位姿渲染 front/left/right/back 四张深度图，并展开 8 个 desired direction 样本。

重新采集前可删除旧数据：
```bash
cd /workspace/YOPO
rm -rf dataset_omni
```

采集正式数据集：
```bash
cd /workspace/YOPO
python3 tools/run_yopo_omni_pipeline.py \
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
  pose-0.csv
  samples-0.csv
  guides-0.csv
  pointcloud-0.ply
```

## 训练 YOPO-Omni
训练 50 epoch：
```bash
cd /workspace/YOPO
python3 tools/run_yopo_omni_pipeline.py \
  --mode train \
  --python python3 \
  --dataset-path ../dataset_omni \
  --train-epoch 50 \
  --batch-size 16 \
  --num-workers 4
```

如果想一键采集并训练：
```bash
cd /workspace/YOPO
python3 tools/run_yopo_omni_pipeline.py \
  --mode all \
  --python python3 \
  --env-num 10 \
  --image-num 10000 \
  --save-path ../dataset_omni \
  --train-epoch 50 \
  --batch-size 16 \
  --num-workers 4
```

训练输出：
```text
YOPO/saved/YOPO_Omni_0/epoch10.pth
YOPO/saved/YOPO_Omni_0/epoch20.pth
YOPO/saved/YOPO_Omni_0/epoch30.pth
YOPO/saved/YOPO_Omni_0/epoch40.pth
YOPO/saved/YOPO_Omni_0/epoch50.pth
```

查看 TensorBoard：
```bash
cd /workspace/YOPO/YOPO/saved
tensorboard --logdir=./
```

## 检查模型
检查 `epoch50.pth` 是否能读取正式数据集并正常推理。这个命令只在终端打印结果，不会打开 RViz 窗口：
```bash
cd /workspace/YOPO
python3 tools/test_yopo_omni_checkpoint.py \
  --weight YOPO/saved/YOPO_Omni_0/epoch50.pth \
  --dataset-path ../dataset_omni \
  --split valid \
  --batch-size 4 \
  --num-batches 8 \
  --device cuda \
  --strict-depth-range
```

正常输出应包含：
```text
Depth normalization check passed
output endstate=(4, 8, 8, 9), score=(4, 8, 8)
```

要看 RViz 可视化，请运行下面“仿真测试”里的 `tools/run_yopo_omni_sim.sh`。

## 仿真测试
启动 roscore、控制器、四向深度传感器、Omni 规划器和 RViz。脚本会按顺序等待各 ROS 节点启动，RViz 图形窗口通常会在命令执行后约 14 秒弹出：
```bash
cd /workspace/YOPO

bash tools/run_yopo_omni_sim.sh \
  --weight /workspace/YOPO/YOPO/saved/YOPO_Omni_0/epoch50.pth \
  --python python3 \
  --velocity 6.0 \
  --rviz-software-gl
```

进入 tmux：
```bash
tmux attach -t yopo_omni_sim
```

停止仿真：
```bash
tmux kill-session -t yopo_omni_sim
```

## RViz 与状态检查
`YOPO/yopo_omni.rviz` 已配置四个深度图面板：
```text
/depth_image_front
/depth_image_left
/depth_image_right
/depth_image_back
```

检查四向深度频率：
```bash
source /opt/ros/noetic/setup.bash
rostopic hz /depth_image_front /depth_image_left /depth_image_right /depth_image_back
```

检查碰撞计数：
```bash
rostopic echo /yopo/collision_counter_total
```

正常运行时可看到：
```text
/depth_image_front 约 33 Hz
/depth_image_left  约 33 Hz
/depth_image_right 约 33 Hz
/depth_image_back  约 33 Hz
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
omni_topology_num: 8
omni_d_model: 128
omni_num_heads: 4
omni_decoder_layers: 2
omni_amp: true
omni_num_workers: 4
sgm_time: 1.4
```

说明：
- `tools/run_yopo_omni_pipeline.py` 的 `--save-path ../dataset_omni` 会同时覆盖训练用的 `dataset_path`。
- 不加 `--keep-config` 时，脚本运行结束会恢复原始配置文件。
- 在线测试订阅四向深度图，输出 8 个 topology 的候选轨迹和 score。
