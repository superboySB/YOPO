# yopo-minco-test：完整复现与双机交互对比

本文对应分支 `yopo-minco-test`。它以本仓库 YOPO-Simple 为基线，合入 TJU-Aerial-Robotics 官方 `YOPO-MINCO`，并保留 Simple 推理运行时，形成可采集、可训练、可评估、可双机交互复测的一套实现。

## 1. 可追溯来源与交付物

- 官方上游：`https://github.com/TJU-Aerial-Robotics/YOPO/tree/YOPO-MINCO`
- 同步的上游 commit：`3295a805f32779941293dc0894b8eee4971bc818`
- 本分支上游合并 commit：`e1ee428`；合并时 58 个上游管理路径已逐字节核对一致。
- Dockerfile 采用 `origin/omni-transformer` 的无代理写死版本。
- 镜像：`dzp_yopo:minco-test-u2004-noetic-py38`
- 保留容器：`dzp-yopo-minco-test`
- 官方 MINCO 模型：`YOPO/saved/yopo-minco/epoch50.pth`
- 本机完整重训：`YOPO/saved/yopo-minco/epoch50-retrained.pth`
- MINCO 最新事件：`YOPO/saved/yopo-minco/events.out.tfevents.minco.latest`
- Simple 模型：`YOPO/saved/yopo-simple/epoch50.pth`
- Simple 最新事件：`YOPO/saved/yopo-simple/events.out.tfevents.simple.latest`

三个 checkpoint 的 SHA-256（用 `sha256sum` 复核）：

```text
09ec31094ed83e09702efc1facf18076564ebbe1afc164960c41036e16ba229f  yopo-simple/epoch50.pth
2d85e712c8874d11cd4e523777698eff78914fdb8f1ffb5068b037f8f972f62f  yopo-minco/epoch50.pth
45d52e0c33e69d0187112cba9eec2eecdf478a23cfdde795fa869a05d5ed4f67  yopo-minco/epoch50-retrained.pth
```

`YOPO/saved/YOPO_<n>/` 是中间训练目录并被忽略；上列最终模型和最新 tfevents 明确不忽略。`dataset*`、编译目录和普通运行结果也被忽略，避免提交几十 GB 的可再生数据。

## 2. Docker：不写死代理

宿主机执行：

```bash
cd /home/dzp/projects/YOPO
docker build -f docker/simulation.dockerfile \
  -t dzp_yopo:minco-test-u2004-noetic-py38 \
  --progress=plain .
```

Dockerfile 没有 `http_proxy`、`https_proxy` 或个人代理地址。如果现场确实需要代理，只在本次 shell/Docker daemon 中配置，不要提交。

本机已按以下等价方式创建最终容器；`XAUTHORITY` 使用宿主机实际路径：

```bash
HOST_XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
docker run --name dzp-yopo-minco-test -itd \
  --privileged --gpus all --network host --entrypoint bash \
  -e DISPLAY -e XAUTHORITY=/root/.Xauthority -e QT_X11_NO_MITSHM=1 \
  -v "$HOST_XAUTHORITY:/root/.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/input:/dev/input \
  --shm-size=4g \
  -v /home/dzp/projects/YOPO:/workspace/YOPO \
  dzp_yopo:minco-test-u2004-noetic-py38
```

```bash
docker start dzp-yopo-minco-test
docker exec -it dzp-yopo-minco-test bash
cd /workspace/YOPO
```

已验证 Python 3.8、Torch 2.4.1+cu118、CUDA 和 RTX 4070 Ti SUPER（16 GB）；OpenCV 4.11、Open3D 0.19、SciPy 1.10.1、scikit-learn 1.3.2 均可 import。

## 3. 编译

```bash
set +u
source /opt/ros/noetic/setup.bash
set -u

cd /workspace/YOPO/Controller
catkin_make -j8

cd /workspace/YOPO/Simulator
source /workspace/YOPO/Controller/devel/setup.bash
catkin_make -j8
```

当前分支已从源码编译通过。CMake 会打印 Ubuntu 20.04/PCL 的 VTK 可选工具警告和 Eigen CUDA annotation 警告，但 `quadrotor_simulator_so3`、`network_control_node`、`sensor_simulator_cuda`、`dataset_generator` 都成功生成。

## 4. 双机 RViz 交互对比（推荐入口）

```bash
cd /workspace/YOPO
tools/launch_compare.sh
```

后台、无 RViz 和停止：

```bash
tools/launch_compare.sh --detach
tools/launch_compare.sh --headless --detach --session yopo-headless
tools/launch_compare.sh --session yopo-headless --stop
```

选择 checkpoint/参数：

```bash
tools/launch_compare.sh \
  --simple-weight /workspace/YOPO/YOPO/saved/yopo-simple/epoch50.pth \
  --minco-weight /workspace/YOPO/YOPO/saved/yopo-minco/epoch50-retrained.pth \
  --velocity 6.0 --safe-radius 0.05 \
  --simple-y -0.4 --minco-y 0.4
```

启动器使用独立 `roscore` tmux 窗口，`--stop` 等待 master 真正退出，连续停止/重启不会发生旧 master 竞态。两个 planner 默认 `--wait-for-goal`，首个 goal 前保持静止。

在 RViz 点击 `2D Nav Goal` 并选一点，只发布一次 `/move_base_simple/goal`，两机同时响应：

- 橙色飞机/轨迹是 `yopo-simple`，青色是 `yopo-minco`；
- 飞机上方显示名称、累计路程和碰撞数；目标上方显示本 run 汇总；
- MINCO 额外显示两段轨迹、候选轨迹和预测 safe corridor；
- 两个深度面板分别来自 `/yopo_simple/depth_image` 和 `/yopo_minco/depth_image`。

实时 JSON：

```bash
source /opt/ros/noetic/setup.bash
rostopic echo /yopo_compare/metrics
```

两机都进入默认 5 m 到达半径后，一行 JSON 追加到 `results/comparison_latest.jsonl`。字段含 `arrival_seconds`、`path_length_m`、`collision_events`、`position`、`elapsed_seconds`、`goal`、`run_id`。

默认横向错开 0.8 m 以免模型重叠；两个传感器使用相同 map seed。严格同起点可传 `--simple-y 0 --minco-y 0`，但 RViz 模型会重叠。严谨性能结论应重复多个 map seed/goal。

已执行双机烟测：odom 均约 100 Hz、depth 约 33 Hz、pos_cmd 50 Hz；等待 goal 成功；向 `(15,0,2)` 发布后两机同时运动，MINCO/Simple 到达约 3.938/3.974 s，碰撞增量均为 0。该数字只证明交互链路，不替代多场景评估。

换成本机 `epoch50-retrained.pth` 后又执行同样的启动/静止/共享 goal 验证：重训 MINCO/Simple 到达约 3.904/3.982 s，到达前碰撞增量仍均为 0。

## 5. 数据采集、完整训练与同流评估

官方规模的一键流程：

```bash
docker exec -it -w /workspace/YOPO dzp-yopo-minco-test bash
tools/run_minco_pipeline.sh --mode full \
  --dataset /workspace/YOPO/dataset_minco \
  --env-num 10 --image-num 10000 \
  --epochs 50 --batch-size 16 --num-workers 8 \
  --seed 0 --overwrite
```

它顺序执行：固定 seed 生成 10 张地图/100,000 样本；90,000/10,000 划分从零训练 50 epoch；提升最终 checkpoint/tfevents；在同一确定性验证流评估官方和本机模型；输出 `results/minco_checkpoint_eval.json`。

分步模式：

```bash
tools/run_minco_pipeline.sh --mode generate --overwrite
tools/run_minco_pipeline.sh --mode train
tools/run_minco_pipeline.sh --mode promote --epochs 50
tools/run_minco_pipeline.sh --mode evaluate --epochs 50
```

采集器默认拒绝覆盖；只有显式 `--overwrite` 才重建。直接调用示例：

```bash
cd /workspace/YOPO/Simulator
set +u; source /opt/ros/noetic/setup.bash; source devel/setup.bash; set -u
rosrun sensor_simulator dataset_generator \
  --save-path /workspace/YOPO/dataset_minco \
  --env-num 10 --image-num 10000 --seed 0 --overwrite
```

单独训练/评估：

```bash
cd /workspace/YOPO/YOPO
python3 train_yopo.py \
  --dataset-path /workspace/YOPO/dataset_minco \
  --epochs 50 --batch-size 16 --num-workers 8 \
  --save-interval 10 --seed 0

python3 evaluate_minco.py \
  official=saved/yopo-minco/epoch50.pth \
  retrained=saved/yopo-minco/epoch50-retrained.pth \
  --dataset-path /workspace/YOPO/dataset_minco \
  --batch-size 16 --seed 0 \
  --output ../results/minco_checkpoint_eval.json
```

```bash
cd /workspace/YOPO/YOPO
tensorboard --logdir saved --bind_all
```

### 本次正式训练结果

本机 RTX 4070 Ti SUPER 上，100,000 样本采集约 90 s，50 epoch 从零训练约 55 min。训练末尾的在线验证 MeanTraj 约 11.1–11.3。随后对两个 checkpoint 重建相同 seed=0、相同 10,000 张验证流，实测如下：

| 指标 | 官方 MINCO | 本机重训 | 重训相对官方 |
|---|---:|---:|---:|
| `mean_traj_loss` | 11.0209 | 11.2532 | +2.11% |
| `perf/MinDist` (m) | 0.4653 | 0.4597 | -1.20% |
| `perf/PathLength` (m) | 9.1648 | 9.1281 | -0.40% |
| `perf/AvgSpeed` (m/s) | 4.8009 | 4.8664 | +1.37% |
| `perf/MaxSpeed` (m/s) | 5.8161 | 5.8681 | +0.89% |
| `perf/Duration` (s) | 1.9186 | 1.8862 | -1.69% |
| `radius_loss` | -1.4532 | -1.4819 | 更低 0.0287 |
| `rank_loss` | 1.4316 | 1.4401 | +0.59% |

总损失差约 2.1%，而轨迹长度、速度、最小障碍距离都在约 0.4%–1.7% 内，corridor radius loss 还略低。因此本机完整重训与官方预训表现相当接近，没有发现 MINCO 同步缺失的迹象。全部 20 个分项数值见已跟踪的 `results/minco_checkpoint_eval.json`。

## 6. YOPO-MINCO 相比 YOPO-Simple 改了什么

Simple 完整推理快照在 `YOPO/simple_runtime/`，与 `saved/yopo-simple/epoch50.pth` 结构匹配，因此对比不依赖切 Git 分支。

### 6.1 单段五次多项式 -> 两段 MINCO

- Simple：`simple_runtime/policy/poly_solver.py` 的 `Poly5Solver` 用起点和网络预测的终点 `p/v/a` 确定一段五次多项式。
- MINCO：`YOPO/minco/minco.py` 实现 MINCO 系数/边界梯度传播；`policy/poly_solver.py` 的 `MincoTraj` 解两段五次轨迹；网络输出中间点、末端 `p/v/a` 和两段持续时间。
- `policy/state_transform.py::pred_to_traj_params` 将 14 个轨迹通道解码为 `inner_pos`、`tail_pva`、`durations`。这就是 “MINCO as trajectory representation” 的代码落点。

MINCO 从少量几何/边界参数恢复整条最小控制代价高阶轨迹，保持位置、速度、加速度连续，且比回归全部系数紧凑。

### 6.2 中间 waypoint 是 homotopy anchor

- Simple primitive 离散 horizon、vertical、radio，锚定单段终点方向/距离。
- MINCO 的 `policy/primitive.py` 为每个图像格构造中间 waypoint lattice，`state_transform.py` 允许格内径向/角度偏移。
- 同一最终目标可经障碍左/右/上/下的不同中间点连接两段轨迹，显式覆盖不同 detour topology，而不是让竞争损失从一个单段初值中自行挤出绕行方向。

这正对应 “using the intermediate waypoint as a homotopy anchor to cover distinct detour topologies”，并缓解 competing costs 导致的局部最优。

### 6.3 richer trajectory expression

- Simple head 每格为 9 个终点状态通道 + 1 score。
- MINCO 每格为 14 个轨迹通道 + 1 score + `2 * radius_num` corridor 通道。当前 `radius_num=10`，实测 shape 为 trajectory `(B,14,3,5)`、score `(B,3,5)`、corridor `(B,20,3,5)`。
- 14 通道含中间点偏移、末端位置方向/距离、末端速度/加速度和两段时间；候选不再局限于起点直连终点的一段曲线。

### 6.4 barrier-form feasibility costs

- `loss/safety_loss.py` 沿完整 MINCO 轨迹查询 ESDF，对安全距离内区域施加指数/势垒安全代价，而非只看终点。
- `loss/feasible_loss.py` 对速度、加速度、jerk、曲率/航向等约束用可微 barrier/hinge，并积分超限程度。
- `loss/smoothness_loss.py` 按时间积分 acceleration/jerk。
- `guidance_loss.py`/`loss_function.py` 组合目标、时间、平滑、安全、动力学可行性。
- `minco/minco.py` 把代价梯度传回中间点、末端状态和 duration，再进入网络。

所以 “adding several barrier-form costs to ensure feasibility” 由轨迹采样、ESDF、动力学约束和 MINCO 梯度链共同实现。

### 6.5 预测 safe flight corridors

- `policy/models/head.py`/`yopo_network.py` 新增 corridor head：每个候选、每个采样点预测 warped radius 均值 `mu` 与不确定性尺度 `b`。
- `policy/yopo_trainer.py` 从 ESDF 得到轨迹可用半径，以 asymmetric Laplace/NLL 风格 `radius_loss` 训练 `mu,b`。
- `test_yopo_ros.py` 在线还原保守半径，先做 corridor admission；无候选满足 `--safe-radius` 就 brake，而非盲选最低 score。
- `_publish_corridor` 与 `yopo_compare.rviz` 直接显示选中轨迹的走廊。

这对应 “predicting safe flight corridors”：把轨迹周围无碰空间作为显式输出，参与选择并可视化。

### 6.6 Simulator/训练接口同步和本分支增强

官方 MINCO 同步了随机地图、CUDA raycast、Simsense 双目及 MINCO 数据/损失配置。本分支另加：

- 两个在线传感器实例的 topic/frame 全参数化；
- 占据栅格主机副本和碰撞事件/采样/状态 topic；
- 采集路径、环境数、图片数、seed、显式覆盖 CLI；
- 训练 epoch、batch、worker、学习率、数据/日志路径、seed CLI；
- 确定性的多 checkpoint 同验证流评估；
- Simple/MINCO 独立网络与权重常驻，杜绝两种 head 错配 checkpoint。

## 7. 验证清单

- Docker 冷构建成功，无硬编码代理；CUDA/GPU 与 Python 依赖通过；
- Controller、Simulator/CUDA 从源码编译成功；
- Simple/MINCO checkpoint 按各自结构加载并完成 GPU dummy forward；
- 200 样本、1 epoch 采集/训练/评估烟测成功；
- 正式 100,000 样本采集成功；
- 正式 50 epoch 与官方/重训同流评估：见最终结果；
- 双机节点、频率、共享 goal、碰撞增量、JSON 指标实际运行成功；
- RViz 实际启动成功；nouveau DRI 不可用时自动使用 llvmpipe，也可传 `--rviz-software-gl`；
- `compileall`、launch XML、`bash -n`、`git diff --check` 均检查。

## 8. 注意事项

- ROS Noetic setup 在 `set -u` 下会读取未定义变量；脚本已临时切换 nounset，手动运行也使用本文写法。
- 双机使用 host network；不要同时运行另一套 roscore，启动器自己管理 master。
- 正式训练前先停止双机 tmux，避免争抢 GPU。
- `collision_events` 是占据体素进入事件，不向动力学施加刚体碰撞反力。
- RViz goal 仅给 x/y；两个 planner 固定目标高度 2 m，与上游一致。
- Git remote 若使用带凭据 URL，应改用 credential helper/SSH；文档、日志和 commit 不保存 token。
