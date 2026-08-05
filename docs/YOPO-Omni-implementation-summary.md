# YOPO-Omni 实现总结

本文档总结当前仓库中 YOPO-Omni 的实现，包括训练数据生成、Dataset 读取、网络结构、拓扑编码、输出解码、轨迹构造、Loss 设计和训练流程。对应的主要代码文件如下：

- 数据生成器：`Simulator/src/src/dataset_generator.cpp`
- 数据生成配置：`Simulator/src/config/config.yaml`
- 训练配置：`YOPO/config/traj_opt.yaml`
- Dataset：`YOPO/policy/yopo_omni_dataset.py`
- 网络：`YOPO/policy/yopo_omni_network.py`
- Loss：`YOPO/loss/omni_loss.py`
- 安全代价：`YOPO/loss/safety_loss.py`
- Trainer：`YOPO/policy/yopo_omni_trainer.py`
- 训练入口：`YOPO/train_yopo_omni.py`
- 一键脚本：`tools/run_yopo_omni_pipeline.py`

## 1. 总体目标

YOPO-Omni 的目标是在一个采样点上同时考虑全向局部拓扑。当前实现采用“方案 B”的数据组织方式：

1. 仿真器在同一个无人机位姿处渲染四个方向的深度图：front、left、right、back。
2. 对同一个采样点随机生成 8 个 desired direction。
3. 对每个 desired direction 做一次局部路径搜索，得到 guide path 和对应的 selected topology。
4. 一个 pose 展开成 8 个训练样本。
5. 训练时网络输入四向深度图和当前状态，同时输出 8 个 topology 的候选轨迹和 score。
6. guide path 始终在数据集中生成和保存，但 `guidance loss` 可通过配置开关关闭。

当前数据规模配置为：

```yaml
env_num: 10
image_num: 10000
omni.direction_num: 8
```

因此理论样本数为：

```text
10 maps * 10000 poses/map * 8 directions/pose = 800000 samples
```

Dataset 默认按 pose 分组划分 train/valid：

```text
train: 720000 samples
valid: 80000 samples
```

这样同一个 pose 的 8 个方向不会同时泄漏到训练集和验证集。

当前 Dataset 已做内存和吞吐优化：

- CSV 不再长期保存为 Python `dict/list/string`，而是加载为紧凑的 numpy 数组。
- train/valid 在同一 Python 进程内共享底层数组缓存，避免重复持有全量 metadata。
- 训练默认使用 pose-level batch：一个 pose 内 8 个 direction 共享同一组四向 depth image tokens。
- 每个 pose 的四向深度图只做一次 backbone 编码，再展开成 8 个 direction 计算 query、decoder 和 loss。
- depth LRU cache 仍用于减少四视角 PNG 重复解码。
- `omni_num_workers` 可按机器内存调整；内存紧张时建议设为 0。
- 默认启用 AMP 混合精度训练：`omni_amp: true`。

## 2. 数据生成流程

数据生成在 `Simulator/src/src/dataset_generator.cpp` 中完成。

### 2.1 地图生成

每个 map 使用原版模拟器的地图生成逻辑：

```cpp
mocka::Maps map;
map.setParam(config);
map.setInfo(info);
map.generate(config["maze_type"].as<int>());
```

其中每张地图使用不同随机种子：

```cpp
info.seed = seed + map_i;
```

地图点云会保存为：

```text
dataset/pointcloud-{map_id}.ply
```

保存前会进行 voxel filter：

```cpp
sor.setLeafSize(ply_res, ply_res, ply_res);
sor.filter(*filtered_cloud);
```

当前 `ply_res = 0.1`。

### 2.2 采样 pose

每张地图采样 `image_num` 个 pose。位置采样范围来自：

```yaml
x_range: 40
y_range: 40
z_range: [0.5, 4]
safe_dist: 0.5
```

位置采样逻辑：

1. 在 `[-x_range/2, x_range/2]`、`[-y_range/2, y_range/2]`、`z_range` 内均匀采样。
2. 用原始点云 KDTree 检查最近障碍距离，要求大于 `safe_dist`。
3. 同时要求该点在引导搜索栅格中是 free。

姿态采样：

```cpp
roll  = clamp(N(0,1) * roll_range / 3, -roll_range, roll_range)
pitch = clamp(N(0,1) * pitch_range / 3, -pitch_range, pitch_range)
yaw   = uniform(0, 360)
```

pose 保存到：

```text
dataset/pose-{map_id}.csv
```

字段：

```text
px,py,pz,qw,qx,qy,qz
```

### 2.3 四方向深度图

每个 pose 渲染 4 张深度图，保存到：

```text
dataset/{map_id}/img_{pose_id}_front.png
dataset/{map_id}/img_{pose_id}_left.png
dataset/{map_id}/img_{pose_id}_right.png
dataset/{map_id}/img_{pose_id}_back.png
```

四个相机朝向为：

```cpp
front: yaw 0 deg
left:  yaw 90 deg
right: yaw -90 deg
back:  yaw 180 deg
```

每个方向还叠加配置中的 camera pitch：

```cpp
quat_bc_view = RPY2Quat(0, camera_pitch_deg, view_yaw)
T_wc = T_wb * T_bc
```

深度图保存为 16-bit PNG：

1. 先除以 `max_depth_dist` 归一化到 `[0, 1]`。
2. clip 到 `[0, 1]`。
3. 乘以 65535 后转为 `CV_16UC1`。

Dataset 读取时再除以 65535 得到 float depth。

## 3. Guide Path 生成

Guide 的目标是给每个 desired direction 生成一个局部可行路径，用于可选的 guidance loss。

### 3.1 每个 pose 展开 8 个方向

当前方向数：

```yaml
omni.direction_num: 8
```

第 `dir_i` 个方向对应扇区中心：

```text
theta_i = 2*pi*i/8
```

为了避免数据过于离散，生成时在扇区内部加随机 jitter：

```cpp
sector_width = 2*pi / direction_num
jitter ~ uniform(-0.35*sector_width, 0.35*sector_width)
theta = direction_idx * sector_width + jitter
```

得到 body frame desired direction：

```cpp
vdes_b = [cos(theta), sin(theta), 0]
```

这里的 `vdes_b` 是单位方向，后续 Dataset 会随机采样速度幅值。

### 3.2 desired direction 到 world goal

为了避免 roll/pitch 影响 desired direction 的水平拓扑，生成目标时只使用 yaw 旋转：

```cpp
R_yaw = RPY2Quat(0, 0, yaw).toRotationMatrix()
vdes_w = R_yaw * vdes_b
goal = pos + goal_length * normalize(vdes_w)
```

当前：

```yaml
omni.goal_length: 10.0
omni.goal_z_margin: 0.2
```

目标高度会 clamp 到：

```text
[z_min + goal_z_margin, z_max - goal_z_margin]
```

当前采样高度范围是 `[0.5, 4]`，所以 goal z 默认限制在 `[0.7, 3.8]`。

### 3.3 引导搜索栅格

虽然配置名仍保留 `dijkstra_*`，当前实现为了性能使用局部 2.5D A*。它可以理解为 Dijkstra 的启发式加速版本：如果把 heuristic 设为 0，就是 Dijkstra。

搜索栅格由 filtered point cloud 构建：

```cpp
HostDijkstraGrid dijkstra_grid(filtered_cloud, dijkstra_resolution, dijkstra_inflation, z_min, z_max);
```

当前参数：

```yaml
omni.dijkstra_resolution: 0.5
omni.dijkstra_inflation: 0.5
omni.astar_local_radius: 16.0
omni.goal_search_radius: 3.0
```

注意这里使用 filtered cloud 而不是原始 cloud，是为了避免森林叶片和地面点云太密导致搜索空间被过度占据。

### 3.4 2.5D A* 搜索

搜索函数：

```cpp
runAStar(...)
```

流程：

1. 将 requested goal 投影到当前飞行高度切片上的 nearest free cell。
2. 如果目标附近 `goal_search_radius` 内有 free cell，则使用最近 free goal。
3. 如果附近没有 free cell，则沿 start-goal 线段向回找最近 free point。
4. 在 start 和 goal 的 XY 包围盒基础上外扩 `astar_local_radius`，得到局部 ROI。
5. 在固定 z voxel 上进行 8 邻域 A* 搜索。
6. 每条边的代价为欧式栅格距离：
   ```text
   straight: resolution
   diagonal: sqrt(2) * resolution
   ```
7. heuristic 使用当前 cell 到 goal cell 的 XY 欧式距离。

路径点保存为 3D world point，但搜索实际在 XY 切片上进行，z 基本保持采样高度。

### 3.5 Path Shorten

A* 得到 raw path 后，会用可见性检查做简化：

```cpp
shortenPathByVisibility2D(...)
```

简化逻辑：

1. 从当前 anchor 开始，尽可能向后找最远的可直连 path point。
2. 如果 anchor 到 candidate 的 2D 线段在当前 z 切片上不穿过障碍，则跳过中间点。
3. 重复直到终点。

这样保存的 guide path 更接近“关键折线”，不会保留大量栅格小步。

### 3.6 selected topology

每个样本都有：

```text
dir_idx
selected_topology
```

`dir_idx` 是原始 desired direction 所在扇区。

`selected_topology` 是根据 A* 路径近场方向重新计算的拓扑：

```cpp
near_field_dir = raw_path[1] - raw_path[0]
init_dir_b = R_yaw.transpose() * near_field_dir
selected_topology = directionSector(init_dir_b, direction_num)
```

这点很重要：如果 desired direction 指向某个方向，但真实可行路径一开始需要绕障碍，`selected_topology` 会变成绕行方向。Guidance loss 只监督这个 selected topology。

### 3.7 CSV 输出

每张地图输出：

```text
dataset/samples-{map_id}.csv
dataset/guides-{map_id}.csv
dataset/pose-{map_id}.csv
dataset/pointcloud-{map_id}.ply
dataset/{map_id}/img_{pose_id}_{view}.png
```

`samples-{map_id}.csv` 字段：

```text
sample_id,pose_id,dir_idx,
px,py,pz,qw,qx,qy,qz,
vdes_bx,vdes_by,vdes_bz,
goal_wx,goal_wy,goal_wz,
guide_offset,guide_len,guide_mask,guide_cost,selected_topology
```

关键字段含义：

- `sample_id`: 当前 map 内的样本 ID，等于 `pose_id * direction_num + dir_idx`。
- `pose_id`: 当前 pose ID。
- `dir_idx`: 原始 desired direction 扇区。
- `vdes_b*`: body frame 单位 desired direction。
- `goal_w*`: world frame goal，若搜索成功，会更新为 guide path 终点。
- `guide_len`: shorten 后 guide path 点数。
- `guide_mask`: 是否成功生成 guide，成功为 1，否则为 0。
- `guide_cost`: guide path 长度；失败时为大数。
- `selected_topology`: guide 近场方向对应的 topology。

`guides-{map_id}.csv` 字段：

```text
sample_id,point_idx,x,y,z
```

保存每个样本的 guide path 点。

## 4. Dataset 读取和训练样本组织

Dataset 实现在 `YOPO/policy/yopo_omni_dataset.py`。

### 4.1 输入文件

Dataset 从 `cfg["dataset_path"]` 读取数据：

```yaml
dataset_path: "../dataset"
```

每个 map 目录结构：

```text
dataset/
  0/
    img_0_front.png
    img_0_left.png
    img_0_right.png
    img_0_back.png
    ...
  pointcloud-0.ply
  pose-0.csv
  samples-0.csv
  guides-0.csv
```

### 4.2 train/valid 划分

Dataset 不是按 sample 随机划分，而是按：

```text
(map_id, pose_id)
```

分组划分。

原因：同一个 pose 会展开出 8 个 direction，如果按 sample 划分，同一个 pose 的不同方向可能同时出现在 train 和 valid，验证集会偏乐观。

当前：

```python
train_test_split(group_keys, test_size=0.1, random_state=0)
```

### 4.3 单个 pose-level 样本返回内容

`__getitem__` 返回：

```python
depth, pos, rot_wb, state_b, goal_w, guide, guide_mask, selected_topology, map_id
```

含义：

- `depth`: `[4, 1, H, W]`
- `pos`: `[3]`，world frame 位置。
- `rot_wb`: `[3, 3]`，body 到 world 的旋转矩阵。
- `state_b`: `[D, 9]`，每个 direction 的 body frame `[vel_b, acc_b, vdes_b]`。
- `goal_w`: `[D, 3]`，world frame goal，目前训练中没有直接使用。
- `guide`: `[D, omni_guide_points, 3]`，world frame guide path。
- `guide_mask`: `[D]`，是否有有效 guide。
- `selected_topology`: `[D]`，guide 近场拓扑。
- `map_id`: scalar，用于 SafetyLoss 查询对应 ESDF。

其中 `D=8`。训练时 DataLoader batch 后 shape 为：

```text
depth:             [B, 4, 1, 96, 160]
state_b:           [B, D, 9]
goal_w:            [B, D, 3]
guide:             [B, D, 32, 3]
guide_mask:        [B, D]
selected_topology: [B, D]
```

### 4.4 深度图处理

四张 16-bit PNG 读取后：

```python
image = cv2.imread(image_path, -1)
image = resize(image, (image_width, image_height))
image = image / 65535.0
```

最终 shape：

```text
[4, 1, 96, 160]
```

DataLoader batch 后：

```text
depth: [B, 4, 1, 96, 160]
```

### 4.5 状态增强

CSV 中的 `vdes_b` 是单位方向。Dataset 每次读取时随机采样 desired speed：

```python
speed ~ uniform(omni_vdes_speed_min, omni_vdes_speed_max)
vdes_b = speed * vdes_unit
```

当前：

```yaml
omni_vdes_speed_min: 0.5
omni_vdes_speed_max: 6.0
```

速度和加速度也会随机增强：

```python
vel_b = clip(vdes_b + N(0, omni_vel_noise_std), -vel_max, vel_max)
acc_b = clip(N(0, omni_acc_noise_std), -acc_max, acc_max)
state_b = [vel_b, acc_b, vdes_b]
```

这样同一个几何样本可以在不同速度/加速度状态下参与训练。

### 4.6 Guide 重采样

每条 guide path 点数不固定，Dataset 会重采样到固定点数：

```yaml
omni_guide_points: 32
```

重采样基于路径弧长插值。若 guide 缺失，则返回全零 guide，并通过 `guide_mask=0` 屏蔽 guidance loss。

## 5. 网络结构

网络实现在 `YOPO/policy/yopo_omni_network.py`。

### 5.1 输入输出

网络 forward 输入：

```python
depth: [B, 4, 1, H, W]
state: [B, D, 9]
```

其中：

```text
state = [vx, vy, vz, ax, ay, az, vdes_x, vdes_y, vdes_z]
```

全部在 body frame 下。

网络输出：

```python
endstate: [B, D, K, 9]
score:    [B, D, K]
```

当前：

```text
K = omni_topology_num = 8
```

每个 topology 输出一个候选终端状态：

```text
endstate[k] = [px, py, pz, vx, vy, vz, ax, ay, az]
```

这些量仍在 body frame 下，表示从当前 body frame 原点出发的终端位置、速度、加速度。

### 5.2 深度 Backbone

四张 depth view 会共享同一个 `OmniDepthBackbone`。

Backbone 使用原版 YOPO 的 ResNet18 风格 `YopoBackbone(d_model)`：

```python
self.net = YopoBackbone(d_model)
```

为了避免 token 太少，修改了最后一个 stage 的 stride：

```python
self.net.cnn.layer4[0].conv1.stride = (1, 1)
self.net.cnn.layer4[0].downsample[0].stride = (1, 1)
```

对于当前输入大小：

```text
H = 96
W = 160
d_model = 128
```

每个 view 输出约：

```text
6 * 10 = 60 tokens
```

四个 view 合计：

```text
4 * 60 = 240 image tokens
```

实际 token 数由 backbone 输出 feature map 的 `h*w` 决定。

### 5.3 View Embedding

网络给四个方向分别加 learnable embedding：

```python
self.view_embedding = nn.Embedding(4, d_model)
```

view 顺序固定为：

```text
front, left, right, back
```

这样网络可以区分“同样的局部图像特征来自哪个朝向”。

### 5.4 Ray Embedding

仅有 view embedding 还不够，因为同一张 feature map 的不同 token 也对应不同相机射线。网络构造每个 token 的近似 ray：

```python
cam_ray = [1, -grid_x, -grid_y]
cam_ray = normalize(cam_ray)
```

然后按 view yaw 旋转到 body frame：

```python
view_yaw = [0, pi/2, -pi/2, pi]
```

旋转后得到：

```text
ray_body: [4, h*w, 3]
```

再通过 MLP 映射到 `d_model`：

```python
ray_embed = ray_mlp(ray_body)
```

最终 image token 为：

```python
tokens = backbone_feature + view_embedding + ray_embedding
```

这让 Transformer 能知道每个 token 大致对应 body frame 下的哪个方向。

### 5.5 State Encoder

输入状态先做归一化：

```python
vel  /= vel_max
acc  /= acc_max
vdes /= vel_max
```

当前：

```yaml
vel_max_train: 6.0
acc_max_train: 6.0
```

然后经过 MLP：

```python
Linear(9, d_model)
LayerNorm
SiLU
Linear(d_model, d_model)
LayerNorm
SiLU
```

得到：

```text
state_feature: [B, D, d_model]
```

### 5.6 拓扑编码

当前 topology 数：

```yaml
omni_topology_num: 8
```

第 `k` 个 topology 的角度：

```text
theta_k = 2*pi*k/K
```

拓扑编码由两部分相加：

1. 几何编码：
   ```python
   geo = [cos(theta_k), sin(theta_k)]
   geo_feature = geo_mlp(geo)
   ```
2. 可学习 ID embedding：
   ```python
   id_feature = topology_embedding(k)
   ```

最终：

```python
topology_feature[k] = geo_mlp([cos(theta_k), sin(theta_k)]) + topology_embedding(k)
```

几何编码保证相邻拓扑在角度上有连续关系，learnable embedding 则允许每个 topology 学到独立偏置。

### 5.7 Transformer Decoder

每个 direction、每个 topology 构造一个 query：

```python
query[b, d, k] = state_feature[b, d] + topology_feature[k]
```

shape：

```text
queries: [B*D, K, d_model]
memory:  [B*D, 4*h*w, d_model]
```

其中 `memory` 来自同一个 pose 的 image tokens 在 direction 维上展开，因此四向 depth backbone 只按 pose 编码一次。

然后使用 Transformer Decoder：

```python
latent = decoder(tgt=queries, memory=image_tokens)
```

这一步含义是：8 个 topology query 同时通过 cross attention 查询四向 depth token，得到每个拓扑下的候选轨迹 latent。

当前配置：

```yaml
omni_d_model: 128
omni_num_heads: 4
omni_decoder_layers: 2
omni_ffn_dim: 512
omni_dropout: 0.1
```

### 5.8 Head 输出

每个 topology latent 经过 MLP：

```python
Linear(d_model, d_model)
SiLU
Linear(d_model, d_model)
SiLU
Linear(d_model, 10)
```

输出 10 个 raw value：

```text
raw[0:9] -> endstate raw
raw[9]   -> score raw
```

## 6. 输出解码

网络不是直接回归任意终端位置，而是用 topology 约束输出范围。

### 6.1 位置解码

对第 k 个 topology：

```python
theta_k = 2*pi*k/K
delta_yaw = tanh(raw[0]) * (pi/K)
delta_pitch = tanh(raw[1]) * pitch_max
radius = radius_min + (radius_max - radius_min) * sigmoid(raw[2])
```

当前：

```yaml
omni_radius_min: 1.0
omni_radius_max: 5.0
omni_pitch_max_deg: 35.0
```

因此：

```text
delta_yaw   in [-pi/K, pi/K]
delta_pitch in [-35 deg, 35 deg]
radius      in [1, 5]
```

终端位置：

```python
yaw = theta_k + delta_yaw
pos_x = radius * cos(yaw) * cos(delta_pitch)
pos_y = radius * sin(yaw) * cos(delta_pitch)
pos_z = radius * sin(delta_pitch)
```

设计含义：

- 每个 topology 覆盖一个角度扇区。
- `delta_yaw` 只允许在当前扇区内偏移。
- 网络仍可通过 `delta_pitch` 输出上升/下降。
- 半径被限制在合理局部规划范围内。

### 6.2 速度和加速度解码

速度：

```python
vel = vel_max * tanh(raw[3:6])
```

加速度：

```python
acc = acc_max * tanh(raw[6:9])
```

因此速度和加速度都被限制在训练上界内：

```text
vel in [-6, 6]
acc in [-6, 6]
```

### 6.3 Score 解码

score 使用：

```python
score = softplus(raw[9])
```

所以 score 非负。训练时 score 不是分类概率，而是学习对应轨迹的 detached total cost。推理时通常选择 score 最低的候选轨迹。

## 7. 轨迹构造

网络输出的是终端状态，不是整条轨迹。Loss 中会根据起点状态和终点状态生成五次多项式轨迹。

### 7.1 起点状态

Trainer 从 batch 里得到：

```python
pos:     [B, 3] world frame
rot_wb:  [B, 3, 3]
state_b: [B, 9]
```

起点 body frame：

```text
position = [0, 0, 0]
velocity = state_b[0:3]
acc      = state_b[3:6]
```

然后转换到 world frame：

```python
start_pos_w, start_vel_w, start_acc_w = state_body2world(...)
```

得到：

```text
start_state_w: [B, 3, 3]
```

其中第二维是：

```text
[position, velocity, acceleration]
```

### 7.2 终点状态

网络输出：

```text
endstate_b: [B, K, 9]
```

拆成：

```text
end_pos_b, end_vel_b, end_acc_b
```

通过 `rot_wb` 转换到 world frame：

```python
end_state_w: [B, K, 3, 3]
```

### 7.3 五次多项式

Loss 内使用 `qp_generation()` 预计算矩阵：

```text
L:    derivative boundary -> polynomial coefficients
R_J:  jerk integral quadratic matrix
R_A:  acceleration integral quadratic matrix
```

对每条候选轨迹，每个轴都是五次多项式：

```text
p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
```

边界条件为：

```text
t = 0:      p, v, a = start
t = T:      p, v, a = end
T = sgm_time
```

`sgm_time` 来自配置。

## 8. Loss 设计

Loss 实现在 `YOPO/loss/omni_loss.py`。

总输入：

```python
start_state_w
end_state_w
endstate_b
state_b
guide_path_w
guide_mask
selected_topology
map_id
pred_score
```

batch 内符号：

```text
B = batch size
K = topology_num = 8
```

### 8.1 基础轨迹代价

每个候选 topology 都计算：

```text
base_cost = ws * smooth_cost
          + wa * acc_cost
          + wc * safety_cost
          + wi * intent_cost
```

当前权重：

```yaml
ws: 10.0
wa: 1.0
wc: 1.0
wi: 1.0
```

其中 smooth 和 acceleration 权重会按速度尺度归一：

```python
smoothness_weight = ws / vel_scale^5
accele_weight     = wa / vel_scale^3
vel_scale = vel_max_train / 1.0
```

当前 `vel_max_train=6`，所以：

```text
smoothness_weight = 10 / 6^5 ~= 0.001286
accele_weight     = 1 / 6^3  ~= 0.00463
```

### 8.2 Smoothness Loss

`SmoothnessLoss` 使用五次多项式边界参数计算 jerk integral：

```text
smooth_cost = dx^T R_J dx + dy^T R_J dy + dz^T R_J dz
```

它惩罚轨迹 jerk，鼓励轨迹平滑。

### 8.3 Acceleration Loss

同样通过二次型计算 acceleration integral：

```text
acc_cost = dx^T R_A dx + dy^T R_A dy + dz^T R_A dz
```

它惩罚过大的加速度。

### 8.4 Safety Loss

SafetyLoss 使用每张地图的 ESDF。

初始化时读取：

```text
dataset/pointcloud-{map_id}.ply
```

构建 occupancy grid，再通过：

```python
distance_transform_edt
```

得到 signed distance field。

当前 ESDF voxel size：

```python
self.voxel_size = 0.2
```

轨迹采样点数：

```yaml
omni_loss_eval_points: 30
```

SafetyLoss 对每条候选轨迹在 `[0, sgm_time]` 上采样 `eval_points` 个点，查询 ESDF 距离 `d`，然后计算：

```python
cost = exp(-(d - d0) / r)
```

当前：

```yaml
d0: 1.2
r: 0.6
```

含义：

- 距离障碍越近，cost 越大。
- 当距离大于 `d0` 时，代价指数衰减。
- 当穿入障碍，signed distance 为负，代价会显著变大。

为了支持 batch 中不同 map，当前实现会按 batch 中 trajectory 覆盖范围 crop 每个 map 的局部 SDF，再用 `grid_sample` 查询距离。

### 8.5 Intent Loss

Intent loss 使用 body frame 下的终端位置和 desired velocity 方向。

```python
vdes_norm = normalize(vdes_b)
progress = dot(end_pos_b, vdes_norm)
lateral = end_pos_b - progress * vdes_norm
```

代价：

```python
intent_cost = softplus(omni_intent_min_progress - progress)
            + 0.1 * norm(lateral)
```

当前：

```yaml
omni_intent_min_progress: 0.5
```

含义：

- 终端点应该沿 desired direction 有足够前进量。
- 横向偏离不要太大。

### 8.6 Guidance Loss

Guidance loss 是可开关的：

```yaml
use_guidance_loss: true
w_guide_path: 1.0
```

如果关闭：

```python
guidance_loss(...) -> zeros([B, K])
guide_weight = 0
```

注意：即使关闭 guidance loss，数据集里仍然会生成和保存 guide path。

如果开启，计算流程如下：

1. 根据 start/end derivative 生成每条候选轨迹的五次多项式。
2. 每条轨迹采样 `omni_loss_eval_points` 个点。
3. 对每条候选轨迹和 guide path 做 `torch.cdist`。
4. 每个轨迹采样点找最近 guide point。
5. 对最近距离求平均，得到该 topology 的 guide distance。
6. 只监督 `selected_topology` 对应的候选轨迹。
7. 如果 `guide_mask=0`，该样本 guide loss 为 0。

代码核心：

```python
nearest = dist.min(dim=-1).values.mean(dim=-1).reshape(B, K)
selected = nearest.gather(1, selected_topology[:, None]).squeeze(1) * guide_mask
guide_cost = zeros(B, K)
guide_cost.scatter_(1, selected_topology[:, None], selected[:, None])
```

也就是说：

- 网络每次输出 8 条候选轨迹。
- 但 guide loss 只拉近 selected topology 的那一条。
- 其他 topology 不受 guide path 直接监督，仍通过 smooth/safety/intent/score/rank 学习。

最终 guide loss 会按有效 guide 样本数归一：

```python
guide_loss = guide_cost.sum() / guide_mask.sum().clamp(min=1)
```

这样不会被 `K=8` 稀释。

### 8.7 Score Loss

网络输出的 score 学习总轨迹代价：

```python
total_cost = base_cost + guide_weight * guide_cost
score_loss = smooth_l1_loss(pred_score, total_cost.detach())
```

这里 `total_cost.detach()` 很重要：

- score head 学习预测代价。
- 不让 score loss 反向改变 cost 本身的计算图。

推理时可以选择 score 最低的 topology。

### 8.8 Ranking Loss

Ranking loss 用来让不同 topology 的 score 保持相对顺序。

如果某个候选 i 的 target cost 比候选 j 小超过阈值：

```text
target_i < target_j - omni_rank_min_gap
```

则希望：

```text
score_i + margin < score_j
```

代码：

```python
diff = target_cost[:, :, None] - target_cost[:, None, :]
score_diff = pred_score[:, :, None] - pred_score[:, None, :]
valid = diff < -omni_rank_min_gap
rank_loss = relu(omni_rank_margin + score_diff[valid]).mean()
```

当前：

```yaml
w_rank: 0.2
omni_rank_margin: 0.1
omni_rank_min_gap: 0.2
```

### 8.9 总 Loss

最终：

```python
trajectory_loss = base_cost.mean() + guide_weight * guide_loss
loss = trajectory_loss + w_score * score_loss + w_rank * rank_loss
```

当前：

```yaml
w_score: 1.0
w_rank: 0.2
```

TensorBoard 中记录：

```text
Train/loss
Train/trajectory
Train/score
Train/rank
Train/smooth
Train/acc
Train/safety
Train/intent
Train/guide
```

Eval 同理在每个 epoch 结束后写入：

```text
Eval/loss
Eval/trajectory
...
```

## 9. 训练流程

训练入口：

```bash
PYTHONPATH=YOPO /home/beiyue/.venvs/yopo/bin/python YOPO/train_yopo_omni.py --train-epoch 50 --batch-size 16
```

或者使用一键脚本：

```bash
tools/run_yopo_omni_pipeline.py --mode train --guidance-loss true --train-epoch 50 --batch-size 16
```

Trainer 参数：

```python
learning_rate = 1.5e-4
optimizer = AdamW
max_grad_norm = omni_max_grad_norm
```

当前：

```yaml
omni_max_grad_norm: 1.0
omni_num_workers: 4
omni_depth_cache_size: 64
omni_amp: true
```

训练循环：

1. 读取 batch。
2. 网络对每个 pose 的四向 depth 编码一次，并对 `D=8` 个 desired direction 分别预测 `K=8` 条 endstate 和 score。
3. Trainer 将 `[B,D,K,...]` reshape 为 `[B*D,K,...]`。
4. 将 body frame 终点状态转换到 world frame。
5. 构造五次多项式轨迹。
6. 计算 smooth、acc、safety、intent、guide、score、rank。
7. 反向传播。
8. gradient clipping。
9. optimizer step。

若 `omni_amp=true` 且 CUDA 可用，forward/loss 在 `torch.amp.autocast` 下执行，backward 使用 `GradScaler`。这会降低显存占用，并通常提升 RTX 级 GPU 上的吞吐。

TensorBoard 每个 epoch 约写 16 次 train scalar：

```python
inspect_interval = max(1, len(train_dataloader) // 16)
```

当前 `batch-size` 表示 pose batch size。对于 90000 train poses、`batch-size=16`：

```text
steps_per_epoch = 90000 / 16 = 5625
effective_direction_supervision_per_step = 16 * 8 = 128
log_interval ~= 5625 / 16 = 351 steps
```

因此刚开始训练时需要等到 epoch 约 6.25% 才会看到第一条 TensorBoard 曲线。

训练样本顺序是 pose-level shuffle：每个 epoch 打乱 pose 顺序，每个 pose 内的 8 个方向作为同一个样本返回。这样一个 pose 的四向深度图只编码一次，8 个 direction 共享 image tokens。

Checkpoint：

```python
trainer.train(epoch=args.train_epoch, save_interval=10)
```

默认每 10 epoch 保存一次，同时注册了退出保存逻辑。

## 10. 当前设计的关键点

### 10.1 为什么四方向深度图

原版 YOPO 主要基于前向深度感知。YOPO-Omni 希望在局部全向规划中同时评估多个拓扑，所以单前向视角不足以覆盖侧向和后向障碍。四方向深度图提供近似 360 度局部感知。

### 10.2 为什么输出 8 个 topology

每次 forward 同时输出 8 条候选轨迹，可以让网络在同一环境状态下比较不同局部拓扑。score head 学习每条轨迹的代价，推理时选择 score 最低的候选。

### 10.3 为什么拓扑编码用几何编码加 embedding

只用 embedding 会让 8 个 topology 完全离散，不知道相邻方向的关系。只用 `[cos, sin]` 又可能表达能力不足。两者相加后：

- `[cos, sin]` 提供连续角度先验。
- embedding 提供每个拓扑独立的可学习偏置。

### 10.4 为什么 guide 只监督 selected topology

每个样本的 guide path 对应一条具体绕行路径。若把同一 guide 同时监督所有 topology，会把不同拓扑都拉向同一条路径，破坏多候选轨迹的多样性。因此只监督 `selected_topology`。

### 10.5 为什么 guide loss 可关闭但 guide 仍生成

Guide 是数据资产，不只是某个 loss 的临时输入。即使训练时关闭 `use_guidance_loss`，仍可以：

- 后续重新开启 guidance loss。
- 可视化采样质量。
- 做数据质量统计。
- 做其他监督或评估。

### 10.6 为什么搜索用 2.5D A*

森林点云中树冠、地面和枝叶很密，完整 3D 栅格搜索容易因为障碍膨胀过保守而失败，并且计算成本更高。当前引导路径的目标主要是提供局部水平拓扑监督，所以使用固定高度切片上的 2.5D A* 更稳定、更快。

### 10.7 为什么 goal 用 yaw-only 旋转

生成 desired direction 时只使用 yaw 把 `vdes_b` 转到 world：

```cpp
vdes_w = R_yaw * vdes_b
```

这样 roll/pitch 不会把“水平期望方向”错误地抬高或压低。pose 的四向深度图仍然使用完整 roll/pitch/yaw 渲染。

## 11. 当前可观察训练状态示例

当前训练启动后，TensorBoard 中应关注：

```text
Train/loss
Train/guide
Train/safety
Train/score
Train/trajectory
```

早期健康状态通常表现为：

- `Train/loss` 下降。
- `Train/score` 下降，说明 score head 在学习 cost。
- `Train/safety` 下降或保持稳定。
- `Train/guide` 在 guidance loss 开启时缓慢下降。
- 第一个 epoch 结束后开始出现 `Eval/...`。

注意不要看旧版 YOPO 的：

```text
Detail/GoalLoss
Detail/SafetyLoss
Detail/SmoothLoss
```

YOPO-Omni 使用新的 tag 命名。

## 12. 常用命令

生成数据并训练：

```bash
tools/run_yopo_omni_pipeline.py --mode all --env-num 10 --image-num 10000 --guidance-loss true --train-epoch 50 --batch-size 16
```

只训练：

```bash
tools/run_yopo_omni_pipeline.py --mode train --guidance-loss true --train-epoch 50 --batch-size 16
```

只生成数据：

```bash
tools/run_yopo_omni_pipeline.py --mode generate --env-num 10 --image-num 10000
```

启动 TensorBoard：

```bash
python3 -m tensorboard.main \
  --logdir /home/beiyue/ubuntu2004-shared/workspace/YOPO/YOPO/saved/YOPO_Omni_0 \
  --host 0.0.0.0 \
  --port 6006 \
  --reload_interval 5
```

可视化 guide：

```bash
PYTHONPATH=YOPO /home/beiyue/.venvs/yopo/bin/python tools/visualize_omni_guides.py \
  --dataset dataset \
  --map-id 0 \
  --pose-id 0 \
  --output docs/yopo_omni_guides_pose0.png
```

## 13. 文件级数据流

完整链路如下：

```text
Simulator/src/src/dataset_generator.cpp
    -> dataset/{map_id}/img_{pose_id}_{front,left,right,back}.png
    -> dataset/samples-{map_id}.csv
    -> dataset/guides-{map_id}.csv
    -> dataset/pointcloud-{map_id}.ply

YOPO/policy/yopo_omni_dataset.py
    -> depth [B,4,1,H,W]
    -> state_b [B,D,9]
    -> guide [B,D,32,3]
    -> selected_topology [B,D]
    -> map_id [B]

YOPO/policy/yopo_omni_network.py
    -> endstate_b [B,D,8,9]
    -> score [B,D,8]

YOPO/policy/yopo_omni_trainer.py
    -> pose-level batch flatten 到 [B*D,...]
    -> body frame endstate 转 world frame
    -> start/end derivative state

YOPO/loss/omni_loss.py
    -> polynomial trajectory
    -> smooth/acc/safety/intent/guide/score/rank
    -> final loss
```

## 14. 需要注意的实现细节

1. `goal_w` 当前由 Dataset 返回，但训练主 loss 没有直接使用它。训练主要依赖 `vdes_b`、guide path、safety 和 score/rank。
2. `guide_mask=0` 的样本不会贡献 guidance loss，但仍贡献 smooth、acc、safety、intent 和 score/rank。
3. `selected_topology` 由 guide path 近场方向决定，不一定等于原始 `dir_idx`。
4. SafetyLoss 在初始化时会为所有 `pointcloud-*.ply` 构建 ESDF，因此地图数量较多时启动训练会有一段预处理时间。
5. TensorBoard train scalar 不是每个 step 写入，而是每个 epoch 约 16 次。
6. 当前网络使用四向深度图，但 ray embedding 只编码 yaw 方向和 feature grid 上的近似射线，没有显式编码真实相机内参。
7. 14GB 内存机器上不建议把 `omni_num_workers` 设回 4；旧实现下每个 worker 会复制 Python metadata，容易把内存和 swap 打满。优化后 metadata 已经明显变小，但 `num_workers=0` 仍是最稳配置。
