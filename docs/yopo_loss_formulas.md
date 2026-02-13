# YOPO Loss 逐步公式说明（纯 Markdown 可读版）

这版不使用 LaTeX 块公式，全部改成纯文本公式，避免你当前预览器不渲染数学表达式的问题。

适用文件：

1. `YOPO/loss/loss_function.py`
2. `YOPO/loss/smoothness_loss.py`
3. `YOPO/loss/safety_loss.py`
4. `YOPO/loss/guidance_loss.py`
5. `YOPO/loss/__init__.py`（空文件）

---

## 1. 统一符号

训练调用入口：`YOPO/policy/yopo_trainer.py:142`

```python
smooth_cost, safety_cost, goal_cost, acc_cost = self.yopo_loss(start_state_w, end_state_w, goal_w, map_id)
```

记号（每条候选轨迹）：

```text
s0 = [p0, v0, a0]      # 起点状态（world）
sT = [pT, vT, aT]      # 终点状态（world）
g  = goal              # 目标点（world）
map_id                 # 这条样本对应哪张地图的 ESDF
```

`YOPO/loss/loss_function.py:102-105` 中：

```text
Df = fixed boundary    # 起点边界导数
Dp = decision boundary # 终点边界导数
```

---

## 2. `loss_function.py`：总损失骨架

### 2.1 五次多项式参数化

代码：`YOPO/loss/loss_function.py:35-73`

```text
p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5

d = [p0, v0, a0, pT, vT, aT]^T
c = [c0, c1, c2, c3, c4, c5]^T

A*c = d  =>  c = A^{-1}*d
L = A^{-1}*Ct
```

`L` 在代码里是 `self._L`（`loss_function.py:67`），后续 `safety_loss.py` 用它把边界导数映射成多项式系数。

### 2.2 平滑二次型矩阵

代码：`YOPO/loss/loss_function.py:44-53`、`69-71`

```text
RJ = C * B^T * H * B * Ct   # jerk 二次型矩阵
RA = C * B^T * Q * B * Ct   # accel 二次型矩阵
```

其中：

```text
H: jerk 积分的 Hessian
Q: accel 积分的 Hessian
```

### 2.3 总损失

代码：`YOPO/loss/loss_function.py:107-111`

```text
L_total =
    ws * L_smooth
  + wc * L_safety
  + wg * L_goal
  + wa * L_acc
```

权重来自：`YOPO/loss/loss_function.py:85-89`。

---

## 3. `smoothness_loss.py`：平滑损失

代码：`YOPO/loss/smoothness_loss.py:19-31`

先构造每轴 6 维边界导数向量：

```text
dx = [x0, vx0, ax0, xT, vxT, axT]^T
dy = [y0, vy0, ay0, yT, vyT, ayT]^T
dz = [z0, vz0, az0, zT, vzT, azT]^T
```

二次型损失：

```text
L_jerk =
    dx^T * RJ * dx
  + dy^T * RJ * dy
  + dz^T * RJ * dz

L_acc =
    dx^T * RA * dx
  + dy^T * RA * dy
  + dz^T * RA * dz
```

对应代码：

1. `jerk_smooth`：`YOPO/loss/smoothness_loss.py:27`
2. `accel_smooth`：`YOPO/loss/smoothness_loss.py:29`

---

## 4. `safety_loss.py`：ESDF 碰撞损失

这部分分两段：先建 ESDF，再用 ESDF 算轨迹代价。

### 4.1 ESDF 构建（从点云到有符号距离）

入口：`YOPO/loss/safety_loss.py:191-239`

#### Step 1) 点云体素化

```text
i = floor((p - min_bound) / h)
```

其中 `h = voxel_size = 0.2`（`safety_loss.py:28`）。

构造占据图：

```text
O(i) = 1  (占据)
O(i) = 0  (空闲)
```

代码：`safety_loss.py:212-213`。

#### Step 2) 距离变换并加符号

代码：`safety_loss.py:218-221`

```text
d_out = EDT(free_mask)     * h   # 空闲体素到最近障碍距离（正）
d_in  = EDT(obstacle_mask) * h   # 障碍内部距离

phi(i) = d_out(i),  if O(i)=0
phi(i) = -d_in(i),  if O(i)=1
```

`phi` 就是 ESDF（外部为正、内部为负）。

---

### 4.2 用 ESDF 算轨迹碰撞代价

入口：`YOPO/loss/safety_loss.py:38`

#### Step 1) 边界导数 -> 多项式系数

代码：`safety_loss.py:108-116`

```text
cx = L * dx
cy = L * dy
cz = L * dz
```

合并成 `coe`（18 维）。

#### Step 2) 时间采样轨迹点

代码：`safety_loss.py:51-57`、`118-130`

```text
dt = T / N
t_n = n*dt, n=1..N, N=30

x(t) = sum_{k=0..5} cx_k * t^k
y(t) = sum_{k=0..5} cy_k * t^k
z(t) = sum_{k=0..5} cz_k * t^k
```

得到每条轨迹采样点 `pos_batch`。

#### Step 3) 在 ESDF 上查询每个采样点距离

代码：`safety_loss.py:87-99`

世界坐标 -> 体素坐标：

```text
g = (x - origin) / h
```

体素坐标 -> `grid_sample` 归一化坐标：

```text
u = 2*g/(shape-1) - 1
```

三线性插值查询（可微）：

```text
d_n = phi(x(t_n))
```

#### Step 4) 距离转代价

代码：`safety_loss.py:105-106`

```text
c_n = exp(-(d_n - d0)/r)
```

解释：

1. `d_n < 0`（在障碍内） => 代价很大  
2. `d_n` 大（远离障碍） => 代价很小

#### Step 5) 聚合成轨迹安全损失

默认 `time_integral=True`（`safety_loss.py:25`）：

```text
L_safety = (1/N) * sum_{n=1..N} c_n
```

代码：`safety_loss.py:65`。

---

## 5. `guidance_loss.py`：朝目标引导

代码：`YOPO/loss/guidance_loss.py:24-37`

定义：

```text
t = pT - p0   # 轨迹方向向量
q = g  - p0   # 目标方向向量
```

默认调用 `similarity_loss`（`guidance_loss.py:32`）。

### 5.1 similarity_loss（默认）

代码：`YOPO/loss/guidance_loss.py:62-78`

```text
q_hat       = q / (||q|| + eps)
t_parallel  = dot(t, q_hat)
parallel    = SmoothL1(||q||, t_parallel)
t_perp      = t - t_parallel * q_hat
perp        = ||t_perp||
L_goal      = parallel + perp_weight * perp
```

`perp_weight = 0.5`（`guidance_loss.py:76`）。

### 5.2 可选速度方向约束

代码：`YOPO/loss/guidance_loss.py:80-88`

```text
L_vel_dir = 1 - cos(vT, q)
```

只有 `vel_dir_weight > 0` 时启用（默认是 0）。

---

## 6. 训练中的连接方式

调用链：

1. `YOPO/policy/yopo_trainer.py:142`
2. `YOPO/policy/yopo_trainer.py:171`
3. `YOPO/loss/loss_function.py:107-109`
4. 回传 `loss.backward()`

并行的 score 监督（`YOPO/policy/yopo_trainer.py:174`）：

```text
score_label = L_smooth + L_safety + L_goal + L_acc
```

---

## 7. 安全代价数值例子

假设：`d0=1.2`, `r=0.6`，公式：

```text
c = exp(-(d-d0)/r)
```

1. `d=2.0` -> `c ≈ 0.26`  
2. `d=0.6` -> `c ≈ 2.72`  
3. `d=-0.2` -> `c ≈ 10.3`

越靠近障碍（特别是进入障碍），惩罚指数上升。

---

## 8. 文件角色一览

1. `YOPO/loss/__init__.py`：空文件，仅包初始化。
2. `YOPO/loss/loss_function.py`：总入口，构造矩阵与加权汇总。
3. `YOPO/loss/smoothness_loss.py`：jerk/acc 二次型代价。
4. `YOPO/loss/safety_loss.py`：ESDF 构建 + 路径碰撞代价。
5. `YOPO/loss/guidance_loss.py`：目标引导代价。

---

## 9. 神经网络输入 / 输出（维度+定义）

这一节对应你图里的 Inference 分支，代码主入口是 `YOPO/test_yopo_ros.py` 和 `YOPO/policy/yopo_network.py`。

### 9.1 原始输入（ROS 侧）

代码：`YOPO/test_yopo_ros.py:93-95`, `YOPO/test_yopo_ros.py:124-143`, `YOPO/test_yopo_ros.py:149-167`

1. 深度图 `depth`  
   来源话题：`/depth_image`（`YOPO/test_yopo_ros.py:380`）  
   编码支持：
   - `32FC1`（仿真，单位米）`YOPO/test_yopo_ros.py:151-153`
   - `16UC1`（实机常见，单位毫米）`YOPO/test_yopo_ros.py:153-155`
   预处理后形状：
   ```text
   depth_input shape = [B, 1, H, W]
   H = cfg["image_height"], W = cfg["image_width"]
   默认 H=96, W=160  (YOPO/config/traj_opt.yaml:17-18)
   ```

2. 状态向量 `obs`（9 维）  
   在 `process_odom()` 构造（`YOPO/test_yopo_ros.py:141`）：
   ```text
   obs = [vel_c(3), acc_c(3), goal_c(3)]  => shape [B, 9]
   ```
   其中：
   - `vel_c`: 机体系速度
   - `acc_c`: 机体系加速度（注意：这里不是 IMU 直接测量，见第 11 节）
   - `goal_c`: 机体系目标方向向量

### 9.2 网络实际输入（进入 forward 前）

代码：`YOPO/test_yopo_ros.py:174-176`, `YOPO/policy/state_transform.py:80-103`

状态先做两步变换：

1. 归一化（速度/加速度按上限缩放）：
```text
obs_norm[:,0:3] = vel / vel_max
obs_norm[:,3:6] = acc / acc_max
obs_norm[:,6:9] = goal / max(||goal||, goal_length)
```
对应 `YOPO/policy/state_transform.py:110-117`。

2. 投影到 primitive 网格坐标：
```text
prepare_input: [B,9] -> [B,9,V,H]
V = vertical_num, H = horizon_num
默认 V=3, H=5   (YOPO/config/traj_opt.yaml:21-22)
```
对应 `YOPO/policy/state_transform.py:86-103`。

所以 `YopoNetwork.forward()` 的输入是：

```text
depth: [B,1,96,160]
obs:   [B,9,3,5]
```

---

## 10. 网络结构（逐层维度）

代码：`YOPO/policy/yopo_network.py`, `YOPO/policy/models/backbone.py`, `YOPO/policy/models/head.py`, `YOPO/policy/models/resnet.py`

### 10.1 Backbone（图像分支）

定义：`YopoBackbone(hidden_state)`，默认 `hidden_state=64`（`YOPO/policy/yopo_network.py:20`, `YOPO/policy/yopo_network.py:26`）。

实现是改造版 ResNet18：

1. 输入通道改为 1（深度图）  
   `YOPO/policy/models/backbone.py:12`
2. 输出层改为 1x1 conv 到 64 通道  
   `YOPO/policy/models/backbone.py:13`
3. `resnet.py` 里 `maxpool` 被注释掉  
   `YOPO/policy/models/resnet.py:238`

维度推导（默认 96x160）：

```text
输入: [B,1,96,160]
conv1 stride2         -> [B,64,48,80]
layer1 stride2        -> [B,64,24,40]
layer2 stride2        -> [B,128,12,20]
layer3 stride2        -> [B,256,6,10]
layer4 stride2        -> [B,512,3,5]
output_layer 1x1 conv -> [B,64,3,5]
```

### 10.2 状态分支

`state_backbone = nn.Sequential()`（恒等映射）  
`YOPO/policy/yopo_network.py:27`

```text
输入: [B,9,3,5]
输出: [B,9,3,5]
```

### 10.3 融合与 Head

代码：`YOPO/policy/yopo_network.py:36`, `YOPO/policy/models/head.py:7-13`

通道拼接：

```text
[B,64,3,5] concat [B,9,3,5] = [B,73,3,5]
```

Head 结构（全 1x1 卷积）：

```text
Conv(73->256) + ReLU
Conv(256->256) + ReLU
Conv(256->10)
输出: [B,10,3,5]
```

### 10.4 网络输出定义

代码：`YOPO/policy/yopo_network.py:38-40`

```text
output[:, :9]  -> endstate_pred_raw, shape [B,9,3,5], 用 tanh 限幅到 [-1,1]
output[:,  9]  -> score_raw,        shape [B,3,5],   用 softplus 保证非负
```

这 9 维语义（在 primitive 局部参数里）：

```text
[delta_yaw, delta_pitch, radius, vx,vy,vz, ax,ay,az]
```

然后通过 `pred_to_endstate()` 变成 body frame 的终态：

```text
endstate: [B,9,3,5]  # [px,py,pz,vx,vy,vz,ax,ay,az]
```
对应 `YOPO/policy/state_transform.py:12-51`。

---

## 11. 你问的 vel/acc 到底从哪来（训练 vs 推理）

### 11.1 训练时

代码：`YOPO/policy/yopo_dataset.py:96-104`, `YOPO/policy/yopo_dataset.py:109-124`

训练输入里的 `vel/acc` 不是传感器真值，而是随机采样：

```text
vel_w ~ 分布采样（含右偏 vx 设计）
acc_w ~ 高斯采样并截断
再旋转到 body frame 得 vel_b, acc_b
```

目标方向也是随机采样得到（`YOPO/policy/yopo_dataset.py:126-136`）。

### 11.2 推理时

代码：`YOPO/test_yopo_ros.py:131-135`

```text
vel_w = odom.twist.linear            # 来自里程计速度
acc_w = self.desire_acc              # 来自上一次发布轨迹的参考加速度
```

关键点：

1. 推理代码没有订阅 IMU（只订阅 odom/depth/goal），见 `YOPO/test_yopo_ros.py:93-95`。  
2. 因此 `acc` 不是“仿真理想 IMU 直接测量值”，而是参考轨迹内部状态 `desire_acc`。  
3. `desire_acc` 在 `control_pub` 每个控制周期由多项式求出并回写，见 `YOPO/test_yopo_ros.py:228-233`。

---

## 12. OOD（分布外）怎么处理？能保证吗？

先说结论：**不能保证绝对不 OOD**，代码里没有显式 OOD 检测器或不确定性门控。

### 12.1 当前代码已有的缓解手段

1. 训练场景随机化（地图、姿态、位置）  
   - 地图随机生成：`Simulator/src/config/config.yaml:13`, `Simulator/src/config/config.yaml:27`
   - 采样姿态/位置：`Simulator/src/config/config.yaml:81-85`

2. 状态随机化（vel/acc/goal）  
   - `YOPO/policy/yopo_dataset.py:96-104`, `YOPO/policy/yopo_dataset.py:109-136`

3. 输入归一化，降低尺度漂移  
   - `YOPO/policy/state_transform.py:110-117`

4. 深度单位兼容与重采样  
   - `32FC1`/`16UC1` 双编码支持：`YOPO/test_yopo_ros.py:151-155`
   - 尺寸不一致时 resize 到训练输入尺寸：`YOPO/test_yopo_ros.py:158-160`

5. README 的实机适配建议  
   - 匹配相机分辨率/FOV：`README.md:199`
   - odom 话题替换为实机话题：`README.md:197`
   - 可切到 `plan_from_reference=True` 做更保守控制接口：`README.md:201`

### 12.2 为什么仍然可能 OOD

1. 训练深度主要来自仿真渲染，实机噪声/失真模型不完全一致。  
2. 推理 `acc` 用的是参考轨迹，不是实测线加速度，存在模型偏差。  
3. 没有在线置信度过滤和异常回退机制（比如不确定性阈值、MPC 接管等）。

所以工程上通常要额外加：

1. 传感器噪声域随机化/实测数据微调  
2. OOD 检测与安全回退策略  
3. 约束更硬的下游控制/安全壳层

---

## 13. 图里说的 `primitive` 到底是什么？

你图里的 “Primitives Definition & Offsets Application” 在代码中对应：

1. primitive 锚点定义：`YOPO/policy/primitive.py:62-81`  
2. 输入坐标变换（Body -> Primitive）：`YOPO/policy/state_transform.py:80-103`  
3. 输出偏移应用（Primitive -> Body）：`YOPO/policy/state_transform.py:30-50`

### 13.1 一句话定义

`primitive` = 一条“候选运动方向模板”（也可以叫锚点轨迹方向）。

YOPO 不直接从连续空间盲搜，而是先离散出一组方向/姿态模板（lattice），网络只需在每个模板上预测“偏移量 + 代价分数”。

### 13.2 这些模板怎么生成

代码：`YOPO/policy/primitive.py:63-72`

配置给出离散个数：

```text
horizon_num = 5
vertical_num = 3
radio_num = 1
traj_num = horizon_num * vertical_num * radio_num = 15
```

每个 primitive 对应一个方向角 `(alpha, beta)` 和半径 `search_radio`：

```text
alpha = 水平离散角（yaw 方向）
beta  = 垂直离散角（pitch 方向）
r     = search_radio
```

锚点位置（在 body 系里）：

```text
px = cos(beta)*cos(alpha)*r
py = cos(beta)*sin(alpha)*r
pz = sin(beta)*r
```

对应 `YOPO/policy/primitive.py:70-72`。  
这就是图里扇形发散出去的橙色“候选方向”。

### 13.3 网络并不是直接输出终点，而是输出“偏移”

代码：`YOPO/policy/state_transform.py:30-38`

网络输出前 3 维先解释为：

```text
delta_yaw   = pred[0] * yaw_diff
delta_pitch = pred[1] * pitch_diff
radius      = (pred[2] + 1.0) * radio_range
```

然后与 primitive 的 `(alpha, beta)` 叠加，得到最终终点位置（body）：

```text
x = cos(beta + delta_pitch) * cos(alpha + delta_yaw) * radius
y = cos(beta + delta_pitch) * sin(alpha + delta_yaw) * radius
z = sin(beta + delta_pitch) * radius
```

对应 `YOPO/policy/state_transform.py:34-38`。

---

## 14. 坐标系怎么来的？为什么要 Body <-> Primitive 变换？

你图里常见 4 个坐标系：

1. `W` World：世界系（odom 所在系）  
2. `B` Body：机体系（无人机本体）  
3. `C` Camera：相机系（深度图射线系）  
4. `P` Primitive：每个 primitive 自己的局部系

### 14.1 World / Body / Camera 在推理中的来源

代码：`YOPO/test_yopo_ros.py:124-143`

1. 从 odom 四元数得到 `R_wb`（Body 到 World 旋转）。  
2. 用机身到相机外参 `R_bc`（由 `pitch_angle_deg` 生成）得到：
```text
R_wc = R_wb * R_bc
R_cw = R_wc^T
```
对应 `YOPO/test_yopo_ros.py:126-129`。

3. 把速度/加速度/目标方向从 world 变到 camera，构造观测 `obs=[vel_c, acc_c, goal_c]`。  
对应 `YOPO/test_yopo_ros.py:131-142`。

### 14.2 为什么还要 Primitive 系？

因为每个 primitive 都代表“朝某个方向看”的局部坐标。  
把同一个状态变到不同 primitive 系后，网络更容易学到“局部偏移规律”，减少直接学习全局多模态。

这一步在 `prepare_input()` 实现：

代码：`YOPO/policy/state_transform.py:86-103`

```text
obs: [B,9] -> [B,3,3]   # 三行分别是 vel/acc/goal，每行 xyz
对每个 primitive 的旋转矩阵做一次变换
输出: [B,9,V,H]
```

### 14.3 输出时怎么从 Primitive 回到 Body

代码：`YOPO/policy/state_transform.py:41-46`

网络预测的 `vel/acc` 先在 primitive 系，随后乘旋转矩阵回到 body 系：

```text
v_b = R * v_p
a_b = R * a_p
```

再与位置 `p_b` 拼成：

```text
endstate_b = [p_b, v_b, a_b], shape [B,9,V,H]
```

对应 `YOPO/policy/state_transform.py:48-50`。

### 14.4 再从 Body 回到 World 用于执行

推理节点里最终会把候选终态从 body 旋到 world：

代码：`YOPO/test_yopo_ros.py:188-190`

```text
endstate_w = R_wc * endstate_c
```

然后选最小 score 的 primitive（`YOPO/test_yopo_ros.py:191`），生成 5 次多项式并发布控制命令。

---

## 15. 一个容易混淆的命名点（Rbp / Rpb）

`YOPO/policy/primitive.py` 里变量名是 `lattice_Rbp`（`primitive.py:81`），  
但在 `state_transform.py` 的用法里，你可以把它理解成“primitive 与 body 间的旋转矩阵集合”。

不同函数里有行向量/列向量写法差异，所以会看到：

1. 输入变换用右乘（`obs_exp @ R...`，`state_transform.py:98`）  
2. 输出变换用左乘（`R... @ v_p`，`state_transform.py:45-46`）

这本质是在不同张量排列下做同一件事：  
**在 Body 系与 Primitive 系之间做坐标旋转。**
