# Fixed-camera YOPO, chicane, 100 epochs

这是报告 `reports/active_camera_ablation/epoch100_chicane_20260820/REPORT.md`
使用的固定相机对照模型。它只能在运行时 `active_camera=false`、相机状态与目标精确为零、
Insight 9 `resize_nearest_full_fov_v1`、深度上限 20 m、网络输入
`160(W)×192(H)` 的合同下使用。

## 文件与完整性

| 文件 | SHA-256 |
|---|---|
| `epoch100.pth` | `6e970cb69c4d8b45e23470205caff5ba1c920651f7bbc408f8962a2f21293afb` |
| `epoch100.training_manifest.json` | `dc712f84c302a9d4489ee512a99e43ccc5ddd7259225ff546b949bc023cfd506` |
| `metrics.csv` | `7ebbaffeeda467d15fa3f0371f5220309067569770c3baf9e5579dfc56f10dac` |
| `metrics.json` | `320cf585bab5ee8c1c3ecf6edb3fe3e20e1cc216dc08e5c7e146e65ced931ac6` |
| `resolved_config.yaml` | `b62ab8af618e45ef6a1e97674ede840a4489791929b3edbe0f3ff19e72b2f939` |

`epoch100.manifest.json` 将 checkpoint、不可变训练 manifest、模式和 resolved config
绑定在一起。不要单独复制或重命名 `.pth`。最终 `.pth` 直接保存在分支中。

## 训练合同

- `active_camera=false`，训练 seed `0`，100 epochs，batch size 16，4 workers。
- 数据为 `maze_type=8`、地图 seeds 3..12、每图 10,000 位姿的
  `dataset_chicane_fixed_100ep`。
- 配对数据审计 SHA-256：
  `2386e2e869679d476f155822dc3994bd0d7d2b9e296e5b5e06e30920da57e4ef`。
- 最终 eval total loss `5.69644664`，score loss `0.12756469`，camera loss
  `2.466e-09`。训练指标不是闭环安全指标。

复现训练：

```bash
python3 tools/run_yopo_pipeline.py \
  --mode train --seed 0 --active-camera false \
  --dataset-path ../dataset_chicane_fixed_100ep \
  --train-epoch 100 --batch-size 16 --num-workers 4 \
  --run-name fixed_camera_chicane_100ep_seed0
```

完整的数据生成、checkpoint 检查、闭环命令和实测限制见实验报告。主评测中该模型
5/10 发生碰撞、8/10 到达、5/10 无碰撞且到达。
