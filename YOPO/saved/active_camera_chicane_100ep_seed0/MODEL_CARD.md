# Active-camera YOPO, chicane, 100 epochs

这是报告 `reports/active_camera_ablation/epoch100_chicane_20260820/REPORT.md`
使用的主动相机模型。它只能在运行时 `active_camera=true`、Insight 9
`resize_nearest_full_fov_v1`、深度上限 20 m、网络输入 `160(W)×192(H)` 的合同下使用。

## 文件与完整性

| 文件 | SHA-256 |
|---|---|
| `epoch100.pth` | `bb59d23ad76e0ba165553f517d1ea05acec0753b256bf5f7d3011c790c638439` |
| `epoch100.training_manifest.json` | `e36c28ea3dc78f61e7705b83b7254c88588a78e1ad53d781c6487f7bc198452b` |
| `metrics.csv` | `8cf2b31184277070bcd525e75bfa562dde8c2e8d69c4f7602192c5dc0d82510f` |
| `metrics.json` | `178ec0ec347247d87ec579001e7d1c74c8194f5a05164f7a05e14f64436c545e` |
| `resolved_config.yaml` | `955721a5e962852805c63c107fc7422698e343cc94549ae2eb205207320355c7` |

`epoch100.manifest.json` 将 checkpoint、不可变训练 manifest、模式和 resolved config
绑定在一起。不要单独复制或重命名 `.pth`。最终 `.pth` 直接保存在分支中。

## 训练合同

- `active_camera=true`，训练 seed `0`，100 epochs，batch size 16，4 workers。
- 数据为 `maze_type=8`、地图 seeds 3..12、每图 10,000 位姿的
  `dataset_chicane_active_100ep`。
- 配对数据审计 SHA-256：
  `2386e2e869679d476f155822dc3994bd0d7d2b9e296e5b5e06e30920da57e4ef`。
- 最终 eval total loss `5.81415056`，score loss `0.14170595`，camera loss
  `0.00778145`。训练指标不是闭环安全指标。

复现训练：

```bash
python3 tools/run_yopo_pipeline.py \
  --mode train --seed 0 --active-camera true \
  --dataset-path ../dataset_chicane_active_100ep \
  --train-epoch 100 --batch-size 16 --num-workers 4 \
  --run-name active_camera_chicane_100ep_seed0
```

完整的数据生成、checkpoint 检查、闭环命令和实测限制见实验报告。主评测中该模型
0/10 碰撞但只有 5/10 到达；它不能被描述为已证明优于固定相机模型。
