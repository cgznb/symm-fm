# 实验索引与命令

| 工作流 | 配置 | 输入与边界 |
|---|---|---|
| I-SPY2 registered DCE0 | `configs/mewm_all_pairs_5090.yaml` | 配准 T0 坐标、共享 T0 ROI、单 DCE0、全部前向访视对 |
| I-SPY2 first-post | ispy2 项目下 `first_post_world_compare_v1.yaml` | 原生单 first-post，调用 `FirstPostSymmFlow` |
| I-SPY2 三相位 | ispy2 项目下 `three_phase_symmflow_all_pairs_t0_v2.yaml` | pre/first-post/late，全部前向访视对，T0 裁剪回退；不是三个纵向访视 |
| I-SPY2 ROI32 | mewm 项目下 `registered_dce0_roi32_firstpostmask_v1.yaml` | 单 DCE0，固定 32×128×128 裁剪，独立 VQ/FM 工作流 |
| MU-Glioma | `configs/mu_glioma_zscore_attn_40k_5090.yaml` | 四模态共享 VQ，拼接连续 latent、训练集通道标准化 |

历史 pilot、不同预算、CFM 和 source-bridge CFM 配置保留为对照。并列配置不意味着同预算、同数据或独立测试。
ROI32 的适配器存在不意味着完整正式 FM 训练已完成；本次只验证合成与 CPU 程序路径。

## 核心训练链

先使用 `import-mewm-latents` 或 `import-mu-glioma-latents` 的 `--help` 核对已有私有缓存与配套 VQ 的输入。
导入产生的 pair manifest 是患者级私有文件，不进入 Git。

```bash
python run.py symm -m ispy2_symmflow.cli import-mewm-latents --help
python run.py symm -m ispy2_symmflow.cli import-mu-glioma-latents --help
python run.py symm -m ispy2_symmflow.cli train-symmflow --config configs/mewm_all_pairs_5090.yaml --pair-manifest /path/to/pairs.jsonl --autoencoder-checkpoint /path/to/vq.ckpt --output-dir /path/to/run
python run.py symm -m ispy2_symmflow.cli train-symmflow --config configs/mewm_all_pairs_5090.yaml --pair-manifest /path/to/pairs.jsonl --autoencoder-checkpoint /path/to/vq.ckpt --output-dir /path/to/run --resume /path/to/checkpoint.pt
python run.py symm -m ispy2_symmflow.cli sample --config configs/mewm_all_pairs_5090.yaml --checkpoint /path/to/checkpoint.pt --autoencoder-checkpoint /path/to/vq.ckpt --source /path/to/source.npz --conditions /path/to/conditions.json --direction forward --output /path/to/prediction.npz --solver euler --steps 20
python run.py symm -m ispy2_symmflow.cli evaluate --prediction /path/to/prediction.npz --target /path/to/target.npz --source /path/to/source.npz --output /path/to/metrics.json --data-range 4
```

`--data-range` 必须与实际图像归一化一致；示例中的 4 不适用于所有历史路径。
输入 `.npz` 应为准备流程生成的影像包，保留几何与来源字段；预测 `.npz` 必须与同名 `.json` 一起保管。
`sample` 的条件字段、输入形状以该子命令帮助和所选配置为准。不得混用不同 VQ 的缓存。

配准 DCE0 的生成器验证集曾参与检查点选择；下游再使用这批患者不构成独立的端到端测试。
