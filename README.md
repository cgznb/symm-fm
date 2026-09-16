# Symm-FM

I-SPY2 乳腺与 MU-Glioma 脑胶质瘤纵向 MRI 的条件 SymmFlow 研究代码，源码快照日期为 2026-09-16。
核心包位于 `src/ispy2_symmflow`；first-post、三相位与 ROI32 的数据和训练适配也随仓库提供。

## 安装与入口

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python run.py symm -m ispy2_symmflow.cli --help
python run.py ispy2 scripts/run_three_phase_symmflow.py --help
python run.py mewm scripts/run_registered_roi32.py --help
```

使用 Python 3.11 或 3.12。GPU 训练需要与设备匹配的 PyTorch；MedGemma 仅用于共享代码里的
BiFM 对照，SymmFlow 的结构化条件编码器不需要该语言模型。
`run.py` 隔离各工作流的源码路径，避免不同版本的 `mewm_ispy2` 包互相覆盖。

## 数据与配置

从 `paths.example.yaml` 建立自己的 `paths.local.yaml`；详见 [路径说明](docs/DATA_AND_PATHS.md)。
仓库不包含 MRI、掩膜、患者级划分、缓存、权重或训练日志。原生数据、分割模型和 VQ 权重须独立取得。
`@repo/` 定位随仓库附带的代码；其余路径别名和单文件映射用于配置私有输入。
历史兼容字段通过 `legacy_config_roots` 在内存中读取，不随示例配置发布。

## 各工作流

```bash
# 核心 CLI：导入、缓存、训练、推理和评估的子命令
python run.py symm -m ispy2_symmflow.cli --help

# first-post：准备 shared bundle 后，只训练 SymmFlow 分支
python run.py ispy2 -m mewm_ispy2.first_post_world_prepare --help
python run.py ispy2 -m mewm_ispy2.first_post_world_training train --config configs/first_post_world_compare_v1.yaml --arm symm

# 原生三相位，包含 T0 裁剪回退策略
python run.py ispy2 scripts/run_three_phase_symmflow.py --config configs/three_phase_symmflow_all_pairs_t0_v2.yaml --stage all

# 配准 ROI32，prepare / VQ / latents / FM / evaluation 阶段见 --help
python run.py mewm scripts/run_registered_roi32.py --config configs/registered_dce0_roi32_firstpostmask_v1.yaml --stage all
```

三相位的数据准备还会使用 `support/pillar` 中的裁剪/元数据帮助函数；这不表示生成模型使用 Pillar 分类器。
远程影像准备必须显式填写 SSH host 和路径；仓库不包含旧机器的 SSH 配置。
训练恢复和生成命令必须继续使用同一配置及配套 VQ，已有旧任务仍在原工作目录管理。
核心训练、`--resume` 恢复、`sample` 推理和 `evaluate` 评估的完整命令见
[实验索引中的核心训练链](docs/EXPERIMENTS.md#核心训练链)。

## 协议与验证

[实验索引](docs/EXPERIMENTS.md) 区分配准 DCE0、原生 first-post、三相位和 ROI32。
方法代码或适配器存在不代表对应正式训练已完成。验证集被用于检查点选择时，不作为未参与选择的测试集。
实际发布验证见 [验证记录](docs/VALIDATION.md)，来源与第三方许可见 [来源说明](docs/SOURCES.md)。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python run.py symm -m pytest -q tests/test_flow_path.py tests/test_solver.py tests/test_sampler.py tests/test_training_run.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python run.py ispy2 -m pytest -q tests/test_three_phase_symmflow.py tests/test_three_phase_all_pairs.py tests/test_three_phase_preparation.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python run.py mewm -m pytest -q tests/test_registered_roi32.py tests/test_first_post_t0_crop.py
```
