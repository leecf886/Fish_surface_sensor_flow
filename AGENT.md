# AGENT.md — `fish_flow` 项目开发交接文档

**更新日期**：2026-11-26  
**工作区**：`e:\code\Fish_surface_sensor_flow`  
**运行环境**：`conda activate surface_cylinder_flow`（`D:\anaconda_envs\surface_cylinder_flow\python.exe`）  
**导入方式**：已建 [src/fish_flow/\_\_init\_\_.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/__init__.py)（尚未建 `pyproject.toml`），运行时需设置 `PYTHONPATH=src`  
**设计参考**：[2026-11-26-fish-flow-migration-design.md](file:///e:/code/Fish_surface_sensor_flow/docs/superpowers/specs/2026-11-26-fish-flow-migration-design.md)（`_ref/` 为原圆柱基线，只读不改）

---

## 1. 核心架构约定与红线

1. **两层解耦与单文件配置约定**：
   - **底层纯参数接口**：`data/`、`collocation.py`、`models.py`、`losses.py`、`training/` 均不读取任何配置文件，全部通过显式函数/类参数传入。
   - **上层单 `config.json` 驱动**：每次训练由 `runs/<exp>/config.json` 单独管控全部参数（`task`、`processed_dir`、`train_re / val_re / test_re`、`data`、`model`、`training`）；[config.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/config.py) 与 [scripts/train.py](file:///e:/code/Fish_surface_sensor_flow/scripts/train.py) 直接读取该文件启动训练，并将 checkpoint（`*.pt`）与 `history.json` 保存至同目录（不覆写原 `config.json`）。`runs/` 已被 `.gitignore` 忽略。
2. **代码风格**：保持精简直接，不写防御性代码与防御性叙述；源码附带清晰中文注释。
3. **物理与数学契约**：
   - 特征体长 $L = 0.200\text{ m}$，来流速度 $U_\infty = 1.0\text{ m/s}$，鱼头位于 $(0, 0)$，尾尖位于 $(L, 0)$。
   - 流场第 3 通道为运动压强 $p_{\text{kin}} = p / \rho$（`pressure_scale_mode = "uinf2"`），无量纲化固定为 $u^* = U_x / U_\infty,\ v^* = U_y / U_\infty,\ p^* = p_{\text{kin}} / U_\infty^2$。
   - 显式 $St$ 建模：编码器输出 4 维 `[Re_norm, St_norm, sinφ, cosφ]`，pPINN 输入 6 维 `[x*, y*, Re_norm, St_norm, sinφ, cosφ]`，PDE 配点独立采样 5 维 `(x*, y*, Re, St, φ)`，不使用任何 $Re \mapsto St$ 经验公式。
4. **科研防泄漏**：16 个雷诺数工况（`13500`–`17500`，无 `17250`）中，默认 `train_re`（12 个）、`val_re`（`14750, 16750`）与锁定 `test_re`（`14250, 16250`）严格互斥；所有归一化统计量仅在 `train_re` 上拟合。

---

## 2. 当前已完成资产与模块

### 2.1 已生成的预处理数据与脚本

- 预处理脚本：[prepare_fish_mesh.py](file:///e:/code/Fish_surface_sensor_flow/scripts/prepare_fish_mesh.py)、[preprocess_fish_npz.py](file:///e:/code/Fish_surface_sensor_flow/scripts/preprocess_fish_npz.py)
- 网格裁剪参数：[configs/mesh_roi_params.json](file:///e:/code/Fish_surface_sensor_flow/configs/mesh_roi_params.json)
- `data/processed/mesh.h5`：
  - `/mesh/roi_xy` `(32084, 2)`、`/mesh/roi_xy_nd` `(32084, 2)`（ROI 覆盖 $x/L \in [-0.5, 4.5],\ y/L \in [-0.8, 0.8]$）
  - `/mesh/roi_cell_ids` `(32084,)`、`/mesh/cell_xy_full` `(60048, 2)`
  - `/sensors/wall_owner_full` `(504,)`、`/sensors/wall_xy` `(504, 2)`、`/sensors/wall_xy_nd` `(504, 2)`、`/sensors/wall_arc` `(504,)`、`/sensors/wall_cand_indices` `(60,)`
  - `/geometry/L` `[0.2]`、`/config/roi_bounds_nd` `[-0.5, 4.5, -0.8, 0.8]`
- `data/processed/Re13500.h5` ~ `Re17500.h5`（共 16 个工况，每工况 `Nt=80` 帧）：
  - `/meta/Re`、`/meta/St`、`/meta/U_inf`、`/meta/L`
  - `/snapshots/phase`（弧度 $[0, 2\pi)$）、`phase_sin`、`phase_cos`、`periodic_mask`
  - `/sensors/wall_uvp` `(80, 504, 3)`（原始物理量纲 `[Ux, Uy, p_kin]`）
  - `/field/uvp` `(80, 32084, 3)`（ROI 全场原始物理量纲 `[Ux, Uy, p_kin]`）

### 2.2 已编写的 `src/fish_flow` 核心模块

| 模块 | 主要对象 / 接口 | 职责与输入输出契约 |
| :--- | :--- | :--- |
| [surface_sensors.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/data/surface_sensors.py) | `BaseSurfaceMesh`, `BaseSurfaceCase`, `load_base_surface_mesh`, `load_base_surface_cases`, `select_surface_sensors` | 仅读取壁面 504 个近壁传感器数据（不读全场 `/field/uvp`），保持原始物理量纲。 |
| [encoder_loader.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/data/encoder_loader.py) | `Normalizer`, `EncoderDataset`, `DeepSetsSubsetCollator`, `LoadedEncoderData`, `select_uniform_arc_indices`, `load_encoder_training_data` | 编码器专用训练数据加载与归一化：MLP 展平 Z-score `[N, 3*K]`；DeepSets 仅在训练集拟合 3 通道 `chan_mean / chan_std` 并完成 `[N, K, 3]` 通道标准化，支持随机子集采样；标签统一为 `[Re_norm, St_norm, sinφ, cosφ]`。 |
| [ppinn_loader.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/data/ppinn_loader.py) | `PPINNMesh`, `PPINNCase`, `PPINNSampler`, `LoadedPPINNData`, `load_ppinn_mesh`, `load_ppinn_training_data` | pPINN 全场数据加载：按 `uinf2` 无量纲化为 `[u*, v*, p*]`，拟合 `re_normalizer / st_normalizer`，提供 `sample_train_batch`（`[B, 6]` 与 `[B, 3]`）及逐帧整场验证迭代 `iter_val_frames`。 |
| [collocation.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/collocation.py) | `fish_half_width_nd`, `points_inside_fish`, `sample_collocation_points`, `collocation_to_tensors`, `collocation_fingerprint` | 5D PDE 配点采样与 Verma et al. (2020) 成鱼轮廓固体剔除；支持 `uniform_random / latin_hypercube / sobol` 与 `continuous_random / discrete_train_re_random`。 |
| [models.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/models.py) | `SensorMLP`, `DynamicSetSensorEncoder`, `PPINNModel` | 纯 PyTorch 网络结构：`SensorMLP` `[B, 3*K] -> [B, 4]`；`DynamicSetSensorEncoder` `([B, K, 3], [B, K, 2], mask) -> [B, 4]`；`PPINNModel` `[N, 6] -> [N, 3]`。 |
| [losses.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/losses.py) | `EncoderLoss` (`MLPLoss`, `DeepSetsLoss`), `gradients`, `compute_pde_residuals`, `PPINNLoss` | 纯参数损失与 Navier-Stokes 无量纲残差计算，返回 `(loss_total, metrics)`。 |
| [training/common.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/training/common.py) | `set_training_seed`, `compute_regression_metrics`, `compute_circular_metrics`, `PressureGaugeAccum`, `EarlyStopper`, `build_lr_scheduler`, `compute_pde_weight` | 训练公共组件：指标计算、Chan/Welford Gauge 压强相对 $L_2$ 累计、早停（默认关闭）、学习率调度（默认 `constant`，支持 `step/multistep/cosine/exponential/plateau`）、PDE 权重调度（默认 `constant`，支持 `linear/cosine/step/exponential`）。 |
| [training/mlp.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/training/mlp.py) | `MLPTrainingConfig`, `MLPTrainHistory`, `MLPTrainer`, `train_mlp` | 独立 MLP 训练器：4 维 `EncoderLoss`、反归一化 Re/St/相位物理评估、最优权重保存与恢复。 |
| [training/deepsets.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/training/deepsets.py) | `DeepSetsTrainingConfig`, `DeepSetsTrainHistory`, `DeepSetsTrainer`, `train_deepsets`, `build_dynamic_validation_panel`, `evaluate_dynamic_validation_panel` | 独立 DeepSets 训练器：训练期随机子集批采样，验证期默认多预算冻结面板（按 `layout -> val_re -> budget` 宏平均选优），兼容全池验证。 |
| [training/ppinn.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/training/ppinn.py) | `PPINNTrainingConfig`, `PPINNTrainHistory`, `PPINNTrainer`, `train_ppinn`, `evaluate_ppinn_validation` | 独立 pPINN 训练器：监督采样 + 5D 配点 PDE 残差（`fixed/epoch/batch`）、PDE 权重动态调度、整场 Gauge 压强验证选优、`best/last` 权重保存。 |
| [config.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/config.py) & [scripts/train.py](file:///e:/code/Fish_surface_sensor_flow/scripts/train.py) | `load_run_config`, `run_from_config`, `run_{mlp,deepsets,ppinn}_from_config` | 单文件配置读取与训练入口：通过 `python scripts/train.py --config runs/<exp>`（或指定 `config.json`）一键加载数据、建模并训练，结果与原 `config.json` 同目录存放。 |

### 2.3 已验证状态与默认训练入口

- `runs/` 下已生成三套本地默认配置模板（Git 忽略）：`runs/mlp/config.json`、`runs/deepsets/config.json`、`runs/ppinn/config.json`，启动示例：
  ```powershell
  $env:PYTHONPATH='src'
  python scripts/train.py --config runs/mlp
  python scripts/train.py --config runs/deepsets
  python scripts/train.py --config runs/ppinn
  ```
- [tests/test_training_pipeline.py](file:///e:/code/Fish_surface_sensor_flow/tests/test_training_pipeline.py) 已在 `surface_cylinder_flow` 环境下通过全部 3 项验证（`3 passed in 10.47s`）：包含全部 LR/PDE 调度与早停逻辑、Gauge 压强偏移不变性，以及基于真实 `data/processed/` 的 MLP / 504 全池 DeepSets / 5D 配点 pPINN 单 `config.json` 端到端训练与 checkpoint 落盘校验。

---

## 3. 下一步待开发与衔接要点（推理 / 选位 / 评估阶段）

1. **闭环推理与全场重构（`inference/`）**：
   - 加载 `runs/<exp>/` 中的 `best_*.pt` 与 `normalizers`（含 MLP `x_normalizer`、DeepSets `chan_mean/chan_std`、编码器与 pPINN 的 `re_normalizer / st_normalizer`），串联传感器编码器预测 `(Re_hat, St_hat, phi_hat)` 与 pPINN 整场重构 `(u*, v*, p*)`。
2. **传感器选位与 POD 基线（`layouts/` & `SurfacePOD`）**：
   - 迁移基于闭合弧长的 `uniform_layout`、鱼体水动力关键区 `physics_layout`、4 维状态 Jacobian 敏感度 `greedy` 选位及 `SurfacePOD` 线性基线。
3. **完整测试集离线评估与工作流（`evaluation/` & `workflows/`）**：
   - 在显式开启测试集的前提下，输出锁定工况 `14250 / 16250` 的全场 Gauge 压强 $L_2$、尾流区统计、涡量 / SSIM、物理残差及鲁棒性对比结果；按需补充 `pyproject.toml`。
