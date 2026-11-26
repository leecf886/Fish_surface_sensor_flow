# AGENT.md — `fish_flow` 项目开发交接文档

**更新日期**：2026-11-26  
**工作区**：`e:\code\Fish_surface_sensor_flow`  
**运行环境**：`conda activate surface_cylinder_flow`（`D:\anaconda_envs\surface_cylinder_flow\python.exe`）  
**导入方式**：当前 `src/fish_flow/` 尚未建 `__init__.py` 与 `pyproject.toml`，运行时需设置 `PYTHONPATH=src`  
**设计参考**：[2026-11-26-fish-flow-migration-design.md](file:///e:/code/Fish_surface_sensor_flow/docs/superpowers/specs/2026-11-26-fish-flow-migration-design.md)（`_ref/` 为原圆柱基线，只读不改）

---

## 1. 核心架构约定与红线

1. **纯参数模块接口**：数据加载、配点采样、模型、损失等底层模块**不读取任何配置文件**，全部通过显式函数/类参数传入；配置文件读取与训练编排统一放在后续 pipeline 中完成。
2. **代码风格**：保持精简直接，不写防御性代码与防御性叙述；源码附带清晰中文注释。
3. **物理与数学契约**：
   - 特征体长 $L = 0.200\text{ m}$，来流速度 $U_\infty = 1.0\text{ m/s}$，鱼头位于 $(0, 0)$，尾尖位于 $(L, 0)$。
   - 流场第 3 通道为运动压强 $p_{\text{kin}} = p / \rho$（`pressure_scale_mode = "uinf2"`），无量纲化固定为 $u^* = U_x / U_\infty,\ v^* = U_y / U_\infty,\ p^* = p_{\text{kin}} / U_\infty^2$。
   - 显式 $St$ 建模：编码器输出 4 维 `[Re_norm, St_norm, sinφ, cosφ]`，pPINN 输入 6 维 `[x*, y*, Re_norm, St_norm, sinφ, cosφ]`，PDE 配点独立采样 5 维 `(x*, y*, Re, St, φ)`，不使用任何 $Re \mapsto St$ 经验公式。
4. **科研防泄漏**：16 个雷诺数工况（`13500`–`17500`，无 `17250`）的 `train_re / val_re / test_re` 严格互斥；所有归一化统计量仅在 `train_re` 上拟合。

---

## 2. 当前已完成资产与模块

### 2.1 已生成的预处理数据与脚本

- 预处理脚本：[prepare_fish_mesh.py](file:///e:/code/Fish_surface_sensor_flow/scripts/prepare_fish_mesh.py)、[preprocess_fish_npz.py](file:///e:/code/Fish_surface_sensor_flow/scripts/preprocess_fish_npz.py)
- 网格裁剪参数：[configs/mesh_roi_params.json](file:///e:/code/Fish_surface_sensor_flow/configs/mesh_roi_params.json)（`configs/experiments/wp_{a,b,c,d}/` 当前为空目录）
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
| [encoder_loader.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/data/encoder_loader.py) | `Normalizer`, `EncoderDataset`, `DeepSetsSubsetCollator`, `LoadedEncoderData`, `select_uniform_arc_indices`, `load_encoder_training_data` | 编码器专用训练数据加载与归一化：MLP 完成展平 Z-score `[N, 3*K]`；DeepSets 在训练集拟合 3 通道 `chan_mean` / `chan_std` 并直接完成 `[N, K, 3]` 通道标准化，同时导出统计量与 `[K, 2]` 坐标；标签统一为 `[Re_norm, St_norm, sinφ, cosφ]`。 |
| [ppinn_loader.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/data/ppinn_loader.py) | `PPINNMesh`, `PPINNCase`, `PPINNSampler`, `LoadedPPINNData`, `load_ppinn_mesh`, `load_ppinn_training_data` | pPINN 全场数据加载：读取 `roi_xy_nd`、`roi_bounds_nd` 与 `/field/uvp` 并按 `uinf2` 无量纲化为 `[u*, v*, p*]`；复用 `Normalizer` 仅在训练集拟合 `re_normalizer` / `st_normalizer`；提供矢量化训练批采样 `sample_train_batch`（直接输出 `[B, 6]` 与 `[B, 3]`）及逐帧整场验证迭代 `iter_val_frames`。 |
| [collocation.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/collocation.py) | `fish_half_width_nd`, `points_inside_fish`, `sample_collocation_points`, `collocation_to_tensors`, `collocation_fingerprint` | 5D PDE 配点采样与鱼体固体剔除：实现 Verma et al. (2020) 无量纲分段半宽函数 `h(s)` 与拒绝采样；支持 `uniform_random` / `latin_hypercube` / `sobol` 与 `continuous_random` / `discrete_train_re_random`；输出 `(x_nd, y_nd, re_val, st_val, phi, re_norm, st_norm)` 与无梯度张量，直接对接 `compute_pde_residuals`。 |
| [models.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/models.py) | `SensorMLP`, `DynamicSetSensorEncoder`, `PPINNModel` | 纯 PyTorch 网络结构：`SensorMLP` `[B, 3*K] -> [B, 4]`；`DynamicSetSensorEncoder` `([B, K, 3], [B, K, 2], mask) -> [B, 4]`；`PPINNModel` `[N, 6] -> [N, 3]`（`[x*, y*, Re_norm, St_norm, sinφ, cosφ] -> [u*, v*, p*]`）。 |
| [losses.py](file:///e:/code/Fish_surface_sensor_flow/src/fish_flow/losses.py) | `EncoderLoss` (`MLPLoss`, `DeepSetsLoss`), `gradients`, `compute_pde_residuals`, `PPINNLoss` | 纯参数损失与 PDE 残差模块。`EncoderLoss` 支持 `lambda_re/st/phase`、`loss_type`、`lambda_unit_circle`；`compute_pde_residuals` 在内部构建 `(x_nd, y_nd, phi)` 计算图并输出 `(r_u, r_v, r_c)`；`PPINNLoss` 组合数据项与物理残差项，均返回 `(loss_total, metrics)`。 |

**已验证状态**：
- `surface_sensors.py`、`encoder_loader.py`、`models.py`、`losses.py` 已在 `surface_cylinder_flow` 环境中完成真实 HDF5 前向/反传与解析流场 PDE 残差验证（`atol=1e-12`）。
- `ppinn_loader.py` 与 `collocation.py` 已完成真实 `data/processed/` 读取、无量纲数值核对、鱼体轮廓判定、3 种 5D 配点方法（含固体剔除与指纹一致性）以及串联 `PPINNModel` + `compute_pde_residuals` + `PPINNLoss` 前向/反传验证，无临时测试文件残留。

---

## 3. 下一步待开发与衔接要点（训练 / Pipeline 阶段）

当前底层数据、配点、模型与损失已全部就绪，可直接开始编写 **MLP 训练**、**DeepSets 训练** 与 **pPINN 训练**；完善为完整工程 pipeline 时需同步补齐以下 3 项：

1. **验证物理指标与最优模型选优**：
   - 编码器验证需补齐反归一化还原 `(Re_hat, St_hat, phi_hat)` 后的 MAE/RMSE 与圆周相位误差，并确定 DeepSets 验证期子集协议（全池 vs 固定多预算子集）。
   - pPINN 验证基于 `iter_val_frames` 整场输出，若按物理误差选优，需补齐速度/全场相对 $L_2$ 及 Gauge 零点对齐压强相对 $L_2$ 计算。
2. **配置文件与工况折划分**：
   - 需在 `configs/` 下补齐 `folds.json`（`train_re / val_re / test_re` 划分）及各训练/工作流配置，由 pipeline 统一读取后传参给底层模块。
3. **Checkpoint 持久化与包入口**：
   - 统一保存 `state_dict`、`normalizers_to_dict()`（含 DeepSets 推理/加噪所需的 `chan_mean / chan_std`）、模型超参及训练历史；按需补充 `src/fish_flow/__init__.py` 与 `pyproject.toml`。
