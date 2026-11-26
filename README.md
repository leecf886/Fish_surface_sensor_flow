# Fish Surface Sensor Flow

## 1. 综述

## 2. 整体流程

### 2.1 数据预处理（`.msh` + `.npz` → `.h5`）

环境准备：

```bash
conda activate surface_cylinder_flow
```

**第一步：网格解析与核心密集区（ROI）参数提取**

读取 `msh/fish_fluent.msh`，转换至鱼头原点无量纲坐标系，截取密集区域 `x/L ∈ [-0.5, 4.5], y/L ∈ [-0.8, 0.8]`（32,084 单元）并沿鱼壁等弧长抽取 60 个候选传感器，生成截取参数、网格文件与尺寸对比图：

```bash
python scripts/prepare_fish_mesh.py
```

输出：
- `configs/mesh_roi_params.json`（ROI 与传感器索引参数）
- `data/processed/mesh.h5`（网格与传感器 HDF5）
- `docs/mesh_roi_overview.png`（全网格与 ROI 尺寸对比图）

**第二步：工况数据裁剪与 HDF5 打包**

单线程循环读取 `data_files/Re*/two_cycles.npz` 与 `cycle_quality.json`，按截取参数提取传感器读数与 ROI 流场，拼接真实 `St` 及弧度相位后输出精悍工况文件：

```bash
python scripts/preprocess_fish_npz.py
```

输出：
- `data/processed/Re13500.h5` ~ `Re17500.h5`（共 16 个工况，总计约 472 MB）
