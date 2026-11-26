"""编码器共用的基础表面传感器提取与加载模块。

以鱼体壁面全部 504 个紧邻流体单元质心作为基础表面传感器池：
- 几何坐标与弧长写入 `data/processed/mesh.h5`（`/sensors/wall_*`）；
- 各雷诺数工况的传感器原始读数写入对应 `data/processed/ReXXX.h5`（`/sensors/wall_uvp`）；
- 所有数据保持原始物理量纲（`Ux [m/s], Uy [m/s], p_kin [m^2/s^2]`），不在此处提前归一化。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Optional, Sequence

import h5py
import numpy as np


@dataclass
class BaseSurfaceMesh:
    """基础表面传感器几何信息（对应全部壁面近壁单元中心）。"""

    L: float  # 特征体长 (m)
    wall_cell_ids: np.ndarray  # [Nwall] 全局网格单元索引
    wall_xy: np.ndarray  # [Nwall, 2] 局部物理坐标 (m)，原点在鱼头
    wall_xy_nd: np.ndarray  # [Nwall, 2] 无量纲坐标 (x/L, y/L)
    wall_arc: np.ndarray  # [Nwall] 沿闭合轮廓逆时针归一化弧长 [0, 1)
    cand60_wall_indices: np.ndarray  # [60] 预抽取的 60 个等弧长候选点在 wall 中的下标


@dataclass
class BaseSurfaceCase:
    """单个雷诺数工况的基础表面传感器时序数据（未归一化）。"""

    re: float
    st: float
    u_inf: float  # 来流速度 (m/s)
    L: float  # 特征体长 (m)
    uvp: np.ndarray  # [Nt, Nwall, 3] 传感器读数 [Ux (m/s), Uy (m/s), p_kin (m^2/s^2)]
    phase: np.ndarray  # [Nt] 瞬时相位 (rad, [0, 2π))
    phase_sin: np.ndarray  # [Nt] sin(phase)
    phase_cos: np.ndarray  # [Nt] cos(phase)
    periodic_mask: np.ndarray  # [Nt] 有效周期帧掩码 (1=有效)


def compute_wall_arc_length(wall_xy_nd: np.ndarray) -> np.ndarray:
    """沿鱼体闭合轮廓计算每个壁面单元中心的逆时针归一化弧长 tau ∈ [0, 1)。"""
    # 下表面从头到尾 (x 递增) + 上表面从尾到头 (x 递减) 构成逆时针闭合环
    lower_idx = np.where(wall_xy_nd[:, 1] < 0.0)[0]
    upper_idx = np.where(wall_xy_nd[:, 1] >= 0.0)[0]
    lower_sorted = lower_idx[np.argsort(wall_xy_nd[lower_idx, 0])]
    upper_sorted = upper_idx[np.argsort(-wall_xy_nd[upper_idx, 0])]
    loop_order = np.concatenate([lower_sorted, upper_sorted])

    loop_xy = wall_xy_nd[loop_order]
    seg_lens = np.linalg.norm(np.roll(loop_xy, -1, axis=0) - loop_xy, axis=1)
    cum_arc = np.concatenate([[0.0], np.cumsum(seg_lens[:-1])])
    tau_loop = cum_arc / seg_lens.sum()

    # 映射回 wall_xy_nd 的原始排列顺序
    wall_arc = np.empty(len(wall_xy_nd), dtype=np.float64)
    wall_arc[loop_order] = tau_loop
    return wall_arc


def _discover_re_files(processed_dir: Path) -> Dict[int, Path]:
    """扫描目录中的 ReXXX.h5 文件并按雷诺数升序返回。"""
    files: Dict[int, Path] = {}
    for fp in processed_dir.glob("Re*.h5"):
        m = re.search(r"(\d+)", fp.stem)
        if m:
            files[int(m.group(1))] = fp
    return dict(sorted(files.items()))


def build_base_surface_sensors(
    processed_dir: str | Path = "data/processed",
    overwrite: bool = False,
) -> Dict[str, object]:
    """从已处理的 mesh.h5 与 ReXXX.h5 提取 504 个壁面单元中心传感器并写回 data/processed。"""
    proc_dir = Path(processed_dir)
    mesh_path = proc_dir / "mesh.h5"

    # 1. 读取全部壁面单元坐标与 ROI 映射索引
    with h5py.File(mesh_path, "r") as h5:
        L = float(h5["/geometry/L"][0])
        cell_xy_full = np.asarray(h5["/mesh/cell_xy_full"][()], dtype=np.float64)
        roi_cell_ids = np.asarray(h5["/mesh/roi_cell_ids"][()], dtype=np.int64)
        wall_cell_ids = np.asarray(h5["/sensors/wall_owner_full"][()], dtype=np.int64)

    wall_xy = cell_xy_full[wall_cell_ids]
    wall_xy_nd = wall_xy / L
    wall_arc = compute_wall_arc_length(wall_xy_nd)

    # 将基础表面传感器坐标与弧长写回 mesh.h5
    with h5py.File(mesh_path, "a") as h5:
        for key, arr, dt in [
            ("/sensors/wall_xy", wall_xy, np.float64),
            ("/sensors/wall_xy_nd", wall_xy_nd, np.float64),
            ("/sensors/wall_arc", wall_arc, np.float64),
        ]:
            if key in h5 and overwrite:
                del h5[key]
            if key not in h5:
                h5.create_dataset(key, data=arr, dtype=dt)

    # 计算壁面单元在 /field/uvp 中的列下标，一次性切片提取各工况读数
    roi_pos = {int(cid): idx for idx, cid in enumerate(roi_cell_ids.tolist())}
    wall_in_roi = np.array([roi_pos[int(cid)] for cid in wall_cell_ids.tolist()], dtype=np.int64)

    re_files = _discover_re_files(proc_dir)
    updated_re: list[int] = []
    for re_val, case_path in re_files.items():
        with h5py.File(case_path, "a") as h5:
            if "/sensors/wall_uvp" in h5 and not overwrite:
                continue
            wall_uvp = np.asarray(h5["/field/uvp"][:, wall_in_roi, :], dtype=np.float32)
            if "/sensors/wall_uvp" in h5:
                del h5["/sensors/wall_uvp"]
            h5.create_dataset("/sensors/wall_uvp", data=wall_uvp, dtype=np.float32)
            updated_re.append(re_val)

    return {
        "n_wall_sensors": int(len(wall_cell_ids)),
        "re_list": list(re_files.keys()),
        "updated_re": updated_re,
    }


def load_base_surface_mesh(processed_dir: str | Path = "data/processed") -> BaseSurfaceMesh:
    """加载基础表面传感器几何信息（504 个壁面单元中心）。"""
    mesh_path = Path(processed_dir) / "mesh.h5"
    with h5py.File(mesh_path, "r") as h5:
        L = float(h5["/geometry/L"][0])
        wall_cell_ids = np.asarray(h5["/sensors/wall_owner_full"][()], dtype=np.int64)
        wall_xy = np.asarray(h5["/sensors/wall_xy"][()], dtype=np.float64)
        wall_xy_nd = np.asarray(h5["/sensors/wall_xy_nd"][()], dtype=np.float64)
        wall_arc = np.asarray(h5["/sensors/wall_arc"][()], dtype=np.float64)
        cand60 = np.asarray(h5["/sensors/wall_cand_indices"][()], dtype=np.int64)

    return BaseSurfaceMesh(
        L=L,
        wall_cell_ids=wall_cell_ids,
        wall_xy=wall_xy,
        wall_xy_nd=wall_xy_nd,
        wall_arc=wall_arc,
        cand60_wall_indices=cand60,
    )


def load_base_surface_cases(
    processed_dir: str | Path = "data/processed",
    re_list: Optional[Sequence[int]] = None,
) -> Dict[int, BaseSurfaceCase]:
    """按雷诺数加载基础表面传感器时序数据（不读取 30MB 全场流场）。"""
    proc_dir = Path(processed_dir)
    re_files = _discover_re_files(proc_dir)
    target_res = sorted(re_files.keys()) if re_list is None else [int(r) for r in re_list]

    cases: Dict[int, BaseSurfaceCase] = {}
    for re_val in target_res:
        with h5py.File(re_files[re_val], "r") as h5:
            cases[re_val] = BaseSurfaceCase(
                re=float(h5["/meta/Re"][0]),
                st=float(h5["/meta/St"][0]),
                u_inf=float(h5["/meta/U_inf"][0]),
                L=float(h5["/meta/L"][0]),
                uvp=np.asarray(h5["/sensors/wall_uvp"][()], dtype=np.float32),
                phase=np.asarray(h5["/snapshots/phase"][()], dtype=np.float64),
                phase_sin=np.asarray(h5["/snapshots/phase_sin"][()], dtype=np.float64),
                phase_cos=np.asarray(h5["/snapshots/phase_cos"][()], dtype=np.float64),
                periodic_mask=np.asarray(h5["/snapshots/periodic_mask"][()], dtype=np.int32),
            )
    return cases


def select_surface_sensors(
    cases: Dict[int, BaseSurfaceCase],
    mesh: BaseSurfaceMesh,
    sensor_indices: Optional[Sequence[int]] = None,
    periodic_only: bool = True,
) -> Dict[str, np.ndarray]:
    """按给定传感器下标从多个工况中抽取并拼接训练所需的原始张量。

    参数:
        cases: 选定雷诺数工况字典（通常由配置文件指定划分）。
        mesh: 基础表面传感器网格。
        sensor_indices: 要取的传感器下标（相对于 504 个壁面单元）；为 None 时取全部 504 个。
        periodic_only: 是否只保留周期有效帧。

    返回:
        包含原始物理读数 `uvp [N, K, 3]`、坐标 `xy_nd [K, 2]` 及目标标签的字典。
    """
    idx = (
        np.arange(len(mesh.wall_cell_ids), dtype=np.int64)
        if sensor_indices is None
        else np.asarray(sensor_indices, dtype=np.int64)
    )

    uvp_list, re_list, st_list = [], [], []
    phase_list, sin_list, cos_list, re_id_list = [], [], [], []

    for re_id in sorted(cases.keys()):
        c = cases[re_id]
        mask = c.periodic_mask > 0 if periodic_only else np.ones(len(c.phase), dtype=bool)
        uvp_case = c.uvp[mask][:, idx, :]
        n = len(uvp_case)

        uvp_list.append(uvp_case)
        re_list.append(np.full(n, c.re, dtype=np.float64))
        st_list.append(np.full(n, c.st, dtype=np.float64))
        phase_list.append(c.phase[mask])
        sin_list.append(c.phase_sin[mask])
        cos_list.append(c.phase_cos[mask])
        re_id_list.append(np.full(n, re_id, dtype=np.int64))

    return {
        "uvp": np.concatenate(uvp_list, axis=0),
        "xy_nd": mesh.wall_xy_nd[idx],
        "arc": mesh.wall_arc[idx],
        "re": np.concatenate(re_list, axis=0),
        "st": np.concatenate(st_list, axis=0),
        "phase": np.concatenate(phase_list, axis=0),
        "phase_sin": np.concatenate(sin_list, axis=0),
        "phase_cos": np.concatenate(cos_list, axis=0),
        "re_id": np.concatenate(re_id_list, axis=0),
        "sensor_indices": idx,
    }


if __name__ == "__main__":
    summary = build_base_surface_sensors("data/processed")
    print(f"基础表面传感器构建完成: 共 {summary['n_wall_sensors']} 个壁面测点, 更新工况 {len(summary['updated_re'])}/{len(summary['re_list'])}")
