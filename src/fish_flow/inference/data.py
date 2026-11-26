"""推理专用数据加载模块：复用基础数据读取接口，提供未归一化物理量纲输入。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from fish_flow.data.ppinn_loader import load_ppinn_mesh
from fish_flow.data.surface_sensors import (
    load_base_surface_cases,
    load_base_surface_mesh,
    select_surface_sensors,
)


@dataclass
class EncoderInferenceInputs:
    """编码器/串联推理所需的原始物理量纲传感器输入。"""

    sensor_uvp: np.ndarray  # [N, K, 3] 物理读数 [Ux, Uy, p_kin]
    sensor_xy: np.ndarray  # [K, 2] 传感器物理坐标 (m)
    sensor_indices: np.ndarray  # [K] 在 504 壁面传感器池中的下标
    L: float
    u_inf: np.ndarray  # [N] 每帧对应的来流速度 (m/s)
    re_id: np.ndarray  # [N] 所属工况编号


@dataclass
class DecoderInferenceInputs:
    """解码器推理所需的原始物理参数空间与 ROI 物理坐标。"""

    xy_phys: np.ndarray  # [M, 2] ROI 物理空间坐标 (m)
    re_phys: np.ndarray  # [N] 物理雷诺数
    st_phys: np.ndarray  # [N] 物理斯特劳哈尔数
    phase_phys: np.ndarray  # [N] 物理相位角 (rad)
    u_inf: np.ndarray  # [N] 来流速度 (m/s)
    re_id: np.ndarray  # [N] 所属工况编号


def load_encoder_inference_inputs(
    processed_dir: str | Path = "data/processed",
    re_list: Optional[Sequence[int]] = None,
    sensor_indices: Optional[Sequence[int]] = None,
    periodic_only: bool = True,
) -> EncoderInferenceInputs:
    """从预处理 .h5 文件读取编码器/串联推理所需的物理传感器读数与坐标。"""
    mesh = load_base_surface_mesh(processed_dir)
    cases = load_base_surface_cases(processed_dir, re_list=re_list)
    idx = (
        np.arange(len(mesh.wall_cell_ids), dtype=np.int64)
        if sensor_indices is None
        else np.asarray(sensor_indices, dtype=np.int64)
    )
    raw = select_surface_sensors(cases, mesh, idx, periodic_only=periodic_only)

    u_inf_list = []
    for re_id in sorted(cases.keys()):
        c = cases[re_id]
        mask = c.periodic_mask > 0 if periodic_only else np.ones(len(c.phase), dtype=bool)
        u_inf_list.append(np.full(int(np.sum(mask)), c.u_inf, dtype=np.float64))

    return EncoderInferenceInputs(
        sensor_uvp=raw["uvp"],
        sensor_xy=mesh.wall_xy[idx],
        sensor_indices=idx,
        L=mesh.L,
        u_inf=np.concatenate(u_inf_list, axis=0),
        re_id=raw["re_id"],
    )


def load_decoder_inference_inputs(
    processed_dir: str | Path = "data/processed",
    re_list: Optional[Sequence[int]] = None,
    periodic_only: bool = True,
) -> DecoderInferenceInputs:
    """从预处理 .h5 文件读取解码器所需的 ROI 物理坐标与物理参数空间 (Re, St, phase)。"""
    mesh = load_ppinn_mesh(processed_dir)
    cases = load_base_surface_cases(processed_dir, re_list=re_list)

    re_arr_list, st_arr_list, phase_list, u_inf_list, re_id_list = [], [], [], [], []
    for re_id in sorted(cases.keys()):
        c = cases[re_id]
        mask = c.periodic_mask > 0 if periodic_only else np.ones(len(c.phase), dtype=bool)
        n = int(np.sum(mask))
        re_arr_list.append(np.full(n, c.re, dtype=np.float64))
        st_arr_list.append(np.full(n, c.st, dtype=np.float64))
        phase_list.append(c.phase[mask])
        u_inf_list.append(np.full(n, c.u_inf, dtype=np.float64))
        re_id_list.append(np.full(n, re_id, dtype=np.int64))

    return DecoderInferenceInputs(
        xy_phys=mesh.roi_xy,
        re_phys=np.concatenate(re_arr_list, axis=0),
        st_phys=np.concatenate(st_arr_list, axis=0),
        phase_phys=np.concatenate(phase_list, axis=0),
        u_inf=np.concatenate(u_inf_list, axis=0),
        re_id=np.concatenate(re_id_list, axis=0),
    )
