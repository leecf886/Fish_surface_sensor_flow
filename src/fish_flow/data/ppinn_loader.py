"""pPINN 全场训练与验证数据加载模块（纯参数接口，不包含配置文件解析）。

工作流程：
1. 从 `mesh.h5` 读取核心 ROI 无量纲网格坐标与范围；
2. 按传入的 `train_re` / `val_re` 读取对应 `ReXXX.h5` 的 ROI 全场流场并无量纲化为 `[u*, v*, p*]`；
3. 仅在训练集上拟合 `Re` 与 `St` 归一化器（复用 `encoder_loader.Normalizer`）；
4. 提供高效矢量化训练批采样器与按工况/帧分块的验证迭代接口，直接对接 `PPINNModel` 与 `PPINNLoss`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from fish_flow.data.encoder_loader import Normalizer
from fish_flow.data.surface_sensors import _discover_re_files


@dataclass
class PPINNMesh:
    """核心 ROI 网格几何信息。"""

    L: float  # 特征体长 (m)
    roi_xy: np.ndarray  # [Nroi, 2] 局部物理坐标 (m)
    roi_xy_nd: np.ndarray  # [Nroi, 2] 无量纲坐标 (x/L, y/L)
    roi_bounds_nd: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)


@dataclass
class PPINNCase:
    """单个雷诺数工况的 ROI 全场无量纲流场数据。"""

    re: float
    st: float
    u_inf: float
    L: float
    re_norm: float
    st_norm: float
    phase: np.ndarray  # [Nt] 瞬时相位 (rad, [0, 2π))
    phase_sin: np.ndarray  # [Nt] sin(phase)
    phase_cos: np.ndarray  # [Nt] cos(phase)
    uvp_nd: np.ndarray  # [Nt, Nroi, 3] 无量纲流场 [u*, v*, p*]

    @property
    def nt(self) -> int:
        return int(self.phase.shape[0])

    @property
    def nroi(self) -> int:
        return int(self.uvp_nd.shape[1])

    @property
    def n_points(self) -> int:
        return self.nt * self.nroi


class PPINNSampler:
    """pPINN 分组随机监督采样器：按工况帧数与网格数等概率抽取 `(t, i)` 配对并拼装 6D 特征。"""

    def __init__(
        self,
        cases: Sequence[PPINNCase],
        roi_xy_nd: np.ndarray,
        seed: int = 42,
    ) -> None:
        self.cases = list(cases)
        self.roi_xy_nd = np.asarray(roi_xy_nd, dtype=np.float32)
        self.rng = np.random.default_rng(seed)

        counts = np.asarray([c.n_points for c in self.cases], dtype=np.float64)
        self.case_prob = counts / counts.sum()
        self.total_points = int(counts.sum())

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样一批监督点，返回 `(feat [B, 6], uvp_true [B, 3])`。"""
        case_idx = self.rng.choice(len(self.cases), size=batch_size, p=self.case_prob)

        feat_np = np.empty((batch_size, 6), dtype=np.float32)
        uvp_np = np.empty((batch_size, 3), dtype=np.float32)

        cursor = 0
        for cid in np.unique(case_idx):
            c = self.cases[int(cid)]
            n = int(np.sum(case_idx == cid))
            t_idx = self.rng.integers(0, c.nt, size=n)
            i_idx = self.rng.integers(0, c.nroi, size=n)

            sl = slice(cursor, cursor + n)
            feat_np[sl, 0:2] = self.roi_xy_nd[i_idx]
            feat_np[sl, 2] = c.re_norm
            feat_np[sl, 3] = c.st_norm
            feat_np[sl, 4] = c.phase_sin[t_idx]
            feat_np[sl, 5] = c.phase_cos[t_idx]
            uvp_np[sl, :] = c.uvp_nd[t_idx, i_idx, :]
            cursor += n

        perm = self.rng.permutation(batch_size)
        feat_t = torch.as_tensor(feat_np[perm], dtype=dtype, device=device)
        uvp_t = torch.as_tensor(uvp_np[perm], dtype=dtype, device=device)
        return feat_t, uvp_t


@dataclass
class LoadedPPINNData:
    """加载完成后的 pPINN 数据对象，供训练 pipeline 采样、验证及保存归一化参数。"""

    mesh: PPINNMesh
    train_cases: Dict[int, PPINNCase]
    val_cases: Dict[int, PPINNCase]
    train_sampler: PPINNSampler
    re_normalizer: Normalizer
    st_normalizer: Normalizer
    train_re: List[int] = field(default_factory=list)
    val_re: List[int] = field(default_factory=list)

    @property
    def re_st_pairs(self) -> np.ndarray:
        """返回训练集真实的 `(Re, St)` 配对矩阵 `[N_cases, 2]`。"""
        return np.asarray(
            [[self.train_cases[r].re, self.train_cases[r].st] for r in sorted(self.train_cases)],
            dtype=np.float64,
        )

    @property
    def re_range(self) -> Tuple[float, float]:
        """返回训练集物理雷诺数上下界 `(re_min, re_max)`。"""
        vals = [c.re for c in self.train_cases.values()]
        return float(min(vals)), float(max(vals))

    @property
    def st_range(self) -> Tuple[float, float]:
        """返回训练集物理斯特劳哈尔数上下界 `(st_min, st_max)`。"""
        vals = [c.st for c in self.train_cases.values()]
        return float(min(vals)), float(max(vals))

    def sample_train_batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """抽取一个训练监督 batch：`(feat [B, 6], uvp_true [B, 3])`。"""
        return self.train_sampler.sample(batch_size, device=device, dtype=dtype)

    def iter_val_frames(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Iterator[Tuple[int, int, torch.Tensor, torch.Tensor]]:
        """逐工况逐帧产出验证集整场 ROI 张量：`(re_id, frame_idx, feat [Nroi, 6], uvp_true [Nroi, 3])`。"""
        xy_nd = self.mesh.roi_xy_nd.astype(np.float32, copy=False)
        nroi = xy_nd.shape[0]
        for re_id in sorted(self.val_cases):
            c = self.val_cases[re_id]
            feat_base = np.empty((nroi, 6), dtype=np.float32)
            feat_base[:, 0:2] = xy_nd
            feat_base[:, 2] = c.re_norm
            feat_base[:, 3] = c.st_norm
            for t in range(c.nt):
                feat_base[:, 4] = c.phase_sin[t]
                feat_base[:, 5] = c.phase_cos[t]
                yield (
                    re_id,
                    t,
                    torch.as_tensor(feat_base.copy(), dtype=dtype, device=device),
                    torch.as_tensor(c.uvp_nd[t], dtype=dtype, device=device),
                )

    def normalizers_to_dict(self) -> Dict[str, Any]:
        """导出训练集拟合的参数归一化器与域信息，方便随模型 checkpoint 持久化。"""
        return {
            "train_re": self.train_re,
            "val_re": self.val_re,
            "L": self.mesh.L,
            "roi_bounds_nd": list(self.mesh.roi_bounds_nd),
            "re_normalizer": self.re_normalizer.to_dict(),
            "st_normalizer": self.st_normalizer.to_dict(),
        }


def load_ppinn_mesh(processed_dir: str | Path = "data/processed") -> PPINNMesh:
    """从 `mesh.h5` 读取核心 ROI 网格坐标、特征体长与无量纲范围。"""
    mesh_path = Path(processed_dir) / "mesh.h5"
    with h5py.File(mesh_path, "r") as h5:
        L = float(h5["/geometry/L"][0])
        roi_xy = np.asarray(h5["/mesh/roi_xy"][()], dtype=np.float64)
        roi_xy_nd = np.asarray(h5["/mesh/roi_xy_nd"][()], dtype=np.float64)
        bounds = tuple(float(v) for v in h5["/config/roi_bounds_nd"][()])
    return PPINNMesh(L=L, roi_xy=roi_xy, roi_xy_nd=roi_xy_nd, roi_bounds_nd=bounds)  # type: ignore[arg-type]


def _load_raw_cases(
    re_files: Dict[int, Path],
    re_list: Sequence[int],
    periodic_only: bool,
) -> Dict[int, Dict[str, Any]]:
    """读取指定雷诺数列表的 ROI 全场并按 `uinf2` 协议无量纲化。"""
    raw: Dict[int, Dict[str, Any]] = {}
    for re_id in re_list:
        with h5py.File(re_files[int(re_id)], "r") as h5:
            re_val = float(h5["/meta/Re"][0])
            st_val = float(h5["/meta/St"][0])
            u_inf = float(h5["/meta/U_inf"][0])
            L = float(h5["/meta/L"][0])
            periodic_mask = np.asarray(h5["/snapshots/periodic_mask"][()], dtype=np.int32)
            mask = periodic_mask > 0 if periodic_only else np.ones(len(periodic_mask), dtype=bool)

            phase = np.asarray(h5["/snapshots/phase"][()], dtype=np.float64)[mask]
            phase_sin = np.asarray(h5["/snapshots/phase_sin"][()], dtype=np.float64)[mask]
            phase_cos = np.asarray(h5["/snapshots/phase_cos"][()], dtype=np.float64)[mask]
            uvp_phys = np.asarray(h5["/field/uvp"][()], dtype=np.float32)[mask]

        # 无量纲化：u* = Ux / U_inf, v* = Uy / U_inf, p* = p_kin / U_inf^2
        uvp_nd = uvp_phys.copy()
        uvp_nd[:, :, 0:2] /= np.float32(u_inf)
        uvp_nd[:, :, 2] /= np.float32(u_inf * u_inf)

        raw[int(re_id)] = {
            "re": re_val,
            "st": st_val,
            "u_inf": u_inf,
            "L": L,
            "phase": phase,
            "phase_sin": phase_sin,
            "phase_cos": phase_cos,
            "uvp_nd": uvp_nd,
        }
    return raw


def _build_cases(
    raw: Dict[int, Dict[str, Any]],
    re_normalizer: Normalizer,
    st_normalizer: Normalizer,
) -> Dict[int, PPINNCase]:
    """注入归一化后的 `(Re_norm, St_norm)` 并构建 `PPINNCase` 字典。"""
    cases: Dict[int, PPINNCase] = {}
    for re_id, d in raw.items():
        re_norm = float(re_normalizer.transform(np.array([d["re"]], dtype=np.float64))[0])
        st_norm = float(st_normalizer.transform(np.array([d["st"]], dtype=np.float64))[0])
        cases[re_id] = PPINNCase(
            re=d["re"],
            st=d["st"],
            u_inf=d["u_inf"],
            L=d["L"],
            re_norm=re_norm,
            st_norm=st_norm,
            phase=d["phase"],
            phase_sin=d["phase_sin"],
            phase_cos=d["phase_cos"],
            uvp_nd=d["uvp_nd"],
        )
    return cases


def load_ppinn_training_data(
    *,
    processed_dir: str | Path = "data/processed",
    train_re: Sequence[int],
    val_re: Optional[Sequence[int]] = None,
    periodic_only: bool = True,
    param_norm_method: str = "standard",
    seed: int = 42,
) -> LoadedPPINNData:
    """pPINN 全场数据加载函数：接收显式参数，返回训练批采样器、验证工况与归一化器。"""
    proc_dir = Path(processed_dir)
    train_re_list = [int(r) for r in train_re]
    val_re_list = [] if val_re is None else [int(r) for r in val_re]

    mesh = load_ppinn_mesh(proc_dir)
    re_files = _discover_re_files(proc_dir)

    # 1. 读取训练集与验证集原始 ROI 流场
    train_raw = _load_raw_cases(re_files, train_re_list, periodic_only=periodic_only)
    val_raw = _load_raw_cases(re_files, val_re_list, periodic_only=periodic_only) if val_re_list else {}

    # 2. 仅在训练集上拟合 Re 与 St 归一化器
    train_re_arr = np.asarray([d["re"] for d in train_raw.values()], dtype=np.float64)
    train_st_arr = np.asarray([d["st"] for d in train_raw.values()], dtype=np.float64)
    re_norm = Normalizer(method=param_norm_method).fit(train_re_arr)
    st_norm = Normalizer(method=param_norm_method).fit(train_st_arr)

    train_cases = _build_cases(train_raw, re_norm, st_norm)
    val_cases = _build_cases(val_raw, re_norm, st_norm)
    train_sampler = PPINNSampler(list(train_cases.values()), mesh.roi_xy_nd, seed=seed)

    val_frames = sum(c.nt for c in val_cases.values())
    train_frames = sum(c.nt for c in train_cases.values())
    print(
        f"pPINN 全场数据已加载 | train={train_frames}帧 ({len(train_re_list)}工况, {train_sampler.total_points}点) | "
        f"val={val_frames}帧 ({len(val_re_list)}工况) | Nroi={len(mesh.roi_xy_nd)}"
    )

    return LoadedPPINNData(
        mesh=mesh,
        train_cases=train_cases,
        val_cases=val_cases,
        train_sampler=train_sampler,
        re_normalizer=re_norm,
        st_normalizer=st_norm,
        train_re=train_re_list,
        val_re=val_re_list,
    )
