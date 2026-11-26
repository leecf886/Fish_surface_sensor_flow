"""pPINN 5 维 PDE 配点采样与成鱼外形固体剔除模块（纯参数接口，不包含配置文件解析）。

提供以下纯函数接口，供训练与评测 pipeline 直接调用并衔接 `losses.compute_pde_residuals`：
1. `fish_half_width_nd` / `points_inside_fish`：Verma et al. (2020) 成鱼静态分段轮廓与固体域掩码；
2. `sample_collocation_points`：在 5 维超立方体 `[0, 1)^5` 上采样 `(x*, y*, Re, St, φ)`，
   支持 `uniform_random`、`latin_hypercube`、`sobol` 三种采样方法，
   支持 `continuous_random`（5D 独立均匀）与 `discrete_train_re_random`（离散工况配对）两种参数模式，
   并通过拒绝采样剔除鱼体内部配点；
3. `collocation_to_tensors` / `collocation_fingerprint`：转换为无梯度 PyTorch 张量与 SHA-256 指纹。
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from fish_flow.data.encoder_loader import Normalizer

TWO_PI: float = 2.0 * math.pi

# Verma et al. (2020) 成鱼轮廓无量纲几何常数 (s = x / L ∈ [0, 1])
_S_HEAD: float = 0.04
_S_TAIL: float = 0.95
_H_HEAD: float = 0.04
_H_TAIL: float = 0.01
_TAIL_B: float = (_H_TAIL - _H_HEAD) / (_S_TAIL - _S_HEAD)
_TAIL_C: float = (-_TAIL_B * (1.0 - _S_TAIL) - _H_TAIL) / ((1.0 - _S_TAIL) ** 2)


def fish_half_width_nd(s: np.ndarray) -> np.ndarray:
    """计算 Verma et al. (2020) 成鱼在无量纲轴向坐标 `s = x/L ∈ [0, 1]` 处的无量纲半宽 `h(s)/L`。"""
    s_arr = np.asarray(s, dtype=np.float64)
    h = np.zeros_like(s_arr, dtype=np.float64)

    head_mask = (s_arr >= 0.0) & (s_arr < _S_HEAD)
    body_mask = (s_arr >= _S_HEAD) & (s_arr < _S_TAIL)
    tail_mask = (s_arr >= _S_TAIL) & (s_arr <= 1.0)

    sh = s_arr[head_mask]
    h[head_mask] = np.sqrt(np.maximum(0.0, 2.0 * sh * _H_HEAD - sh * sh))

    sb = s_arr[body_mask]
    h[body_mask] = _H_HEAD - (_H_HEAD - _H_TAIL) * ((sb - _S_HEAD) / (_S_TAIL - _S_HEAD)) ** 2

    st = s_arr[tail_mask] - _S_TAIL
    h[tail_mask] = _H_TAIL + _TAIL_B * st + _TAIL_C * st * st
    return h


def points_inside_fish(x_nd: np.ndarray, y_nd: np.ndarray) -> np.ndarray:
    """判断无量纲坐标 `(x/L, y/L)` 是否落在成鱼固体轮廓内部。"""
    x_arr = np.asarray(x_nd, dtype=np.float64)
    y_arr = np.asarray(y_nd, dtype=np.float64)
    in_x = (x_arr >= 0.0) & (x_arr <= 1.0)
    inside = np.zeros_like(in_x, dtype=bool)
    if np.any(in_x):
        half_w = fish_half_width_nd(x_arr[in_x])
        inside[in_x] = np.abs(y_arr[in_x]) <= half_w
    return inside


def _sample_unit_hypercube(
    n: int,
    dim: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """在 `[0, 1)^dim` 内生成 `n` 个样本点（支持 uniform_random / latin_hypercube / sobol）。"""
    m = method.strip().lower()
    if m == "uniform_random":
        return rng.random((n, dim), dtype=np.float64)
    if m == "latin_hypercube":
        out = np.empty((n, dim), dtype=np.float64)
        for j in range(dim):
            perm = rng.permutation(n)
            out[:, j] = (perm + rng.random(n)) / n
        return out
    # sobol：使用 scramble=True 的低差异序列
    seed = int(rng.integers(0, 2**31 - 1))
    eng = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    return eng.draw(n).cpu().numpy().astype(np.float64)


def collocation_fingerprint(col: Mapping[str, np.ndarray]) -> str:
    """计算配点字典（按 `x_nd, y_nd, re_val, st_val, phi` 物理量）的 SHA-256 指纹。"""
    digest = hashlib.sha256()
    for key in ("x_nd", "y_nd", "re_val", "st_val", "phi"):
        arr = np.ascontiguousarray(np.asarray(col[key], dtype=np.float64))
        digest.update(key.encode("ascii"))
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def sample_collocation_points(
    n_points: int,
    roi_bounds_nd: Tuple[float, float, float, float],
    re_normalizer: Normalizer,
    st_normalizer: Normalizer,
    *,
    re_range: Optional[Tuple[float, float]] = None,
    st_range: Optional[Tuple[float, float]] = None,
    re_st_pairs: Optional[Sequence[Tuple[float, float]] | np.ndarray] = None,
    method: str = "latin_hypercube",
    re_st_mode: str = "continuous_random",
    exclude_solid: bool = True,
    seed: int | np.random.Generator = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """采样 5 维 PDE 配点 `(x*, y*, Re, St, φ)` 并映射得到对应的归一化参数特征。

    参数:
        n_points: 最终保留的流体域配点数量。
        roi_bounds_nd: 无量纲空间采样边界 `(xmin, xmax, ymin, ymax)`。
        re_normalizer: 训练集拟合的雷诺数归一化器。
        st_normalizer: 训练集拟合的斯特劳哈尔数归一化器。
        re_range: 连续采样时的 `(re_min, re_max)` 物理区间。
        st_range: 连续采样时的 `(st_min, st_max)` 物理区间。
        re_st_pairs: 离散采样时的训练集真实 `(Re, St)` 配对序列 `[N_cases, 2]`。
        method: 5D 超立方体采样方法，`'latin_hypercube'` / `'uniform_random'` / `'sobol'`。
        re_st_mode: 参数采样模式，`'continuous_random'`（5D 独立）或 `'discrete_train_re_random'`。
        exclude_solid: 是否通过拒绝采样剔除鱼体固体内部配点。
        seed: 随机种子或 `np.random.Generator` 实例。

    返回:
        `(col_dict, stats)`：
        - `col_dict` 包含 `x_nd, y_nd, re_val, st_val, phi, re_norm, st_norm`，形状均为 `[N, 1]`；
        - `stats` 包含采样计数、剔除数量与 SHA-256 指纹。
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(int(seed))
    xmin, xmax, ymin, ymax = roi_bounds_nd

    proposed = 0
    rejected_solid = 0

    # 1. 在 5 维 [0, 1)^5 上采样空间与参数坐标，按需进行鱼体固体拒绝采样
    if not exclude_solid:
        u = _sample_unit_hypercube(n_points, 5, method, rng)
        proposed = n_points
        x_nd = xmin + (xmax - xmin) * u[:, 0]
        y_nd = ymin + (ymax - ymin) * u[:, 1]
    else:
        x_parts: List[np.ndarray] = []
        y_parts: List[np.ndarray] = []
        u_parts: List[np.ndarray] = []
        need = int(n_points)
        while need > 0:
            chunk = max(need * 4, 64)
            u_chunk = _sample_unit_hypercube(chunk, 5, method, rng)
            proposed += chunk

            x_chunk = xmin + (xmax - xmin) * u_chunk[:, 0]
            y_chunk = ymin + (ymax - ymin) * u_chunk[:, 1]
            inside = points_inside_fish(x_chunk, y_chunk)
            rejected_solid += int(np.sum(inside))

            valid = ~inside
            if not np.any(valid):
                continue
            take = min(int(np.sum(valid)), need)
            valid_idx = np.flatnonzero(valid)[:take]
            x_parts.append(x_chunk[valid_idx])
            y_parts.append(y_chunk[valid_idx])
            u_parts.append(u_chunk[valid_idx])
            need -= take

        u = np.concatenate(u_parts, axis=0)
        x_nd = np.concatenate(x_parts, axis=0)
        y_nd = np.concatenate(y_parts, axis=0)

    # 2. 映射 Re 与 St（不使用任何经验公式）
    mode = re_st_mode.strip().lower()
    if mode == "continuous_random":
        re_min, re_max = re_range  # type: ignore[misc]
        st_min, st_max = st_range  # type: ignore[misc]
        re_val = re_min + (re_max - re_min) * u[:, 2]
        st_val = st_min + (st_max - st_min) * u[:, 3]
    else:
        pairs = np.asarray(re_st_pairs, dtype=np.float64)
        idx = np.minimum((u[:, 2] * len(pairs)).astype(np.int64), len(pairs) - 1)
        re_val = pairs[idx, 0]
        st_val = pairs[idx, 1]

    # 3. 映射周期相位 φ ∈ [0, 2π) 并计算归一化特征
    phi = np.mod(u[:, 4] * TWO_PI, TWO_PI)
    re_norm = re_normalizer.transform(re_val)
    st_norm = st_normalizer.transform(st_val)

    col: Dict[str, np.ndarray] = {
        "x_nd": x_nd.reshape(-1, 1).astype(np.float64),
        "y_nd": y_nd.reshape(-1, 1).astype(np.float64),
        "re_val": re_val.reshape(-1, 1).astype(np.float64),
        "st_val": st_val.reshape(-1, 1).astype(np.float64),
        "phi": phi.reshape(-1, 1).astype(np.float64),
        "re_norm": re_norm.reshape(-1, 1).astype(np.float64),
        "st_norm": st_norm.reshape(-1, 1).astype(np.float64),
    }
    stats: Dict[str, Any] = {
        "proposed": proposed,
        "rejected_solid": rejected_solid,
        "accepted": int(n_points),
        "exclude_solid": exclude_solid,
        "method": method.strip().lower(),
        "re_st_mode": mode,
        "fingerprint": collocation_fingerprint(col),
    }
    return col, stats


def collocation_to_tensors(
    col: Mapping[str, np.ndarray],
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """将配点字典转为无梯度 PyTorch 张量，直接解包传给 `compute_pde_residuals`。"""
    return {
        key: torch.as_tensor(arr, dtype=dtype, device=device)
        for key, arr in col.items()
    }


__all__ = [
    "TWO_PI",
    "collocation_fingerprint",
    "collocation_to_tensors",
    "fish_half_width_nd",
    "points_inside_fish",
    "sample_collocation_points",
]
