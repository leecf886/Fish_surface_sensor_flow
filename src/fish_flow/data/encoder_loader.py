"""编码器公用训练数据加载与归一化模块。

工作流程：
1. 调用基础表面传感器模块，加载 504 个壁面单元中心坐标与各雷诺数原始传感器读数；
2. 读取模型配置文件（可与 checkpoint 同目录存放），按划分选取 train/val 雷诺数与传感器布局；
3. 仅在训练集上拟合统计量并完成归一化（MLP 展平逐列标准化，DeepSets 保留集合与 3 通道均值方差），
   构造 4 维回归目标 [Re_norm, St_norm, sin(phase), cos(phase)]；
4. 打印加载完成摘要并输出供后续 model / trainer 使用的 DataLoader 与归一化器。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from fish_flow.data.surface_sensors import (
    BaseSurfaceMesh,
    load_base_surface_cases,
    load_base_surface_mesh,
    select_surface_sensors,
)

# 16 个鱼体绕流工况的默认折划分
DEFAULT_FOLDS: Dict[str, Dict[str, List[int]]] = {
    "default": {
        "train_re": [13500, 13750, 14000, 14500, 15000, 15250, 15500, 15750, 16000, 16500, 17000, 17500],
        "val_re": [14750, 16750],
        "test_re": [14250, 16250],
    },
    "loo_14750": {
        "train_re": [13500, 13750, 14000, 14500, 15000, 15250, 15500, 15750, 16000, 16500, 16750, 17000, 17500],
        "val_re": [14750],
        "test_re": [14250, 16250],
    },
    "loo_16750": {
        "train_re": [13500, 13750, 14000, 14500, 14750, 15000, 15250, 15500, 15750, 16000, 16500, 17000, 17500],
        "val_re": [16750],
        "test_re": [14250, 16250],
    },
}


@dataclass
class Normalizer:
    """Z-score / MinMax / 无变换归一化器（仅由训练集拟合）。"""

    method: str = "standard"
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None

    def fit(self, arr: np.ndarray) -> "Normalizer":
        x = np.asarray(arr, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if self.method == "standard":
            self.mean = np.mean(x, axis=0)
            std = np.std(x, axis=0)
            self.std = np.where(np.abs(std) < 1e-12, 1.0, std)
        elif self.method == "minmax":
            mn = np.min(x, axis=0)
            mx = np.max(x, axis=0)
            span = np.where(np.abs(mx - mn) < 1e-12, 1.0, mx - mn)
            self.min_val = mn
            self.max_val = mn + span
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float64)
        is_1d = x.ndim == 1
        if is_1d:
            x = x[:, None]
        if self.method == "standard":
            out = (x - self.mean) / self.std
        elif self.method == "minmax":
            out = (x - self.min_val) / (self.max_val - self.min_val)
        else:
            out = x
        return out[:, 0] if is_1d else out

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float64)
        is_1d = x.ndim == 1
        if is_1d:
            x = x[:, None]
        if self.method == "standard":
            out = x * self.std + self.mean
        elif self.method == "minmax":
            out = x * (self.max_val - self.min_val) + self.min_val
        else:
            out = x
        return out[:, 0] if is_1d else out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "min": None if self.min_val is None else self.min_val.tolist(),
            "max": None if self.max_val is None else self.max_val.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Normalizer":
        def _arr(v: Any) -> Optional[np.ndarray]:
            return None if v is None else np.asarray(v, dtype=np.float64)

        return cls(
            method=str(data.get("method", "standard")),
            mean=_arr(data.get("mean")),
            std=_arr(data.get("std")),
            min_val=_arr(data.get("min")),
            max_val=_arr(data.get("max")),
        )


class EncoderDataset(Dataset):
    """编码器统一数据集：支持 MLP 展平输入 [N, 3*K] 与 DeepSets 集合输入 [N, K, 3] + [K, 2]。"""

    def __init__(
        self,
        encoder_type: str,
        x_or_uvp: np.ndarray,
        xy_nd: np.ndarray,
        re_norm: np.ndarray,
        st_norm: np.ndarray,
        phase_sin: np.ndarray,
        phase_cos: np.ndarray,
        re_raw: np.ndarray,
        st_raw: np.ndarray,
        phase_true: np.ndarray,
        re_id: np.ndarray,
    ) -> None:
        self.encoder_type = encoder_type
        self.x = torch.as_tensor(x_or_uvp, dtype=torch.float32)
        self.xy_nd = torch.as_tensor(xy_nd, dtype=torch.float32)
        # 目标固定为 4 维: [Re_norm, St_norm, sin(phi), cos(phi)]
        self.y = torch.stack(
            [
                torch.as_tensor(re_norm, dtype=torch.float32),
                torch.as_tensor(st_norm, dtype=torch.float32),
                torch.as_tensor(phase_sin, dtype=torch.float32),
                torch.as_tensor(phase_cos, dtype=torch.float32),
            ],
            dim=1,
        )
        self.re_raw = torch.as_tensor(re_raw, dtype=torch.float32)
        self.st_raw = torch.as_tensor(st_raw, dtype=torch.float32)
        self.phase_true = torch.as_tensor(phase_true, dtype=torch.float32)
        self.re_id = torch.as_tensor(re_id, dtype=torch.int64)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.encoder_type == "mlp":
            return self.x[idx], self.y[idx], self.re_raw[idx], self.st_raw[idx], self.phase_true[idx], self.re_id[idx]
        return self.x[idx], self.xy_nd, self.y[idx], self.re_raw[idx], self.st_raw[idx], self.phase_true[idx], self.re_id[idx]


class DeepSetsSubsetCollator:
    """DeepSets 动态子集采样与可选 dropout / 噪声增强整理器。"""

    def __init__(
        self,
        pool_size: int,
        sensor_counts: Sequence[int],
        chan_std: np.ndarray,
        seed: int = 42,
        token_dropout_prob: float = 0.0,
        sensor_noise_rel_std: float = 0.0,
    ) -> None:
        self.pool_size = int(pool_size)
        self.sensor_counts = tuple(sorted({int(c) for c in sensor_counts}))
        self.chan_std = np.asarray(chan_std, dtype=np.float32).reshape(1, 1, 3)
        self.token_dropout_prob = float(token_dropout_prob)
        self.sensor_noise_rel_std = float(sensor_noise_rel_std)
        self._rng = np.random.default_rng(seed)
        self._cycle: List[int] = []

    def __call__(self, batch: Sequence[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        uvp, xy_nd, y, re_raw, st_raw, phase_true, re_id = default_collate(batch)
        if not self._cycle:
            self._cycle = self._rng.permutation(self.sensor_counts).tolist()
        k = int(self._cycle.pop())
        subset = np.sort(self._rng.choice(self.pool_size, size=k, replace=False))
        gather = torch.from_numpy(subset.astype(np.int64))

        uvp = uvp.index_select(1, gather).clone()
        xy_nd = xy_nd.index_select(1, gather)
        bsz = int(uvp.shape[0])

        mask_np = np.ones((bsz, k), dtype=bool)
        if self.token_dropout_prob > 0.0:
            mask_np = self._rng.random((bsz, k)) >= self.token_dropout_prob
            # 保证每个样本至少保留 1 个有效传感器
            for row in np.flatnonzero(~mask_np.any(axis=1)):
                mask_np[int(row), int(self._rng.integers(0, k))] = True

        if self.sensor_noise_rel_std > 0.0:
            noise = self._rng.normal(0.0, 1.0, size=(bsz, k, 3)).astype(np.float32)
            noise *= self.sensor_noise_rel_std * self.chan_std * mask_np[:, :, None]
            uvp = uvp + torch.from_numpy(noise)

        mask = torch.from_numpy(mask_np)
        return uvp, xy_nd, mask, y, re_raw, st_raw, phase_true, re_id


@dataclass
class LoadedEncoderData:
    """加载完成后的统一数据打包对象，直接对接 model 与 trainer。"""

    config: Dict[str, Any]
    encoder_type: str
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    train_dataset: EncoderDataset
    val_dataset: Optional[EncoderDataset]
    sensor_indices: np.ndarray
    xy_nd: np.ndarray
    re_normalizer: Normalizer
    st_normalizer: Normalizer
    x_normalizer: Optional[Normalizer] = None  # 仅 MLP 使用
    chan_mean: Optional[np.ndarray] = None  # 仅 DeepSets 使用
    chan_std: Optional[np.ndarray] = None  # 仅 DeepSets 使用
    train_re: List[int] = field(default_factory=list)
    val_re: List[int] = field(default_factory=list)

    def save_meta_near_ckpt(self, ckpt_dir: str | Path, filename: str = "data_config.json") -> Path:
        """将模型配置、实际传感器布局与拟合好的归一化参数保存到 ckpt 同目录。"""
        out_path = Path(ckpt_dir) / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.config)
        payload.update(
            {
                "encoder_type": self.encoder_type,
                "train_re": self.train_re,
                "val_re": self.val_re,
                "sensor_indices": self.sensor_indices.tolist(),
                "xy_nd": self.xy_nd.tolist(),
                "re_normalizer": self.re_normalizer.to_dict(),
                "st_normalizer": self.st_normalizer.to_dict(),
                "x_normalizer": None if self.x_normalizer is None else self.x_normalizer.to_dict(),
                "chan_mean": None if self.chan_mean is None else self.chan_mean.tolist(),
                "chan_std": None if self.chan_std is None else self.chan_std.tolist(),
            }
        )
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out_path


def _resolve_sensor_indices(mesh: BaseSurfaceMesh, cfg: Mapping[str, Any]) -> np.ndarray:
    """根据配置解析要从 504 个基础表面传感器中选取的下标。"""
    mode = str(cfg.get("layout_mode", "uniform")).lower()
    if mode in ("all", "random_subset"):
        return np.arange(len(mesh.wall_cell_ids), dtype=np.int64)
    if mode == "cand60":
        return mesh.cand60_wall_indices.astype(np.int64)
    if mode == "indices":
        return np.asarray(cfg["sensor_indices"], dtype=np.int64)

    # uniform 模式：按闭合轮廓归一化弧长等距选 K 个点
    k = int(cfg.get("n_sensors", 12))
    targets = np.linspace(0.0, 1.0, k, endpoint=False)
    used = np.zeros(len(mesh.wall_arc), dtype=bool)
    chosen: List[int] = []
    for t in targets:
        dist = np.abs(mesh.wall_arc - t)
        dist = np.minimum(dist, 1.0 - dist)
        dist[used] = np.inf
        best = int(np.argmin(dist))
        used[best] = True
        chosen.append(best)
    return np.asarray(chosen, dtype=np.int64)


def _resolve_splits(cfg: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
    """从配置直接读取或按 fold 名称解析 train_re 与 val_re。"""
    if "train_re" in cfg and "val_re" in cfg:
        return [int(r) for r in cfg["train_re"]], [int(r) for r in cfg["val_re"]]
    fold_name = str(cfg.get("fold", "default"))
    fold = DEFAULT_FOLDS[fold_name]
    return list(fold["train_re"]), list(fold["val_re"])


def load_encoder_training_data(config: Union[str, Path, Mapping[str, Any]]) -> LoadedEncoderData:
    """公用训练数据加载入口：读配置 -> 调基础表面传感器 -> 训练集拟合归一化 -> 构造 DataLoader。"""
    if isinstance(config, (str, Path)):
        cfg: Dict[str, Any] = json.loads(Path(config).read_text(encoding="utf-8"))
    else:
        cfg = dict(config)

    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    encoder_type = str(cfg.get("encoder_type", "mlp")).lower()
    batch_size = int(cfg.get("batch_size", 256))
    seed = int(cfg.get("seed", 42))
    periodic_only = bool(cfg.get("periodic_only", True))
    train_re, val_re = _resolve_splits(cfg)

    # 1. 调用基础表面传感器模块，加载 504 点几何与训练/验证工况原始读数
    mesh = load_base_surface_mesh(processed_dir)
    sensor_indices = _resolve_sensor_indices(mesh, cfg)
    train_cases = load_base_surface_cases(processed_dir, re_list=train_re)
    train_raw = select_surface_sensors(train_cases, mesh, sensor_indices, periodic_only=periodic_only)

    val_raw: Optional[Dict[str, np.ndarray]] = None
    if val_re:
        val_cases = load_base_surface_cases(processed_dir, re_list=val_re)
        val_raw = select_surface_sensors(val_cases, mesh, sensor_indices, periodic_only=periodic_only)

    # 2. 仅在训练集上拟合目标归一化器 (Re, St)
    target_method = str(cfg.get("target_norm_method", "standard"))
    re_norm = Normalizer(method=target_method).fit(train_raw["re"])
    st_norm = Normalizer(method=target_method).fit(train_raw["st"])

    train_re_n = re_norm.transform(train_raw["re"])
    train_st_n = st_norm.transform(train_raw["st"])

    # 3. 按编码器类型分别处理输入特征归一化
    x_norm: Optional[Normalizer] = None
    chan_mean: Optional[np.ndarray] = None
    chan_std: Optional[np.ndarray] = None
    train_collate = None

    if encoder_type == "mlp":
        # 原代码约定：MLP 将传感器展平为 [N, 3*K]，直接做逐列 Z-score 标准化
        x_train_flat = train_raw["uvp"].reshape(len(train_raw["uvp"]), -1)
        x_norm = Normalizer(method=str(cfg.get("x_norm_method", "standard"))).fit(x_train_flat)
        train_x = x_norm.transform(x_train_flat)
    else:
        # 原代码约定：DeepSets 保留 [N, K, 3] 原始读数，仅计算 3 通道均值方差供模型 forward 内标准化
        flat_3 = train_raw["uvp"].reshape(-1, 3).astype(np.float64)
        chan_mean = flat_3.mean(axis=0)
        std_3 = flat_3.std(axis=0)
        chan_std = np.where(np.abs(std_3) < 1e-12, 1.0, std_3)
        train_x = train_raw["uvp"]

        if str(cfg.get("layout_mode", "")).lower() == "random_subset":
            train_collate = DeepSetsSubsetCollator(
                pool_size=len(sensor_indices),
                sensor_counts=cfg.get("training_sensor_counts", (6, 12, 24, 48)),
                chan_std=chan_std,
                seed=seed,
                token_dropout_prob=float(cfg.get("token_dropout_prob", 0.0)),
                sensor_noise_rel_std=float(cfg.get("sensor_noise_rel_std", 0.0)),
            )

    train_ds = EncoderDataset(
        encoder_type=encoder_type,
        x_or_uvp=train_x,
        xy_nd=train_raw["xy_nd"],
        re_norm=train_re_n,
        st_norm=train_st_n,
        phase_sin=train_raw["phase_sin"],
        phase_cos=train_raw["phase_cos"],
        re_raw=train_raw["re"],
        st_raw=train_raw["st"],
        phase_true=train_raw["phase"],
        re_id=train_raw["re_id"],
    )
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,
        collate_fn=train_collate,
        num_workers=0,
    )

    val_ds: Optional[EncoderDataset] = None
    val_loader: Optional[DataLoader] = None
    if val_raw is not None:
        if encoder_type == "mlp":
            assert x_norm is not None
            val_x = x_norm.transform(val_raw["uvp"].reshape(len(val_raw["uvp"]), -1))
        else:
            val_x = val_raw["uvp"]

        val_ds = EncoderDataset(
            encoder_type=encoder_type,
            x_or_uvp=val_x,
            xy_nd=val_raw["xy_nd"],
            re_norm=re_norm.transform(val_raw["re"]),
            st_norm=st_norm.transform(val_raw["st"]),
            phase_sin=val_raw["phase_sin"],
            phase_cos=val_raw["phase_cos"],
            re_raw=val_raw["re"],
            st_raw=val_raw["st"],
            phase_true=val_raw["phase"],
            re_id=val_raw["re_id"],
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 4. 打印加载完成摘要，后续直接衔接 model 模块
    val_count = 0 if val_ds is None else len(val_ds)
    print(
        f"数据已加载完成 | encoder={encoder_type} | "
        f"train={len(train_ds)}帧 ({len(train_re)}工况) | val={val_count}帧 ({len(val_re)}工况) | "
        f"sensors={len(sensor_indices)}"
    )

    return LoadedEncoderData(
        config=cfg,
        encoder_type=encoder_type,
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        sensor_indices=sensor_indices,
        xy_nd=train_raw["xy_nd"],
        re_normalizer=re_norm,
        st_normalizer=st_norm,
        x_normalizer=x_norm,
        chan_mean=chan_mean,
        chan_std=chan_std,
        train_re=train_re,
        val_re=val_re,
    )
