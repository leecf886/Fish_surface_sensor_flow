"""训练公共基础组件：随机种子、指标计算、早停、学习率调度、PDE 权重调度与落盘工具（纯参数接口）。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

TWO_PI: float = 2.0 * math.pi


def set_training_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 随机种子。"""
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> torch.device:
    """解析目标计算设备。"""
    dev = device.strip().lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算标量回归 MAE 与 RMSE。"""
    t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    p = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = p - t
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err * err))),
    }


def compute_circular_metrics(phase_true: np.ndarray, phase_pred: np.ndarray) -> Dict[str, float]:
    """计算周期相位角（弧度）在 [-pi, pi) 上的圆周 MAE 与 RMSE。"""
    t = np.asarray(phase_true, dtype=np.float64).reshape(-1)
    p = np.asarray(phase_pred, dtype=np.float64).reshape(-1)
    diff = (p - t + math.pi) % TWO_PI - math.pi
    return {
        "circular_mae": float(np.mean(np.abs(diff))),
        "circular_rmse": float(np.sqrt(np.mean(diff * diff))),
    }


class PressureGaugeAccum:
    """基于 Welford / Chan 并行合并的逐帧压强 Gauge 零点对齐误差累计器。"""

    __slots__ = (
        "raw_err_sse",
        "raw_ref_energy",
        "gauge_err_sse",
        "true_fluct_sse",
        "offset_abs_sum",
        "n_snapshots",
        "n_points",
    )

    def __init__(self) -> None:
        self.raw_err_sse: float = 0.0
        self.raw_ref_energy: float = 0.0
        self.gauge_err_sse: float = 0.0
        self.true_fluct_sse: float = 0.0
        self.offset_abs_sum: float = 0.0
        self.n_snapshots: int = 0
        self.n_points: int = 0

    @staticmethod
    def chan_merge(
        n_a: int,
        mean_a: float,
        m2_a: float,
        n_b: int,
        mean_b: float,
        m2_b: float,
    ) -> Tuple[int, float, float]:
        """Chan 并行均值与二阶中心矩合并公式。"""
        if n_a == 0:
            return n_b, mean_b, m2_b
        if n_b == 0:
            return n_a, mean_a, m2_a
        n = n_a + n_b
        delta = mean_b - mean_a
        mean = mean_a + delta * n_b / n
        m2 = m2_a + m2_b + delta * delta * n_a * n_b / n
        return n, mean, m2

    def add_snapshot_welford(
        self,
        mean_e: float,
        m2_e: float,
        mean_p: float,
        m2_p: float,
        n: int,
    ) -> None:
        """累计单帧统计量（e = pred_p - true_p）。"""
        self.raw_err_sse += m2_e + n * (mean_e * mean_e)
        self.raw_ref_energy += m2_p + n * (mean_p * mean_p)
        self.gauge_err_sse += m2_e
        self.true_fluct_sse += m2_p
        self.offset_abs_sum += abs(mean_e)
        self.n_snapshots += 1
        self.n_points += n

    def metrics(self, eps: float = 1e-12) -> Dict[str, float]:
        """汇总全部快照上的原始与 Gauge 对齐相对 L2 误差。"""
        return {
            "p_rel_l2_raw": math.sqrt(self.raw_err_sse / (self.raw_ref_energy + eps)),
            "p_rel_l2_gauge_sameden": math.sqrt(self.gauge_err_sse / (self.raw_ref_energy + eps)),
            "p_rel_l2_gauge_fluctden": math.sqrt(self.gauge_err_sse / (self.true_fluct_sse + eps)),
            "p_gauge_offset_mean_abs": self.offset_abs_sum / max(self.n_snapshots, 1),
        }


class EarlyStopper:
    """早停状态机（默认关闭，按连续未改善轮数控制训练终止）。"""

    def __init__(self, enabled: bool = False, patience: int = 20, min_delta: float = 0.0) -> None:
        self.enabled = bool(enabled)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score: float = float("inf")
        self.bad_epochs: int = 0

    def update(self, score: float) -> Tuple[bool, bool]:
        """更新本轮监控指标，返回 `(是否刷新最优, 是否触发早停)`。"""
        if score < (self.best_score - self.min_delta):
            self.best_score = float(score)
            self.bad_epochs = 0
            return True, False
        self.bad_epochs += 1
        should_stop = self.enabled and self.bad_epochs >= self.patience
        return False, should_stop


def build_lr_scheduler(
    optimizer: Optimizer,
    strategy: str = "constant",
    num_epochs: int = 200,
    params: Optional[Mapping[str, Any]] = None,
) -> Optional[Any]:
    """构建学习率调度器；`'constant'` 返回 None 表示保持学习率不变。"""
    mode = strategy.strip().lower()
    cfg = dict(params or {})
    if mode in ("", "constant", "none"):
        return None
    if mode == "step":
        return StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 50)),
            gamma=float(cfg.get("gamma", 0.5)),
        )
    if mode == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=[int(m) for m in cfg.get("milestones", [100, 150])],
            gamma=float(cfg.get("gamma", 0.5)),
        )
    if mode == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.get("t_max", num_epochs)),
            eta_min=float(cfg.get("eta_min", 1e-6)),
        )
    if mode == "exponential":
        return ExponentialLR(optimizer, gamma=float(cfg.get("gamma", 0.99)))
    if mode == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 10)),
            threshold=float(cfg.get("threshold", 1e-4)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
        )
    raise ValueError(
        f"不支持的 lr_scheduler.strategy: {strategy!r}，"
        "可选: ['constant', 'step', 'multistep', 'cosine', 'exponential', 'plateau']"
    )


def step_lr_scheduler(
    scheduler: Optional[Any],
    strategy: str,
    val_metric: Optional[float] = None,
) -> None:
    """在每个 epoch 结束后更新学习率调度器。"""
    if scheduler is None:
        return
    mode = strategy.strip().lower()
    if mode == "plateau":
        if val_metric is not None:
            scheduler.step(val_metric)
    elif mode in ("step", "multistep", "cosine", "exponential"):
        scheduler.step()
    else:
        raise ValueError(f"不支持的 lr_scheduler.strategy: {strategy!r}")


def compute_pde_weight(
    epoch: int,
    num_epochs: int,
    base_weight: float,
    strategy: str = "constant",
    params: Optional[Mapping[str, Any]] = None,
) -> float:
    """计算 pPINN 第 `epoch` 轮（1-indexed）的 PDE 物理残差权重 `lambda_phys`。

    支持策略：`constant`（恒定）、`linear`（线性过渡）、`cosine`（余弦过渡）、
    `step`（阶梯变化）、`exponential`（指数变化）。
    """
    mode = strategy.strip().lower()
    cfg = dict(params or {})
    if mode in ("", "constant", "none"):
        return float(base_weight)

    start_w = float(cfg.get("start_weight", 0.0))
    end_w = float(cfg.get("end_weight", base_weight))
    start_ep = int(cfg.get("start_epoch", 1))
    end_ep = int(cfg.get("end_epoch", num_epochs))

    if mode in ("linear", "cosine"):
        if epoch <= start_ep:
            return start_w
        if epoch >= end_ep:
            return end_w
        t = (epoch - start_ep) / float(end_ep - start_ep)
        alpha = t if mode == "linear" else 0.5 * (1.0 - math.cos(math.pi * t))
        return float(start_w + (end_w - start_w) * alpha)

    if mode == "step":
        gamma = float(cfg.get("gamma", 2.0))
        init_w = float(cfg.get("start_weight", base_weight))
        if init_w <= 0.0 and float(base_weight) > 0.0:
            raise ValueError("pde_weight_schedule.strategy='step' 时 start_weight 必须大于 0")
        if "milestones" in cfg:
            steps = sum(1 for m in cfg["milestones"] if epoch >= int(m))
        else:
            step_size = int(cfg.get("step_size", 50))
            if step_size <= 0:
                raise ValueError(f"step_size 必须为正整数，实际为 {step_size}")
            steps = (epoch - 1) // step_size
        w = init_w * (gamma ** steps)
        return float(min(w, float(cfg["max_weight"]))) if "max_weight" in cfg else float(w)

    if mode == "exponential":
        gamma = float(cfg.get("gamma", 1.02))
        init_w = float(cfg.get("start_weight", base_weight))
        if init_w <= 0.0 and float(base_weight) > 0.0:
            raise ValueError("pde_weight_schedule.strategy='exponential' 时 start_weight 必须大于 0")
        w = init_w * (gamma ** (epoch - 1))
        return float(min(w, float(cfg["max_weight"]))) if "max_weight" in cfg else float(w)

    raise ValueError(
        f"不支持的 pde_weight_schedule.strategy: {strategy!r}，"
        "可选: ['constant', 'linear', 'cosine', 'step', 'exponential']"
    )


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """保存模型权重、归一化统计量与训练元信息到磁盘。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), out)
    return out


def write_json(path: str | Path, data: Any) -> Path:
    """将配置或训练历史写入 JSON 文件。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return out
