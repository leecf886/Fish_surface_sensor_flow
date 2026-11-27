"""训练过程损失可视化模块（纯参数接口，支持 MLP / DeepSets / pPINN 训练历史绘图）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_VALID_TASKS = ("mlp", "deepsets", "ppinn")
_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf")


def infer_history_task(history: Mapping[str, Any]) -> str:
    """根据 history 字段自动识别训练任务类型 (`mlp` | `deepsets` | `ppinn`)。"""
    if "val_by_budget" in history or "val_selection_score" in history:
        return "deepsets"
    if "train_loss_data" in history or "val_data_mse" in history:
        return "ppinn"
    if "train_loss_re" in history or "val_loss_total" in history:
        return "mlp"
    raise ValueError(f"无法从 history 字段自动识别任务类型，当前键: {sorted(history.keys())}")


def _to_float_array(values: Sequence[Any]) -> np.ndarray:
    """将序列安全转换为 float64 一维数组。"""
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _plot_finite_line(
    ax: plt.Axes,
    epochs: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.6,
    alpha: float = 0.92,
) -> bool:
    """仅绘制有限数值点，返回是否至少绘制了一个有效点。"""
    mask = np.isfinite(epochs) & np.isfinite(values)
    if not np.any(mask):
        return False
    marker = "o" if int(np.sum(mask)) <= 15 else None
    ax.plot(
        epochs[mask],
        values[mask],
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=3.5 if marker else None,
        alpha=alpha,
    )
    return True


def _apply_panel_style(
    ax: plt.Axes,
    *,
    title: str,
    ylabel: str,
    xlabel: str = "",
    log_scale: bool = True,
    best_epoch: Optional[int] = None,
    best_label: Optional[str] = None,
) -> None:
    """统一设置坐标轴刻度、网格、最优轮次竖线与图例。"""
    if log_scale:
        has_pos = any(
            np.any(np.asarray(line.get_ydata(), dtype=np.float64) > 0.0)
            for line in ax.get_lines()
        )
        if has_pos:
            ax.set_yscale("log")

    if best_epoch is not None and int(best_epoch) > 0:
        ax.axvline(
            int(best_epoch),
            color="#444444",
            linestyle=":",
            linewidth=1.2,
            label=best_label or f"Best Epoch ({int(best_epoch)})",
        )

    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5)
    ax.grid(alpha=0.25, which="both" if ax.get_yscale() == "log" else "major")
    if ax.get_lines():
        ax.legend(loc="best", fontsize=8, framealpha=0.85)


def _finalize_figure(fig: plt.Figure, out_path: str | Path, suptitle: str, dpi: int) -> Path:
    """保存图像并释放 Matplotlib 句柄。"""
    fig.suptitle(suptitle, fontsize=12.5)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=int(dpi))
    plt.close(fig)
    return out_file


def _extract_budget_series(
    val_by_budget: Sequence[Mapping[str, Mapping[str, float]]],
    metric_key: str,
) -> List[Tuple[str, np.ndarray]]:
    """从逐轮 val_by_budget 列表中提取各传感器预算下的指标曲线。"""
    if not val_by_budget:
        return []
    budgets = sorted(
        {str(k) for item in val_by_budget if item for k in item.keys()},
        key=lambda s: (int(s) if s.isdigit() else 0, s),
    )
    out: List[Tuple[str, np.ndarray]] = []
    for b in budgets:
        vals = [
            float(item[b][metric_key]) if (item and b in item and metric_key in item[b]) else float("nan")
            for item in val_by_budget
        ]
        out.append((b, np.asarray(vals, dtype=np.float64)))
    return out


def plot_deepsets_history(
    out_path: str | Path,
    *,
    epochs: Sequence[int],
    lr: Sequence[float],
    train_loss_total: Sequence[float],
    train_loss_re: Sequence[float],
    train_loss_st: Sequence[float],
    train_loss_phase: Sequence[float],
    val_selection_score: Sequence[float],
    val_by_budget: Optional[Sequence[Mapping[str, Mapping[str, float]]]] = None,
    val_standard_metrics: Optional[Sequence[Mapping[str, float]]] = None,
    best_epoch: Optional[int] = None,
    best_selection_loss: Optional[float] = None,
    log_scale: bool = True,
    dpi: int = 180,
) -> Path:
    """绘制 DeepSets 训练损失、学习率与多预算验证面板收敛曲线。"""
    ep = _to_float_array(epochs)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5), sharex=True, constrained_layout=True)

    # 1. 总训练损失与验证选优分数
    ax = axes[0, 0]
    _plot_finite_line(ax, ep, _to_float_array(train_loss_total), label="Train Total Loss", color=_COLORS[0])
    _plot_finite_line(ax, ep, _to_float_array(val_selection_score), label="Val Selection Score", color=_COLORS[1])
    best_lbl = (
        f"Best Epoch {best_epoch} ({best_selection_loss:.4f})"
        if best_epoch is not None and best_selection_loss is not None
        else None
    )
    _apply_panel_style(
        ax,
        title="Total Loss & Selection Score",
        ylabel="Loss",
        log_scale=log_scale,
        best_epoch=best_epoch,
        best_label=best_lbl,
    )

    # 2. 训练分项损失与学习率曲线
    ax = axes[0, 1]
    _plot_finite_line(ax, ep, _to_float_array(train_loss_re), label="Train Re Loss", color=_COLORS[0])
    _plot_finite_line(ax, ep, _to_float_array(train_loss_st), label="Train St Loss", color=_COLORS[2])
    _plot_finite_line(ax, ep, _to_float_array(train_loss_phase), label="Train Phase Loss", color=_COLORS[3])
    _apply_panel_style(ax, title="Train Loss Components & LR", ylabel="Component Loss", log_scale=log_scale)
    ax_lr = ax.twinx()
    lr_arr = _to_float_array(lr)
    _plot_finite_line(ax_lr, ep, lr_arr, label="LR", color="#555555", linestyle="--", linewidth=1.2)
    ax_lr.set_ylabel("Learning Rate", fontsize=9, color="#555555")
    if log_scale and np.any(lr_arr > 0):
        ax_lr.set_yscale("log")
    ax_lr.legend(loc="lower left", fontsize=8, framealpha=0.85)

    # 3~6. 多预算验证面板分项指标（若关闭冻结面板则退化展示标准验证曲线）
    has_budget = bool(val_by_budget and any(bool(item) for item in val_by_budget))
    if has_budget:
        assert val_by_budget is not None
        budget_panels = [
            (axes[1, 0], "val_loss_total", "Val Total Loss by Budget", "Loss", True),
            (axes[1, 1], "phase_crmse", "Val Phase cRMSE by Budget", "Phase cRMSE (rad)", log_scale),
            (axes[2, 0], "re_rmse", "Val Re RMSE by Budget", "Re RMSE", log_scale),
            (axes[2, 1], "st_rmse", "Val St RMSE by Budget", "St RMSE", log_scale),
        ]
        for b_ax, key, title, ylabel, use_log in budget_panels:
            for i, (b_name, vals) in enumerate(_extract_budget_series(val_by_budget, key)):
                _plot_finite_line(
                    b_ax,
                    ep,
                    vals,
                    label=f"K={b_name}",
                    color=_COLORS[i % len(_COLORS)],
                )
            _apply_panel_style(
                b_ax,
                title=title,
                ylabel=ylabel,
                xlabel="Epoch" if b_ax in axes[2] else "",
                log_scale=use_log and log_scale,
                best_epoch=best_epoch,
            )
    else:
        std_list = list(val_standard_metrics or [])
        std_panels = [
            (axes[1, 0], "loss_total", "Val Standard Loss", "Loss"),
            (axes[1, 1], "phase_crmse", "Val Phase cRMSE", "Phase cRMSE (rad)"),
            (axes[2, 0], "re_rmse", "Val Re RMSE", "Re RMSE"),
            (axes[2, 1], "st_rmse", "Val St RMSE", "St RMSE"),
        ]
        for s_ax, key, title, ylabel in std_panels:
            vals = np.asarray(
                [float(item.get(key, float("nan"))) if item else float("nan") for item in std_list],
                dtype=np.float64,
            )
            _plot_finite_line(s_ax, ep, vals, label=key, color=_COLORS[0])
            _apply_panel_style(
                s_ax,
                title=title,
                ylabel=ylabel,
                xlabel="Epoch" if s_ax in axes[2] else "",
                log_scale=log_scale,
                best_epoch=best_epoch,
            )

    return _finalize_figure(fig, out_path, "DeepSets Training & Validation Curves", dpi)


def plot_mlp_history(
    out_path: str | Path,
    *,
    epochs: Sequence[int],
    lr: Sequence[float],
    train_loss_total: Sequence[float],
    train_loss_re: Sequence[float],
    train_loss_st: Sequence[float],
    train_loss_phase: Sequence[float],
    val_loss_total: Sequence[float],
    val_loss_re: Sequence[float],
    val_loss_st: Sequence[float],
    val_loss_phase: Sequence[float],
    val_re_rmse: Sequence[float],
    val_st_rmse: Sequence[float],
    val_phase_crmse: Sequence[float],
    best_epoch: Optional[int] = None,
    best_selection_loss: Optional[float] = None,
    log_scale: bool = True,
    dpi: int = 180,
) -> Path:
    """绘制 SensorMLP 训练与验证收敛曲线。"""
    ep = _to_float_array(epochs)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True, constrained_layout=True)

    # 1. 总损失
    ax = axes[0, 0]
    _plot_finite_line(ax, ep, _to_float_array(train_loss_total), label="Train Total Loss", color=_COLORS[0])
    _plot_finite_line(ax, ep, _to_float_array(val_loss_total), label="Val Total Loss", color=_COLORS[1])
    best_lbl = (
        f"Best Epoch {best_epoch} ({best_selection_loss:.4f})"
        if best_epoch is not None and best_selection_loss is not None
        else None
    )
    _apply_panel_style(
        ax,
        title="Total Loss",
        ylabel="Loss",
        log_scale=log_scale,
        best_epoch=best_epoch,
        best_label=best_lbl,
    )

    # 2. 分项损失与学习率
    ax = axes[0, 1]
    _plot_finite_line(ax, ep, _to_float_array(train_loss_re), label="Train Re", color=_COLORS[0])
    _plot_finite_line(ax, ep, _to_float_array(train_loss_st), label="Train St", color=_COLORS[2])
    _plot_finite_line(ax, ep, _to_float_array(train_loss_phase), label="Train Phase", color=_COLORS[3])
    _plot_finite_line(ax, ep, _to_float_array(val_loss_re), label="Val Re", color=_COLORS[0], linestyle="--")
    _plot_finite_line(ax, ep, _to_float_array(val_loss_st), label="Val St", color=_COLORS[2], linestyle="--")
    _plot_finite_line(ax, ep, _to_float_array(val_loss_phase), label="Val Phase", color=_COLORS[3], linestyle="--")
    _apply_panel_style(ax, title="Loss Components & LR", ylabel="Loss", log_scale=log_scale)
    ax_lr = ax.twinx()
    lr_arr = _to_float_array(lr)
    _plot_finite_line(ax_lr, ep, lr_arr, label="LR", color="#555555", linestyle=":", linewidth=1.2)
    ax_lr.set_ylabel("Learning Rate", fontsize=9, color="#555555")
    if log_scale and np.any(lr_arr > 0):
        ax_lr.set_yscale("log")

    # 3. Re 与 St 验证 RMSE
    ax = axes[1, 0]
    _plot_finite_line(ax, ep, _to_float_array(val_re_rmse), label="Val Re RMSE", color=_COLORS[0])
    _apply_panel_style(ax, title="Val Re & St RMSE", ylabel="Re RMSE", xlabel="Epoch", log_scale=log_scale, best_epoch=best_epoch)
    ax_st = ax.twinx()
    _plot_finite_line(ax_st, ep, _to_float_array(val_st_rmse), label="Val St RMSE", color=_COLORS[2], linestyle="--")
    ax_st.set_ylabel("St RMSE", fontsize=9, color=_COLORS[2])
    ax_st.legend(loc="lower left", fontsize=8, framealpha=0.85)

    # 4. 相位验证误差
    ax = axes[1, 1]
    _plot_finite_line(ax, ep, _to_float_array(val_phase_crmse), label="Val Phase cRMSE", color=_COLORS[3])
    _apply_panel_style(ax, title="Val Phase cRMSE", ylabel="Phase cRMSE (rad)", xlabel="Epoch", log_scale=log_scale, best_epoch=best_epoch)

    return _finalize_figure(fig, out_path, "SensorMLP Training & Validation Curves", dpi)


def plot_ppinn_history(
    out_path: str | Path,
    *,
    epochs: Sequence[int],
    lr: Sequence[float],
    lambda_phys: Sequence[float],
    train_loss_total: Sequence[float],
    train_loss_data: Sequence[float],
    train_loss_phys: Sequence[float],
    val_data_mse: Sequence[float],
    val_data_mse_gauge: Sequence[float],
    val_u_rel_l2: Sequence[float],
    val_v_rel_l2: Sequence[float],
    val_p_rel_l2: Sequence[float],
    val_overall_rel_l2_gauge_sameden: Sequence[float],
    best_epoch: Optional[int] = None,
    best_metric_value: Optional[float] = None,
    best_metric_name: str = "data_mse",
    log_scale: bool = True,
    dpi: int = 180,
) -> Path:
    """绘制 pPINN 训练损失、PDE 权重调度与验证流场相对 L2 曲线。"""
    ep = _to_float_array(epochs)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True, constrained_layout=True)

    # 1. 训练损失分项
    ax = axes[0, 0]
    _plot_finite_line(ax, ep, _to_float_array(train_loss_total), label="Train Total", color=_COLORS[0])
    _plot_finite_line(ax, ep, _to_float_array(train_loss_data), label="Train Data", color=_COLORS[2])
    _plot_finite_line(ax, ep, _to_float_array(train_loss_phys), label="Train PDE", color=_COLORS[3])
    _apply_panel_style(ax, title="Training Losses", ylabel="Loss", log_scale=log_scale, best_epoch=best_epoch)

    # 2. 学习率与 PDE 权重调度
    ax = axes[0, 1]
    lr_arr = _to_float_array(lr)
    _plot_finite_line(ax, ep, lr_arr, label="LR", color=_COLORS[0])
    _apply_panel_style(ax, title="LR & PDE Weight Schedule", ylabel="Learning Rate", log_scale=log_scale and bool(np.any(lr_arr > 0)))
    ax_w = ax.twinx()
    _plot_finite_line(ax_w, ep, _to_float_array(lambda_phys), label="lambda_phys", color=_COLORS[1], linestyle="--")
    ax_w.set_ylabel("lambda_phys", fontsize=9, color=_COLORS[1])
    ax_w.legend(loc="lower left", fontsize=8, framealpha=0.85)

    # 3. 验证 Data MSE
    ax = axes[1, 0]
    _plot_finite_line(ax, ep, _to_float_array(val_data_mse), label="Val Data MSE", color=_COLORS[0])
    _plot_finite_line(ax, ep, _to_float_array(val_data_mse_gauge), label="Val Data MSE (Gauge)", color=_COLORS[2])
    best_lbl = (
        f"Best Epoch {best_epoch} ({best_metric_name}={best_metric_value:.4e})"
        if best_epoch is not None and best_metric_value is not None
        else None
    )
    _apply_panel_style(
        ax,
        title="Validation Data MSE",
        ylabel="MSE",
        xlabel="Epoch",
        log_scale=log_scale,
        best_epoch=best_epoch,
        best_label=best_lbl,
    )

    # 4. 验证流场相对 L2 误差
    ax = axes[1, 1]
    _plot_finite_line(ax, ep, _to_float_array(val_u_rel_l2), label="u rel L2", color=_COLORS[0])
    _plot_finite_line(ax, ep, _to_float_array(val_v_rel_l2), label="v rel L2", color=_COLORS[2])
    _plot_finite_line(ax, ep, _to_float_array(val_p_rel_l2), label="p gauge rel L2", color=_COLORS[3])
    _plot_finite_line(
        ax,
        ep,
        _to_float_array(val_overall_rel_l2_gauge_sameden),
        label="overall gauge rel L2",
        color=_COLORS[1],
        linewidth=2.0,
    )
    _apply_panel_style(
        ax,
        title="Validation Relative L2 Errors",
        ylabel="Relative L2",
        xlabel="Epoch",
        log_scale=log_scale,
        best_epoch=best_epoch,
    )

    return _finalize_figure(fig, out_path, "pPINN Training & Validation Curves", dpi)


def _resolve_history_path(history_or_dir: str | Path) -> Path:
    """支持传入 `runs/<exp>` 目录或 `history.json` 文件路径。"""
    p = Path(history_or_dir)
    return p / "history.json" if p.is_dir() else p


def _resolve_output_image_path(output_path: Optional[str | Path], default_dir: Optional[Path]) -> Path:
    """解析输出图片路径：默认保存到 history/ckpt 同目录下的 `loss_curve.png`。"""
    if output_path is None:
        if default_dir is None:
            raise ValueError("传入内存 history 字典时必须显式指定 output_path")
        return default_dir / "loss_curve.png"
    p = Path(output_path)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
        return p
    return p / "loss_curve.png"


def plot_training_history(
    history: Mapping[str, Any] | str | Path,
    output_path: Optional[str | Path] = None,
    *,
    task: Optional[str] = None,
    log_scale: bool = True,
    dpi: int = 180,
) -> Path:
    """统一纯参数训练损失可视化入口：支持直接传入 history 字典或 history.json / 实验目录路径。"""
    if isinstance(history, (str, Path)):
        hist_file = _resolve_history_path(history)
        if not hist_file.exists():
            raise FileNotFoundError(f"未找到训练历史文件: {hist_file}")
        hist_data: Mapping[str, Any] = json.loads(hist_file.read_text(encoding="utf-8"))
        out_file = _resolve_output_image_path(output_path, hist_file.parent)
    else:
        hist_data = history
        out_file = _resolve_output_image_path(output_path, None)

    task_key = infer_history_task(hist_data) if task is None or str(task).strip().lower() == "auto" else str(task).strip().lower()
    if task_key not in _VALID_TASKS:
        raise ValueError(f"不支持的 task: {task!r}，可选: {list(_VALID_TASKS)}")

    if task_key == "deepsets":
        return plot_deepsets_history(
            out_file,
            epochs=hist_data["epochs"],
            lr=hist_data["lr"],
            train_loss_total=hist_data["train_loss_total"],
            train_loss_re=hist_data["train_loss_re"],
            train_loss_st=hist_data["train_loss_st"],
            train_loss_phase=hist_data["train_loss_phase"],
            val_selection_score=hist_data["val_selection_score"],
            val_by_budget=hist_data.get("val_by_budget"),
            val_standard_metrics=hist_data.get("val_standard_metrics"),
            best_epoch=hist_data.get("best_epoch"),
            best_selection_loss=hist_data.get("best_selection_loss"),
            log_scale=log_scale,
            dpi=dpi,
        )

    if task_key == "mlp":
        return plot_mlp_history(
            out_file,
            epochs=hist_data["epochs"],
            lr=hist_data["lr"],
            train_loss_total=hist_data["train_loss_total"],
            train_loss_re=hist_data["train_loss_re"],
            train_loss_st=hist_data["train_loss_st"],
            train_loss_phase=hist_data["train_loss_phase"],
            val_loss_total=hist_data["val_loss_total"],
            val_loss_re=hist_data["val_loss_re"],
            val_loss_st=hist_data["val_loss_st"],
            val_loss_phase=hist_data["val_loss_phase"],
            val_re_rmse=hist_data["val_re_rmse"],
            val_st_rmse=hist_data["val_st_rmse"],
            val_phase_crmse=hist_data["val_phase_crmse"],
            best_epoch=hist_data.get("best_epoch"),
            best_selection_loss=hist_data.get("best_selection_loss"),
            log_scale=log_scale,
            dpi=dpi,
        )

    return plot_ppinn_history(
        out_file,
        epochs=hist_data["epochs"],
        lr=hist_data["lr"],
        lambda_phys=hist_data["lambda_phys"],
        train_loss_total=hist_data["train_loss_total"],
        train_loss_data=hist_data["train_loss_data"],
        train_loss_phys=hist_data["train_loss_phys"],
        val_data_mse=hist_data["val_data_mse"],
        val_data_mse_gauge=hist_data["val_data_mse_gauge"],
        val_u_rel_l2=hist_data["val_u_rel_l2"],
        val_v_rel_l2=hist_data["val_v_rel_l2"],
        val_p_rel_l2=hist_data["val_p_rel_l2"],
        val_overall_rel_l2_gauge_sameden=hist_data["val_overall_rel_l2_gauge_sameden"],
        best_epoch=hist_data.get("best_epoch"),
        best_metric_value=hist_data.get("best_metric_value"),
        best_metric_name=str(hist_data.get("best_metric_name", "data_mse")),
        log_scale=log_scale,
        dpi=dpi,
    )
