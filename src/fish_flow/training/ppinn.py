"""参数化物理信息神经网络（pPINN）训练模块（纯参数接口，不读取配置文件）。

训练与验证协议与原代码一致：
- 监督流场数据批采样 + 5 维无量纲 PDE 配点残差协同训练（支持 fixed / epoch / batch 三种配点模式）；
- 整场 ROI 验证评估（含基于 Welford/Chan 合并的 Gauge 零点对齐压强相对 L2 误差）；
- 默认恒定学习率、恒定 PDE 权重、无早停；支持通过参数配置 PDE 权重变化、学习率调度与早停策略。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from fish_flow.collocation import collocation_to_tensors, sample_collocation_points
from fish_flow.data.ppinn_loader import LoadedPPINNData
from fish_flow.losses import PPINNLoss, compute_pde_residuals
from fish_flow.training.common import (
    EarlyStopper,
    PressureGaugeAccum,
    build_lr_scheduler,
    compute_pde_weight,
    resolve_device,
    save_checkpoint,
    set_training_seed,
    step_lr_scheduler,
    write_json,
)


@dataclass
class PPINNTrainingConfig:
    """pPINN 纯参数训练配置。"""

    batch_size: int = 2048
    steps_per_epoch: int = 200
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    lambda_data: float = 1.0
    lambda_phys: float = 1.0
    loss_type: str = "mse"
    data_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    phys_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # 5D PDE 配点参数
    n_collocation: int = 4096
    collocation_method: str = "latin_hypercube"
    collocation_re_st_mode: str = "continuous_random"
    exclude_solid_collocation: bool = True
    resample_collocation_mode: str = "fixed"  # fixed | epoch | batch

    # PDE 权重调度策略（默认 constant 保持不变）
    pde_weight_strategy: str = "constant"
    pde_weight_params: Dict[str, Any] = field(default_factory=dict)

    # 学习率调度（默认 constant 保持不变）
    lr_strategy: str = "constant"
    lr_params: Dict[str, Any] = field(default_factory=dict)

    # 早停策略（默认关闭）
    early_stopping_enabled: bool = False
    patience: int = 20
    min_delta: float = 0.0

    # 验证与选优
    checkpoint_selection_metric: str = "data_mse"
    eval_chunk_size: int = 16384
    val_every_n_epochs: int = 1

    seed: int = 42
    device: str = "auto"
    save_best: bool = True
    save_last: bool = True
    best_checkpoint_name: str = "best_ppinn.pt"
    last_checkpoint_name: str = "last_ppinn.pt"
    verbose: bool = False


@dataclass
class PPINNTrainHistory:
    """pPINN 训练与验证历史记录。"""

    epochs: List[int] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    lambda_phys: List[float] = field(default_factory=list)
    train_loss_total: List[float] = field(default_factory=list)
    train_loss_data: List[float] = field(default_factory=list)
    train_loss_phys: List[float] = field(default_factory=list)
    val_data_mse: List[float] = field(default_factory=list)
    val_data_mse_gauge: List[float] = field(default_factory=list)
    val_u_rel_l2: List[float] = field(default_factory=list)
    val_v_rel_l2: List[float] = field(default_factory=list)
    val_p_rel_l2: List[float] = field(default_factory=list)
    val_overall_rel_l2_gauge_sameden: List[float] = field(default_factory=list)
    wall_time_seconds: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_metric_value: Optional[float] = None
    best_metric_name: str = "data_mse"
    collocation_mode: str = "fixed"
    collocation_fingerprint: Optional[str] = None

    def add_epoch_record(
        self,
        epoch: int,
        lr: float,
        lambda_phys: float,
        train_metrics: Mapping[str, float],
        val_metrics: Optional[Mapping[str, float]],
        wall_time: float,
    ) -> None:
        """记录单个 epoch 的训练损失、调度参数与验证指标。"""
        self.epochs.append(int(epoch))
        self.lr.append(float(lr))
        self.lambda_phys.append(float(lambda_phys))
        self.train_loss_total.append(float(train_metrics["loss_total"]))
        self.train_loss_data.append(float(train_metrics["loss_data"]))
        self.train_loss_phys.append(float(train_metrics["loss_phys"]))
        self.wall_time_seconds.append(float(wall_time))

        nan = float("nan")
        vm = val_metrics or {}
        self.val_data_mse.append(float(vm.get("data_mse", nan)))
        self.val_data_mse_gauge.append(float(vm.get("data_mse_gauge", nan)))
        self.val_u_rel_l2.append(float(vm.get("u_rel_l2", nan)))
        self.val_v_rel_l2.append(float(vm.get("v_rel_l2", nan)))
        self.val_p_rel_l2.append(float(vm.get("p_rel_l2", nan)))
        self.val_overall_rel_l2_gauge_sameden.append(float(vm.get("overall_rel_l2_gauge_sameden", nan)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def evaluate_ppinn_validation(
    model: nn.Module,
    data: LoadedPPINNData,
    device: torch.device,
    chunk_size: int = 16384,
) -> Dict[str, float]:
    """在验证集整场 ROI 上评估速度相对 L2 与 Gauge 零点对齐压强相对 L2 误差。"""
    model.eval()
    u_err2 = v_err2 = u_ref2 = v_ref2 = 0.0
    data_err2_sum = 0.0
    n_total = 0
    p_gauge = PressureGaugeAccum()

    for _, _, feat_cpu, true_cpu in data.iter_val_frames(device="cpu"):
        nroi = int(feat_cpu.shape[0])
        true_np = true_cpu.numpy().astype(np.float64)

        n_e = n_p = 0
        mean_e_w = m2_e_w = mean_p_w = m2_p_w = 0.0

        # 分块前向并使用 Chan 公式合并同帧多块压强统计量
        for st in range(0, nroi, chunk_size):
            ed = min(st + chunk_size, nroi)
            n = ed - st
            pred = model(feat_cpu[st:ed].to(device)).cpu().numpy().astype(np.float64)
            true = true_np[st:ed]

            diff = pred - true
            data_err2_sum += float(np.sum(diff * diff))
            n_total += n

            u_err2 += float(np.sum((pred[:, 0] - true[:, 0]) ** 2))
            v_err2 += float(np.sum((pred[:, 1] - true[:, 1]) ** 2))
            u_ref2 += float(np.sum(true[:, 0] ** 2))
            v_ref2 += float(np.sum(true[:, 1] ** 2))

            p_true = true[:, 2]
            p_diff = pred[:, 2] - p_true
            mean_e_c = float(np.mean(p_diff))
            m2_e_c = float(np.sum((p_diff - mean_e_c) ** 2))
            mean_p_c = float(np.mean(p_true))
            m2_p_c = float(np.sum((p_true - mean_p_c) ** 2))

            n_e, mean_e_w, m2_e_w = PressureGaugeAccum.chan_merge(n_e, mean_e_w, m2_e_w, n, mean_e_c, m2_e_c)
            n_p, mean_p_w, m2_p_w = PressureGaugeAccum.chan_merge(n_p, mean_p_w, m2_p_w, n, mean_p_c, m2_p_c)

        p_gauge.add_snapshot_welford(mean_e_w, m2_e_w, mean_p_w, m2_p_w, n_e)

    eps = 1e-12
    u_rel = math.sqrt(u_err2 / (u_ref2 + eps))
    v_rel = math.sqrt(v_err2 / (v_ref2 + eps))
    pgm = p_gauge.metrics(eps=eps)
    p_rel_gsd = pgm["p_rel_l2_gauge_sameden"]
    p_rel_raw = pgm["p_rel_l2_raw"]

    total_dof = max(n_total * 3, 1)
    return {
        "u_rel_l2": float(u_rel),
        "v_rel_l2": float(v_rel),
        "p_rel_l2": float(p_rel_gsd),
        "p_rel_l2_raw": float(p_rel_raw),
        "p_rel_l2_gauge_sameden": float(p_rel_gsd),
        "p_rel_l2_gauge_fluctden": float(pgm["p_rel_l2_gauge_fluctden"]),
        "p_gauge_offset_mean_abs": float(pgm["p_gauge_offset_mean_abs"]),
        "overall_rel_l2": float((u_rel + v_rel + p_rel_gsd) / 3.0),
        "overall_rel_l2_raw": float((u_rel + v_rel + p_rel_raw) / 3.0),
        "overall_rel_l2_gauge_sameden": float((u_rel + v_rel + p_rel_gsd) / 3.0),
        "data_mse": float(data_err2_sum / total_dof),
        "data_mse_gauge": float((u_err2 + v_err2 + p_gauge.gauge_err_sse) / total_dof),
        "n_points": int(n_total),
    }


class PPINNTrainer:
    """参数化物理信息神经网络（pPINN）训练器。"""

    def __init__(self, config: Optional[PPINNTrainingConfig] = None) -> None:
        self.config = config or PPINNTrainingConfig()

    def _sample_collocation_tensors(
        self,
        data: LoadedPPINNData,
        rng: np.random.Generator,
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """采样 5D PDE 配点并转换为设备张量。"""
        cfg = self.config
        col, stats = sample_collocation_points(
            n_points=cfg.n_collocation,
            roi_bounds_nd=data.mesh.roi_bounds_nd,
            re_normalizer=data.re_normalizer,
            st_normalizer=data.st_normalizer,
            re_range=data.re_range,
            st_range=data.st_range,
            re_st_pairs=data.re_st_pairs,
            method=cfg.collocation_method,
            re_st_mode=cfg.collocation_re_st_mode,
            exclude_solid=cfg.exclude_solid_collocation,
            seed=rng,
        )
        return collocation_to_tensors(col, device=device, dtype=torch.float32), stats

    def train_one_epoch(
        self,
        model: nn.Module,
        data: LoadedPPINNData,
        criterion: PPINNLoss,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int,
        collocation_rng: np.random.Generator,
        device: torch.device,
        cached_col: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """执行单个 pPINN 训练 epoch，联合反传数据损失与 Navier-Stokes 残差损失。"""
        model.train()
        cfg = self.config
        sum_total = sum_data = sum_phys = 0.0

        for _ in range(steps_per_epoch):
            feat_sup, uvp_true = data.sample_train_batch(cfg.batch_size, device=device)
            uvp_pred = model(feat_sup)

            residuals = None
            if criterion.lambda_phys > 0.0:
                if cfg.resample_collocation_mode == "batch":
                    col_t, _ = self._sample_collocation_tensors(data, collocation_rng, device)
                else:
                    col_t = cached_col  # type: ignore[assignment]
                residuals = compute_pde_residuals(model, **col_t)

            loss, metrics = criterion(uvp_pred, uvp_true, residuals=residuals)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sum_total += metrics["loss_total"]
            sum_data += metrics["loss_data"]
            sum_phys += metrics["loss_phys"]

        denom = float(steps_per_epoch)
        return {
            "loss_total": sum_total / denom,
            "loss_data": sum_data / denom,
            "loss_phys": sum_phys / denom,
        }

    def fit(
        self,
        model: nn.Module,
        data: LoadedPPINNData,
        output_dir: Optional[str | Path] = None,
        *,
        model_hparams: Optional[Mapping[str, Any]] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[nn.Module, PPINNTrainHistory]:
        """执行 pPINN 训练、PDE 权重调度、验证选优、学习率调度、早停与权重保存。"""
        cfg = self.config

        # 按原代码协议派生独立随机流：模型=seed，监督采样=seed+101，配点=seed+202
        set_training_seed(cfg.seed)
        data.train_sampler.rng = np.random.default_rng(int(cfg.seed) + 101)
        collocation_rng = np.random.default_rng(int(cfg.seed) + 202)

        device = resolve_device(cfg.device)
        model = model.to(device)

        steps_per_epoch = (
            int(cfg.steps_per_epoch)
            if cfg.steps_per_epoch > 0
            else max(1, int(math.ceil(data.train_sampler.total_points / cfg.batch_size)))
        )

        criterion = PPINNLoss(
            lambda_data=cfg.lambda_data,
            lambda_phys=cfg.lambda_phys,
            loss_type=cfg.loss_type,
            data_weights=cfg.data_weights,
            phys_weights=cfg.phys_weights,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = build_lr_scheduler(
            optimizer,
            strategy=cfg.lr_strategy,
            num_epochs=cfg.num_epochs,
            params=cfg.lr_params,
        )
        stopper = EarlyStopper(
            enabled=cfg.early_stopping_enabled,
            patience=cfg.patience,
            min_delta=cfg.min_delta,
        )

        resample_mode = cfg.resample_collocation_mode.strip().lower()
        fixed_col_tensors: Optional[Dict[str, torch.Tensor]] = None
        col_fingerprint: Optional[str] = None
        if resample_mode == "fixed":
            fixed_col_tensors, col_stats = self._sample_collocation_tensors(data, collocation_rng, device)
            col_fingerprint = str(col_stats["fingerprint"])

        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        sel_metric = cfg.checkpoint_selection_metric.strip().lower()
        history = PPINNTrainHistory(
            best_metric_name=sel_metric,
            collocation_mode=resample_mode,
            collocation_fingerprint=col_fingerprint,
        )
        best_epoch = -1
        best_score = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())
        norm_dict = data.normalizers_to_dict()
        has_val = len(data.val_cases) > 0

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.perf_counter()
            curr_lr = float(optimizer.param_groups[0]["lr"])
            curr_lambda_phys = compute_pde_weight(
                epoch=epoch,
                num_epochs=cfg.num_epochs,
                base_weight=cfg.lambda_phys,
                strategy=cfg.pde_weight_strategy,
                params=cfg.pde_weight_params,
            )
            criterion.lambda_phys = curr_lambda_phys

            if resample_mode == "epoch" and curr_lambda_phys > 0.0:
                epoch_col, _ = self._sample_collocation_tensors(data, collocation_rng, device)
            else:
                epoch_col = fixed_col_tensors

            train_out = self.train_one_epoch(
                model=model,
                data=data,
                criterion=criterion,
                optimizer=optimizer,
                steps_per_epoch=steps_per_epoch,
                collocation_rng=collocation_rng,
                device=device,
                cached_col=epoch_col,
            )

            run_val = has_val and (epoch % cfg.val_every_n_epochs == 0 or epoch == cfg.num_epochs)
            val_out = (
                evaluate_ppinn_validation(model, data, device, chunk_size=cfg.eval_chunk_size)
                if run_val
                else None
            )

            wall_time = float(time.perf_counter() - t0)
            history.add_epoch_record(epoch, curr_lr, curr_lambda_phys, train_out, val_out, wall_time)

            score: Optional[float] = None
            should_stop = False
            active_metric = sel_metric if has_val else "train_loss_total"
            if val_out is not None:
                score = float(val_out[sel_metric])
            elif not has_val:
                score = float(train_out["loss_total"])

            if score is not None:
                is_best, should_stop = stopper.update(score)
                if is_best:
                    best_score = score
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(model.state_dict())

                    if out_path is not None and cfg.save_best:
                        save_checkpoint(
                            out_path / cfg.best_checkpoint_name,
                            {
                                "model_type": "PPINNModel",
                                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                "model_hparams": dict(model_hparams or {}),
                                "normalizers": norm_dict,
                                "training_config": asdict(cfg),
                                "epoch": epoch,
                                "best_epoch": best_epoch,
                                "selection_metric": active_metric,
                                "selection_score": best_score,
                                "val_metrics": val_out,
                                "collocation_fingerprint": col_fingerprint,
                                "extra_metadata": dict(extra_metadata or {}),
                            },
                        )

            if out_path is not None and cfg.save_last:
                save_checkpoint(
                    out_path / cfg.last_checkpoint_name,
                    {
                        "model_type": "PPINNModel",
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "model_hparams": dict(model_hparams or {}),
                        "normalizers": norm_dict,
                        "training_config": asdict(cfg),
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "selection_metric": active_metric,
                        "selection_score": score if score is not None else float("nan"),
                        "val_metrics": val_out,
                        "collocation_fingerprint": col_fingerprint,
                        "extra_metadata": dict(extra_metadata or {}),
                    },
                )

            if cfg.verbose:
                val_msg = (
                    f"val_mse={val_out['data_mse']:.6f} uL2={val_out['u_rel_l2']:.4f} "
                    f"vL2={val_out['v_rel_l2']:.4f} pL2_g={val_out['p_rel_l2_gauge_sameden']:.4f}"
                    if val_out is not None
                    else "val=skipped"
                )
                print(
                    f"[pPINN] Epoch {epoch}/{cfg.num_epochs} | lr={curr_lr:.2e} | w_pde={curr_lambda_phys:.3f} | "
                    f"data={train_out['loss_data']:.6f} phys={train_out['loss_phys']:.6f} | {val_msg} | {wall_time:.2f}s"
                )

            step_lr_scheduler(scheduler, cfg.lr_strategy, score)
            if should_stop:
                if cfg.verbose:
                    print(f"[pPINN] 早停触发于第 {epoch} 轮 (best_epoch={best_epoch}, best={best_score:.6f})")
                break

        # 恢复验证最优权重并输出训练历史
        model.load_state_dict(best_state_dict)
        history.best_epoch = best_epoch
        history.best_metric_value = best_score

        if out_path is not None:
            write_json(out_path / "history.json", history.to_dict())

        return model, history


def train_ppinn(
    model: nn.Module,
    data: LoadedPPINNData,
    *,
    output_dir: Optional[str | Path] = None,
    model_hparams: Optional[Mapping[str, Any]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    config: Optional[PPINNTrainingConfig] = None,
    **kwargs: Any,
) -> Tuple[nn.Module, PPINNTrainHistory]:
    """函数式 pPINN 训练入口，支持传入 `PPINNTrainingConfig` 或显式关键字参数。"""
    cfg = config if config is not None else PPINNTrainingConfig(**kwargs)
    return PPINNTrainer(cfg).fit(
        model=model,
        data=data,
        output_dir=output_dir,
        model_hparams=model_hparams,
        extra_metadata=extra_metadata,
    )
