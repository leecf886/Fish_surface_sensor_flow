"""固定布局表面传感器 MLP 编码器训练模块（纯参数接口，不读取配置文件）。"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fish_flow.data.encoder_loader import LoadedEncoderData, Normalizer
from fish_flow.losses import MLPLoss
from fish_flow.training.common import (
    TWO_PI,
    EarlyStopper,
    build_lr_scheduler,
    compute_circular_metrics,
    compute_regression_metrics,
    resolve_device,
    save_checkpoint,
    set_training_seed,
    step_lr_scheduler,
    write_json,
)


@dataclass
class MLPTrainingConfig:
    """MLP 编码器纯参数训练配置。"""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 200

    lambda_re: float = 1.0
    lambda_st: float = 1.0
    lambda_phase: float = 1.0
    loss_type: str = "mse"
    lambda_unit_circle: float = 0.0

    # 学习率调度（默认 constant 保持不变）
    lr_strategy: str = "constant"
    lr_params: Dict[str, Any] = field(default_factory=dict)

    # 早停策略（默认关闭）
    early_stopping_enabled: bool = False
    patience: int = 20
    min_delta: float = 0.0

    seed: int = 42
    device: str = "auto"
    save_best: bool = True
    save_last: bool = False
    checkpoint_name: str = "best_mlp.pt"
    last_checkpoint_name: str = "last_mlp.pt"
    verbose: bool = False


@dataclass
class MLPTrainHistory:
    """MLP 训练与验证历史记录。"""

    epochs: List[int] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    train_loss_total: List[float] = field(default_factory=list)
    train_loss_re: List[float] = field(default_factory=list)
    train_loss_st: List[float] = field(default_factory=list)
    train_loss_phase: List[float] = field(default_factory=list)
    val_loss_total: List[float] = field(default_factory=list)
    val_loss_re: List[float] = field(default_factory=list)
    val_loss_st: List[float] = field(default_factory=list)
    val_loss_phase: List[float] = field(default_factory=list)
    val_re_mae: List[float] = field(default_factory=list)
    val_re_rmse: List[float] = field(default_factory=list)
    val_st_mae: List[float] = field(default_factory=list)
    val_st_rmse: List[float] = field(default_factory=list)
    val_phase_cmae: List[float] = field(default_factory=list)
    val_phase_crmse: List[float] = field(default_factory=list)
    epoch_wall_time_seconds: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_selection_loss: Optional[float] = None

    def add_epoch_record(
        self,
        epoch: int,
        lr: float,
        train_metrics: Mapping[str, float],
        val_metrics: Optional[Mapping[str, Any]] = None,
        wall_time: float = 0.0,
    ) -> None:
        """记录单个 epoch 的训练与验证指标。"""
        self.epochs.append(int(epoch))
        self.lr.append(float(lr))
        self.train_loss_total.append(float(train_metrics["loss_total"]))
        self.train_loss_re.append(float(train_metrics["loss_re"]))
        self.train_loss_st.append(float(train_metrics["loss_st"]))
        self.train_loss_phase.append(float(train_metrics["loss_phase"]))
        self.epoch_wall_time_seconds.append(float(wall_time))

        nan = float("nan")
        vm = val_metrics or {}
        self.val_loss_total.append(float(vm.get("loss_total", nan)))
        self.val_loss_re.append(float(vm.get("loss_re", nan)))
        self.val_loss_st.append(float(vm.get("loss_st", nan)))
        self.val_loss_phase.append(float(vm.get("loss_phase", nan)))
        self.val_re_mae.append(float(vm.get("re_mae", nan)))
        self.val_re_rmse.append(float(vm.get("re_rmse", nan)))
        self.val_st_mae.append(float(vm.get("st_mae", nan)))
        self.val_st_rmse.append(float(vm.get("st_rmse", nan)))
        self.val_phase_cmae.append(float(vm.get("phase_cmae", nan)))
        self.val_phase_crmse.append(float(vm.get("phase_crmse", nan)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MLPTrainer:
    """固定布局 SensorMLP 训练器。"""

    def __init__(self, config: Optional[MLPTrainingConfig] = None) -> None:
        self.config = config or MLPTrainingConfig()

    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: MLPLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float]:
        """执行单个训练 epoch，按样本数加权累计各项损失。"""
        model.train()
        sum_total = sum_re = sum_st = sum_phase = 0.0
        n_seen = 0

        for x, y, _, _, _, _ in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, metrics = criterion(pred, y)
            loss.backward()
            optimizer.step()

            bs = int(x.shape[0])
            n_seen += bs
            sum_total += metrics["loss_total"] * bs
            sum_re += metrics["loss_re"] * bs
            sum_st += metrics["loss_st"] * bs
            sum_phase += metrics["loss_phase"] * bs

        return {
            "loss_total": sum_total / n_seen,
            "loss_re": sum_re / n_seen,
            "loss_st": sum_st / n_seen,
            "loss_phase": sum_phase / n_seen,
        }

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: MLPLoss,
        re_normalizer: Normalizer,
        st_normalizer: Normalizer,
        device: torch.device,
    ) -> Dict[str, Any]:
        """在验证集上评估损失及反归一化后的 Re / St / 相位物理误差。"""
        model.eval()
        sum_total = sum_re = sum_st = sum_phase = 0.0
        n_seen = 0

        re_true_list: List[np.ndarray] = []
        re_pred_list: List[np.ndarray] = []
        st_true_list: List[np.ndarray] = []
        st_pred_list: List[np.ndarray] = []
        phase_true_list: List[np.ndarray] = []
        phase_pred_list: List[np.ndarray] = []

        for x, y, re_raw, st_raw, phase_true, _ in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            _, metrics = criterion(pred, y)

            bs = int(x.shape[0])
            n_seen += bs
            sum_total += metrics["loss_total"] * bs
            sum_re += metrics["loss_re"] * bs
            sum_st += metrics["loss_st"] * bs
            sum_phase += metrics["loss_phase"] * bs

            # 4 维输出顺序为 [Re_norm, St_norm, sin(phi), cos(phi)]
            pred_np = pred.detach().cpu().numpy().astype(np.float64)
            re_pred_list.append(re_normalizer.inverse_transform(pred_np[:, 0]))
            st_pred_list.append(st_normalizer.inverse_transform(pred_np[:, 1]))
            phase_pred_list.append(np.mod(np.arctan2(pred_np[:, 2], pred_np[:, 3]), TWO_PI))

            re_true_list.append(re_raw.cpu().numpy().astype(np.float64))
            st_true_list.append(st_raw.cpu().numpy().astype(np.float64))
            phase_true_list.append(phase_true.cpu().numpy().astype(np.float64))

        re_m = compute_regression_metrics(np.concatenate(re_true_list), np.concatenate(re_pred_list))
        st_m = compute_regression_metrics(np.concatenate(st_true_list), np.concatenate(st_pred_list))
        ph_m = compute_circular_metrics(np.concatenate(phase_true_list), np.concatenate(phase_pred_list))

        return {
            "loss_total": sum_total / n_seen,
            "loss_re": sum_re / n_seen,
            "loss_st": sum_st / n_seen,
            "loss_phase": sum_phase / n_seen,
            "re_mae": re_m["mae"],
            "re_rmse": re_m["rmse"],
            "st_mae": st_m["mae"],
            "st_rmse": st_m["rmse"],
            "phase_cmae": ph_m["circular_mae"],
            "phase_crmse": ph_m["circular_rmse"],
        }

    def fit(
        self,
        model: nn.Module,
        data: LoadedEncoderData,
        output_dir: Optional[str | Path] = None,
        *,
        model_hparams: Optional[Mapping[str, Any]] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[nn.Module, MLPTrainHistory]:
        """执行 MLP 训练、验证选优、学习率调度、早停与权重保存。"""
        cfg = self.config
        set_training_seed(cfg.seed)
        if getattr(data.train_loader, "generator", None) is not None:
            data.train_loader.generator.manual_seed(int(cfg.seed))

        device = resolve_device(cfg.device)
        model = model.to(device)

        criterion = MLPLoss(
            lambda_re=cfg.lambda_re,
            lambda_st=cfg.lambda_st,
            lambda_phase=cfg.lambda_phase,
            loss_type=cfg.loss_type,
            lambda_unit_circle=cfg.lambda_unit_circle,
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

        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        history = MLPTrainHistory()
        best_epoch = -1
        best_score = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())
        norm_dict = data.normalizers_to_dict()

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.perf_counter()
            curr_lr = float(optimizer.param_groups[0]["lr"])
            train_out = self.train_one_epoch(model, data.train_loader, criterion, optimizer, device)

            val_out = (
                self.evaluate(model, data.val_loader, criterion, data.re_normalizer, data.st_normalizer, device)
                if data.val_loader is not None
                else None
            )

            wall_time = float(time.perf_counter() - t0)
            history.add_epoch_record(epoch, curr_lr, train_out, val_out, wall_time)

            selection_metric = "val_loss_total" if val_out is not None else "train_loss_total"
            score = float(val_out["loss_total"] if val_out is not None else train_out["loss_total"])
            is_best, should_stop = stopper.update(score)

            if is_best:
                best_score = score
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

                if out_path is not None and cfg.save_best:
                    save_checkpoint(
                        out_path / cfg.checkpoint_name,
                        {
                            "model_type": "SensorMLP",
                            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "model_hparams": dict(model_hparams or {}),
                            "normalizers": norm_dict,
                            "training_config": asdict(cfg),
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "selection_metric": selection_metric,
                            "selection_score": best_score,
                            "val_metrics": val_out,
                            "extra_metadata": dict(extra_metadata or {}),
                        },
                    )

            if out_path is not None and cfg.save_last:
                save_checkpoint(
                    out_path / cfg.last_checkpoint_name,
                    {
                        "model_type": "SensorMLP",
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "model_hparams": dict(model_hparams or {}),
                        "normalizers": norm_dict,
                        "training_config": asdict(cfg),
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "selection_metric": selection_metric,
                        "selection_score": score,
                        "val_metrics": val_out,
                        "extra_metadata": dict(extra_metadata or {}),
                    },
                )

            if cfg.verbose:
                val_msg = (
                    f"val_loss={val_out['loss_total']:.6f} Re_RMSE={val_out['re_rmse']:.2f} "
                    f"St_RMSE={val_out['st_rmse']:.4f} phi_cRMSE={val_out['phase_crmse']:.4f}"
                    if val_out is not None
                    else "val=None"
                )
                print(
                    f"[MLP] Epoch {epoch}/{cfg.num_epochs} | lr={curr_lr:.2e} | "
                    f"train_loss={train_out['loss_total']:.6f} | {val_msg} | {wall_time:.2f}s"
                )

            step_lr_scheduler(scheduler, cfg.lr_strategy, score)
            if should_stop:
                if cfg.verbose:
                    print(f"[MLP] 早停触发于第 {epoch} 轮 (best_epoch={best_epoch}, best={best_score:.6f})")
                break

        # 恢复验证最优权重并输出训练历史
        model.load_state_dict(best_state_dict)
        history.best_epoch = best_epoch
        history.best_selection_loss = best_score

        if out_path is not None:
            write_json(out_path / "history.json", history.to_dict())

        return model, history


def train_mlp(
    model: nn.Module,
    data: LoadedEncoderData,
    *,
    output_dir: Optional[str | Path] = None,
    model_hparams: Optional[Mapping[str, Any]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    config: Optional[MLPTrainingConfig] = None,
    **kwargs: Any,
) -> Tuple[nn.Module, MLPTrainHistory]:
    """函数式 MLP 训练入口，支持传入 `MLPTrainingConfig` 或显式关键字参数。"""
    cfg = config if config is not None else MLPTrainingConfig(**kwargs)
    return MLPTrainer(cfg).fit(
        model=model,
        data=data,
        output_dir=output_dir,
        model_hparams=model_hparams,
        extra_metadata=extra_metadata,
    )
