"""动态传感器集合 DeepSets 编码器训练模块（纯参数接口，不读取配置文件）。

训练与验证机制与原代码保持一致：
- 训练期：每个 batch 通过 `DeepSetsSubsetCollator` 随机采样不同传感器数量子集；
- 验证期：默认采用冻结多预算随机子集验证面板（宏平均 layout -> val_re -> budget），同时支持标准全池评估。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fish_flow.data.encoder_loader import (
    DeepSetsSubsetCollator,
    EncoderDataset,
    LoadedEncoderData,
    Normalizer,
)
from fish_flow.losses import DeepSetsLoss
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


def _layout_id(indices: Sequence[int]) -> str:
    """根据排序后的传感器下标生成唯一短哈希标识。"""
    arr = np.asarray(sorted(int(i) for i in indices), dtype=np.int64)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()[:16]


def build_dynamic_validation_panel(
    candidate_count: int,
    budgets: Sequence[int] = (12, 24, 48, 96),
    layouts_per_budget: int = 8,
    seed: int = 400042,
) -> Dict[str, Any]:
    """在指定候选传感器总池上生成固定多预算验证面板。"""
    n_cand = int(candidate_count)
    budget_list = [int(b) for b in budgets]
    rng = np.random.default_rng(int(seed))

    layouts: List[Dict[str, Any]] = []
    for nsensor in budget_list:
        seen: set[Tuple[int, ...]] = set()
        while len(seen) < int(layouts_per_budget):
            idx = tuple(int(x) for x in np.sort(rng.choice(n_cand, size=nsensor, replace=False)).tolist())
            if idx in seen:
                continue
            seen.add(idx)
            layouts.append(
                {
                    "nsensor": nsensor,
                    "layout_id": _layout_id(idx),
                    "indices": list(idx),
                }
            )

    return {
        "panel_seed": int(seed),
        "candidate_count": n_cand,
        "budgets": budget_list,
        "layouts_per_budget": int(layouts_per_budget),
        "layouts": layouts,
    }


@torch.no_grad()
def evaluate_dynamic_validation_panel(
    model: nn.Module,
    val_dataset: EncoderDataset,
    panel: Mapping[str, Any],
    criterion: DeepSetsLoss,
    re_normalizer: Normalizer,
    st_normalizer: Normalizer,
    device: torch.device,
    chunk_size: int = 8192,
) -> Dict[str, Any]:
    """按 layout -> val_re -> budget 三级等权宏平均评估冻结多预算验证面板。"""
    model.eval()
    re_ids_all = val_dataset.re_id.cpu().numpy()
    unique_res = sorted(int(r) for r in np.unique(re_ids_all).tolist())
    budgets = [int(b) for b in panel["budgets"]]

    layouts_by_budget: Dict[int, List[List[int]]] = {b: [] for b in budgets}
    for rec in panel["layouts"]:
        layouts_by_budget[int(rec["nsensor"])].append(list(rec["indices"]))

    by_budget: Dict[str, Dict[str, float]] = {}
    budget_scores: List[float] = []

    for nsensor in budgets:
        layout_list = layouts_by_budget[nsensor]
        re_loss_list, re_rmse_list, st_rmse_list, ph_crmse_list = [], [], [], []

        for re_val in unique_res:
            case_mask = torch.as_tensor(re_ids_all == re_val, dtype=torch.bool)
            uvp_case = val_dataset.x[case_mask]
            y_case = val_dataset.y[case_mask]
            re_raw_case = val_dataset.re_raw[case_mask].cpu().numpy().astype(np.float64)
            st_raw_case = val_dataset.st_raw[case_mask].cpu().numpy().astype(np.float64)
            ph_true_case = val_dataset.phase_true[case_mask].cpu().numpy().astype(np.float64)
            n_snap = int(uvp_case.shape[0])

            lay_losses, lay_re_rmses, lay_st_rmses, lay_ph_crmses = [], [], [], []
            for idx in layout_list:
                idx_t = torch.as_tensor(idx, dtype=torch.long)
                uvp_sub = uvp_case.index_select(1, idx_t)
                pos_sub = val_dataset.xy_nd.index_select(0, idx_t)

                preds: List[torch.Tensor] = []
                loss_sum = 0.0
                for st in range(0, n_snap, chunk_size):
                    ed = min(st + chunk_size, n_snap)
                    bs = ed - st
                    pos_b = pos_sub.unsqueeze(0).expand(bs, -1, -1).to(device)
                    pred_b = model(uvp_sub[st:ed].to(device), pos_b, None)
                    loss_b, _ = criterion(pred_b, y_case[st:ed].to(device))
                    loss_sum += float(loss_b.item()) * bs
                    preds.append(pred_b.detach().cpu())

                pred_np = torch.cat(preds, dim=0).numpy().astype(np.float64)
                re_pred = re_normalizer.inverse_transform(pred_np[:, 0])
                st_pred = st_normalizer.inverse_transform(pred_np[:, 1])
                ph_pred = np.mod(np.arctan2(pred_np[:, 2], pred_np[:, 3]), TWO_PI)

                lay_losses.append(loss_sum / n_snap)
                lay_re_rmses.append(compute_regression_metrics(re_raw_case, re_pred)["rmse"])
                lay_st_rmses.append(compute_regression_metrics(st_raw_case, st_pred)["rmse"])
                lay_ph_crmses.append(compute_circular_metrics(ph_true_case, ph_pred)["circular_rmse"])

            # 先对同一工况下的多个布局取平均
            re_loss_list.append(float(np.mean(lay_losses)))
            re_rmse_list.append(float(np.mean(lay_re_rmses)))
            st_rmse_list.append(float(np.mean(lay_st_rmses)))
            ph_crmse_list.append(float(np.mean(lay_ph_crmses)))

        # 再对同一预算下的多个验证工况取平均
        b_loss = float(np.mean(re_loss_list))
        budget_scores.append(b_loss)
        by_budget[str(nsensor)] = {
            "val_loss_total": b_loss,
            "re_rmse": float(np.mean(re_rmse_list)),
            "st_rmse": float(np.mean(st_rmse_list)),
            "phase_crmse": float(np.mean(ph_crmse_list)),
        }

    # 最后对全部传感器预算等权平均作为选优分数
    return {
        "val_equal_budget_selection_score": float(np.mean(budget_scores)),
        "by_budget": by_budget,
    }


@dataclass
class DeepSetsTrainingConfig:
    """DeepSets 动态集合编码器纯参数训练配置。"""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 200

    lambda_re: float = 1.0
    lambda_st: float = 1.0
    lambda_phase: float = 1.0
    loss_type: str = "mse"
    lambda_unit_circle: float = 0.0

    # 冻结多预算验证面板参数
    use_frozen_panel: bool = True
    panel_budgets: Sequence[int] = (12, 24, 48, 96)
    panel_layouts_per_budget: int = 8
    panel_seed: int = 400042
    eval_chunk_size: int = 8192

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
    checkpoint_name: str = "best_deepsets.pt"
    last_checkpoint_name: str = "last_deepsets.pt"
    verbose: bool = False


@dataclass
class DeepSetsTrainHistory:
    """DeepSets 训练与多预算面板验证历史记录。"""

    epochs: List[int] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    train_loss_total: List[float] = field(default_factory=list)
    train_loss_re: List[float] = field(default_factory=list)
    train_loss_st: List[float] = field(default_factory=list)
    train_loss_phase: List[float] = field(default_factory=list)
    val_selection_score: List[float] = field(default_factory=list)
    val_by_budget: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    val_standard_metrics: List[Dict[str, float]] = field(default_factory=list)
    epoch_wall_time_seconds: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None
    best_selection_loss: Optional[float] = None

    def add_epoch_record(
        self,
        epoch: int,
        lr: float,
        train_metrics: Mapping[str, float],
        selection_score: float,
        by_budget: Optional[Mapping[str, Dict[str, float]]] = None,
        standard_metrics: Optional[Mapping[str, float]] = None,
        wall_time: float = 0.0,
    ) -> None:
        """记录单个 epoch 的训练损失与验证指标。"""
        self.epochs.append(int(epoch))
        self.lr.append(float(lr))
        self.train_loss_total.append(float(train_metrics["loss_total"]))
        self.train_loss_re.append(float(train_metrics["loss_re"]))
        self.train_loss_st.append(float(train_metrics["loss_st"]))
        self.train_loss_phase.append(float(train_metrics["loss_phase"]))
        self.val_selection_score.append(float(selection_score))
        self.val_by_budget.append(dict(by_budget) if by_budget is not None else {})
        self.val_standard_metrics.append(dict(standard_metrics) if standard_metrics is not None else {})
        self.epoch_wall_time_seconds.append(float(wall_time))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeepSetsTrainer:
    """动态传感器集合 DeepSets 训练器。"""

    def __init__(self, config: Optional[DeepSetsTrainingConfig] = None) -> None:
        self.config = config or DeepSetsTrainingConfig()

    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: DeepSetsLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float]:
        """执行单个训练 epoch（支持随机子集 8 元组与全池 7 元组 batch）。"""
        model.train()
        sum_total = sum_re = sum_st = sum_phase = 0.0
        n_seen = 0

        for batch in loader:
            if len(batch) == 8:
                uvp, xy_nd, mask, y, _, _, _, _ = batch
                mask_t: Optional[torch.Tensor] = mask.to(device)
            else:
                uvp, xy_nd, y, _, _, _, _ = batch
                mask_t = None

            uvp = uvp.to(device)
            xy_nd = xy_nd.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(uvp, xy_nd, mask_t)
            loss, metrics = criterion(pred, y)
            loss.backward()
            optimizer.step()

            bs = int(uvp.shape[0])
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
    def evaluate_standard(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: DeepSetsLoss,
        re_normalizer: Normalizer,
        st_normalizer: Normalizer,
        device: torch.device,
    ) -> Dict[str, float]:
        """在标准全池 val_loader 上评估物理回归与圆周相位指标。"""
        model.eval()
        sum_total = 0.0
        n_seen = 0
        re_t_all, re_p_all, st_t_all, st_p_all, ph_t_all, ph_p_all = [], [], [], [], [], []

        for batch in loader:
            if len(batch) == 8:
                uvp, xy_nd, mask, y, re_raw, st_raw, phase_true, _ = batch
                mask_t: Optional[torch.Tensor] = mask.to(device)
            else:
                uvp, xy_nd, y, re_raw, st_raw, phase_true, _ = batch
                mask_t = None

            pred = model(uvp.to(device), xy_nd.to(device), mask_t)
            _, metrics = criterion(pred, y.to(device))
            bs = int(uvp.shape[0])
            n_seen += bs
            sum_total += metrics["loss_total"] * bs

            pred_np = pred.detach().cpu().numpy().astype(np.float64)
            re_p_all.append(re_normalizer.inverse_transform(pred_np[:, 0]))
            st_p_all.append(st_normalizer.inverse_transform(pred_np[:, 1]))
            ph_p_all.append(np.mod(np.arctan2(pred_np[:, 2], pred_np[:, 3]), TWO_PI))

            re_t_all.append(re_raw.cpu().numpy().astype(np.float64))
            st_t_all.append(st_raw.cpu().numpy().astype(np.float64))
            ph_t_all.append(phase_true.cpu().numpy().astype(np.float64))

        re_m = compute_regression_metrics(np.concatenate(re_t_all), np.concatenate(re_p_all))
        st_m = compute_regression_metrics(np.concatenate(st_t_all), np.concatenate(st_p_all))
        ph_m = compute_circular_metrics(np.concatenate(ph_t_all), np.concatenate(ph_p_all))
        return {
            "loss_total": sum_total / n_seen,
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
    ) -> Tuple[nn.Module, DeepSetsTrainHistory]:
        """执行 DeepSets 训练、多预算验证面板选优、学习率调度、早停与权重保存。"""
        cfg = self.config
        set_training_seed(cfg.seed)
        if getattr(data.train_loader, "generator", None) is not None:
            data.train_loader.generator.manual_seed(int(cfg.seed))

        device = resolve_device(cfg.device)
        model = model.to(device)

        criterion = DeepSetsLoss(
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

        # 构建多预算冻结验证面板
        cand_count = int(len(data.sensor_indices))
        val_panel: Optional[Dict[str, Any]] = None
        if cfg.use_frozen_panel and data.val_dataset is not None:
            val_panel = build_dynamic_validation_panel(
                candidate_count=cand_count,
                budgets=cfg.panel_budgets,
                layouts_per_budget=cfg.panel_layouts_per_budget,
                seed=cfg.panel_seed,
            )

        out_path = Path(output_dir) if output_dir is not None else None
        if out_path is not None:
            out_path.mkdir(parents=True, exist_ok=True)

        history = DeepSetsTrainHistory()
        best_epoch = -1
        best_score = float("inf")
        best_state_dict = copy.deepcopy(model.state_dict())
        norm_dict = data.normalizers_to_dict()

        collator = getattr(data.train_loader, "collate_fn", None)
        train_counts = (
            list(collator.sensor_counts)
            if isinstance(collator, DeepSetsSubsetCollator)
            else [cand_count]
        )

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.perf_counter()
            curr_lr = float(optimizer.param_groups[0]["lr"])
            train_out = self.train_one_epoch(model, data.train_loader, criterion, optimizer, device)

            by_budget: Optional[Dict[str, Dict[str, float]]] = None
            std_metrics: Optional[Dict[str, float]] = None
            if val_panel is not None and data.val_dataset is not None:
                panel_out = evaluate_dynamic_validation_panel(
                    model=model,
                    val_dataset=data.val_dataset,
                    panel=val_panel,
                    criterion=criterion,
                    re_normalizer=data.re_normalizer,
                    st_normalizer=data.st_normalizer,
                    device=device,
                    chunk_size=cfg.eval_chunk_size,
                )
                score = float(panel_out["val_equal_budget_selection_score"])
                by_budget = panel_out["by_budget"]
                selection_metric = "val_equal_budget_selection_score"
            elif data.val_loader is not None:
                std_metrics = self.evaluate_standard(
                    model=model,
                    loader=data.val_loader,
                    criterion=criterion,
                    re_normalizer=data.re_normalizer,
                    st_normalizer=data.st_normalizer,
                    device=device,
                )
                score = float(std_metrics["loss_total"])
                selection_metric = "val_loss_total"
            else:
                score = float(train_out["loss_total"])
                selection_metric = "train_loss_total"

            wall_time = float(time.perf_counter() - t0)
            history.add_epoch_record(
                epoch=epoch,
                lr=curr_lr,
                train_metrics=train_out,
                selection_score=score,
                by_budget=by_budget,
                standard_metrics=std_metrics,
                wall_time=wall_time,
            )

            is_best, should_stop = stopper.update(score)
            if is_best:
                best_score = score
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

                if out_path is not None and cfg.save_best:
                    save_checkpoint(
                        out_path / cfg.checkpoint_name,
                        {
                            "model_type": "DynamicSetSensorEncoder",
                            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            "model_hparams": dict(model_hparams or {}),
                            "normalizers": norm_dict,
                            "training_sensor_counts": train_counts,
                            "validation_panel": val_panel,
                            "training_config": asdict(cfg),
                            "epoch": epoch,
                            "best_epoch": best_epoch,
                            "selection_metric": selection_metric,
                            "selection_score": best_score,
                            "val_by_budget": by_budget,
                            "val_standard_metrics": std_metrics,
                            "extra_metadata": dict(extra_metadata or {}),
                        },
                    )

            if out_path is not None and cfg.save_last:
                save_checkpoint(
                    out_path / cfg.last_checkpoint_name,
                    {
                        "model_type": "DynamicSetSensorEncoder",
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "model_hparams": dict(model_hparams or {}),
                        "normalizers": norm_dict,
                        "training_sensor_counts": train_counts,
                        "validation_panel": val_panel,
                        "training_config": asdict(cfg),
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "selection_metric": selection_metric,
                        "selection_score": score,
                        "val_by_budget": by_budget,
                        "val_standard_metrics": std_metrics,
                        "extra_metadata": dict(extra_metadata or {}),
                    },
                )

            if cfg.verbose:
                print(
                    f"[DeepSets] Epoch {epoch}/{cfg.num_epochs} | lr={curr_lr:.2e} | "
                    f"train_loss={train_out['loss_total']:.6f} | {selection_metric}={score:.6f} | {wall_time:.2f}s"
                )

            step_lr_scheduler(scheduler, cfg.lr_strategy, score)
            if should_stop:
                if cfg.verbose:
                    print(f"[DeepSets] 早停触发于第 {epoch} 轮 (best_epoch={best_epoch}, best={best_score:.6f})")
                break

        # 恢复验证最优权重并输出训练历史
        model.load_state_dict(best_state_dict)
        history.best_epoch = best_epoch
        history.best_selection_loss = best_score

        if out_path is not None:
            write_json(out_path / "history.json", history.to_dict())

        return model, history


def train_deepsets(
    model: nn.Module,
    data: LoadedEncoderData,
    *,
    output_dir: Optional[str | Path] = None,
    model_hparams: Optional[Mapping[str, Any]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    config: Optional[DeepSetsTrainingConfig] = None,
    **kwargs: Any,
) -> Tuple[nn.Module, DeepSetsTrainHistory]:
    """函数式 DeepSets 训练入口，支持传入 `DeepSetsTrainingConfig` 或显式关键字参数。"""
    cfg = config if config is not None else DeepSetsTrainingConfig(**kwargs)
    return DeepSetsTrainer(cfg).fit(
        model=model,
        data=data,
        output_dir=output_dir,
        model_hparams=model_hparams,
        extra_metadata=extra_metadata,
    )
