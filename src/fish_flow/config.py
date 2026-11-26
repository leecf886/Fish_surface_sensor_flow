"""上层单文件配置解析与训练编排模块：直接读取 runs/ 下的单个 config.json 驱动底层纯参数接口。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from fish_flow.data.encoder_loader import load_encoder_training_data, select_uniform_arc_indices
from fish_flow.data.ppinn_loader import load_ppinn_training_data
from fish_flow.data.surface_sensors import load_base_surface_mesh
from fish_flow.models import DynamicSetSensorEncoder, PPINNModel, SensorMLP
from fish_flow.training.deepsets import DeepSetsTrainHistory, DeepSetsTrainingConfig, train_deepsets
from fish_flow.training.mlp import MLPTrainHistory, MLPTrainingConfig, train_mlp
from fish_flow.training.ppinn import PPINNTrainHistory, PPINNTrainingConfig, train_ppinn


def resolve_config_path(config_or_run_dir: str | Path) -> Path:
    """支持传入 `runs/<exp>/config.json` 文件或 `runs/<exp>` 目录，返回 config.json 路径。"""
    p = Path(config_or_run_dir)
    return p / "config.json" if p.is_dir() else p


def load_run_config(
    config_or_run_dir: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Path]:
    """读取单个实验的 config.json，并确定与 config.json 同目录的默认输出路径。"""
    cfg_path = resolve_config_path(config_or_run_dir)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = dict(json.load(f))

    if overrides:
        for k, v in overrides.items():
            if v is None:
                continue
            if k in ("output_dir", "processed_dir"):
                cfg[k] = v
            elif k in ("num_epochs", "seed", "device"):
                cfg.setdefault("training", {})[k] = v

    # 未显式指定 output_dir 时，checkpoint 与 history.json 直接保存在 config.json 所在目录
    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else cfg_path.parent
    return cfg, out_dir


def _resolve_mlp_sensors(processed_dir: str | Path, data_cfg: Mapping[str, Any]) -> Optional[np.ndarray]:
    """根据配置解析 MLP 固定传感器下标。"""
    if data_cfg.get("sensor_indices") is not None:
        return np.asarray(data_cfg["sensor_indices"], dtype=np.int64)
    mode = str(data_cfg.get("sensor_mode", "uniform_arc")).strip().lower()
    if mode == "all":
        return None
    mesh = load_base_surface_mesh(processed_dir)
    if mode == "cand60":
        return mesh.cand60_wall_indices
    if mode == "uniform_arc":
        n_sensors = int(data_cfg.get("n_sensors", 24))
        if not (1 <= n_sensors <= len(mesh.wall_cell_ids)):
            raise ValueError(f"n_sensors={n_sensors} 超出壁面传感器范围 [1, {len(mesh.wall_cell_ids)}]")
        return select_uniform_arc_indices(mesh.wall_arc, n_sensors)
    raise ValueError(f"不支持的 MLP sensor_mode: {mode!r}，可选: ['uniform_arc', 'cand60', 'all']")


def _resolve_deepsets_pool(processed_dir: str | Path, data_cfg: Mapping[str, Any]) -> Optional[np.ndarray]:
    """根据配置解析 DeepSets 候选传感器总池下标。"""
    if data_cfg.get("sensor_indices") is not None:
        return np.asarray(data_cfg["sensor_indices"], dtype=np.int64)
    pool = str(data_cfg.get("sensor_pool", "all")).strip().lower()
    if pool == "all":
        return None
    if pool == "cand60":
        return load_base_surface_mesh(processed_dir).cand60_wall_indices
    raise ValueError(f"不支持的 DeepSets sensor_pool: {pool!r}，可选: ['all', 'cand60']")


def _validate_re_splits(cfg: Mapping[str, Any]) -> Tuple[list[int], Optional[list[int]]]:
    """校验 train_re / val_re / test_re 划分非空且严格两两互斥，防止数据泄漏。"""
    train_re = [int(r) for r in cfg["train_re"]]
    if not train_re:
        raise ValueError("train_re 不能为空")
    val_re = [int(r) for r in cfg["val_re"]] if cfg.get("val_re") is not None else None
    test_re = [int(r) for r in cfg["test_re"]] if cfg.get("test_re") is not None else None

    train_set = set(train_re)
    val_set = set(val_re or [])
    test_set = set(test_re or [])
    if train_set & val_set:
        raise ValueError(f"train_re 与 val_re 存在重叠工况: {sorted(train_set & val_set)}")
    if train_set & test_set:
        raise ValueError(f"train_re 与 test_re 存在重叠工况（禁止数据泄漏）: {sorted(train_set & test_set)}")
    if val_set & test_set:
        raise ValueError(f"val_re 与 test_re 存在重叠工况（禁止数据泄漏）: {sorted(val_set & test_set)}")
    return train_re, val_re


def run_mlp_from_config(
    config_or_run_dir: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[SensorMLP, MLPTrainHistory]:
    """读取单个 config.json 并执行 MLP 训练。"""
    cfg, out_dir = load_run_config(config_or_run_dir, overrides)
    processed_dir = cfg.get("processed_dir", "data/processed")
    train_re, val_re = _validate_re_splits(cfg)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})
    lr_cfg = tr_cfg.get("lr_scheduler", {})
    es_cfg = tr_cfg.get("early_stopping", {})

    sensor_indices = _resolve_mlp_sensors(processed_dir, data_cfg)
    data = load_encoder_training_data(
        processed_dir=processed_dir,
        train_re=train_re,
        val_re=val_re,
        encoder_type="mlp",
        sensor_indices=sensor_indices,
        batch_size=int(data_cfg.get("batch_size", 256)),
        seed=int(tr_cfg.get("seed", 42)),
        periodic_only=bool(data_cfg.get("periodic_only", True)),
        x_norm_method=str(data_cfg.get("x_norm_method", "standard")),
        target_norm_method=str(data_cfg.get("target_norm_method", "standard")),
    )

    model_hparams = {
        "input_dim": int(len(data.sensor_indices) * 3),
        "hidden_dims": tuple(int(h) for h in model_cfg.get("hidden_dims", (128, 128, 128))),
        "output_dim": int(model_cfg.get("output_dim", 4)),
        "activation": str(model_cfg.get("activation", "relu")),
    }
    model = SensorMLP(**model_hparams)

    train_cfg = MLPTrainingConfig(
        learning_rate=float(tr_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        num_epochs=int(tr_cfg.get("num_epochs", 200)),
        lambda_re=float(tr_cfg.get("lambda_re", 1.0)),
        lambda_st=float(tr_cfg.get("lambda_st", 1.0)),
        lambda_phase=float(tr_cfg.get("lambda_phase", 1.0)),
        loss_type=str(tr_cfg.get("loss_type", "mse")),
        lambda_unit_circle=float(tr_cfg.get("lambda_unit_circle", 0.0)),
        lr_strategy=str(lr_cfg.get("strategy", "constant")),
        lr_params=dict(lr_cfg.get("params", {})),
        early_stopping_enabled=bool(es_cfg.get("enabled", False)),
        patience=int(es_cfg.get("patience", 20)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
        seed=int(tr_cfg.get("seed", 42)),
        device=str(tr_cfg.get("device", "auto")),
        save_best=bool(tr_cfg.get("save_best", True)),
        save_last=bool(tr_cfg.get("save_last", False)),
        checkpoint_name=str(tr_cfg.get("checkpoint_name", "best_mlp.pt")),
        last_checkpoint_name=str(tr_cfg.get("last_checkpoint_name", "last_mlp.pt")),
        verbose=bool(tr_cfg.get("verbose", True)),
    )
    trained_model, history = train_mlp(
        model=model,
        data=data,
        output_dir=out_dir,
        model_hparams=model_hparams,
        config=train_cfg,
    )
    return trained_model, history  # type: ignore[return-value]


def run_deepsets_from_config(
    config_or_run_dir: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[DynamicSetSensorEncoder, DeepSetsTrainHistory]:
    """读取单个 config.json 并执行 DeepSets 训练。"""
    cfg, out_dir = load_run_config(config_or_run_dir, overrides)
    processed_dir = cfg.get("processed_dir", "data/processed")
    train_re, val_re = _validate_re_splits(cfg)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})
    lr_cfg = tr_cfg.get("lr_scheduler", {})
    es_cfg = tr_cfg.get("early_stopping", {})

    sensor_indices = _resolve_deepsets_pool(processed_dir, data_cfg)
    sensor_counts = tuple(int(c) for c in data_cfg.get("training_sensor_counts", (12, 24, 48, 96)))
    data = load_encoder_training_data(
        processed_dir=processed_dir,
        train_re=train_re,
        val_re=val_re,
        encoder_type="deepsets",
        sensor_indices=sensor_indices,
        batch_size=int(data_cfg.get("batch_size", 256)),
        seed=int(tr_cfg.get("seed", 42)),
        periodic_only=bool(data_cfg.get("periodic_only", True)),
        target_norm_method=str(data_cfg.get("target_norm_method", "standard")),
        random_subset=bool(data_cfg.get("random_subset", True)),
        training_sensor_counts=sensor_counts,
        token_dropout_prob=float(data_cfg.get("token_dropout_prob", 0.0)),
        sensor_noise_rel_std=float(data_cfg.get("sensor_noise_rel_std", 0.0)),
    )

    model_hparams = {
        "in_dim": int(model_cfg.get("in_dim", 5)),
        "embed_dim": int(model_cfg.get("embed_dim", 128)),
        "phi_hidden": tuple(int(h) for h in model_cfg.get("phi_hidden", (128, 128))),
        "rho_hidden": tuple(int(h) for h in model_cfg.get("rho_hidden", (128, 128))),
        "out_dim": int(model_cfg.get("out_dim", 4)),
        "pool": str(model_cfg.get("pool", "mean+max")),
        "activation": str(model_cfg.get("activation", "relu")),
        "use_count_feature": bool(model_cfg.get("use_count_feature", True)),
        "candidate_count": int(len(data.sensor_indices)),
    }
    model = DynamicSetSensorEncoder(**model_hparams)

    panel_budgets = tuple(int(b) for b in tr_cfg.get("panel_budgets", sensor_counts))
    train_cfg = DeepSetsTrainingConfig(
        learning_rate=float(tr_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        num_epochs=int(tr_cfg.get("num_epochs", 200)),
        lambda_re=float(tr_cfg.get("lambda_re", 1.0)),
        lambda_st=float(tr_cfg.get("lambda_st", 1.0)),
        lambda_phase=float(tr_cfg.get("lambda_phase", 1.0)),
        loss_type=str(tr_cfg.get("loss_type", "mse")),
        lambda_unit_circle=float(tr_cfg.get("lambda_unit_circle", 0.0)),
        use_frozen_panel=bool(tr_cfg.get("use_frozen_panel", True)),
        panel_budgets=panel_budgets,
        panel_layouts_per_budget=int(tr_cfg.get("panel_layouts_per_budget", 8)),
        panel_seed=int(tr_cfg.get("panel_seed", 400042)),
        eval_chunk_size=int(tr_cfg.get("eval_chunk_size", 8192)),
        lr_strategy=str(lr_cfg.get("strategy", "constant")),
        lr_params=dict(lr_cfg.get("params", {})),
        early_stopping_enabled=bool(es_cfg.get("enabled", False)),
        patience=int(es_cfg.get("patience", 20)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
        seed=int(tr_cfg.get("seed", 42)),
        device=str(tr_cfg.get("device", "auto")),
        save_best=bool(tr_cfg.get("save_best", True)),
        save_last=bool(tr_cfg.get("save_last", False)),
        checkpoint_name=str(tr_cfg.get("checkpoint_name", "best_deepsets.pt")),
        last_checkpoint_name=str(tr_cfg.get("last_checkpoint_name", "last_deepsets.pt")),
        verbose=bool(tr_cfg.get("verbose", True)),
    )
    trained_model, history = train_deepsets(
        model=model,
        data=data,
        output_dir=out_dir,
        model_hparams=model_hparams,
        config=train_cfg,
    )
    return trained_model, history  # type: ignore[return-value]


def run_ppinn_from_config(
    config_or_run_dir: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[PPINNModel, PPINNTrainHistory]:
    """读取单个 config.json 并执行 pPINN 训练。"""
    cfg, out_dir = load_run_config(config_or_run_dir, overrides)
    processed_dir = cfg.get("processed_dir", "data/processed")
    train_re, val_re = _validate_re_splits(cfg)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})
    pde_cfg = tr_cfg.get("pde_weight_schedule", {})
    lr_cfg = tr_cfg.get("lr_scheduler", {})
    es_cfg = tr_cfg.get("early_stopping", {})

    data = load_ppinn_training_data(
        processed_dir=processed_dir,
        train_re=train_re,
        val_re=val_re,
        periodic_only=bool(data_cfg.get("periodic_only", True)),
        param_norm_method=str(data_cfg.get("param_norm_method", "standard")),
        seed=int(tr_cfg.get("seed", 42)),
    )

    model_hparams = {
        "in_dim": int(model_cfg.get("in_dim", 6)),
        "hidden_dims": tuple(int(h) for h in model_cfg.get("hidden_dims", (128,) * 6)),
        "out_dim": int(model_cfg.get("out_dim", 3)),
        "activation": str(model_cfg.get("activation", "tanh")),
    }
    model = PPINNModel(**model_hparams)

    train_cfg = PPINNTrainingConfig(
        batch_size=int(tr_cfg.get("batch_size", 2048)),
        steps_per_epoch=int(tr_cfg.get("steps_per_epoch", 200)),
        num_epochs=int(tr_cfg.get("num_epochs", 200)),
        learning_rate=float(tr_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(tr_cfg.get("weight_decay", 0.0)),
        lambda_data=float(tr_cfg.get("lambda_data", 1.0)),
        lambda_phys=float(tr_cfg.get("lambda_phys", 1.0)),
        loss_type=str(tr_cfg.get("loss_type", "mse")),
        data_weights=tuple(float(w) for w in tr_cfg.get("data_weights", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
        phys_weights=tuple(float(w) for w in tr_cfg.get("phys_weights", (1.0, 1.0, 1.0))),  # type: ignore[arg-type]
        n_collocation=int(tr_cfg.get("n_collocation", 4096)),
        collocation_method=str(tr_cfg.get("collocation_method", "latin_hypercube")),
        collocation_re_st_mode=str(tr_cfg.get("collocation_re_st_mode", "continuous_random")),
        exclude_solid_collocation=bool(tr_cfg.get("exclude_solid_collocation", True)),
        resample_collocation_mode=str(tr_cfg.get("resample_collocation_mode", "fixed")),
        pde_weight_strategy=str(pde_cfg.get("strategy", "constant")),
        pde_weight_params=dict(pde_cfg.get("params", {})),
        lr_strategy=str(lr_cfg.get("strategy", "constant")),
        lr_params=dict(lr_cfg.get("params", {})),
        early_stopping_enabled=bool(es_cfg.get("enabled", False)),
        patience=int(es_cfg.get("patience", 20)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
        checkpoint_selection_metric=str(tr_cfg.get("checkpoint_selection_metric", "data_mse")),
        eval_chunk_size=int(tr_cfg.get("eval_chunk_size", 16384)),
        val_every_n_epochs=int(tr_cfg.get("val_every_n_epochs", 1)),
        seed=int(tr_cfg.get("seed", 42)),
        device=str(tr_cfg.get("device", "auto")),
        save_best=bool(tr_cfg.get("save_best", True)),
        save_last=bool(tr_cfg.get("save_last", True)),
        best_checkpoint_name=str(tr_cfg.get("best_checkpoint_name", "best_ppinn.pt")),
        last_checkpoint_name=str(tr_cfg.get("last_checkpoint_name", "last_ppinn.pt")),
        verbose=bool(tr_cfg.get("verbose", True)),
    )
    trained_model, history = train_ppinn(
        model=model,
        data=data,
        output_dir=out_dir,
        model_hparams=model_hparams,
        config=train_cfg,
    )
    return trained_model, history  # type: ignore[return-value]


def run_from_config(
    config_or_run_dir: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Any]:
    """根据单个 config.json 中的 `task` 字段统一分发到 MLP / DeepSets / pPINN 训练。"""
    cfg, _ = load_run_config(config_or_run_dir, overrides)
    task = str(cfg["task"]).strip().lower()
    if task == "mlp":
        return run_mlp_from_config(config_or_run_dir, overrides=overrides)
    if task == "deepsets":
        return run_deepsets_from_config(config_or_run_dir, overrides=overrides)
    if task == "ppinn":
        return run_ppinn_from_config(config_or_run_dir, overrides=overrides)
    raise ValueError(f"不支持的训练 task: {cfg['task']!r}，可选: ['mlp', 'deepsets', 'ppinn']")
