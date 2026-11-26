"""训练模块单元测试与单 config.json 端到端连通性验证。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fish_flow.config import run_from_config
from fish_flow.training.common import (
    EarlyStopper,
    PressureGaugeAccum,
    build_lr_scheduler,
    compute_pde_weight,
    step_lr_scheduler,
)


def test_schedules_and_early_stopping() -> None:
    # PDE 权重调度：constant / linear / cosine / step / exponential
    assert compute_pde_weight(1, 10, 1.0, "constant") == pytest.approx(1.0)
    assert compute_pde_weight(5, 10, 1.0, "constant") == pytest.approx(1.0)

    assert compute_pde_weight(1, 10, 1.0, "linear", {"start_weight": 0.0, "end_weight": 2.0, "start_epoch": 1, "end_epoch": 5}) == pytest.approx(0.0)
    assert compute_pde_weight(3, 10, 1.0, "linear", {"start_weight": 0.0, "end_weight": 2.0, "start_epoch": 1, "end_epoch": 5}) == pytest.approx(1.0)
    assert compute_pde_weight(5, 10, 1.0, "linear", {"start_weight": 0.0, "end_weight": 2.0, "start_epoch": 1, "end_epoch": 5}) == pytest.approx(2.0)

    c_mid = compute_pde_weight(3, 10, 1.0, "cosine", {"start_weight": 0.0, "end_weight": 2.0, "start_epoch": 1, "end_epoch": 5})
    assert c_mid == pytest.approx(1.0)
    assert compute_pde_weight(5, 10, 1.0, "step", {"start_weight": 0.5, "step_size": 2, "gamma": 2.0}) == pytest.approx(2.0)
    assert compute_pde_weight(3, 10, 1.0, "exponential", {"start_weight": 0.5, "gamma": 2.0}) == pytest.approx(2.0)

    # 学习率调度器
    param = torch.nn.Parameter(torch.tensor([1.0]))
    opt_const = torch.optim.SGD([param], lr=1e-2)
    assert build_lr_scheduler(opt_const, "constant", 10) is None

    for strat, kw in [
        ("step", {"step_size": 1, "gamma": 0.5}),
        ("multistep", {"milestones": [1], "gamma": 0.5}),
        ("cosine", {"t_max": 4, "eta_min": 1e-4}),
        ("exponential", {"gamma": 0.5}),
        ("plateau", {"patience": 1, "factor": 0.5}),
    ]:
        opt = torch.optim.SGD([param], lr=1e-2)
        sched = build_lr_scheduler(opt, strat, 4, kw)
        assert sched is not None
        for _ in range(3):
            opt.zero_grad()
            (param * 1.0).sum().backward()
            opt.step()
            step_lr_scheduler(sched, strat, val_metric=1.0)
        assert opt.param_groups[0]["lr"] < 1e-2

    # 早停开关
    es_off = EarlyStopper(enabled=False, patience=2)
    for score in [1.0, 2.0, 3.0, 4.0]:
        _, stop = es_off.update(score)
        assert stop is False

    es_on = EarlyStopper(enabled=True, patience=2, min_delta=0.0)
    assert es_on.update(1.0) == (True, False)
    assert es_on.update(1.1) == (False, False)
    assert es_on.update(1.2) == (False, True)


def test_gauge_invariance() -> None:
    # Gauge 对齐压强误差：常数偏移 +5.0 应当被完全消除
    rng = np.random.default_rng(0)
    p_true = rng.normal(0.0, 1.0, size=1000)
    p_pred = p_true + 5.0
    acc = PressureGaugeAccum()
    mean_e = float(np.mean(p_pred - p_true))
    m2_e = float(np.sum(((p_pred - p_true) - mean_e) ** 2))
    mean_p = float(np.mean(p_true))
    m2_p = float(np.sum((p_true - mean_p) ** 2))
    acc.add_snapshot_welford(mean_e, m2_e, mean_p, m2_p, len(p_true))
    m = acc.metrics()
    assert m["p_rel_l2_gauge_sameden"] < 1e-12
    assert m["p_rel_l2_raw"] > 1.0


def test_single_run_config_mlp_deepsets_ppinn(tmp_path: Path) -> None:
    # 1. MLP：直接传入 runs/mlp_exp 目录，ckpt 与 config.json 存放于同一目录且原 config.json 不被覆盖
    mlp_dir = tmp_path / "runs" / "mlp_exp"
    mlp_dir.mkdir(parents=True)
    mlp_cfg = json.loads((ROOT / "runs" / "mlp" / "config.json").read_text(encoding="utf-8"))
    mlp_cfg["processed_dir"] = str(ROOT / "data" / "processed")
    mlp_cfg["training"]["num_epochs"] = 2
    mlp_cfg["training"]["verbose"] = False
    mlp_cfg["training"]["lr_scheduler"]["strategy"] = "step"
    mlp_cfg["training"]["lr_scheduler"]["params"]["step_size"] = 1
    mlp_raw_text = json.dumps(mlp_cfg, indent=2)
    (mlp_dir / "config.json").write_text(mlp_raw_text, encoding="utf-8")

    _, mlp_hist = run_from_config(mlp_dir)
    assert len(mlp_hist.epochs) == 2
    assert mlp_hist.lr[1] < mlp_hist.lr[0]
    assert (mlp_dir / "config.json").read_text(encoding="utf-8") == mlp_raw_text
    assert (mlp_dir / "history.json").exists()
    mlp_ckpt = torch.load(mlp_dir / "best_mlp.pt", map_location="cpu", weights_only=False)
    assert mlp_ckpt["normalizers"]["st_normalizer"] is not None
    assert mlp_ckpt["normalizers"]["x_normalizer"] is not None

    # 2. DeepSets：504 全池随机子集 [12, 24, 48, 96] + 冻结验证面板
    ds_dir = tmp_path / "runs" / "deepsets_exp"
    ds_dir.mkdir(parents=True)
    ds_cfg = json.loads((ROOT / "runs" / "deepsets" / "config.json").read_text(encoding="utf-8"))
    ds_cfg["processed_dir"] = str(ROOT / "data" / "processed")
    ds_cfg["training"]["num_epochs"] = 2
    ds_cfg["training"]["panel_layouts_per_budget"] = 2
    ds_cfg["training"]["verbose"] = False
    ds_raw_text = json.dumps(ds_cfg, indent=2)
    (ds_dir / "config.json").write_text(ds_raw_text, encoding="utf-8")

    _, ds_hist = run_from_config(ds_dir / "config.json")
    assert len(ds_hist.epochs) == 2
    assert ds_hist.lr[0] == ds_hist.lr[1]
    assert set(ds_hist.val_by_budget[0].keys()) == {"12", "24", "48", "96"}
    assert (ds_dir / "config.json").read_text(encoding="utf-8") == ds_raw_text
    ds_ckpt = torch.load(ds_dir / "best_deepsets.pt", map_location="cpu", weights_only=False)
    assert ds_ckpt["normalizers"]["chan_mean"] is not None
    assert ds_ckpt["normalizers"]["st_normalizer"] is not None
    assert ds_ckpt["validation_panel"] is not None

    # 3. pPINN：5D 配点 + PDE 权重线性调度 + 余弦 LR + 整场 Gauge 验证
    pp_dir = tmp_path / "runs" / "ppinn_exp"
    pp_dir.mkdir(parents=True)
    pp_cfg = json.loads((ROOT / "runs" / "ppinn" / "config.json").read_text(encoding="utf-8"))
    pp_cfg["processed_dir"] = str(ROOT / "data" / "processed")
    pp_cfg["model"]["hidden_dims"] = [32, 32]
    pp_cfg["training"]["num_epochs"] = 2
    pp_cfg["training"]["steps_per_epoch"] = 3
    pp_cfg["training"]["batch_size"] = 256
    pp_cfg["training"]["n_collocation"] = 64
    pp_cfg["training"]["verbose"] = False
    pp_cfg["training"]["pde_weight_schedule"] = {
        "strategy": "linear",
        "params": {"start_weight": 0.1, "end_weight": 1.0, "start_epoch": 1, "end_epoch": 2},
    }
    pp_cfg["training"]["lr_scheduler"] = {
        "strategy": "cosine",
        "params": {"t_max": 2, "eta_min": 1e-5},
    }
    pp_raw_text = json.dumps(pp_cfg, indent=2)
    (pp_dir / "config.json").write_text(pp_raw_text, encoding="utf-8")

    _, pp_hist = run_from_config(pp_dir)
    assert len(pp_hist.epochs) == 2
    assert pp_hist.lambda_phys == pytest.approx([0.1, 1.0])
    assert pp_hist.lr[1] < pp_hist.lr[0]
    assert (pp_dir / "config.json").read_text(encoding="utf-8") == pp_raw_text
    assert (pp_dir / "best_ppinn.pt").exists()
    assert (pp_dir / "last_ppinn.pt").exists()
    assert (pp_dir / "history.json").exists()
