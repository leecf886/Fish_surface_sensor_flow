"""训练过程损失可视化模块单元测试与 CLI 验证。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fish_flow.plot_loss import (
    infer_history_task,
    plot_deepsets_history,
    plot_mlp_history,
    plot_ppinn_history,
    plot_training_history,
)


def test_plot_loss_pure_param_all_tasks(tmp_path: Path) -> None:
    """校验 MLP / DeepSets / pPINN 纯参数绘图接口均可正常输出图像。"""
    ep = [1, 2, 3, 4]
    lr = [1e-3, 8e-4, 5e-4, 2e-4]

    # 1. MLP 纯参数调用
    mlp_png = plot_mlp_history(
        tmp_path / "mlp_loss.png",
        epochs=ep,
        lr=lr,
        train_loss_total=[2.0, 1.2, 0.7, 0.4],
        train_loss_re=[0.8, 0.5, 0.3, 0.15],
        train_loss_st=[0.6, 0.3, 0.2, 0.10],
        train_loss_phase=[0.6, 0.4, 0.2, 0.15],
        val_loss_total=[2.2, 1.3, 0.8, 0.5],
        val_loss_re=[0.9, 0.5, 0.3, 0.2],
        val_loss_st=[0.6, 0.4, 0.2, 0.1],
        val_loss_phase=[0.7, 0.4, 0.3, 0.2],
        val_re_rmse=[500.0, 300.0, 180.0, 90.0],
        val_st_rmse=[0.04, 0.02, 0.01, 0.005],
        val_phase_crmse=[1.5, 1.0, 0.6, 0.2],
        best_epoch=4,
        best_selection_loss=0.5,
    )
    assert mlp_png.exists() and mlp_png.stat().st_size > 1000

    # 2. DeepSets 纯参数调用（含 val_by_budget）
    budgets = [
        {"6": {"val_loss_total": 1.8 - 0.3 * i, "re_rmse": 400.0 - 40 * i, "st_rmse": 0.03 - 0.003 * i, "phase_crmse": 1.4 - 0.2 * i},
         "24": {"val_loss_total": 1.2 - 0.25 * i, "re_rmse": 250.0 - 40 * i, "st_rmse": 0.02 - 0.003 * i, "phase_crmse": 1.0 - 0.2 * i}}
        for i in range(4)
    ]
    ds_png = plot_deepsets_history(
        tmp_path / "ds_loss.png",
        epochs=ep,
        lr=lr,
        train_loss_total=[2.4, 1.5, 0.9, 0.5],
        train_loss_re=[0.9, 0.5, 0.3, 0.15],
        train_loss_st=[0.8, 0.5, 0.3, 0.15],
        train_loss_phase=[0.7, 0.5, 0.3, 0.2],
        val_selection_score=[1.5, 1.225, 0.95, 0.675],
        val_by_budget=budgets,
        best_epoch=4,
        best_selection_loss=0.675,
    )
    assert ds_png.exists() and ds_png.stat().st_size > 1000

    # 3. pPINN 纯参数调用（含稀疏验证 NaN）
    nan = float("nan")
    pp_png = plot_ppinn_history(
        tmp_path / "ppinn_loss.png",
        epochs=ep,
        lr=lr,
        lambda_phys=[0.1, 0.3, 0.6, 1.0],
        train_loss_total=[1.0, 0.7, 0.4, 0.2],
        train_loss_data=[0.8, 0.5, 0.3, 0.15],
        train_loss_phys=[2.0, 0.6, 0.15, 0.05],
        val_data_mse=[0.9, nan, 0.35, 0.18],
        val_data_mse_gauge=[0.8, nan, 0.30, 0.15],
        val_u_rel_l2=[0.5, nan, 0.25, 0.12],
        val_v_rel_l2=[0.6, nan, 0.30, 0.15],
        val_p_rel_l2=[0.9, nan, 0.45, 0.22],
        val_overall_rel_l2_gauge_sameden=[0.66, nan, 0.33, 0.16],
        best_epoch=4,
        best_metric_value=0.18,
    )
    assert pp_png.exists() and pp_png.stat().st_size > 1000


def test_plot_training_history_cli_and_default_dir(tmp_path: Path) -> None:
    """校验通过 history.json 目录自动识别任务并默认输出到同目录 loss_curve.png。"""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    hist = {
        "epochs": [1, 2],
        "lr": [1e-3, 5e-4],
        "train_loss_total": [1.5, 0.8],
        "train_loss_re": [0.5, 0.3],
        "train_loss_st": [0.5, 0.25],
        "train_loss_phase": [0.5, 0.25],
        "val_selection_score": [1.4, 0.75],
        "val_by_budget": [{}, {}],
        "val_standard_metrics": [{"loss_total": 1.4, "re_rmse": 300.0, "st_rmse": 0.02, "phase_crmse": 1.1},
                                 {"loss_total": 0.75, "re_rmse": 150.0, "st_rmse": 0.01, "phase_crmse": 0.6}],
        "best_epoch": 2,
        "best_selection_loss": 0.75,
    }
    assert infer_history_task(hist) == "deepsets"
    (exp_dir / "history.json").write_text(json.dumps(hist), encoding="utf-8")

    with pytest.raises(ValueError):
        plot_training_history(hist)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_loss.py"), "--history", str(exp_dir)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert (exp_dir / "loss_curve.png").exists()
