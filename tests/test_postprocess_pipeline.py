"""后处理模块单元测试与三种推理模式端到端验证。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fish_flow.config import run_from_config
from fish_flow.inference import run_inference_from_config
from fish_flow.postprocess import (
    evaluate_encoder_predictions,
    evaluate_field_predictions,
    run_postprocess,
)


def test_postprocess_metrics_math() -> None:
    """校验编码器圆周相位误差与流场 Gauge 零点对齐 / 尾流区指标数学正确性。"""
    # 1. 相位跨 0/2π 边界误差应为 0.02 rad
    re_t = np.array([14000.0, 16000.0])
    re_p = np.array([14100.0, 15900.0])
    st_t = np.array([2.50, 2.60])
    st_p = np.array([2.50, 2.60])
    ph_t = np.array([0.01, 2.0 * math.pi - 0.01])
    ph_p = np.array([2.0 * math.pi - 0.01, 0.01])
    re_id = np.array([14000, 16000])

    enc_res = evaluate_encoder_predictions(re_t, re_p, st_t, st_p, ph_t, ph_p, re_id=re_id)
    assert enc_res["overall"]["re_mae"] == pytest.approx(100.0)
    assert enc_res["overall"]["st_rmse"] == pytest.approx(0.0)
    assert enc_res["overall"]["phase_cmae"] == pytest.approx(0.02, abs=1e-9)
    assert set(enc_res["by_re"].keys()) == {"14000", "16000"}

    # 2. 流场压强叠加常数偏置 +10.0，Gauge 对齐误差与尾流区误差应为 0
    xy_phys = np.array([[0.3, 0.0], [0.5, 0.02], [0.8, -0.05], [0.05, 0.12]], dtype=np.float64)  # 前 3 点在尾流区
    rng = np.random.default_rng(7)
    uvp_true = rng.normal(0.0, 1.0, size=(3, 4, 3)).astype(np.float32)
    uvp_pred = uvp_true.copy()
    uvp_pred[:, :, 2] += 10.0

    fld_res = evaluate_field_predictions(uvp_true, uvp_pred, xy_phys, L=0.2, re_id=np.array([14000, 14000, 16000]))
    ov = fld_res["overall"]
    assert ov["u_rel_l2"] < 1e-6
    assert ov["p_rel_l2_raw"] > 1.0
    assert ov["p_rel_l2_gauge_sameden"] < 1e-6
    assert ov["wake_n_points"] == 3
    assert ov["wake_p_rel_l2_gauge_sameden"] < 1e-6


def test_postprocess_all_modes_and_cli(tmp_path: Path) -> None:
    """基于真实预处理数据验证 encoder / decoder / cascade 三种后处理模式与对比云图输出。"""
    proc_dir = str(ROOT / "data" / "processed")

    # 1. 快速训练 1 轮 MLP 与 pPINN
    mlp_dir = tmp_path / "train_mlp"
    mlp_dir.mkdir()
    mlp_cfg = json.loads((ROOT / "runs" / "example" / "mlp_uniform_24" / "config.json").read_text(encoding="utf-8"))
    mlp_cfg["processed_dir"] = proc_dir
    mlp_cfg["training"]["num_epochs"] = 1
    mlp_cfg["training"]["verbose"] = False
    (mlp_dir / "config.json").write_text(json.dumps(mlp_cfg, indent=2), encoding="utf-8")
    run_from_config(mlp_dir)

    pp_dir = tmp_path / "train_ppinn"
    pp_dir.mkdir()
    pp_cfg = json.loads((ROOT / "runs" / "example" / "ppinn" / "config.json").read_text(encoding="utf-8"))
    pp_cfg["processed_dir"] = proc_dir
    pp_cfg["model"]["hidden_dims"] = [32, 32]
    pp_cfg["training"]["num_epochs"] = 1
    pp_cfg["training"]["steps_per_epoch"] = 2
    pp_cfg["training"]["batch_size"] = 128
    pp_cfg["training"]["n_collocation"] = 32
    pp_cfg["training"]["verbose"] = False
    (pp_dir / "config.json").write_text(json.dumps(pp_cfg, indent=2), encoding="utf-8")
    run_from_config(pp_dir)

    # 2. 生成三种模式的推理结果（仅使用验证工况 14750）
    for mode, extra in [
        ("encoder", {"encoder_ckpt": str(mlp_dir / "best_mlp.pt")}),
        ("decoder", {"ppinn_ckpt": str(pp_dir / "best_ppinn.pt"), "chunk_size": 16384}),
        ("cascade", {"encoder_ckpt": str(mlp_dir / "best_mlp.pt"), "ppinn_ckpt": str(pp_dir / "best_ppinn.pt"), "chunk_size": 16384}),
    ]:
        out_dir = tmp_path / f"infer_{mode}"
        cfg_file = tmp_path / f"cfg_{mode}.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mode": mode,
                    "processed_dir": proc_dir,
                    "re_list": [14750],
                    "output_dir": str(out_dir),
                    "device": "cpu",
                    **extra,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        run_inference_from_config(cfg_file)

    # 3. 分别执行 encoder / decoder 后处理，并用 CLI 跑 cascade 后处理
    rep_enc = run_postprocess("encoder", tmp_path / "infer_encoder", proc_dir, tmp_path / "post_encoder")
    assert "encoder_metrics" in rep_enc and "field_metrics" not in rep_enc
    assert (tmp_path / "post_encoder" / "metrics.json").exists()
    assert (tmp_path / "post_encoder" / "encoder_comparison.png").exists()

    rep_dec = run_postprocess("decoder", tmp_path / "infer_decoder", proc_dir, tmp_path / "post_decoder", frame_idx=5)
    assert "field_metrics" in rep_dec and "encoder_metrics" not in rep_dec
    assert (tmp_path / "post_decoder" / "metrics.json").exists()
    assert (tmp_path / "post_decoder" / "field_comparison_Re14750_f5.png").exists()

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "postprocess.py"),
            "--mode",
            "cascade",
            "--pred",
            str(tmp_path / "infer_cascade"),
            "--raw",
            proc_dir,
            "--output-dir",
            str(tmp_path / "post_cascade"),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "后处理完成" in proc.stdout
    cas_metrics = json.loads((tmp_path / "post_cascade" / "metrics.json").read_text(encoding="utf-8"))
    assert "encoder_metrics" in cas_metrics and "field_metrics" in cas_metrics
    assert (tmp_path / "post_cascade" / "encoder_comparison.png").exists()
    assert (tmp_path / "post_cascade" / "field_comparison_Re14750_f0.png").exists()
