"""推理模块单元测试与单 config.json 端到端闭环验证。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fish_flow.config import run_from_config
from fish_flow.data.encoder_loader import Normalizer
from fish_flow.inference import (
    infer_cascade,
    infer_decoder,
    infer_encoder,
    load_encoder_checkpoint,
    load_ppinn_checkpoint,
    run_inference_from_config,
)
from fish_flow.models import DynamicSetSensorEncoder, PPINNModel, SensorMLP
from fish_flow.training.common import save_checkpoint


def test_inference_normalization_and_cascade_bridge(tmp_path: Path) -> None:
    """校验物理量纲输入输出、归一化/反归一化及串联双套归一化器桥接一致性。"""
    torch.manual_seed(0)

    # 1. 构造具有不同统计量的编码器与 pPINN ckpt，验证物理量桥接
    enc_re_norm = Normalizer("standard").fit(np.array([13500.0, 17500.0]))
    enc_st_norm = Normalizer("standard").fit(np.array([2.40, 2.70]))
    x_norm = Normalizer("standard").fit(np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]]))
    mlp = SensorMLP(input_dim=6, hidden_dims=(8,), output_dim=4, activation="tanh")
    enc_ckpt = tmp_path / "best_mlp.pt"
    save_checkpoint(
        enc_ckpt,
        {
            "model_type": "SensorMLP",
            "model_state_dict": mlp.state_dict(),
            "model_hparams": {"input_dim": 6, "hidden_dims": (8,), "output_dim": 4, "activation": "tanh"},
            "normalizers": {
                "encoder_type": "mlp",
                "sensor_indices": [0, 1],
                "xy_nd": [[0.1, 0.0], [0.5, 0.0]],
                "re_normalizer": enc_re_norm.to_dict(),
                "st_normalizer": enc_st_norm.to_dict(),
                "x_normalizer": x_norm.to_dict(),
            },
        },
    )

    dec_re_norm = Normalizer("standard").fit(np.array([14000.0, 16000.0]))
    dec_st_norm = Normalizer("standard").fit(np.array([2.50, 2.60]))
    ppinn = PPINNModel(in_dim=6, hidden_dims=(8,), out_dim=3, activation="tanh")
    dec_ckpt = tmp_path / "best_ppinn.pt"
    save_checkpoint(
        dec_ckpt,
        {
            "model_type": "PPINNModel",
            "model_state_dict": ppinn.state_dict(),
            "model_hparams": {"in_dim": 6, "hidden_dims": (8,), "out_dim": 3, "activation": "tanh"},
            "normalizers": {
                "L": 0.2,
                "roi_bounds_nd": [-0.5, 4.5, -0.8, 0.8],
                "re_normalizer": dec_re_norm.to_dict(),
                "st_normalizer": dec_st_norm.to_dict(),
            },
        },
    )

    enc_bundle = load_encoder_checkpoint(enc_ckpt, device="cpu")
    dec_bundle = load_ppinn_checkpoint(dec_ckpt, device="cpu")

    sensor_uvp = np.array([[[1.5, 3.0, 4.5], [6.0, 7.5, 9.0]]], dtype=np.float64)
    xy_phys = np.array([[0.1, 0.04], [0.4, -0.04], [0.8, 0.0]], dtype=np.float64)
    u_inf = 2.0

    # 手工计算编码器与解码器期望输出
    with torch.no_grad():
        x_in = torch.as_tensor(x_norm.transform(sensor_uvp.reshape(1, -1)), dtype=torch.float32)
        raw_enc = mlp(x_in).numpy()[0]
    exp_re = float(enc_re_norm.inverse_transform(np.array([raw_enc[0]]))[0])
    exp_st = float(enc_st_norm.inverse_transform(np.array([raw_enc[1]]))[0])
    exp_phase = float(np.mod(np.arctan2(raw_enc[2], raw_enc[3]), 2.0 * math.pi))

    enc_pred = infer_encoder(enc_bundle, sensor_uvp, device="cpu")
    assert enc_pred.re[0] == pytest.approx(exp_re, rel=1e-5)
    assert enc_pred.st[0] == pytest.approx(exp_st, rel=1e-5)
    assert enc_pred.phase[0] == pytest.approx(exp_phase, rel=1e-5)

    with torch.no_grad():
        feat = np.column_stack(
            [
                xy_phys / 0.2,
                np.full(3, dec_re_norm.transform(np.array([exp_re]))[0]),
                np.full(3, dec_st_norm.transform(np.array([exp_st]))[0]),
                np.full(3, math.sin(exp_phase)),
                np.full(3, math.cos(exp_phase)),
            ]
        ).astype(np.float32)
        raw_dec = ppinn(torch.from_numpy(feat)).numpy()
    raw_dec[:, 0:2] *= u_inf
    raw_dec[:, 2] *= u_inf * u_inf

    dec_pred = infer_decoder(
        dec_bundle,
        xy_phys,
        np.array([exp_re]),
        np.array([exp_st]),
        np.array([exp_phase]),
        u_inf=u_inf,
        chunk_size=2,
        device="cpu",
    )
    np.testing.assert_allclose(dec_pred.uvp[0], raw_dec, rtol=1e-5, atol=1e-6)

    cas_pred = infer_cascade(
        enc_bundle,
        dec_bundle,
        sensor_uvp,
        xy_phys,
        u_inf=u_inf,
        decoder_chunk_size=2,
        device="cpu",
    )
    np.testing.assert_allclose(cas_pred.encoder.re, enc_pred.re, rtol=1e-6)
    np.testing.assert_allclose(cas_pred.decoder.uvp, dec_pred.uvp, rtol=1e-5, atol=1e-6)

    # 2. 校验 DeepSets 物理坐标无量纲化路径
    ds = DynamicSetSensorEncoder(in_dim=5, embed_dim=8, phi_hidden=(8,), rho_hidden=(8,), out_dim=4, candidate_count=4)
    ds_ckpt = tmp_path / "best_deepsets.pt"
    save_checkpoint(
        ds_ckpt,
        {
            "model_type": "DynamicSetSensorEncoder",
            "model_state_dict": ds.state_dict(),
            "model_hparams": {
                "in_dim": 5,
                "embed_dim": 8,
                "phi_hidden": (8,),
                "rho_hidden": (8,),
                "out_dim": 4,
                "pool": "mean+max",
                "activation": "relu",
                "use_count_feature": True,
                "candidate_count": 4,
            },
            "normalizers": {
                "encoder_type": "deepsets",
                "sensor_indices": [0, 1, 2, 3],
                "xy_nd": [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
                "re_normalizer": enc_re_norm.to_dict(),
                "st_normalizer": enc_st_norm.to_dict(),
                "chan_mean": [0.5, 0.1, -0.2],
                "chan_std": [0.2, 0.3, 0.4],
            },
        },
    )
    ds_bundle = load_encoder_checkpoint(ds_ckpt, device="cpu")
    sub_uvp = np.ones((2, 2, 3), dtype=np.float64)
    sub_xy = np.array([[0.02, 0.0], [0.06, 0.0]], dtype=np.float64)
    ds_pred = infer_encoder(ds_bundle, sub_uvp, sensor_xy=sub_xy, L=0.2, device="cpu")
    assert ds_pred.re.shape == (2,)
    assert np.all(np.isfinite(ds_pred.phase))


def test_end_to_end_infer_configs_and_cli(tmp_path: Path) -> None:
    """基于真实预处理数据完成 1 轮训练，并端到端验证 encoder / decoder / cascade 与 CLI 入口。"""
    proc_dir = str(ROOT / "data" / "processed")

    # 1. 训练极简 MLP / DeepSets / pPINN 获取真实格式 ckpt
    mlp_train_dir = tmp_path / "train_mlp"
    mlp_train_dir.mkdir()
    mlp_cfg = json.loads((ROOT / "runs" / "example" / "mlp" / "config.json").read_text(encoding="utf-8"))
    mlp_cfg["processed_dir"] = proc_dir
    mlp_cfg["training"]["num_epochs"] = 1
    mlp_cfg["training"]["verbose"] = False
    (mlp_train_dir / "config.json").write_text(json.dumps(mlp_cfg, indent=2), encoding="utf-8")
    run_from_config(mlp_train_dir)

    ds_train_dir = tmp_path / "train_ds"
    ds_train_dir.mkdir()
    ds_cfg = json.loads((ROOT / "runs" / "example" / "deepsets" / "config.json").read_text(encoding="utf-8"))
    ds_cfg["processed_dir"] = proc_dir
    ds_cfg["training"]["num_epochs"] = 1
    ds_cfg["training"]["panel_layouts_per_budget"] = 1
    ds_cfg["training"]["verbose"] = False
    (ds_train_dir / "config.json").write_text(json.dumps(ds_cfg, indent=2), encoding="utf-8")
    run_from_config(ds_train_dir)

    pp_train_dir = tmp_path / "train_ppinn"
    pp_train_dir.mkdir()
    pp_cfg = json.loads((ROOT / "runs" / "example" / "ppinn" / "config.json").read_text(encoding="utf-8"))
    pp_cfg["processed_dir"] = proc_dir
    pp_cfg["model"]["hidden_dims"] = [32, 32]
    pp_cfg["training"]["num_epochs"] = 1
    pp_cfg["training"]["steps_per_epoch"] = 2
    pp_cfg["training"]["batch_size"] = 128
    pp_cfg["training"]["n_collocation"] = 32
    pp_cfg["training"]["verbose"] = False
    (pp_train_dir / "config.json").write_text(json.dumps(pp_cfg, indent=2), encoding="utf-8")
    run_from_config(pp_train_dir)

    # 2. 编码器单独推理（MLP，验证工况 14750）
    enc_cfg_path = tmp_path / "cfg_encoder.json"
    enc_out_dir = tmp_path / "out_encoder"
    enc_cfg_text = json.dumps(
        {
            "mode": "encoder",
            "encoder_ckpt": str(mlp_train_dir / "best_mlp.pt"),
            "processed_dir": proc_dir,
            "re_list": [14750],
            "output_dir": str(enc_out_dir),
            "output_name": "encoder_pred.npz",
            "device": "cpu",
        },
        indent=2,
    )
    enc_cfg_path.write_text(enc_cfg_text, encoding="utf-8")
    enc_pred, enc_npz = run_inference_from_config(enc_cfg_path)
    assert (enc_out_dir / "config.json").read_text(encoding="utf-8") == enc_cfg_text
    assert enc_npz.exists()
    with np.load(enc_npz) as z:
        assert set(z.files) >= {"re_pred", "st_pred", "phase_pred", "phase_sin_pred", "phase_cos_pred", "re_id"}
        assert z["re_pred"].shape == (80,)
        assert np.all(np.isfinite(z["re_pred"]))

    # 3. 解码器单独推理（同时验证预处理工况 re_list 与自定义 params 两条路径）
    dec_re_cfg_path = tmp_path / "cfg_decoder_re.json"
    dec_re_out_dir = tmp_path / "out_decoder_re"
    dec_re_cfg_path.write_text(
        json.dumps(
            {
                "mode": "decoder",
                "ppinn_ckpt": str(pp_train_dir / "best_ppinn.pt"),
                "processed_dir": proc_dir,
                "re_list": [14750],
                "output_dir": str(dec_re_out_dir),
                "output_name": "decoder_pred.npz",
                "chunk_size": 16384,
                "device": "cpu",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _, dec_re_npz = run_inference_from_config(dec_re_cfg_path)
    with np.load(dec_re_npz) as z:
        assert z["uvp_pred"].shape == (80, 32084, 3)
        assert z["re_input"].shape == (80,)
        assert np.all(np.isfinite(z["uvp_pred"]))

    dec_cfg_path = tmp_path / "cfg_decoder.json"
    dec_out_dir = tmp_path / "out_decoder"
    dec_cfg_text = json.dumps(
        {
            "mode": "decoder",
            "ppinn_ckpt": str(pp_train_dir / "best_ppinn.pt"),
            "processed_dir": proc_dir,
            "params": {"re": [14750.0, 16750.0], "st": [2.52, 2.61], "phase": [0.0, 1.57], "u_inf": 1.0},
            "output_dir": str(dec_out_dir),
            "output_name": "decoder_pred.npz",
            "chunk_size": 8192,
            "device": "cpu",
        },
        indent=2,
    )
    dec_cfg_path.write_text(dec_cfg_text, encoding="utf-8")
    _, dec_npz = run_inference_from_config(dec_cfg_path)
    assert (dec_out_dir / "config.json").read_text(encoding="utf-8") == dec_cfg_text
    with np.load(dec_npz) as z:
        assert z["uvp_pred"].shape == (2, 32084, 3)
        assert z["xy"].shape == (32084, 2)
        assert np.all(np.isfinite(z["uvp_pred"]))

    # 4. 串联推理通过 CLI scripts/infer.py 调用（DeepSets 24 传感器子集 + pPINN，验证工况 14750）
    cas_cfg_path = tmp_path / "cfg_cascade.json"
    cas_out_dir = tmp_path / "out_cascade"
    cas_cfg_text = json.dumps(
        {
            "mode": "cascade",
            "encoder_ckpt": str(ds_train_dir / "best_deepsets.pt"),
            "ppinn_ckpt": str(pp_train_dir / "best_ppinn.pt"),
            "processed_dir": proc_dir,
            "re_list": [14750],
            "sensor_mode": "uniform_arc",
            "n_sensors": 24,
            "output_dir": str(cas_out_dir),
            "output_name": "cascade_pred.npz",
            "chunk_size": 16384,
            "device": "cpu",
        },
        indent=2,
    )
    cas_cfg_path.write_text(cas_cfg_text, encoding="utf-8")

    env = {"PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "infer.py"), "--config", str(cas_cfg_path)],
        cwd=str(ROOT),
        env={**dict(**__import__("os").environ), **env},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "串联推理完成" in proc.stdout
    assert (cas_out_dir / "config.json").read_text(encoding="utf-8") == cas_cfg_text
    with np.load(cas_out_dir / "cascade_pred.npz") as z:
        assert set(z.files) >= {"re_pred", "st_pred", "phase_pred", "uvp_pred", "xy", "re_id", "sensor_indices"}
        assert z["re_pred"].shape == (80,)
        assert z["uvp_pred"].shape == (80, 32084, 3)
        assert z["sensor_indices"].shape == (24,)
        assert np.all(np.isfinite(z["uvp_pred"]))
