"""上层单配置文件推理编排模块：读取 config.json -> 加载 ckpt 与数据 -> 前向推理 -> 复制原配置并导出 .npz。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from fish_flow.config import resolve_config_path
from fish_flow.data.encoder_loader import select_uniform_arc_indices
from fish_flow.data.ppinn_loader import load_ppinn_mesh
from fish_flow.data.surface_sensors import load_base_surface_mesh
from fish_flow.inference.core import (
    CascadePrediction,
    DecoderPrediction,
    EncoderPrediction,
    infer_cascade,
    infer_decoder,
    infer_encoder,
    load_encoder_checkpoint,
    load_ppinn_checkpoint,
)
from fish_flow.inference.data import (
    load_decoder_inference_inputs,
    load_encoder_inference_inputs,
)

_DEFAULT_OUT_NAMES = {
    "encoder": "encoder_pred.npz",
    "decoder": "decoder_pred.npz",
    "cascade": "cascade_pred.npz",
}


def load_infer_config(
    config_or_dir: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Path, Path]:
    """读取推理 config.json 并确定配置文件路径与输出目录。"""
    cfg_path = resolve_config_path(config_or_dir)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = dict(json.load(f))

    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v

    out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else cfg_path.parent
    return cfg, cfg_path, out_dir


def _resolve_output_npz_path(cfg: Mapping[str, Any], out_dir: Path, mode: str) -> Path:
    """解析输出 .npz 文件路径（支持传入 output_path 或 output_dir + output_name）。"""
    if cfg.get("output_path"):
        return Path(cfg["output_path"])
    name = str(cfg.get("output_name") or _DEFAULT_OUT_NAMES[mode])
    return out_dir / name


def _copy_config_to_output(cfg_path: Path, out_dir: Path) -> None:
    """将原 config.json 原样复制到输出目录（同目录时不重复覆盖）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "config.json"
    if cfg_path.resolve() != dst.resolve():
        shutil.copy2(cfg_path, dst)


def _resolve_infer_sensor_indices(
    processed_dir: str | Path,
    ckpt_indices: np.ndarray,
    cfg: Mapping[str, Any],
) -> np.ndarray:
    """解析推理阶段使用的传感器下标（默认沿用 ckpt 内保存的下标）。"""
    if cfg.get("sensor_indices") is not None:
        return np.asarray(cfg["sensor_indices"], dtype=np.int64)
    mode = cfg.get("sensor_mode")
    if mode is None:
        return ckpt_indices
    mode_s = str(mode).strip().lower()
    mesh = load_base_surface_mesh(processed_dir)
    if mode_s == "all":
        return np.arange(len(mesh.wall_cell_ids), dtype=np.int64)
    if mode_s == "cand60":
        return mesh.cand60_wall_indices
    return select_uniform_arc_indices(mesh.wall_arc, int(cfg.get("n_sensors", 24)))


def run_inference_from_config(
    config_or_dir: str | Path,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[EncoderPrediction | DecoderPrediction | CascadePrediction, Path]:
    """根据单一推理 config.json 执行编码器 / 解码器 / 串联推理，复制原配置并输出 .npz。"""
    cfg, cfg_path, out_dir = load_infer_config(config_or_dir, overrides)
    mode = str(cfg["mode"]).strip().lower()
    processed_dir = cfg.get("processed_dir", "data/processed")
    re_list = [int(r) for r in cfg["re_list"]] if cfg.get("re_list") is not None else None
    periodic_only = bool(cfg.get("periodic_only", True))
    device = str(cfg.get("device", "auto"))

    out_npz = _resolve_output_npz_path(cfg, out_dir, mode)
    _copy_config_to_output(cfg_path, out_npz.parent)

    if mode == "encoder":
        enc_bundle = load_encoder_checkpoint(cfg["encoder_ckpt"], device=device)
        sensor_idx = _resolve_infer_sensor_indices(processed_dir, enc_bundle.sensor_indices, cfg)
        inputs = load_encoder_inference_inputs(
            processed_dir=processed_dir,
            re_list=re_list,
            sensor_indices=sensor_idx,
            periodic_only=periodic_only,
        )
        pred = infer_encoder(
            enc_bundle,
            inputs.sensor_uvp,
            sensor_xy=inputs.sensor_xy,
            L=inputs.L,
            batch_size=int(cfg.get("batch_size", 1024)),
            device=device,
        )
        np.savez(
            out_npz,
            re_pred=pred.re,
            st_pred=pred.st,
            phase_pred=pred.phase,
            phase_sin_pred=pred.phase_sin,
            phase_cos_pred=pred.phase_cos,
            re_id=inputs.re_id,
            sensor_indices=inputs.sensor_indices,
        )
        return pred, out_npz

    if mode == "decoder":
        ppinn_bundle = load_ppinn_checkpoint(cfg["ppinn_ckpt"], device=device)
        if cfg.get("params") is not None:
            # 支持配置中直接传入自定义物理参数空间 (Re, St, phase)
            p_cfg = cfg["params"]
            mesh = load_ppinn_mesh(processed_dir)
            xy_phys = mesh.roi_xy
            re_phys = np.asarray(p_cfg["re"], dtype=np.float64)
            st_phys = np.asarray(p_cfg["st"], dtype=np.float64)
            phase_phys = np.asarray(p_cfg["phase"], dtype=np.float64)
            u_inf = np.broadcast_to(np.asarray(p_cfg.get("u_inf", 1.0), dtype=np.float64), re_phys.shape)
            re_id = np.asarray(re_phys, dtype=np.int64)
        else:
            dec_inputs = load_decoder_inference_inputs(
                processed_dir=processed_dir,
                re_list=re_list,
                periodic_only=periodic_only,
            )
            xy_phys = dec_inputs.xy_phys
            re_phys = dec_inputs.re_phys
            st_phys = dec_inputs.st_phys
            phase_phys = dec_inputs.phase_phys
            u_inf = dec_inputs.u_inf
            re_id = dec_inputs.re_id

        pred = infer_decoder(
            ppinn_bundle,
            xy_phys,
            re_phys,
            st_phys,
            phase_phys,
            u_inf=u_inf,
            chunk_size=int(cfg.get("chunk_size", 16384)),
            device=device,
        )
        np.savez(
            out_npz,
            uvp_pred=pred.uvp,
            xy=pred.xy,
            re_input=pred.re,
            st_input=pred.st,
            phase_input=pred.phase,
            re_id=re_id,
        )
        return pred, out_npz

    # 串联推理：同时输出编码器预测状态与解码器重构物理流场
    enc_bundle = load_encoder_checkpoint(cfg["encoder_ckpt"], device=device)
    ppinn_bundle = load_ppinn_checkpoint(cfg["ppinn_ckpt"], device=device)
    sensor_idx = _resolve_infer_sensor_indices(processed_dir, enc_bundle.sensor_indices, cfg)
    enc_inputs = load_encoder_inference_inputs(
        processed_dir=processed_dir,
        re_list=re_list,
        sensor_indices=sensor_idx,
        periodic_only=periodic_only,
    )
    mesh = load_ppinn_mesh(processed_dir)
    pred = infer_cascade(
        enc_bundle,
        ppinn_bundle,
        enc_inputs.sensor_uvp,
        mesh.roi_xy,
        sensor_xy=enc_inputs.sensor_xy,
        u_inf=enc_inputs.u_inf,
        encoder_batch_size=int(cfg.get("batch_size", 1024)),
        decoder_chunk_size=int(cfg.get("chunk_size", 16384)),
        device=device,
    )
    np.savez(
        out_npz,
        re_pred=pred.encoder.re,
        st_pred=pred.encoder.st,
        phase_pred=pred.encoder.phase,
        phase_sin_pred=pred.encoder.phase_sin,
        phase_cos_pred=pred.encoder.phase_cos,
        uvp_pred=pred.decoder.uvp,
        xy=pred.decoder.xy,
        re_id=enc_inputs.re_id,
        sensor_indices=enc_inputs.sensor_indices,
    )
    return pred, out_npz
