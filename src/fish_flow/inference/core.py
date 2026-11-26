"""鱼体绕流推理核心模块（纯参数接口，输入输出均为未归一化的原始物理量纲）。

支持三种推理路径：
1. 编码器单独推理（SensorMLP / DynamicSetSensorEncoder）：物理传感器读数 -> 物理 (Re, St, phase)；
2. 解码器单独推理（PPINNModel）：物理参数空间 (Re, St, phase) + 物理坐标 -> 物理流场 (Ux, Uy, p_kin)；
3. 串联推理：物理传感器读数 -> 编码器预测物理状态 -> pPINN 重归一化桥接与分块前向 -> 同时返回编码器与解码器结果。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from fish_flow.data.encoder_loader import Normalizer
from fish_flow.models import DynamicSetSensorEncoder, PPINNModel, SensorMLP
from fish_flow.training.common import TWO_PI, resolve_device


@dataclass
class LoadedEncoderBundle:
    """已加载的编码器模型及其拟合归一化器。"""

    model_type: str
    model: nn.Module
    sensor_indices: np.ndarray
    xy_nd: np.ndarray
    re_normalizer: Normalizer
    st_normalizer: Normalizer
    x_normalizer: Optional[Normalizer] = None
    chan_mean: Optional[np.ndarray] = None
    chan_std: Optional[np.ndarray] = None


@dataclass
class LoadedPPINNBundle:
    """已加载的 pPINN 解码器模型及其参数归一化器。"""

    model: PPINNModel
    L: float
    roi_bounds_nd: Tuple[float, float, float, float]
    re_normalizer: Normalizer
    st_normalizer: Normalizer


@dataclass
class EncoderPrediction:
    """编码器物理量纲预测结果。"""

    re: np.ndarray  # [N] 物理雷诺数
    st: np.ndarray  # [N] 物理斯特劳哈尔数
    phase: np.ndarray  # [N] 物理相位角 (rad, [0, 2π))
    phase_sin: np.ndarray  # [N]
    phase_cos: np.ndarray  # [N]


@dataclass
class DecoderPrediction:
    """解码器物理量纲流场重构结果。"""

    uvp: np.ndarray  # [N, M, 3] 物理量纲流场 [Ux (m/s), Uy (m/s), p_kin (m^2/s^2)]
    xy: np.ndarray  # [M, 2] 物理空间坐标 (m)
    re: np.ndarray  # [N] 输入物理雷诺数
    st: np.ndarray  # [N] 输入物理斯特劳哈尔数
    phase: np.ndarray  # [N] 输入物理相位角 (rad)


@dataclass
class CascadePrediction:
    """串联推理物理量纲结果（同时保留编码器与解码器输出）。"""

    encoder: EncoderPrediction
    decoder: DecoderPrediction


def load_encoder_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device = "auto",
) -> LoadedEncoderBundle:
    """加载最优编码器 checkpoint，还原模型结构与归一化器。"""
    dev = resolve_device(device) if isinstance(device, str) else device
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_type = str(ckpt["model_type"])
    hparams = dict(ckpt["model_hparams"])
    norm = ckpt["normalizers"]

    if model_type == "SensorMLP":
        model: nn.Module = SensorMLP(**hparams)
        x_norm = Normalizer.from_dict(norm["x_normalizer"])
        chan_mean = None
        chan_std = None
    else:
        model = DynamicSetSensorEncoder(**hparams)
        x_norm = None
        chan_mean = np.asarray(norm["chan_mean"], dtype=np.float64)
        chan_std = np.asarray(norm["chan_std"], dtype=np.float64)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()

    return LoadedEncoderBundle(
        model_type=model_type,
        model=model,
        sensor_indices=np.asarray(norm["sensor_indices"], dtype=np.int64),
        xy_nd=np.asarray(norm["xy_nd"], dtype=np.float32),
        re_normalizer=Normalizer.from_dict(norm["re_normalizer"]),
        st_normalizer=Normalizer.from_dict(norm["st_normalizer"]),
        x_normalizer=x_norm,
        chan_mean=chan_mean,
        chan_std=chan_std,
    )


def load_ppinn_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device = "auto",
) -> LoadedPPINNBundle:
    """加载最优 pPINN checkpoint，还原解码器模型与参数归一化器。"""
    dev = resolve_device(device) if isinstance(device, str) else device
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    norm = ckpt["normalizers"]

    model = PPINNModel(**dict(ckpt["model_hparams"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev).eval()

    return LoadedPPINNBundle(
        model=model,
        L=float(norm["L"]),
        roi_bounds_nd=tuple(float(v) for v in norm["roi_bounds_nd"]),  # type: ignore[arg-type]
        re_normalizer=Normalizer.from_dict(norm["re_normalizer"]),
        st_normalizer=Normalizer.from_dict(norm["st_normalizer"]),
    )


@torch.no_grad()
def infer_encoder(
    bundle: LoadedEncoderBundle,
    sensor_uvp: np.ndarray,
    sensor_xy: Optional[np.ndarray] = None,
    *,
    L: float = 0.2,
    batch_size: int = 1024,
    device: str | torch.device = "auto",
) -> EncoderPrediction:
    """编码器前向推理：输入物理传感器读数 `[N, K, 3]`，输出物理 `(Re, St, phase)`。"""
    dev = resolve_device(device) if isinstance(device, str) else device
    model = bundle.model.to(dev).eval()

    uvp_np = np.asarray(sensor_uvp, dtype=np.float64)
    n_samples = int(uvp_np.shape[0])
    preds = []

    if bundle.model_type == "SensorMLP":
        assert bundle.x_normalizer is not None
        x_norm = bundle.x_normalizer.transform(uvp_np.reshape(n_samples, -1)).astype(np.float32)
        for st in range(0, n_samples, batch_size):
            xb = torch.as_tensor(x_norm[st : st + batch_size], dtype=torch.float32, device=dev)
            preds.append(model(xb).cpu())
    else:
        # DeepSets：按训练集通道统计量标准化，并无量纲化传感器坐标
        uvp_norm = ((uvp_np - bundle.chan_mean) / bundle.chan_std).astype(np.float32)
        xy_nd_np = (
            np.asarray(sensor_xy, dtype=np.float32) / float(L)
            if sensor_xy is not None
            else bundle.xy_nd.astype(np.float32, copy=False)
        )
        pos_base = torch.as_tensor(xy_nd_np, dtype=torch.float32, device=dev)
        for st in range(0, n_samples, batch_size):
            ed = min(st + batch_size, n_samples)
            xb = torch.as_tensor(uvp_norm[st:ed], dtype=torch.float32, device=dev)
            pb = pos_base.unsqueeze(0).expand(ed - st, -1, -1)
            preds.append(model(xb, pb, None).cpu())

    pred_np = torch.cat(preds, dim=0).numpy().astype(np.float64)
    re_hat = bundle.re_normalizer.inverse_transform(pred_np[:, 0])
    st_hat = bundle.st_normalizer.inverse_transform(pred_np[:, 1])
    sin_raw = pred_np[:, 2]
    cos_raw = pred_np[:, 3]
    phase_hat = np.mod(np.arctan2(sin_raw, cos_raw), TWO_PI)

    return EncoderPrediction(
        re=re_hat,
        st=st_hat,
        phase=phase_hat,
        phase_sin=sin_raw,
        phase_cos=cos_raw,
    )


@torch.no_grad()
def infer_decoder(
    bundle: LoadedPPINNBundle,
    xy_phys: np.ndarray,
    re_phys: np.ndarray,
    st_phys: np.ndarray,
    phase_phys: np.ndarray,
    *,
    u_inf: Union[float, np.ndarray] = 1.0,
    chunk_size: int = 16384,
    device: str | torch.device = "auto",
) -> DecoderPrediction:
    """解码器分块前向推理：输入物理坐标与物理参数 `(Re, St, phase)`，输出物理流场 `[N, M, 3]`。"""
    dev = resolve_device(device) if isinstance(device, str) else device
    model = bundle.model.to(dev).eval()

    xy_arr = np.asarray(xy_phys, dtype=np.float64)
    re_arr = np.asarray(re_phys, dtype=np.float64).reshape(-1)
    st_arr = np.asarray(st_phys, dtype=np.float64).reshape(-1)
    phase_arr = np.asarray(phase_phys, dtype=np.float64).reshape(-1)
    u_inf_arr = np.broadcast_to(np.asarray(u_inf, dtype=np.float64), (len(re_arr),))

    # 无量纲化空间坐标，并用 pPINN 自身归一化器变换物理参数
    xy_nd_t = torch.as_tensor((xy_arr / bundle.L).astype(np.float32), dtype=torch.float32, device=dev)
    re_norm = bundle.re_normalizer.transform(re_arr).astype(np.float32)
    st_norm = bundle.st_normalizer.transform(st_arr).astype(np.float32)
    sin_phi = np.sin(phase_arr).astype(np.float32)
    cos_phi = np.cos(phase_arr).astype(np.float32)

    n_frames = len(re_arr)
    n_pts = int(xy_arr.shape[0])
    uvp_out = np.empty((n_frames, n_pts, 3), dtype=np.float32)

    feat = torch.empty((n_pts, 6), dtype=torch.float32, device=dev)
    feat[:, 0:2] = xy_nd_t

    for i in range(n_frames):
        feat[:, 2] = float(re_norm[i])
        feat[:, 3] = float(st_norm[i])
        feat[:, 4] = float(sin_phi[i])
        feat[:, 5] = float(cos_phi[i])

        frame_chunks = []
        for st in range(0, n_pts, chunk_size):
            frame_chunks.append(model(feat[st : st + chunk_size]).cpu())
        uvp_nd = torch.cat(frame_chunks, dim=0).numpy()

        # 反无量纲化还原物理速度与运动压强：u = u* U_inf, p_kin = p* U_inf^2
        u0 = np.float32(u_inf_arr[i])
        uvp_nd[:, 0:2] *= u0
        uvp_nd[:, 2] *= u0 * u0
        uvp_out[i] = uvp_nd

    return DecoderPrediction(
        uvp=uvp_out,
        xy=xy_arr,
        re=re_arr,
        st=st_arr,
        phase=phase_arr,
    )


def infer_cascade(
    encoder_bundle: LoadedEncoderBundle,
    ppinn_bundle: LoadedPPINNBundle,
    sensor_uvp: np.ndarray,
    xy_phys: np.ndarray,
    *,
    sensor_xy: Optional[np.ndarray] = None,
    u_inf: Union[float, np.ndarray] = 1.0,
    encoder_batch_size: int = 1024,
    decoder_chunk_size: int = 16384,
    device: str | torch.device = "auto",
) -> CascadePrediction:
    """串联推理：物理传感器读数 -> 编码器物理状态预测 -> pPINN 物理流场重构。"""
    enc_pred = infer_encoder(
        encoder_bundle,
        sensor_uvp,
        sensor_xy=sensor_xy,
        L=ppinn_bundle.L,
        batch_size=encoder_batch_size,
        device=device,
    )
    # 通过物理量 (Re_hat, St_hat, phase_hat) 桥接两套模型各自的归一化器
    dec_pred = infer_decoder(
        ppinn_bundle,
        xy_phys,
        enc_pred.re,
        enc_pred.st,
        enc_pred.phase,
        u_inf=u_inf,
        chunk_size=decoder_chunk_size,
        device=device,
    )
    return CascadePrediction(encoder=enc_pred, decoder=dec_pred)
