"""鱼体绕流模型损失与物理残差模块（纯参数接口，不包含配置文件解析）。

提供三种模型对应的纯参数损失封装，以及 pPINN 所需的 Navier-Stokes 残差计算：
1. `EncoderLoss`（别名 `MLPLoss` / `DeepSetsLoss`）：用于 `SensorMLP` 与 `DynamicSetSensorEncoder`，
   计算 4 维状态 `[Re_norm, St_norm, sinφ, cosφ]` 的加权回归损失与可选单位圆约束；
2. `gradients` / `compute_pde_residuals`：基于 PyTorch 自动微分构建高阶计算图，
   计算二维不可压缩非定常 Navier-Stokes 无量纲控制方程残差 `(r_u, r_v, r_c)`；
3. `PPINNLoss`：组合 `PPINNModel` 的监督流场数据损失与物理残差损失，输出总损失及各分项指标。
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI: float = 2.0 * math.pi

_LOSS_FNS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "mse": F.mse_loss,
    "l1": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
}


class EncoderLoss(nn.Module):
    """表面传感器编码器损失：对齐 4 维输出 `[Re_norm, St_norm, sinφ, cosφ]`。"""

    def __init__(
        self,
        lambda_re: float = 1.0,
        lambda_st: float = 1.0,
        lambda_phase: float = 1.0,
        loss_type: str = "mse",
        lambda_unit_circle: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_re = float(lambda_re)
        self.lambda_st = float(lambda_st)
        self.lambda_phase = float(lambda_phase)
        self.lambda_unit_circle = float(lambda_unit_circle)
        self.criterion = _LOSS_FNS[loss_type.strip().lower()]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算编码器总损失及分项指标字典。"""
        loss_re = self.criterion(pred[:, 0], target[:, 0])
        loss_st = self.criterion(pred[:, 1], target[:, 1])
        loss_phase = self.criterion(pred[:, 2:4], target[:, 2:4])

        loss_total = (
            self.lambda_re * loss_re
            + self.lambda_st * loss_st
            + self.lambda_phase * loss_phase
        )

        metrics: Dict[str, float] = {
            "loss_re": float(loss_re.detach()),
            "loss_st": float(loss_st.detach()),
            "loss_phase": float(loss_phase.detach()),
        }

        if self.lambda_unit_circle > 0.0:
            # 约束相位正余弦预测落在单位圆上：sin^2(φ) + cos^2(φ) = 1
            norm_sq = pred[:, 2] * pred[:, 2] + pred[:, 3] * pred[:, 3]
            loss_unit = torch.mean((norm_sq - 1.0) ** 2)
            loss_total = loss_total + self.lambda_unit_circle * loss_unit
            metrics["loss_unit_circle"] = float(loss_unit.detach())

        metrics["loss_total"] = float(loss_total.detach())
        return loss_total, metrics


# 为固定布局 MLP 与动态 DeepSets 编码器提供显式别名入口
MLPLoss = EncoderLoss
DeepSetsLoss = EncoderLoss


def gradients(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """计算输出对输入的一阶偏导数，保留计算图以支持二阶导与参数反传。"""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def compute_pde_residuals(
    model: nn.Module,
    x_nd: torch.Tensor,
    y_nd: torch.Tensor,
    re_norm: torch.Tensor,
    st_norm: torch.Tensor,
    phi: torch.Tensor,
    re_val: torch.Tensor,
    st_val: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 6 维输入 pPINN 的无量纲二维不可压缩 Navier-Stokes 残差。

    参数:
        model: 接收 `[N, 6]` 特征 `[x*, y*, Re_norm, St_norm, sinφ, cosφ]` 并输出 `[u*, v*, p*]` 的网络。
        x_nd: 无量纲空间坐标 `x/L` `[N]` 或 `[N, 1]`。
        y_nd: 无量纲空间坐标 `y/L` `[N]` 或 `[N, 1]`。
        re_norm: 归一化后的雷诺数特征，用于网络前向输入。
        st_norm: 归一化后的斯特劳哈尔数特征，用于网络前向输入。
        phi: 相位角（弧度 `[0, 2π)`），在内部通过 `sin/cos` 进入计算图以支持对 `φ` 求导。
        re_val: 物理真实雷诺数，用于粘性扩散项 `1 / Re`。
        st_val: 物理真实斯特劳哈尔数，用于非定常项 `2π * St * ∂/∂φ`。

    返回:
        `(r_u, r_v, r_c)`：分别为 X 动量、Y 动量与连续性方程残差 `[N, 1]`。
    """
    x_col = x_nd.reshape(-1, 1).detach().requires_grad_(True)
    y_col = y_nd.reshape(-1, 1).detach().requires_grad_(True)
    phi_col = phi.reshape(-1, 1).detach().requires_grad_(True)

    re_n = re_norm.reshape(-1, 1).expand_as(x_col)
    st_n = st_norm.reshape(-1, 1).expand_as(x_col)
    re_phys = re_val.reshape(-1, 1).expand_as(x_col)
    st_phys = st_val.reshape(-1, 1).expand_as(x_col)

    # 组装 6 维输入并预测无量纲流场 (u*, v*, p*)
    feat = torch.cat(
        [x_col, y_col, re_n, st_n, torch.sin(phi_col), torch.cos(phi_col)],
        dim=-1,
    )
    pred = model(feat)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]

    # 一阶空间导数与相位导数
    u_x = gradients(u, x_col)
    u_y = gradients(u, y_col)
    u_phi = gradients(u, phi_col)

    v_x = gradients(v, x_col)
    v_y = gradients(v, y_col)
    v_phi = gradients(v, phi_col)

    p_x = gradients(p, x_col)
    p_y = gradients(p, y_col)

    # 二阶空间粘性导数
    u_xx = gradients(u_x, x_col)
    u_yy = gradients(u_y, y_col)
    v_xx = gradients(v_x, x_col)
    v_yy = gradients(v_y, y_col)

    # 组装无量纲 Navier-Stokes 方程残差
    inv_re = 1.0 / re_phys
    omega = TWO_PI * st_phys

    r_u = omega * u_phi + u * u_x + v * u_y + p_x - inv_re * (u_xx + u_yy)
    r_v = omega * v_phi + u * v_x + v * v_y + p_y - inv_re * (v_xx + v_yy)
    r_c = u_x + v_y
    return r_u, r_v, r_c


class PPINNLoss(nn.Module):
    """pPINN 组合损失：融合流场监督数据项 `(u*, v*, p*)` 与 PDE 物理残差项 `(r_u, r_v, r_c)`。"""

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_phys: float = 1.0,
        loss_type: str = "mse",
        data_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        phys_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.lambda_data = float(lambda_data)
        self.lambda_phys = float(lambda_phys)
        self.data_weights = tuple(float(w) for w in data_weights)
        self.phys_weights = tuple(float(w) for w in phys_weights)
        self.criterion = _LOSS_FNS[loss_type.strip().lower()]

    def forward(
        self,
        uvp_pred: torch.Tensor,
        uvp_true: torch.Tensor,
        residuals: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算 pPINN 总损失及各通道/各方程分项标量。"""
        w_u, w_v, w_p = self.data_weights
        loss_u = self.criterion(uvp_pred[:, 0], uvp_true[:, 0])
        loss_v = self.criterion(uvp_pred[:, 1], uvp_true[:, 1])
        loss_p = self.criterion(uvp_pred[:, 2], uvp_true[:, 2])
        loss_data = (w_u * loss_u + w_v * loss_v + w_p * loss_p) / 3.0

        if residuals is not None and self.lambda_phys > 0.0:
            r_u, r_v, r_c = residuals
            w_ru, w_rv, w_rc = self.phys_weights
            loss_mom_x = torch.mean(r_u * r_u)
            loss_mom_y = torch.mean(r_v * r_v)
            loss_cont = torch.mean(r_c * r_c)
            loss_phys = w_ru * loss_mom_x + w_rv * loss_mom_y + w_rc * loss_cont
        else:
            zero = uvp_pred.new_zeros(())
            loss_mom_x = zero
            loss_mom_y = zero
            loss_cont = zero
            loss_phys = zero

        loss_total = self.lambda_data * loss_data + self.lambda_phys * loss_phys

        metrics: Dict[str, float] = {
            "loss_total": float(loss_total.detach()),
            "loss_data": float(loss_data.detach()),
            "loss_u": float(loss_u.detach()),
            "loss_v": float(loss_v.detach()),
            "loss_p": float(loss_p.detach()),
            "loss_phys": float(loss_phys.detach()),
            "loss_mom_x": float(loss_mom_x.detach()),
            "loss_mom_y": float(loss_mom_y.detach()),
            "loss_cont": float(loss_cont.detach()),
        }
        return loss_total, metrics


__all__ = [
    "TWO_PI",
    "DeepSetsLoss",
    "EncoderLoss",
    "MLPLoss",
    "PPINNLoss",
    "compute_pde_residuals",
    "gradients",
]
