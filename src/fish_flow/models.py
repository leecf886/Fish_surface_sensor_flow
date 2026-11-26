"""鱼体绕流神经网络架构定义（MLP、DeepSets、pPINN），仅包含纯网络结构。"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Type

import torch
import torch.nn as nn

_ACTIVATIONS: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "identity": nn.Identity,
}


def _make_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: str,
) -> nn.Sequential:
    """构建 Linear -> Act -> ... -> Linear 全连接网络。"""
    act_cls = _ACTIVATIONS[activation.strip().lower()]
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(d, h), act_cls()])
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class SensorMLP(nn.Module):
    """固定传感器布局 MLP 编码器：展平输入 [B, 3*K] -> [B, 4]。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 128, 128),
        output_dim: int = 4,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = _make_mlp(input_dim, hidden_dims, output_dim, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicSetSensorEncoder(nn.Module):
    """置换不变动态传感器集合编码器（DeepSets）：([B, K, 3], [B, K, 2], mask) -> [B, 4]。"""

    def __init__(
        self,
        in_dim: int = 5,
        embed_dim: int = 128,
        phi_hidden: Sequence[int] = (128, 128),
        rho_hidden: Sequence[int] = (128, 128),
        out_dim: int = 4,
        pool: str = "mean+max",
        activation: str = "relu",
        use_count_feature: bool = True,
        candidate_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.pool = pool.strip().lower()
        self.use_count_feature = use_count_feature
        self.candidate_count = candidate_count

        # 逐点特征映射 phi 与集合聚合映射 rho
        self.phi = _make_mlp(in_dim, phi_hidden, embed_dim, activation)
        pooled_dim = embed_dim * (2 if self.pool == "mean+max" else 1) + (1 if use_count_feature else 0)
        self.rho = _make_mlp(pooled_dim, rho_hidden, out_dim, activation)

    def forward(
        self,
        values: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = values if pos is None else torch.cat([values, pos], dim=-1)
        emb = self.phi(tokens)

        if mask is None:
            cnt = torch.full((emb.shape[0], 1), emb.shape[1], dtype=emb.dtype, device=emb.device)
            mean_pool = emb.mean(dim=1)
            max_pool = emb.max(dim=1).values
            sum_pool = emb.sum(dim=1)
        else:
            valid = mask.unsqueeze(-1)
            valid_f = valid.to(emb.dtype)
            cnt = valid_f.sum(dim=1)
            masked_emb = emb * valid_f
            sum_pool = masked_emb.sum(dim=1)
            mean_pool = sum_pool / cnt
            max_pool = emb.masked_fill(~valid, -torch.inf).max(dim=1).values

        if self.pool == "mean":
            pooled = mean_pool
        elif self.pool == "sum":
            pooled = sum_pool
        elif self.pool == "max":
            pooled = max_pool
        else:
            pooled = torch.cat([mean_pool, max_pool], dim=-1)

        if self.use_count_feature:
            count_feat = cnt / float(self.candidate_count) if self.candidate_count is not None else cnt
            pooled = torch.cat([pooled, count_feat], dim=-1)

        return self.rho(pooled)


class PPINNModel(nn.Module):
    """参数化物理信息神经网络（pPINN）：6 维特征 [x*, y*, Re*, St*, sinφ, cosφ] -> [u*, v*, p*]。"""

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dims: Sequence[int] = (128, 128, 128, 128, 128, 128),
        out_dim: int = 3,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.net = _make_mlp(in_dim, hidden_dims, out_dim, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
