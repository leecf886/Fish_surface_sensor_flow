"""推理结果后处理模块：加载推理输出(.npz)与原始工况(.h5)，计算编码器/流场物理指标并绘制对比云图。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from fish_flow.collocation import fish_half_width_nd, points_inside_fish
from fish_flow.data.ppinn_loader import load_ppinn_mesh
from fish_flow.data.surface_sensors import _discover_re_files, load_base_surface_cases
from fish_flow.training.common import (
    PressureGaugeAccum,
    compute_circular_metrics,
    compute_regression_metrics,
    write_json,
)

_DEFAULT_PRED_NAMES = {
    "encoder": "encoder_pred.npz",
    "decoder": "decoder_pred.npz",
    "cascade": "cascade_pred.npz",
}


def _resolve_pred_npz(pred_path: str | Path, mode: str) -> Path:
    """支持传入 .npz 文件或推理输出目录，返回对应 .npz 路径。"""
    p = Path(pred_path)
    return p / _DEFAULT_PRED_NAMES[mode] if p.is_dir() else p


def _read_periodic_flag(pred_npz: Path, periodic_only: Optional[bool]) -> bool:
    """优先使用显式参数，否则从同目录复制的 config.json 读取 periodic_only。"""
    if periodic_only is not None:
        return bool(periodic_only)
    cfg_file = pred_npz.parent / "config.json"
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        if "periodic_only" in cfg:
            return bool(cfg["periodic_only"])
    return True


def load_raw_reference(
    raw_dir: str | Path,
    re_ids: Sequence[int],
    *,
    need_field: bool,
    periodic_only: bool = True,
) -> Dict[str, Any]:
    """按给定工况列表加载原始物理真值（保持未归一化原始量纲）。"""
    proc_dir = Path(raw_dir)
    cases = load_base_surface_cases(proc_dir, re_list=re_ids)

    re_list, st_list, phase_list, re_id_list = [], [], [], []
    for re_val in re_ids:
        c = cases[int(re_val)]
        mask = c.periodic_mask > 0 if periodic_only else np.ones(len(c.phase), dtype=bool)
        n = int(np.sum(mask))
        re_list.append(np.full(n, c.re, dtype=np.float64))
        st_list.append(np.full(n, c.st, dtype=np.float64))
        phase_list.append(c.phase[mask])
        re_id_list.append(np.full(n, int(re_val), dtype=np.int64))

    out: Dict[str, Any] = {
        "re": np.concatenate(re_list, axis=0),
        "st": np.concatenate(st_list, axis=0),
        "phase": np.concatenate(phase_list, axis=0),
        "re_id": np.concatenate(re_id_list, axis=0),
        "L": float(cases[int(re_ids[0])].L),
    }

    if need_field:
        mesh = load_ppinn_mesh(proc_dir)
        re_files = _discover_re_files(proc_dir)
        uvp_list = []
        for re_val in re_ids:
            with h5py.File(re_files[int(re_val)], "r") as h5:
                pm = np.asarray(h5["/snapshots/periodic_mask"][()], dtype=np.int32)
                mask = pm > 0 if periodic_only else np.ones(len(pm), dtype=bool)
                uvp_list.append(np.asarray(h5["/field/uvp"][()], dtype=np.float32)[mask])
        out["uvp"] = np.concatenate(uvp_list, axis=0)
        out["xy"] = mesh.roi_xy
        out["L"] = float(mesh.L)

    return out


def _encoder_subset_metrics(
    re_true: np.ndarray,
    re_pred: np.ndarray,
    st_true: np.ndarray,
    st_pred: np.ndarray,
    phase_true: np.ndarray,
    phase_pred: np.ndarray,
) -> Dict[str, float]:
    """计算单个子集上的编码器物理误差（Re/St 回归误差 + 相位圆周误差）。"""
    re_m = compute_regression_metrics(re_true, re_pred)
    st_m = compute_regression_metrics(st_true, st_pred)
    ph_m = compute_circular_metrics(phase_true, phase_pred)

    eps = 1e-12
    re_rel_l2 = math.sqrt(float(np.sum((re_pred - re_true) ** 2)) / (float(np.sum(re_true ** 2)) + eps))
    st_rel_l2 = math.sqrt(float(np.sum((st_pred - st_true) ** 2)) / (float(np.sum(st_true ** 2)) + eps))
    re_mape = float(np.mean(np.abs(re_pred - re_true) / (np.abs(re_true) + eps)))
    st_mape = float(np.mean(np.abs(st_pred - st_true) / (np.abs(st_true) + eps)))

    return {
        "n_samples": int(len(re_true)),
        "re_mae": re_m["mae"],
        "re_rmse": re_m["rmse"],
        "re_rel_l2": float(re_rel_l2),
        "re_mape": re_mape,
        "st_mae": st_m["mae"],
        "st_rmse": st_m["rmse"],
        "st_rel_l2": float(st_rel_l2),
        "st_mape": st_mape,
        "phase_cmae": ph_m["circular_mae"],
        "phase_crmse": ph_m["circular_rmse"],
    }


def evaluate_encoder_predictions(
    re_true: np.ndarray,
    re_pred: np.ndarray,
    st_true: np.ndarray,
    st_pred: np.ndarray,
    phase_true: np.ndarray,
    phase_pred: np.ndarray,
    re_id: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """计算编码器总体与分工况物理预测误差。"""
    re_t = np.asarray(re_true, dtype=np.float64).reshape(-1)
    re_p = np.asarray(re_pred, dtype=np.float64).reshape(-1)
    st_t = np.asarray(st_true, dtype=np.float64).reshape(-1)
    st_p = np.asarray(st_pred, dtype=np.float64).reshape(-1)
    ph_t = np.asarray(phase_true, dtype=np.float64).reshape(-1)
    ph_p = np.asarray(phase_pred, dtype=np.float64).reshape(-1)

    result: Dict[str, Any] = {
        "overall": _encoder_subset_metrics(re_t, re_p, st_t, st_p, ph_t, ph_p),
    }
    if re_id is not None:
        rid = np.asarray(re_id, dtype=np.int64).reshape(-1)
        by_re: Dict[str, Dict[str, float]] = {}
        for r in sorted(int(v) for v in np.unique(rid).tolist()):
            m = rid == r
            by_re[str(r)] = _encoder_subset_metrics(re_t[m], re_p[m], st_t[m], st_p[m], ph_t[m], ph_p[m])
        result["by_re"] = by_re
    return result


def _field_subset_metrics(
    uvp_true: np.ndarray,
    uvp_pred: np.ndarray,
    xy_nd: np.ndarray,
) -> Dict[str, float]:
    """计算全 ROI 与尾流区流场对比指标（含 Gauge 零点对齐压强相对 L2）。"""
    n_frames, n_pts, _ = uvp_true.shape
    eps = 1e-12

    u_t, v_t, p_t = uvp_true[:, :, 0].astype(np.float64), uvp_true[:, :, 1].astype(np.float64), uvp_true[:, :, 2].astype(np.float64)
    u_p, v_p, p_p = uvp_pred[:, :, 0].astype(np.float64), uvp_pred[:, :, 1].astype(np.float64), uvp_pred[:, :, 2].astype(np.float64)

    u_err = u_p - u_t
    v_err = v_p - v_t
    u_rel_l2 = math.sqrt(float(np.sum(u_err * u_err)) / (float(np.sum(u_t * u_t)) + eps))
    v_rel_l2 = math.sqrt(float(np.sum(v_err * v_err)) / (float(np.sum(v_t * v_t)) + eps))

    # 全 ROI 逐帧 Gauge 对齐压强误差累计
    p_gauge = PressureGaugeAccum()
    p_gauge_abs_sum = 0.0
    for i in range(n_frames):
        pe = p_p[i] - p_t[i]
        mean_e = float(np.mean(pe))
        m2_e = float(np.sum((pe - mean_e) ** 2))
        mean_p = float(np.mean(p_t[i]))
        m2_p = float(np.sum((p_t[i] - mean_p) ** 2))
        p_gauge.add_snapshot_welford(mean_e, m2_e, mean_p, m2_p, n_pts)
        p_gauge_abs_sum += float(np.sum(np.abs(pe - mean_e)))

    pgm = p_gauge.metrics(eps=eps)
    p_rel_gsd = pgm["p_rel_l2_gauge_sameden"]
    total_pts = max(n_frames * n_pts, 1)

    metrics: Dict[str, float] = {
        "n_frames": int(n_frames),
        "n_points": int(n_pts),
        "u_rel_l2": float(u_rel_l2),
        "v_rel_l2": float(v_rel_l2),
        "p_rel_l2_raw": pgm["p_rel_l2_raw"],
        "p_rel_l2_gauge_sameden": p_rel_gsd,
        "p_rel_l2_gauge_fluctden": pgm["p_rel_l2_gauge_fluctden"],
        "p_gauge_offset_mean_abs": pgm["p_gauge_offset_mean_abs"],
        "overall_rel_l2_gauge_sameden": float((u_rel_l2 + v_rel_l2 + p_rel_gsd) / 3.0),
        "u_mae": float(np.mean(np.abs(u_err))),
        "u_rmse": float(math.sqrt(float(np.mean(u_err * u_err)))),
        "v_mae": float(np.mean(np.abs(v_err))),
        "v_rmse": float(math.sqrt(float(np.mean(v_err * v_err)))),
        "p_gauge_mae": float(p_gauge_abs_sum / total_pts),
        "p_gauge_rmse": float(math.sqrt(p_gauge.gauge_err_sse / total_pts)),
    }

    # 鱼体标准尾流参考区：x/L ∈ [1.0, 4.5], |y/L| ≤ 0.4
    wake = (xy_nd[:, 0] >= 1.0) & (xy_nd[:, 0] <= 4.5) & (np.abs(xy_nd[:, 1]) <= 0.4)
    n_wake = int(np.sum(wake))
    if n_wake > 0:
        wu_t, wv_t, wp_t = u_t[:, wake], v_t[:, wake], p_t[:, wake]
        wu_p, wv_p, wp_p = u_p[:, wake], v_p[:, wake], p_p[:, wake]
        wu_rel = math.sqrt(float(np.sum((wu_p - wu_t) ** 2)) / (float(np.sum(wu_t * wu_t)) + eps))
        wv_rel = math.sqrt(float(np.sum((wv_p - wv_t) ** 2)) / (float(np.sum(wv_t * wv_t)) + eps))

        w_gauge = PressureGaugeAccum()
        for i in range(n_frames):
            wpe = wp_p[i] - wp_t[i]
            w_me = float(np.mean(wpe))
            w_m2e = float(np.sum((wpe - w_me) ** 2))
            w_mp = float(np.mean(wp_t[i]))
            w_m2p = float(np.sum((wp_t[i] - w_mp) ** 2))
            w_gauge.add_snapshot_welford(w_me, w_m2e, w_mp, w_m2p, n_wake)
        wp_rel_gsd = w_gauge.metrics(eps=eps)["p_rel_l2_gauge_sameden"]
        metrics.update(
            {
                "wake_n_points": n_wake,
                "wake_u_rel_l2": float(wu_rel),
                "wake_v_rel_l2": float(wv_rel),
                "wake_p_rel_l2_gauge_sameden": wp_rel_gsd,
                "wake_overall_rel_l2_gauge_sameden": float((wu_rel + wv_rel + wp_rel_gsd) / 3.0),
            }
        )

    return metrics


def evaluate_field_predictions(
    uvp_true: np.ndarray,
    uvp_pred: np.ndarray,
    xy_phys: np.ndarray,
    *,
    L: float = 0.2,
    re_id: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """计算流场全 ROI 与尾流区物理对比指标（支持分工况统计）。"""
    xy_nd = np.asarray(xy_phys, dtype=np.float64) / float(L)
    result: Dict[str, Any] = {
        "overall": _field_subset_metrics(uvp_true, uvp_pred, xy_nd),
    }
    if re_id is not None:
        rid = np.asarray(re_id, dtype=np.int64).reshape(-1)
        by_re: Dict[str, Dict[str, float]] = {}
        for r in sorted(int(v) for v in np.unique(rid).tolist()):
            m = rid == r
            by_re[str(r)] = _field_subset_metrics(uvp_true[m], uvp_pred[m], xy_nd)
        result["by_re"] = by_re
    return result


def plot_encoder_comparison(
    out_path: str | Path,
    re_true: np.ndarray,
    re_pred: np.ndarray,
    st_true: np.ndarray,
    st_pred: np.ndarray,
    phase_true: np.ndarray,
    phase_pred: np.ndarray,
) -> Path:
    """绘制编码器 Re / St / 相位角真值与预测值对比图。"""
    idx = np.arange(len(re_true))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    pairs = [
        (axes[0], re_true, re_pred, "Re"),
        (axes[1], st_true, st_pred, "St"),
        (axes[2], phase_true, phase_pred, "Phase (rad)"),
    ]
    for ax, yt, yp, label in pairs:
        ax.plot(idx, yt, label="Ground Truth", linewidth=1.6, color="#1f77b4")
        ax.plot(idx, yp, label="Prediction", linewidth=1.2, linestyle="--", color="#d62728")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

    axes[2].set_xlabel("Sample Index")
    fig.suptitle("Encoder Physical Prediction vs Ground Truth", fontsize=12)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    return out_file


def plot_field_comparison(
    out_path: str | Path,
    xy_phys: np.ndarray,
    uvp_true_frame: np.ndarray,
    uvp_pred_frame: np.ndarray,
    *,
    L: float = 0.2,
    title: str = "",
) -> Path:
    """绘制单帧流场 (Ux, Uy, Gauge-aligned p_kin) 真值 / 预测 / 绝对误差 3x3 对比云图。"""
    xy_nd = np.asarray(xy_phys, dtype=np.float64) / float(L)
    tri = mtri.Triangulation(xy_nd[:, 0], xy_nd[:, 1])
    cx = xy_nd[tri.triangles, 0].mean(axis=1)
    cy = xy_nd[tri.triangles, 1].mean(axis=1)
    tri.set_mask(points_inside_fish(cx, cy))

    # 鱼体闭合轮廓用于叠加遮罩
    s_line = np.linspace(0.0, 1.0, 200)
    h_line = fish_half_width_nd(s_line)
    fish_x = np.concatenate([s_line, s_line[::-1]])
    fish_y = np.concatenate([h_line, -h_line[::-1]])

    # 压强做单帧均值零点对齐后再可视化，消除任意常数偏置
    true_f = np.asarray(uvp_true_frame, dtype=np.float64)
    pred_f = np.asarray(uvp_pred_frame, dtype=np.float64).copy()
    pred_f[:, 2] -= float(np.mean(pred_f[:, 2] - true_f[:, 2]))

    fields = [
        ("Ux (m/s)", true_f[:, 0], pred_f[:, 0]),
        ("Uy (m/s)", true_f[:, 1], pred_f[:, 1]),
        ("p_kin gauge (m^2/s^2)", true_f[:, 2], pred_f[:, 2]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 7.5), sharex=True, sharey=True, constrained_layout=True)
    col_titles = ("Ground Truth", "Prediction", "|Error|")

    for row, (name, vt, vp) in enumerate(fields):
        err = np.abs(vp - vt)
        vmin = float(min(np.min(vt), np.min(vp)))
        vmax = float(max(np.max(vt), np.max(vp)))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6
        levels_val = np.linspace(vmin, vmax, 41)

        emax = float(np.max(err))
        if emax < 1e-12:
            emax = 1e-6
        levels_err = np.linspace(0.0, emax, 41)

        for col, (vals, cmap, lv) in enumerate(
            [
                (vt, "RdBu_r", levels_val),
                (vp, "RdBu_r", levels_val),
                (err, "inferno", levels_err),
            ]
        ):
            ax = axes[row, col]
            cf = ax.tricontourf(tri, vals, levels=lv, cmap=cmap, extend="both")
            ax.fill(fish_x, fish_y, facecolor="#d9d9d9", edgecolor="#222222", linewidth=0.8, zorder=5)
            ax.set_aspect("equal", adjustable="box")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{name}\ny/L", fontsize=9)
            if row == 2:
                ax.set_xlabel("x/L", fontsize=9)
            fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)

    if title:
        fig.suptitle(title, fontsize=12)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)
    return out_file


def run_postprocess(
    mode: str,
    pred_path: str | Path,
    raw_dir: str | Path = "data/processed",
    output_dir: Optional[str | Path] = None,
    *,
    re_list: Optional[Sequence[int]] = None,
    periodic_only: Optional[bool] = None,
    frame_idx: int = 0,
    plot: bool = True,
) -> Dict[str, Any]:
    """统一后处理入口：按推理模式加载预测与真值，按需计算编码器/流场指标并绘制对比云图。"""
    mode_s = mode.strip().lower()
    pred_npz = _resolve_pred_npz(pred_path, mode_s)
    out_dir = Path(output_dir) if output_dir is not None else pred_npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    p_only = _read_periodic_flag(pred_npz, periodic_only)

    with np.load(pred_npz) as z:
        pred_data = {k: z[k] for k in z.files}

    pred_re_id = np.asarray(pred_data["re_id"], dtype=np.int64).reshape(-1)
    if re_list is not None:
        target_res = sorted(int(r) for r in re_list)
        sel = np.isin(pred_re_id, target_res)
        pred_re_id = pred_re_id[sel]
    else:
        target_res = sorted(int(r) for r in np.unique(pred_re_id).tolist())
        sel = np.ones(len(pred_re_id), dtype=bool)

    raw = load_raw_reference(
        raw_dir,
        target_res,
        need_field=(mode_s in ("decoder", "cascade")),
        periodic_only=p_only,
    )

    report: Dict[str, Any] = {
        "mode": mode_s,
        "pred_path": str(pred_npz),
        "raw_dir": str(raw_dir),
        "re_ids": target_res,
        "periodic_only": p_only,
        "figures": [],
    }

    if mode_s in ("encoder", "cascade"):
        re_p = pred_data["re_pred"][sel]
        st_p = pred_data["st_pred"][sel]
        ph_p = pred_data["phase_pred"][sel]
        report["encoder_metrics"] = evaluate_encoder_predictions(
            raw["re"],
            re_p,
            raw["st"],
            st_p,
            raw["phase"],
            ph_p,
            re_id=pred_re_id,
        )
        if plot:
            fig_enc = plot_encoder_comparison(
                out_dir / "encoder_comparison.png",
                raw["re"],
                re_p,
                raw["st"],
                st_p,
                raw["phase"],
                ph_p,
            )
            report["figures"].append(str(fig_enc))

    if mode_s in ("decoder", "cascade"):
        uvp_p = pred_data["uvp_pred"][sel]
        report["field_metrics"] = evaluate_field_predictions(
            raw["uvp"],
            uvp_p,
            raw["xy"],
            L=raw["L"],
            re_id=pred_re_id,
        )
        if plot:
            for r in target_res:
                idx_r = np.flatnonzero(pred_re_id == r)
                local_i = int(idx_r[min(int(frame_idx), len(idx_r) - 1)])
                fig_field = plot_field_comparison(
                    out_dir / f"field_comparison_Re{r}_f{int(frame_idx)}.png",
                    raw["xy"],
                    raw["uvp"][local_i],
                    uvp_p[local_i],
                    L=raw["L"],
                    title=f"{mode_s.upper()} | Re={r} | Frame={int(frame_idx)}",
                )
                report["figures"].append(str(fig_field))

    write_json(out_dir / "metrics.json", report)
    return report
