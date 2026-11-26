"""文件一：解析鱼体 Fluent 网格，确定核心密集 ROI 区域与候选传感器并生成尺寸对比图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
import numpy as np


def parse_fluent_msh(msh_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """解析 Fluent ASCII .msh 文件中的节点、单元有序顶点、质心与壁面所属单元索引。"""
    nodes_list: list[list[float]] = []
    # 记录每个单元的4条边：(n0, n1)，节点转为 0-based
    cell_edges: list[list[tuple[int, int]]] = []
    wall_owner_list: list[int] = []
    n_cells = 0

    with msh_path.open("r", encoding="utf-8") as f:
        mode: str | None = None
        cur_zone = -1
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 读取总单元数声明：(12 (0 1 ea90 0))
            if line.startswith("(12 (0 "):
                tokens = line.replace("(", " ").replace(")", " ").split()
                n_cells = int(tokens[3], 16)
                cell_edges = [[] for _ in range(n_cells)]
                continue

            # 进入节点坐标段：(10 (1 1 edaa 1 2)(
            if line.startswith("(10 (1 ") and line.endswith("("):
                mode = "nodes"
                continue

            # 进入面段：(13 (zone_id start end bc_type face_type)(
            if line.startswith("(13 (") and line.endswith("("):
                tokens = line.replace("(", " ").replace(")", " ").split()
                cur_zone = int(tokens[1], 16)
                mode = "faces"
                continue

            if line.startswith(")"):
                mode = None
                continue

            if mode == "nodes":
                x_str, y_str = line.split()[:2]
                nodes_list.append([float(x_str), float(y_str)])
            elif mode == "faces":
                vals = [int(v, 16) for v in line.split()]
                n0, n1, c0, c1 = vals[0] - 1, vals[1] - 1, vals[2] - 1, vals[3] - 1
                if c0 >= 0:
                    cell_edges[c0].append((n0, n1))
                if c1 >= 0:
                    cell_edges[c1].append((n0, n1))
                # zone 8 为 fish_wall 边界，记录其邻接流体单元 c0
                if cur_zone == 8:
                    wall_owner_list.append(c0)

    nodes_msh = np.asarray(nodes_list, dtype=np.float64)
    wall_owner = np.asarray(wall_owner_list, dtype=np.int64)

    # 按边连通顺序重构每个四边形单元的 4 个顶点，并保证逆时针顺序
    cell_verts_idx = np.empty((n_cells, 4), dtype=np.int64)
    for cid, edges in enumerate(cell_edges):
        adj: dict[int, list[int]] = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        v0 = edges[0][0]
        v1 = adj[v0][0]
        v2 = adj[v1][0] if adj[v1][0] != v0 else adj[v1][1]
        v3 = adj[v2][0] if adj[v2][0] != v1 else adj[v2][1]
        cell_verts_idx[cid] = (v0, v1, v2, v3)

    verts_msh = nodes_msh[cell_verts_idx]  # [N_cell, 4, 2]

    # 使用 Shoelace 公式计算面积加权有向面积，若局部顶点为顺时针则翻转统一为逆时针
    x = verts_msh[:, :, 0]
    y = verts_msh[:, :, 1]
    signed_area = 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)
    cw_mask = signed_area < 0
    if np.any(cw_mask):
        verts_msh[cw_mask] = verts_msh[cw_mask, ::-1, :]

    # 统一逆时针顶点顺序后，再计算精确的面积加权单元质心
    x = verts_msh[:, :, 0]
    y = verts_msh[:, :, 1]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * cross.sum(axis=1)
    cx = ((x + x_next) * cross).sum(axis=1) / (6.0 * signed_area)
    cy = ((y + y_next) * cross).sum(axis=1) / (6.0 * signed_area)
    cell_xy_msh = np.stack([cx, cy], axis=1)

    return nodes_msh, verts_msh, cell_xy_msh, wall_owner


def select_arc_length_sensors(
    wall_xy_nd: np.ndarray,
    wall_owner: np.ndarray,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """沿鱼体闭合轮廓按逆时针归一化弧长等距抽取候选近壁传感器。"""
    # 下表面从头到尾 (x 递增) + 上表面从尾到头 (x 递减)，组成完整逆时针闭合环
    lower_idx = np.where(wall_xy_nd[:, 1] < 0.0)[0]
    upper_idx = np.where(wall_xy_nd[:, 1] >= 0.0)[0]
    lower_sorted = lower_idx[np.argsort(wall_xy_nd[lower_idx, 0])]
    upper_sorted = upper_idx[np.argsort(-wall_xy_nd[upper_idx, 0])]
    loop_order = np.concatenate([lower_sorted, upper_sorted])

    loop_xy = wall_xy_nd[loop_order]
    diffs = np.roll(loop_xy, -1, axis=0) - loop_xy
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_arc = np.concatenate([[0.0], np.cumsum(seg_lens[:-1])])
    tau = cum_arc / seg_lens.sum()

    # 按目标弧长逐点匹配最近的未占用壁面单元
    target_tau = np.linspace(0.0, 1.0, n_candidates, endpoint=False)
    used = np.zeros(len(loop_order), dtype=bool)
    picked_loop_pos = np.empty(n_candidates, dtype=np.int64)
    for i, t in enumerate(target_tau):
        dist = np.abs(tau - t)
        dist = np.minimum(dist, 1.0 - dist)
        dist[used] = np.inf
        best = int(np.argmin(dist))
        used[best] = True
        picked_loop_pos[i] = best

    wall_cand_indices = loop_order[picked_loop_pos]
    sensor_cell_ids = wall_owner[wall_cand_indices]
    sensor_xy_nd = wall_xy_nd[wall_cand_indices]
    sensor_arc = tau[picked_loop_pos]
    return wall_cand_indices, sensor_cell_ids, sensor_xy_nd, sensor_arc


def plot_mesh_and_roi(
    verts_nd: np.ndarray,
    cell_xy_nd: np.ndarray,
    roi_mask: np.ndarray,
    roi_bounds_nd: tuple[float, float, float, float],
    sensor_xy_nd: np.ndarray,
    L_m: float,
    out_png: Path,
) -> None:
    """绘制全计算域 .msh 网格与核心密集 ROI 截取框的尺寸对比图。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    xmin, xmax, ymin, ymax = roi_bounds_nd
    full_xmin, full_ymin = cell_xy_nd.min(axis=0)
    full_xmax, full_ymax = cell_xy_nd.max(axis=0)

    # 计算每个单元的特征边长 sqrt(面积)，用于着色对比网格疏密分布
    dx_box = verts_nd[:, :, 0].max(axis=1) - verts_nd[:, :, 0].min(axis=1)
    dy_box = verts_nd[:, :, 1].max(axis=1) - verts_nd[:, :, 1].min(axis=1)
    cell_size_nd = np.sqrt(dx_box * dy_box)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.2), gridspec_kw={"height_ratios": [1.0, 1.15]})

    # 上图：全计算域网格尺寸分布 + 核心密集 ROI 截取框
    ax0 = axes[0]
    poly_all = PolyCollection(verts_nd, array=np.log10(cell_size_nd), cmap="viridis", edgecolors="none")
    ax0.add_collection(poly_all)
    cbar = fig.colorbar(poly_all, ax=ax0, pad=0.015, fraction=0.025)
    cbar.set_label(r"$\log_{10}(\Delta / L)$ 单元无量纲尺度")

    # 标出待截取的密集 ROI 框
    roi_rect = Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=2.0,
        edgecolor="#ff3b30",
        facecolor="none",
        linestyle="-",
        zorder=5,
        label=(
            f"截取密集 ROI 区域: x/L∈[{xmin}, {xmax}], y/L∈[{ymin}, {ymax}] "
            f"({roi_mask.sum():,} 单元, 尺寸 {xmax - xmin:.1f}L × {ymax - ymin:.1f}L)"
        ),
    )
    ax0.add_patch(roi_rect)
    ax0.set_xlim(-4.6, 25.6)
    ax0.set_ylim(-3.1, 3.1)
    ax0.set_aspect("equal")
    ax0.set_title(
        f"原始 Fluent 全域网格（{len(cell_xy_nd):,} 单元，覆盖 x/L∈[-4.5, 25.5], y/L∈[-3.0, 3.0]，尺寸 6.0m × 1.2m）",
        fontsize=11.5,
        pad=8,
    )
    ax0.set_xlabel("x / L")
    ax0.set_ylabel("y / L")
    ax0.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=9.5)

    # 在全域图中标注物理尺寸对比
    ax0.text(
        13.5,
        -2.55,
        f"全域尺寸: 30.0L × 6.0L ({30.0 * L_m:.1f}m × {6.0 * L_m:.1f}m), {len(cell_xy_nd):,} 单元\n"
        f"核心 ROI: {(xmax - xmin):.1f}L × {(ymax - ymin):.1f}L ({(xmax - xmin) * L_m:.2f}m × {(ymax - ymin) * L_m:.2f}m), "
        f"{roi_mask.sum():,} 单元 ({roi_mask.mean() * 100:.1f}%)",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    # 下图：ROI 局部放大视图（展示真实四边形网格、标准尾迹区及 60 个近壁候选传感器）
    ax1 = axes[1]
    poly_roi = PolyCollection(
        verts_nd[roi_mask],
        facecolors="#f3f7fb",
        edgecolors="#4c78a8",
        linewidths=0.22,
        zorder=1,
    )
    ax1.add_collection(poly_roi)

    # 标出 cycle_quality.json 的标准尾迹参考区 (1 <= x/L <= 4.5, |y/L| <= 0.4)
    wake_rect = Rectangle(
        (1.0, -0.4),
        3.5,
        0.8,
        linewidth=1.6,
        edgecolor="#2ca02c",
        facecolor="none",
        linestyle="--",
        zorder=4,
        label="标准尾迹评估区: x/L∈[1.0, 4.5], |y/L|≤0.4",
    )
    ax1.add_patch(wake_rect)

    # 绘制候选传感器位置
    ax1.scatter(
        sensor_xy_nd[:, 0],
        sensor_xy_nd[:, 1],
        s=16,
        c="#ff3b30",
        edgecolors="white",
        linewidths=0.4,
        zorder=6,
        label=f"近壁等弧长候选传感器 ({len(sensor_xy_nd)} 个)",
    )

    ax1.set_xlim(xmin - 0.1, xmax + 0.1)
    ax1.set_ylim(ymin - 0.08, ymax + 0.08)
    ax1.set_aspect("equal")
    ax1.set_title(
        f"截取核心密集 ROI 细节展示（物理范围 x∈[{xmin * L_m:.2f}, {xmax * L_m:.2f}]m, "
        f"y∈[{ymin * L_m:.2f}, {ymax * L_m:.2f}]m；单元尺度中位数 {np.median(cell_size_nd[roi_mask]):.4f}L）",
        fontsize=11.5,
        pad=8,
    )
    ax1.set_xlabel("x / L")
    ax1.set_ylabel("y / L")
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=9.5)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_mesh_and_params(
    out_h5: Path,
    out_json: Path,
    nodes_m: np.ndarray,
    cell_xy_m: np.ndarray,
    cell_xy_nd: np.ndarray,
    roi_cell_ids: np.ndarray,
    roi_mask: np.ndarray,
    roi_bounds_nd: tuple[float, float, float, float],
    origin_msh: tuple[float, float],
    L_m: float,
    wall_owner: np.ndarray,
    wall_cand_indices: np.ndarray,
    sensor_cell_ids: np.ndarray,
    sensor_xy_nd: np.ndarray,
    sensor_arc: np.ndarray,
) -> None:
    """保存 mesh.h5（供下游训练加载）与截取参数 JSON（供文件二循环裁剪 npz 使用）。"""
    xmin, xmax, ymin, ymax = roi_bounds_nd
    ox, oy = origin_msh

    # 1. 保存供文件二和配置参考的截取参数 JSON
    params = {
        "L_m": L_m,
        "origin_msh_m": [ox, oy],
        "roi_bounds_nd": [xmin, xmax, ymin, ymax],
        "roi_bounds_local_m": [xmin * L_m, xmax * L_m, ymin * L_m, ymax * L_m],
        "roi_bounds_msh_m": [ox + xmin * L_m, ox + xmax * L_m, oy + ymin * L_m, oy + ymax * L_m],
        "n_cells_full": int(cell_xy_nd.shape[0]),
        "n_cells_roi": int(roi_cell_ids.shape[0]),
        "n_wall_faces": int(wall_owner.shape[0]),
        "n_candidates": int(sensor_cell_ids.shape[0]),
        "roi_cell_ids": roi_cell_ids.tolist(),
        "wall_cand_indices": wall_cand_indices.tolist(),
        "sensor_cell_ids": sensor_cell_ids.tolist(),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. 保存标准 HDF5 网格文件 mesh.h5
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as h5:
        h5.create_dataset("/mesh/roi_xy", data=cell_xy_m[roi_cell_ids], dtype=np.float64)
        h5.create_dataset("/mesh/roi_xy_nd", data=cell_xy_nd[roi_cell_ids], dtype=np.float64)
        h5.create_dataset("/mesh/cell_xy_full", data=cell_xy_m, dtype=np.float64)
        h5.create_dataset("/mesh/node_xy_full", data=nodes_m, dtype=np.float64)
        h5.create_dataset("/mesh/roi_cell_ids", data=roi_cell_ids, dtype=np.int32)
        h5.create_dataset("/mesh/roi_mask", data=roi_mask.astype(np.int32), dtype=np.int32)

        h5.create_dataset("/sensors/sensor_xy_nd", data=sensor_xy_nd, dtype=np.float64)
        h5.create_dataset("/sensors/sensor_xy_actual", data=sensor_xy_nd * L_m, dtype=np.float64)
        h5.create_dataset("/sensors/sensor_arc", data=sensor_arc, dtype=np.float64)
        h5.create_dataset("/sensors/sensor_cell_id", data=sensor_cell_ids, dtype=np.int32)
        h5.create_dataset("/sensors/wall_owner_full", data=wall_owner, dtype=np.int32)
        h5.create_dataset("/sensors/wall_cand_indices", data=wall_cand_indices, dtype=np.int32)

        h5.create_dataset("/geometry/L", data=np.array([L_m], dtype=np.float64))
        h5.create_dataset("/geometry/fish_head_msh", data=np.array([ox, oy], dtype=np.float64))
        h5.create_dataset("/config/roi_bounds_nd", data=np.array(roi_bounds_nd, dtype=np.float64))
        h5.create_dataset("/config/n_candidate_sensor", data=np.array([len(sensor_cell_ids)], dtype=np.int32))


def main() -> None:
    parser = argparse.ArgumentParser(description="提取鱼体网格密集 ROI 区域、保存参数并绘制尺寸对比图")
    parser.add_argument("--msh", type=Path, default=Path("msh/fish_fluent.msh"), help="输入 Fluent .msh 文件路径")
    parser.add_argument("--out-h5", type=Path, default=Path("data/processed/mesh.h5"), help="输出 mesh.h5 路径")
    parser.add_argument("--out-json", type=Path, default=Path("configs/mesh_roi_params.json"), help="输出截取参数 JSON 路径")
    parser.add_argument("--out-fig", type=Path, default=Path("docs/mesh_roi_overview.png"), help="输出对比图路径")
    parser.add_argument(
        "--roi-bounds-nd",
        type=float,
        nargs=4,
        default=[-0.5, 4.5, -0.8, 0.8],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="无量纲 ROI 截取框 [x/L_min, x/L_max, y/L_min, y/L_max]",
    )
    parser.add_argument("--n-candidates", type=int, default=60, help="近壁等弧长候选传感器数量")
    args = parser.parse_args()

    L_m = 0.2
    origin_msh = (0.9, 0.6)
    roi_bounds_nd = tuple(float(v) for v in args.roi_bounds_nd)

    # 第一步：读取原网格 msh，并转换到以鱼头为原点的局部物理坐标 (m) 和无量纲坐标 (x/L, y/L)
    nodes_msh, verts_msh, cell_xy_msh, wall_owner = parse_fluent_msh(args.msh)
    origin_arr = np.asarray(origin_msh, dtype=np.float64)
    nodes_m = nodes_msh - origin_arr
    cell_xy_m = cell_xy_msh - origin_arr
    verts_nd = (verts_msh - origin_arr) / L_m
    cell_xy_nd = cell_xy_m / L_m
    wall_xy_nd = cell_xy_nd[wall_owner]

    # 第二步：截取中间核心密集区 ROI，并沿鱼壁等弧长抽取候选传感器
    xmin, xmax, ymin, ymax = roi_bounds_nd
    roi_mask = (
        (cell_xy_nd[:, 0] >= xmin)
        & (cell_xy_nd[:, 0] <= xmax)
        & (cell_xy_nd[:, 1] >= ymin)
        & (cell_xy_nd[:, 1] <= ymax)
    )
    roi_cell_ids = np.where(roi_mask)[0].astype(np.int32)

    wall_cand_indices, sensor_cell_ids, sensor_xy_nd, sensor_arc = select_arc_length_sensors(
        wall_xy_nd=wall_xy_nd,
        wall_owner=wall_owner,
        n_candidates=args.n_candidates,
    )

    # 第三步：输出截取坐标与规模统计，并保存参数文件与 mesh.h5
    save_mesh_and_params(
        out_h5=args.out_h5,
        out_json=args.out_json,
        nodes_m=nodes_m,
        cell_xy_m=cell_xy_m,
        cell_xy_nd=cell_xy_nd,
        roi_cell_ids=roi_cell_ids,
        roi_mask=roi_mask,
        roi_bounds_nd=roi_bounds_nd,
        origin_msh=origin_msh,
        L_m=L_m,
        wall_owner=wall_owner,
        wall_cand_indices=wall_cand_indices,
        sensor_cell_ids=sensor_cell_ids,
        sensor_xy_nd=sensor_xy_nd,
        sensor_arc=sensor_arc,
    )

    # 第四步：绘图展示总网格与提取密集块（ROI）的尺寸大小，保存至 docs/
    plot_mesh_and_roi(
        verts_nd=verts_nd,
        cell_xy_nd=cell_xy_nd,
        roi_mask=roi_mask,
        roi_bounds_nd=roi_bounds_nd,
        sensor_xy_nd=sensor_xy_nd,
        L_m=L_m,
        out_png=args.out_fig,
    )

    print(f"网格总单元数: {len(cell_xy_nd)} | 总节点数: {len(nodes_msh)} | 壁面单元数: {len(wall_owner)}")
    print(f"截取 ROI 无量纲坐标 (x/L, y/L): x∈[{xmin}, {xmax}], y∈[{ymin}, {ymax}]")
    print(f"截取 ROI 局部物理坐标 (m): x∈[{xmin * L_m:.3f}, {xmax * L_m:.3f}], y∈[{ymin * L_m:.3f}, {ymax * L_m:.3f}]")
    print(
        f"截取 ROI 原msh坐标 (m): x∈[{origin_msh[0] + xmin * L_m:.3f}, {origin_msh[0] + xmax * L_m:.3f}], "
        f"y∈[{origin_msh[1] + ymin * L_m:.3f}, {origin_msh[1] + ymax * L_m:.3f}]"
    )
    print(f"截取后 ROI 单元数: {len(roi_cell_ids)} ({len(roi_cell_ids) / len(cell_xy_nd) * 100:.2f}%)")
    print(f"候选传感器数: {len(sensor_cell_ids)}")
    print(f"已保存参数: {args.out_json}")
    print(f"已保存网格: {args.out_h5}")
    print(f"已保存图像: {args.out_fig}")


if __name__ == "__main__":
    main()
